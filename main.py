"""
UR5 Face Tracking Controller
=============================
Main Tkinter application for controlling a UR5 robot arm to follow a detected face.

Architecture:
    Camera → face detector → FaceState → StateMachine → PositionSmoother → RobotController
    All robot commands run in a daemon thread at RTDE_FREQUENCY Hz via servoL().
    The main (Tkinter) thread handles camera capture, detection, and UI.

Run:
    python main.py
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional
import tkinter as tk
from PIL import Image, ImageTk

from config import (
    CAM_INDEX, CAM_W, CAM_H, PLANE_MODE, DEFAULT_POSE, 
    PLANE_LOCKED_Y, FREEZE_DURATION, RTDE_AVAILABLE, EMOTION_UPDATE_INTERVAL
)
from models import FaceState
from state_machine import StateMachine
from smoothing import PositionSmoother, FaceCalibrator
from coordinate_mapping import cam_to_robot, clamp_workspace, servo_track_face
from robot_controller import RobotController
from face_detector import FaceDetectorWrapper
from emotion_service import EmotionServiceClient


class FaceControllerApp(tk.Tk):
    """
    Main Tkinter application. The main thread handles:
      - Camera frame capture (via after() non-blocking scheduler)
      - Face detection
      - State machine evaluation
      - Writing smoothed target poses to RobotController

    The robot's servo loop runs independently in a daemon thread.
    """

    def __init__(self):
        super().__init__()
        self.title("UR5 Face Controller")
        self.resizable(False, False)
        self.configure(bg="#1a1a1a")

        self.detector   = FaceDetectorWrapper()
        self.emotion    = EmotionServiceClient()
        self.state_m    = StateMachine()
        self.smoother   = PositionSmoother()
        self.calibrator = FaceCalibrator()
        self.robot      = RobotController()

        self.cap        = None
        self.running    = False
        self._after_id  = None
        self._prev_tick = None
        self._emotion_frame_count = 0
        self._emotion_display = ""
        self._emotion_confidence = 0.0
        self._emotion_svc_ready = False  # Track if emotion service is ready
        self._plane_mode = PLANE_MODE

        # Plane mode as a runtime Tkinter variable
        self._plane_mode_var = tk.BooleanVar(value=PLANE_MODE)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Check emotion service startup in background
        self._check_emotion_service_ready()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the GUI layout."""
        feed = tk.Frame(self, bg="#111", bd=2, relief="sunken")
        feed.grid(row=0, column=0, padx=(16, 8), pady=16)
        self.video_label = tk.Label(feed, bg="#111")
        self.video_label.pack()

        panel = tk.Frame(self, bg="#1a1a1a", width=260)
        panel.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        panel.grid_propagate(False)

        tk.Label(panel, text="UR5 Face Controller",
                 font=("Helvetica", 14, "bold"),
                 bg="#1a1a1a", fg="#e0e0e0").pack(pady=(0, 4))
        tk.Label(panel, text="cvzone · OpenCV · ur-rtde",
                 font=("Helvetica", 9), bg="#1a1a1a", fg="#555").pack(pady=(0, 14))

        self.cam_btn = tk.Button(
            panel, text="▶  Start Camera",
            font=("Helvetica", 11, "bold"), bg="#2ecc71", fg="white",
            activebackground="#27ae60", relief="flat", padx=12, pady=8,
            cursor="hand2", command=self._toggle_camera)
        self.cam_btn.pack(fill="x", pady=(0, 6))

        self.robot_btn = tk.Button(
            panel, text="⚡  Connect & Home Robot",
            font=("Helvetica", 11, "bold"), bg="#3498db", fg="white",
            activebackground="#2980b9", relief="flat", padx=12, pady=8,
            cursor="hand2", command=self._connect_robot)
        self.robot_btn.pack(fill="x", pady=(0, 14))

        # Status section
        sf = tk.Frame(panel, bg="#222", bd=1, relief="sunken")
        sf.pack(fill="x", pady=(0, 8))
        tk.Label(sf, text="STATUS", font=("Courier", 8, "bold"),
                 bg="#222", fg="#555").pack(anchor="w", padx=8, pady=(6, 2))
        self.ui_state = self._row(sf, "State")
        self.ui_faces = self._row(sf, "Faces")
        self.ui_emotion = self._row(sf, "Emotion")
        self.ui_emotion_svc = self._row(sf, "Emotion Svc")
        self.ui_robot = self._row(sf, "Robot")
        self.ui_fps   = self._row(sf, "FPS")

        # Position section
        pf = tk.Frame(panel, bg="#222", bd=1, relief="sunken")
        pf.pack(fill="x", pady=(0, 8))
        tk.Label(pf, text="TARGET POSE", font=("Courier", 8, "bold"),
                 bg="#222", fg="#555").pack(anchor="w", padx=8, pady=(6, 2))
        self.ui_px        = self._row(pf, "X depth")
        self.ui_py        = self._row(pf, "Y lateral")
        self.ui_pz        = self._row(pf, "Z height")
        self.ui_face_size = self._row(pf, "Face size")

        # Plane mode section
        mf = tk.Frame(panel, bg="#222", bd=1, relief="sunken")
        mf.pack(fill="x", pady=(0, 8))
        tk.Label(mf, text="TRACKING MODE", font=("Courier", 8, "bold"),
                 bg="#222", fg="#555").pack(anchor="w", padx=8, pady=(6, 2))

        cb_frame = tk.Frame(mf, bg="#222")
        cb_frame.pack(fill="x", padx=8, pady=(2, 8))
        self._plane_cb = tk.Checkbutton(
            cb_frame,
            text="Plane mode  (Y / Z only, X locked)",
            variable=self._plane_mode_var,
            command=self._on_plane_mode_toggle,
            font=("Helvetica", 9),
            bg="#222", fg="#e0e0e0",
            selectcolor="#333",
            activebackground="#222",
            activeforeground="#e0e0e0",
            cursor="hand2",
        )
        self._plane_cb.pack(anchor="w")
        self.ui_mode_label = tk.Label(
            cb_frame,
            text=self._mode_label_text(),
            font=("Courier", 8),
            bg="#222", fg="#aaa",
        )
        self.ui_mode_label.pack(anchor="w", pady=(2, 0))

        # Face calibration
        tk.Button(
            panel, text="↺  Recalibrate Face",
            font=("Helvetica", 10), bg="#8e44ad", fg="white",
            activebackground="#6c3483", relief="flat", padx=12, pady=6,
            cursor="hand2", command=lambda: self.calibrator.reset()
        ).pack(fill="x", pady=(0, 4))
        tk.Label(panel,
                 text="Recalibrate when face is at\nnominal working distance.\nInactive in plane mode.",
                 font=("Helvetica", 8), bg="#1a1a1a", fg="#444",
                 justify="left").pack(anchor="w")

        self._set_frame(Image.new("RGB", (CAM_W, CAM_H), (20, 20, 20)))

    def _mode_label_text(self) -> str:
        if self._plane_mode_var.get():
            return f"Y locked at {PLANE_LOCKED_Y*1000:.0f} mm  (X/Z plane)"
        return "Face-centering look-at mode"

    def _on_plane_mode_toggle(self):
        """Called when the checkbox is clicked."""
        self._plane_mode = self._plane_mode_var.get()
        self.ui_mode_label.config(text=self._mode_label_text())
        if self.state_m.get() == StateMachine.TRACKING:
            self._reset_smoother()
            self.state_m.transition(StateMachine.FROZEN, "plane mode toggled")
        print(f"[Mode] Plane mode {'ON — X/Z plane' if self._plane_mode else 'OFF — 3D'}")

    def _row(self, parent, label: str) -> tk.StringVar:
        """Create a status row label."""
        row = tk.Frame(parent, bg="#222")
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label + ":", font=("Courier", 8),
                 bg="#222", fg="#666", width=12, anchor="w").pack(side="left")
        var = tk.StringVar(value="—")
        tk.Label(row, textvariable=var, font=("Courier", 8, "bold"),
                 bg="#222", fg="#e0e0e0", anchor="w").pack(side="left")
        return var

    # ── Camera ─────────────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        """Start camera capture with fallback to different indices if needed."""
        # Try the configured camera index first, then fall back to others
        for attempt_idx in [CAM_INDEX, 0, 1, 2]:
            print(f"[Camera] Attempting to open camera index {attempt_idx}...")
            self.cap = cv2.VideoCapture(attempt_idx)
            
            if not self.cap.isOpened():
                print(f"[Camera] Failed to open camera {attempt_idx}")
                continue
            
            # Try to set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            
            # Verify we can read at least one frame
            ok, test_frame = self.cap.read()
            if ok and test_frame is not None:
                actual_w, actual_h = test_frame.shape[1], test_frame.shape[0]
                print(f"[Camera] SUCCESS! Opened camera {attempt_idx} @ {actual_w}x{actual_h}")
                self.running    = True
                self._prev_tick = None
                self._emotion_frame_count = 0
                self._emotion_display = ""
                self.cam_btn.config(text="■  Stop Camera", bg="#e74c3c",
                                    activebackground="#c0392b")
                self._after_id = self.after(0, self._update_frame)
                return
            else:
                print(f"[Camera] Camera {attempt_idx} opened but cannot read frames")
                self.cap.release()
        
        # All attempts failed
        print("[Camera] ERROR: Could not open any camera!")
        self.ui_faces.set("no camera")
        self.cap = None

    def _stop_camera(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
        if self.cap:
            self.cap.release()
        self.cam_btn.config(text="▶  Start Camera", bg="#2ecc71",
                            activebackground="#27ae60")
        self._set_frame(Image.new("RGB", (CAM_W, CAM_H), (20, 20, 20)))

    def _check_emotion_service_ready(self):
        """Check if emotion service is ready, update GUI status."""
        if self.emotion.ready:
            self.ui_emotion_svc.set("✓ PyTorch")
            self._emotion_svc_ready = True
            print("[Main] ✓ Emotion service ready (PyTorch)!", flush=True)
        else:
            self.ui_emotion_svc.set("✗ Failed")

    def _connect_robot(self):
        if not RTDE_AVAILABLE:
            self.ui_robot.set("ur-rtde missing")
            return

        self.ui_robot.set("connecting...")
        self.robot_btn.config(text="…  Connecting", bg="#f39c12",
                              activebackground="#e67e22", state="disabled")

        # Run connection & homing in background thread
        threading.Thread(target=self._connect_and_home_worker,
                         daemon=True, name="connect-home").start()

    def _connect_and_home_worker(self):
        """Connect to robot and home it in background thread."""
        if not self.robot.connect():
            self.after(0, lambda: self._on_connect_failed())
            return

        ok = self.robot.home()
        self.after(0, lambda: self._on_home_complete(ok))

    def _on_connect_failed(self):
        self.ui_robot.set("FAILED")
        self.robot_btn.config(text="✗  Connection Failed", bg="#e74c3c",
                              state="normal")

    def _on_home_complete(self, ok: bool):
        if not ok:
            self.ui_robot.set("home FAILED")
            self.robot_btn.config(text="✗  Home Failed", bg="#e74c3c",
                                  state="normal")
            return
        self.robot.start()
        self.ui_robot.set("connected")
        self.robot_btn.config(text="✓  Robot Ready", bg="#27ae60",
                              state="normal")

    # ── Frame loop ─────────────────────────────────────────────────────────────

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            print("[Camera] Frame read failed, retrying...")
            self._after_id = self.after(10, self._update_frame)
            return

        now = time.perf_counter()
        dt  = min((now - self._prev_tick) if self._prev_tick else 0.033, 0.1)
        self._prev_tick = now

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Detect faces
        faces, frame = self.detector.find_faces(frame, draw=False)
        face = self.detector.parse_faces(faces, w, h)

        # Update robot target
        target = self._tick(face, dt)

        # Update emotion only while tracking, and only every N frames
        current_state = self.state_m.get()
        if current_state == StateMachine.TRACKING:
            self._emotion_frame_count += 1
            if self._emotion_frame_count % EMOTION_UPDATE_INTERVAL == 0:
                # Run emotion detection on first detected face
                if len(faces) > 0 and 'bbox' in faces[0]:
                    bbox = faces[0]['bbox']
                    emotion = self.emotion.detect_emotion_from_frame(frame, bbox)
                    if emotion is not None:
                        self._emotion_display = emotion.display
                        self._emotion_confidence = emotion.confidence
        else:
            self._emotion_frame_count = 0
            self._emotion_display = ""
            self._emotion_confidence = 0.0

        if self.robot.connected and target:
            self.robot.set_target(np.array(target[:3]), target[3:])

        # Draw annotations
        frame = self._draw(frame, face, target)

        # Update UI
        self.ui_state.set(self.state_m.get())
        self.ui_faces.set(str(face.count))
        
        # Emotion display with confidence
        if current_state == StateMachine.TRACKING:
            if self._emotion_display:
                emotion_text = f"{self._emotion_display} ({self._emotion_confidence*100:.0f}%)"
            else:
                emotion_text = "—"
        else:
            emotion_text = "—"
        self.ui_emotion.set(emotion_text)
        
        # Emotion service status
        svc_status = "✓ PyTorch" if self.emotion.ready else "✗ Failed"
        self.ui_emotion_svc.set(svc_status)
        
        self.ui_fps.set(f"{1/dt:.1f}")
        if target:
            self.ui_px.set(f"{target[0]:.3f} m")
            self.ui_py.set(f"{target[1]:.3f} m")
            self.ui_pz.set(f"{target[2]:.3f} m")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._set_frame(img)
        self._after_id = self.after(1, self._update_frame)

    # ── State machine ──────────────────────────────────────────────────────────

    def _tick(self, face: FaceState, dt: float) -> Optional[list]:
        """
        Evaluate the state machine for this frame.
        Returns a full 6-DOF target pose.
        """
        sm  = self.state_m
        cur = sm.get()

        home_xyz    = self.robot.home_pose[:3]
        home_orient = self.robot.home_pose[3:]
        plane_mode  = self._plane_mode

        # ── Edge cases (only checked while actively tracking) ─────────────────

        if cur == StateMachine.TRACKING:
            if not face.detected:
                sm.transition(StateMachine.FROZEN, "face lost")
                self._reset_smoother()
                return self._default_target()

            if face.count > 1:
                sm.transition(StateMachine.FROZEN, "multiple faces")
                self._reset_smoother()
                return self._default_target()

            if self.smoother.lasso_pos is not None:
                raw_xy = cam_to_robot(face.cx, face.cy, 1.0,
                                      center_x=home_xyz[0],
                                      plane_mode=plane_mode)[:2]
                delta  = np.linalg.norm(raw_xy - self.smoother.lasso_pos[:2])
                from config import JUMP_THRESHOLD_M
                if delta > JUMP_THRESHOLD_M:
                    sm.transition(StateMachine.FROZEN, f"jump {delta:.3f} m")
                    self._reset_smoother()
                    return self._default_target()

        # ── State handlers ────────────────────────────────────────────────────

        if cur == StateMachine.FROZEN:
            if sm.freeze_elapsed() >= FREEZE_DURATION:
                if face.detected and face.count == 1:
                    sm.transition(StateMachine.STABILIZING, "freeze done, face present")
                else:
                    sm.transition(StateMachine.RETURNING, "freeze done, no face")
            return self._default_target()

        if cur == StateMachine.RETURNING:
            if face.detected and face.count == 1:
                sm.transition(StateMachine.STABILIZING, "face appeared")
            elif self.smoother.lasso_pos is not None:
                dist = np.linalg.norm(self.smoother.lasso_pos - np.array(home_xyz))
                if dist < 0.01:
                    sm.transition(StateMachine.IDLE, "reached home")
            return self._default_target()

        if cur == StateMachine.IDLE:
            if face.detected and face.count == 1:
                sm.transition(StateMachine.STABILIZING, "face appeared")
                self.calibrator.reset()
            return self._default_target()

        if cur == StateMachine.STABILIZING:
            if not face.detected or face.count > 1:
                sm.transition(StateMachine.IDLE, "lost during stabilize")
                return self._default_target()
            if sm.tick_stabilize():
                init = servo_track_face(face.cx, face.cy, self.robot.home_pose)
                self.smoother.reset(init)
                sm.transition(StateMachine.TRACKING, "stabilized")
            return self._default_target()

        if cur == StateMachine.TRACKING:
            face_ratio = self.calibrator.update(face.face_size)

            if plane_mode:
                self.ui_face_size.set("locked")
            else:
                self.ui_face_size.set(f"{face_ratio:.2f}×")

            raw = servo_track_face(face.cx, face.cy, self.robot.home_pose)
            xyz = clamp_workspace(self.smoother.update(raw, dt))

            return list(xyz) + list(home_orient)

        return self._default_target()

    def _reset_smoother(self):
        if self.smoother.lasso_pos is not None:
            self.smoother.ema_pos = self.smoother.lasso_pos.copy()
        self.calibrator.reset()

    def _default_target(self) -> list:
        home = self.robot.home_pose
        if self.smoother.lasso_pos is not None:
            xyz = clamp_workspace(self.smoother.update(np.array(home[:3]), 0.033))
            return list(xyz) + list(home[3:])
        return list(home)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self, frame, face: FaceState, target: Optional[list]):
        state = self.state_m.get()
        COLORS = {
            StateMachine.TRACKING:    (0, 220, 100),
            StateMachine.STABILIZING: (0, 200, 255),
            StateMachine.FROZEN:      (30, 80, 255),
            StateMachine.RETURNING:   (255, 140, 0),
            StateMachine.IDLE:        (130, 130, 130),
        }
        c = COLORS.get(state, (200, 200, 200))

        if face.detected:
            bx, by, bw, bh = face.bbox
            cx_px = bx + bw // 2
            cy_px = by + bh // 2

            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), c, 2)

            # Draw corner markers
            acc = 16
            for dx, dy in [(0,0),(bw,0),(0,bh),(bw,bh)]:
                ox, oy = bx+dx, by+dy
                sx = 1 if dx == 0 else -1
                sy = 1 if dy == 0 else -1
                cv2.line(frame, (ox, oy), (ox+sx*acc, oy), c, 3)
                cv2.line(frame, (ox, oy), (ox, oy+sy*acc), c, 3)

            # Draw center crosshair
            cv2.circle(frame, (cx_px, cy_px), 5, c, -1)
            cv2.line(frame, (cx_px-14, cy_px), (cx_px+14, cy_px), c, 1)
            cv2.line(frame, (cx_px, cy_px-14), (cx_px, cy_px+14), c, 1)

            # Draw face size depth indicator (if not plane mode)
            face_ratio = (face.face_size / self.calibrator.baseline
                          if self.calibrator.baseline else 1.0)

            if not self._plane_mode:
                bar_w = int(np.clip(face_ratio / 2.0, 0, 1) * bw)
                cv2.rectangle(frame, (bx, by+bh+8), (bx+bw, by+bh+16), (50,50,50), -1)
                cv2.rectangle(frame, (bx, by+bh+8), (bx+bar_w, by+bh+16), (100,180,255), -1)
                cv2.putText(frame, f"face {face_ratio:.2f}x",
                            (bx, by+bh+28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,180,255), 1)
            else:
                cv2.putText(frame, "plane  X/Z",
                            (bx, by+bh+28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,100), 1)

        # Draw state label
        cv2.putText(frame, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, c, 2)

        # Draw mode indicator
        mode_txt = "PLANE X/Z" if self._plane_mode else "3D  X/Y/Z"
        mode_col = (180, 180, 80) if self._plane_mode else (100, 180, 255)
        cv2.putText(frame, mode_txt, (CAM_W - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_col, 2)

        if state == StateMachine.FROZEN:
            t = max(0.0, FREEZE_DURATION - self.state_m.freeze_elapsed())
            cv2.putText(frame, f"hold  {t:.1f}s",
                        (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,120,255), 2)

        if state == StateMachine.TRACKING and self._emotion_display:
            # Show emotion with confidence percentage
            confidence_pct = f" ({self._emotion_confidence*100:.0f}%)" if self._emotion_confidence > 0 else ""
            emotion_text = self._emotion_display + confidence_pct
            cv2.putText(frame, emotion_text,
                        (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (120, 220, 255), 2)
        elif state == StateMachine.TRACKING:
            # Show "Analyzing..." every 50 frames during tracking
            if (self._emotion_frame_count % EMOTION_UPDATE_INTERVAL) in range(0, 10):
                cv2.putText(frame, "🔍 Analyzing emotion...",
                            (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 180, 255), 1)

        if target:
            cv2.putText(frame,
                        f"X{target[0]:.3f}  Y{target[1]:.3f}  Z{target[2]:.3f}",
                        (12, CAM_H-14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (150,150,150), 1)

        return frame

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_frame(self, img: Image.Image):
        tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=tk_img, width=CAM_W, height=CAM_H)
        self.video_label.image = tk_img

    def _on_close(self):
        self._stop_camera()
        self.robot.stop()
        self.emotion.shutdown()  # Gracefully shutdown emotion service
        self.destroy()


if __name__ == "__main__":
    app = FaceControllerApp()
    app.mainloop()
