"""
Robot controller with RTDE interface
"""
import threading
import time
from typing import Optional
import numpy as np

from config import (
    ROBOT_IP, RTDE_FREQUENCY, HOME_JOINTS, 
    HOME_JOINT_VELOCITY, HOME_JOINT_ACCELERATION,
    DEFAULT_POSE, WORKSPACE, SERVO_VELOCITY, SERVO_ACCELERATION,
    SERVO_LOOKAHEAD, SERVO_GAIN
)

try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("[Warning] ur-rtde not found — running in camera-only mode.")


class RobotController:
    """
    Wraps the RTDE connection. Runs a dedicated servo loop thread.

    Why servoL, not moveL?
    ----------------------
    moveL plans a complete trajectory to a fixed target. It's unsuitable for
    real-time following because each position update would interrupt and restart
    the motion — resulting in jerky stop-start behavior.

    servoL sends incremental pose targets at a fixed rate. The UR controller
    interpolates between them using its own path planner (lookahead_time, gain).
    This produces smooth, continuous motion even at modest command rates.

    Threading:
    ----------
    The servo loop is a daemon thread running at RTDE_FREQUENCY Hz.
    The main thread writes targets via set_target() under a lock.
    The servo thread reads the latest target each cycle. This decouples
    camera frame rate from robot command rate — important because Tkinter's
    after() scheduler is not perfectly timed.

    Startup:
    --------
    home() must be called after connect() and before start(). It uses moveJ
    (joint-space) to reach HOME_JOINTS, then reads FK to store home_pose.
    Only after that do we start the servo loop.
    """

    SAFETY_MODES = {
        1:  "NORMAL",
        2:  "REDUCED",
        3:  "PROTECTIVE_STOP",
        4:  "RECOVERY",
        5:  "SAFEGUARD_STOP",
        6:  "SYSTEM_EMERGENCY_STOP",
        7:  "ROBOT_EMERGENCY_STOP",
        8:  "VIOLATION",
        9:  "FAULT",
        10: "VALIDATE_JOINT_ID",
        11: "UNDEFINED_SAFETY_MODE",
        12: "STOPPED_DUE_TO_SAFETY",
    }
    ROBOT_MODES = {
        -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY",
         2: "BOOTING",        3: "POWER_OFF",    4: "POWER_ON",
         5: "IDLE",           6: "BACKDRIVE",    7: "RUNNING",
         8: "UPDATING_FIRMWARE",
    }

    def __init__(self):
        self.rtde_c    = None
        self.rtde_r    = None
        self.connected = False
        self._target   = list(DEFAULT_POSE)
        self.home_pose = list(DEFAULT_POSE)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

    def connect(self, ip: str = ROBOT_IP) -> bool:
        try:
            self.rtde_c    = rtde_control.RTDEControlInterface(ip)
            self.rtde_r    = rtde_receive.RTDEReceiveInterface(ip)
            self.connected = True
            print(f"[Robot] Connected to {ip}")
            return True
        except Exception as e:
            print(f"[Robot] Connection failed: {e}")
            return False

    def _log_status(self, label: str):
        if not self.rtde_r:
            return
        try:
            q     = self.rtde_r.getActualQ()
            pose  = self.rtde_r.getActualTCPPose()
            smode = self.rtde_r.getSafetyMode()
            rmode = self.rtde_r.getRobotMode()
            prot  = (self.rtde_r.isProtectiveStopped()
                     if hasattr(self.rtde_r, "isProtectiveStopped") else None)
            print(f"[Robot] --- status: {label} ---")
            print(f"  q     = [{', '.join(f'{v:+.3f}' for v in q)}] rad")
            print(f"  TCP   = [{', '.join(f'{v:+.3f}' for v in pose)}]")
            print(f"  safety= {smode} ({self.SAFETY_MODES.get(smode, '?')})")
            print(f"  robot = {rmode} ({self.ROBOT_MODES.get(rmode, '?')})")
            if prot is not None:
                print(f"  protective_stopped = {prot}")
        except Exception as e:
            print(f"[Robot] status read failed: {e}")

    def _safety_ok(self, where: str) -> bool:
        if not self.rtde_r:
            return True
        try:
            if (hasattr(self.rtde_r, "isProtectiveStopped")
                    and self.rtde_r.isProtectiveStopped()):
                print(f"[Robot] PROTECTIVE STOP active ({where})")
                self._log_status(where)
                return False
            smode = self.rtde_r.getSafetyMode()
            if smode not in (1, 2):
                name = self.SAFETY_MODES.get(smode, "?")
                print(f"[Robot] Safety fault: mode={smode} ({name}) at {where}")
                self._log_status(where)
                return False
        except Exception as e:
            print(f"[Robot] safety check failed: {e}")
        return True

    def unlock_protective_stop(self) -> bool:
        if not self.rtde_c:
            return False
        try:
            self.rtde_c.unlockProtectiveStop()
            time.sleep(0.5)
            print("[Robot] Protective stop unlocked.")
            return True
        except Exception as e:
            print(f"[Robot] Unlock failed: {e}")
            return False

    def home(self) -> bool:
        if not self.connected:
            return False

        self._log_status("pre-home")

        try:
            if (hasattr(self.rtde_r, "isProtectiveStopped")
                    and self.rtde_r.isProtectiveStopped()):
                print("[Robot] Clearing pre-existing protective stop")
                self.unlock_protective_stop()
                time.sleep(0.5)
        except Exception:
            pass

        try:
            print(f"[Robot] moveJ → HOME_JOINTS "
                  f"({', '.join(f'{v:+.3f}' for v in HOME_JOINTS)})")
            self.rtde_c.moveJ(HOME_JOINTS,
                              HOME_JOINT_VELOCITY,
                              HOME_JOINT_ACCELERATION)
            if not self._safety_ok("after moveJ to HOME_JOINTS"):
                return False

            actual = self.rtde_r.getActualTCPPose()
            self.home_pose = list(DEFAULT_POSE)   # use known pose, not FK

            print(f"[Robot] FK confirmation: "
                  f"({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})")
            print(f"[Robot] Using hardcoded home_pose: "
                  f"({DEFAULT_POSE[0]:+.3f}, {DEFAULT_POSE[1]:+.3f}, {DEFAULT_POSE[2]:+.3f})")

            print("[Robot] Home pose (FK of HOME_JOINTS):")
            print(f"  position    = "
                  f"({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f}) m")
            print(f"  orientation = "
                  f"({actual[3]:+.3f}, {actual[4]:+.3f}, {actual[5]:+.3f}) rad")
            print("[Robot] Workspace (absolute, fixed):")
            print(f"  x: ({WORKSPACE['x'][0]:+.3f}, {WORKSPACE['x'][1]:+.3f}) m")
            print(f"  y: ({WORKSPACE['y'][0]:+.3f}, {WORKSPACE['y'][1]:+.3f}) m")
            print(f"  z: ({WORKSPACE['z'][0]:+.3f}, {WORKSPACE['z'][1]:+.3f}) m")

            with self._lock:
                self._target = list(actual)

            print("[Robot] Home complete.")
            return True

        except Exception as e:
            print(f"[Robot] Homing failed: {e}")
            self._log_status("at exception")
            return False

    def start(self):
        if not self.connected:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop,
                                         daemon=True, name="rtde-servo")
        self._thread.start()
        print(f"[Robot] Servo loop started at {RTDE_FREQUENCY} Hz")

    def stop(self):
        self._running = False
        if self.rtde_c:
            try:
                self.rtde_c.servoStop()
                time.sleep(0.1)
                self.rtde_c.stopScript()
            except Exception as e:
                print(f"[Robot] Stop error: {e}")

    def set_target(self, xyz: np.ndarray, orientation: Optional[list] = None):
        with self._lock:
            self._target[:3] = xyz.tolist()
            if orientation is not None:
                self._target[3:] = orientation

    def get_actual_pose(self) -> Optional[list]:
        if self.rtde_r:
            try:
                return self.rtde_r.getActualTCPPose()
            except Exception:
                return None
        return None

    def _loop(self):
        """
        Servo loop. Each iteration:
          1. Read latest target (lock held briefly)
          2. Send servoL command
          3. Sleep for the remaining portion of the timestep
        """
        dt = 1.0 / RTDE_FREQUENCY

        while self._running:
            t0 = time.perf_counter()

            try:
                with self._lock:
                    pose = list(self._target)

                self.rtde_c.servoL(pose,
                                   SERVO_VELOCITY,
                                   SERVO_ACCELERATION,
                                   dt,
                                   SERVO_LOOKAHEAD,
                                   SERVO_GAIN)
            except Exception as e:
                print(f"[Robot] servoL error: {e}")
                self._running  = False
                self.connected = False
                break

            elapsed = time.perf_counter() - t0
            remain  = dt - elapsed
            if remain > 0:
                time.sleep(remain)
