"""
Smoothing filters and calibrators for face tracking
"""
import numpy as np
from typing import Optional, List
from config import EMA_ALPHA, LASSO_MAX_VEL, DEADBAND_M, PALM_BASELINE_FRAMES


class PositionSmoother:
    """
    Two-stage filter between raw face position and robot target.

    Stage 1 — EMA (Exponential Moving Average):
        new = alpha * raw + (1 - alpha) * previous
        Acts as a low-pass filter: attenuates detector noise, preserves trend.
        The price is lag — lower alpha = smoother signal but later arrival.

    Stage 2 — Lasso velocity clamp:
        Caps how fast the robot target can move per second.
        Even a perfectly smooth EMA signal can shift faster than we want
        the robot to respond. The lasso makes the commanded position
        "catch up" to the EMA at a bounded speed, producing the characteristic
        delayed-follow feel: the robot trails the face, then smoothly closes in.

    Deadband:
        Movements below DEADBAND_M are skipped entirely.
        Without this, the robot would constantly micro-correct — which causes
        vibration and unnecessary joint wear.

    Reset protocol:
        Always call reset() when transitioning INTO tracking from a stopped state.
        This primes both stages at the current face position so tracking begins
        smoothly rather than lurching from wherever the smoother last was.
    """

    def __init__(self):
        self.ema_pos   = None
        self.lasso_pos = None

    def reset(self, pos: np.ndarray):
        arr = np.array(pos, dtype=float)
        self.ema_pos   = arr.copy()
        self.lasso_pos = arr.copy()

    def update(self, raw_pos: np.ndarray, dt: float) -> np.ndarray:
        raw = np.array(raw_pos, dtype=float)
        dt  = max(dt, 0.001)

        if self.ema_pos is None:
            self.reset(raw)
            return self.lasso_pos.copy()

        # Stage 1: EMA — suppresses high-frequency noise
        self.ema_pos = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * self.ema_pos

        # Stage 2: lasso — caps velocity so robot can't lurch
        delta     = self.ema_pos - self.lasso_pos
        max_delta = LASSO_MAX_VEL * dt
        clamped   = np.clip(delta, -max_delta, max_delta)
        candidate = self.lasso_pos + clamped

        # Deadband: skip micro-movements
        if np.linalg.norm(candidate - self.lasso_pos) > DEADBAND_M:
            self.lasso_pos = candidate

        return self.lasso_pos.copy()


class FaceCalibrator:
    """
    Measures baseline face size at session start, then returns a ratio.

    ratio > 1: face grew → face moved closer → robot follows (Y decreases)
    ratio < 1: face shrank → face further    → robot retreats (Y increases)
    ratio = 1: no change → robot stays at default Y

    Resets on each new tracking session so different operators and distances
    are handled automatically. Inactive when PLANE_MODE = True.
    """

    def __init__(self, n_frames: int = PALM_BASELINE_FRAMES):
        self.n_frames = n_frames
        self.baseline: Optional[float] = None
        self._buf: List[float] = []

    def reset(self):
        self.baseline = None
        self._buf = []

    def update(self, face_size: float) -> float:
        if self.baseline is None:
            self._buf.append(face_size)
            if len(self._buf) >= self.n_frames:
                self.baseline = float(np.mean(self._buf))
                print(f"[Face] Baseline: {self.baseline:.5f}")
            return 1.0
        return face_size / self.baseline if self.baseline > 1e-6 else 1.0
