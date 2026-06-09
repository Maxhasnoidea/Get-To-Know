"""
State machine for robot control logic
"""
import threading
import time
from config import STABILIZE_FRAMES, FREEZE_DURATION


class StateMachine:
    """
    Five states govern robot behavior:

    IDLE        → face detected              → STABILIZING
    STABILIZING → N stable frames            → TRACKING
    STABILIZING → face lost                  → IDLE
    TRACKING    → edge case (lost/multi/jump)→ FROZEN
    FROZEN      → timer expired + face OK    → STABILIZING
    FROZEN      → timer expired, no face     → RETURNING
    RETURNING   → near default pose          → IDLE
    RETURNING   → face detected              → STABILIZING

    All unsafe transitions route through FROZEN so the robot has time to
    settle and the operator has a clear visual cue before motion resumes.
    """

    IDLE        = "IDLE"
    STABILIZING = "STABILIZING"
    TRACKING    = "TRACKING"
    FROZEN      = "FROZEN"
    RETURNING   = "RETURNING"

    def __init__(self):
        self.state            = self.IDLE
        self._freeze_start    = 0.0
        self._stabilize_count = 0
        self._lock            = threading.Lock()

    def get(self) -> str:
        with self._lock:
            return self.state

    def transition(self, new_state: str, reason: str = ""):
        with self._lock:
            if new_state == self.state:
                return
            print(f"[State] {self.state} → {new_state}  ({reason})")
            self.state = new_state
            if new_state == self.FROZEN:
                self._freeze_start = time.time()
            if new_state == self.STABILIZING:
                self._stabilize_count = 0

    def freeze_elapsed(self) -> float:
        return time.time() - self._freeze_start

    def tick_stabilize(self) -> bool:
        """Call once per frame. Returns True when stabilization is complete."""
        with self._lock:
            self._stabilize_count += 1
            return self._stabilize_count >= STABILIZE_FRAMES
