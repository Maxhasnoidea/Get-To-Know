"""
Coordinate system mapping: camera → robot base frame
"""
import numpy as np
from config import (
    WORKSPACE, DEFAULT_POSE, PLANE_LOCKED_Y, PALM_Z_SENSITIVITY,
    SERVO_TRACKING_ENABLED, SERVO_H_GAIN, SERVO_V_GAIN
)


def cam_to_robot(cx: float, cy: float,
                 palm_ratio: float,
                 center_x: float = None,
                 plane_mode: bool = True) -> np.ndarray:
    """
    Map normalized camera coords + palm ratio to robot base frame (meters).

    Camera (after horizontal flip):
        cx: 0 = left,  1 = right
        cy: 0 = top,   1 = bottom

    Y is the depth axis (points toward the operator). It is frozen at
    PLANE_LOCKED_Y in plane mode so the robot moves only in the X/Z plane.

    Plane mode (default):
        cx  -> robot X  (lateral across the workspace)
        cy  -> robot Z  (height)
        Y   -> PLANE_LOCKED_Y, palm ignored

    3D mode:
        cx  -> robot X  (lateral)
        cy  -> robot Z  (height)
        Y   -> driven by palm_ratio via PALM_Z_SENSITIVITY
               palm_ratio > 1 (closer) -> Y decreases (toward operator)
               palm_ratio < 1 (further) -> Y increases (away)
    """
    ws = WORKSPACE
    if center_x is None:
        center_x = DEFAULT_POSE[0]

    robot_x = ws['x'][0] + cx * (ws['x'][1] - ws['x'][0])
    robot_z = ws['z'][1] - cy * (ws['z'][1] - ws['z'][0])

    if plane_mode:
        robot_y = PLANE_LOCKED_Y
    else:
        robot_y = PLANE_LOCKED_Y - (palm_ratio - 1.0) * PALM_Z_SENSITIVITY

    return np.array([robot_x, robot_y, robot_z], dtype=float)


def clamp_workspace(pos: np.ndarray) -> np.ndarray:
    """Hard-clamp to safe workspace. Applied before AND after smoothing."""
    ws = WORKSPACE
    return np.array([
        np.clip(pos[0], *ws['x']),
        np.clip(pos[1], *ws['y']),
        np.clip(pos[2], *ws['z']),
    ])


def servo_track_face(cx: float, cy: float, home_pos: np.ndarray) -> np.ndarray:
    """
    Servo tracking: Calculate robot target to CENTER the face in the camera frame.
    
    This creates a 'look-at' behavior where the robot tries to keep the detected face
    centered in the screen by moving the camera toward or away from the face.
    
    Camera coordinates (normalized to [0,1]):
        cx: 0=left, 0.5=center, 1=right
        cy: 0=top, 0.5=center, 1=bottom
    
    Control logic:
        error_h = cx - 0.5   (horizontal: negative=too left, positive=too right)
        error_v = cy - 0.5   (vertical: negative=too high, positive=too low)
        
        Robot moves to reduce the error and center the face:
        - If face is on the right (cx > 0.5), robot moves right to look at it
        - If face is too low (cy > 0.5), robot moves down to look at it
    
    Args:
        cx, cy: Normalized face center position [0,1]
        home_pos: Home position (reference point where camera looks straight ahead)
    
    Returns:
        Target position [x, y, z] in robot frame (meters)
    """
    if not SERVO_TRACKING_ENABLED:
        return home_pos[:3].copy()
    
    # Calculate screen center error
    error_h = cx - 0.5  # Range: -0.5 to +0.5
    error_v = cy - 0.5  # Range: -0.5 to +0.5
    
    # Convert normalized error to robot workspace movement
    # Positive error_h (face right) → increase X (move right)
    # Positive error_v (face down) → increase Z (move down)
    delta_x = error_h * SERVO_H_GAIN * 2.0  # Scale by 2 to cover full range
    delta_z = error_v * SERVO_V_GAIN * 2.0
    
    # Start from home position and apply servo correction
    target = home_pos[:3].copy()
    target[0] += delta_x   # X (lateral)
    target[2] -= delta_z   # Z (height) — inverted because lower cy means move up
    
    return clamp_workspace(target)
