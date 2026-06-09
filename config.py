"""
Configuration and constants for UR5 Hand Controller
"""

# ============================================================================
# ROBOT CONNECTION
# ============================================================================
ROBOT_IP = "127.0.0.1"  # URSim via localhost default; change for physical robot
#ROBOT_IP = "192.168.12.1"
RTDE_FREQUENCY = 50  # Hz — servoL command rate

# ============================================================================
# CAMERA
# ============================================================================
CAM_INDEX = 1  # USB camera (0=built-in, 1=USB, 2=USB if multiple)
CAM_W, CAM_H = 1280, 720  # Display resolution (good balance for GUI + content)

# ============================================================================
# WORKSPACE (absolute bounds in meters)
# ============================================================================
# Center of this box: X=0.025, Y=-0.425, Z=0.325
WORKSPACE = {
    'x': (-0.430,  0.480),
    'y': (-0.750, -0.100),
    'z': ( 0.050,  0.600),
}

# ============================================================================
# HOME CONFIGURATION
# ============================================================================
# Known TCP pose at HOME_JOINTS — from freedrive read.
# [x, y, z, rx, ry, rz]
DEFAULT_POSE    = [+0.081, -0.536, +0.494, +1.051, -1.424, +1.318]
PLANE_LOCKED_Y  = -0.536   # meters — Y frozen here; derived from DEFAULT_POSE[1]

# Homing pose in joint space. Captured by free-driving the robot to a
# desired starting pose and reading getActualQ() on the pendant.
# [base, shoulder, elbow, wrist1, wrist2, wrist3]
HOME_JOINTS = [+1.525, -1.410, +1.198, -2.904, -1.558, +1.809]  # rad

# Homing motion parameters — kept conservative.
HOME_JOINT_VELOCITY     = 0.5    # rad/s
HOME_JOINT_ACCELERATION = 0.3    # rad/s^2

# ============================================================================
# TRACKING MODE
# ============================================================================
# Plane mode — when True, Y is locked at PLANE_LOCKED_Y (toward operator)
# and the robot tracks only in the X/Z plane. Palm ratio is ignored.
PLANE_MODE = True

# ============================================================================
# PALM CALIBRATION
# ============================================================================
# Palm size → robot Y (depth) mapping — only active when PLANE_MODE = False
PALM_Z_SENSITIVITY   = 0.20   # meters of robot Y travel per unit palm ratio
PALM_BASELINE_FRAMES = 30     # frames averaged to establish initial baseline

# ============================================================================
# SMOOTHING & FILTERING
# ============================================================================
# Exponential Moving Average (EMA) — lower = smoother + more lag
EMA_ALPHA     = 0.10    # good range: 0.08-0.20
LASSO_MAX_VEL = 0.12    # max robot speed (m/s) under lasso constraint
DEADBAND_M    = 0.003   # movements below this (m) are ignored — prevents micro-jitter

# ============================================================================
# EDGE CASE HANDLING
# ============================================================================
FREEZE_DURATION  = 0.5   # seconds robot holds position after edge case triggers
STABILIZE_FRAMES = 5     # frames of stable single-face detection before tracking
JUMP_THRESHOLD_M = 0.40  # position jump larger than this (m) triggers a freeze

# ============================================================================
# ROBOT SERVO PARAMETERS
# ============================================================================
SERVO_VELOCITY     = 0.5    # m/s ceiling (robot profiles internally)
SERVO_ACCELERATION = 0.5    # m/s^2
SERVO_LOOKAHEAD    = 0.15   # seconds — robot path-planning horizon (larger = smoother)
SERVO_GAIN         = 200    # position gain (lower = smoother, higher = stiffer)

# ============================================================================
# FACE DETECTOR PARAMETERS
# ============================================================================
FACE_DETECTION_CONFIDENCE = 0.7
FACE_MAX_FACES = 1

# ============================================================================
# EMOTION DETECTION
# ============================================================================
EMOTION_UPDATE_INTERVAL = 50  # frames between emotion updates while tracking

# ============================================================================
# SERVO TRACKING (Camera Look-at Behavior)
# ============================================================================
# When enabled, robot tries to keep face centered in camera frame by moving the camera
# This creates a "look-at" behavior: robot looks toward the face to center it
SERVO_TRACKING_ENABLED = True
SERVO_H_GAIN = 0.50  # horizontal sensitivity — controls X/Y movement for centering (m per 0.5 screen offset)
SERVO_V_GAIN = 0.40  # vertical sensitivity — controls Z movement for vertical centering (m per 0.5 screen offset)
# Higher gains = more aggressive tracking, lower = smoother but slower to center

# ============================================================================
# RTDE AVAILABILITY CHECK
# ============================================================================
try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("[Warning] ur-rtde not found — running in camera-only mode.")
