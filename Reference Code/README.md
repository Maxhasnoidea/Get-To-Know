# UR5 Face Tracking + Emotion Recognition

**Status:** ✅ Production Ready

A real-time system for controlling a UR5 robot to track faces and recognize emotions. Uses a Vision Transformer (ViT) for emotion detection with MPS acceleration on Apple Silicon.

## Quick Start

```bash
conda activate face
cd "/Users/macklaus/Desktop/get to know/Reference Code"
python main.py
```

**Then:**
1. Click "▶ Start Camera"
2. Show your face — watch emotions update in real-time
3. (Optional) Click "⚡ Connect & Home Robot" to control UR5

## Features

- 🎥 Real-time face detection (MediaPipe + cvzone)
- 😊 Emotion recognition (ViT fine-tuned on FER+, ~72% accuracy)
- 🤖 Automatic servo tracking (centers face in frame)
- ⚡ MPS acceleration on M-series Mac (CPU fallback)
- 🎮 Tkinter GUI with live status
- 🔄 Async emotion inference — never blocks the camera loop

## System Requirements

- **Python:** 3.10 (conda `face` environment)
- **Camera:** USB webcam (640×480 or higher)
- **Robot:** UR5 with network connectivity (optional)
- **GPU:** MPS on Apple Silicon (CPU fallback works too)

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Tkinter GUI + camera loop + servo control |
| `emotion_service.py` | ViT emotion detector (background thread) |
| `face_detector.py` | MediaPipe face detection wrapper |
| `robot_controller.py` | UR5 RTDE interface |
| `state_machine.py` | Behavior state management |
| `coordinate_mapping.py` | Camera → robot coordinate transform |
| `smoothing.py` | EMA + velocity lasso for smooth motion |
| `config.py` | All tunable parameters |

---

## 🧠 Emotion Detection

### How It Works

1. Main loop captures frames at ~100 FPS
2. Every 50 frames (while TRACKING), face crop is submitted to a background thread
3. Background thread runs ViT inference (~100–300 ms on MPS)
4. Result is available on the next frame — main loop never waits
5. Emotion label + confidence displayed on screen and in GUI panel

### Model

- **Architecture:** Vision Transformer (ViT-base)
- **Source:** `trpakov/vit-face-expression` (Hugging Face)
- **Training data:** FER+ dataset
- **Classes:** angry, disgust, fear, happy, sad, surprise, neutral
- **Confidence threshold:** 0.35 — below this, shows neutral instead of a low-confidence guess

### Output Format

```python
EmotionResult(
    label="happy",
    emoji="😄",
    confidence=0.82   # 0.0 to 1.0
)
```

### Performance

- Inference: ~100–300 ms on MPS (Apple Silicon)
- Runs in background thread — zero impact on camera/tracking FPS
- Updates approximately every 0.5–1 second during TRACKING

---

## 🤖 Robot Control

### Tracking Modes

1. **Servo Tracking (default):** Robot moves to keep face centered in frame
   - Horizontal error → X motion
   - Vertical error → Z motion
   - Adjustable gains: `SERVO_H_GAIN`, `SERVO_V_GAIN`

2. **Plane Mode:** Lock X axis, move only in Y/Z plane
   - Toggle in GUI checkbox
   - Useful for tabletop setups

### State Machine

```
IDLE
  ↓ (face detected)
STABILIZING (15 frames)
  ↓ (stable)
TRACKING          ← emotion detection runs here
  ↕ (face lost / multiple faces / jump > 0.3m)
FROZEN (2 seconds)
  ↓ (timeout)
RETURNING (move to home)
  ↓ (reached)
IDLE
```

### Safety Features

- Workspace bounds enforced
- Position jump detection (sudden moves freeze system)
- Velocity limiting (smooth motion)
- Home position return on disconnect

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Camera
CAM_INDEX = 0            # Camera device index
CAM_W, CAM_H = 640, 480  # Resolution

# Servo tracking
SERVO_TRACKING_ENABLED = True
SERVO_H_GAIN = 0.50      # Horizontal sensitivity
SERVO_V_GAIN = 0.40      # Vertical sensitivity

# State machine
STABILIZE_FRAMES = 15
FREEZE_DURATION = 2.0
JUMP_THRESHOLD_M = 0.3

# Emotion updates
EMOTION_UPDATE_INTERVAL = 50  # Frames between submissions
```

Confidence threshold for emotion (in `emotion_service.py`):

```python
CONFIDENCE_THRESHOLD = 0.35  # Lower = more expressive, Higher = more conservative
```

---

## 🚨 Troubleshooting

### Emotion Svc shows "✗ Failed"

The emotion service failed to load. Check the terminal for the error. Most likely cause: wrong Python environment.

```bash
# Always run with the conda face env
conda activate face
python main.py

# Verify transformers is installed
python -c "import transformers; print(transformers.__version__)"
```

### Emotion always shows Neutral

The model is uncertain about the expression. Try:
- Better lighting on your face
- More exaggerated expressions
- Lower `CONFIDENCE_THRESHOLD` in `emotion_service.py` (e.g. `0.25`)

### Face detection not working

```bash
# Check cvzone
python -c "import cvzone; print(cvzone.__version__)"

# Try different camera index
# Edit CAM_INDEX in config.py (0, 1, 2, etc.)
```

### Robot connection fails

```bash
# Verify ur-rtde installed
pip list | grep ur-rtde

# Check UR5 IP in config.py
python -c "from rtde_control import RTDEControlInterface"
```

### Application crashes on startup

```bash
python main.py 2>&1 | head -50

python -c "from emotion_service import EmotionServiceClient"
python -c "from main import FaceControllerApp"
```

---

## 📊 Performance

| Component | Speed | Notes |
|---|---|---|
| Camera loop | ~100 FPS | Hardware limited |
| Face detection | Every frame | MediaPipe, very fast |
| Emotion inference | ~100–300 ms | Background thread, MPS |
| Emotion UI update | Every ~50 frames | Non-blocking |
| Servo loop | 50 Hz | Separate daemon thread |

---

## 📚 Dependencies

All in the `face` conda environment (Python 3.10):

```
cvzone 1.6.1         # Face detection
mediapipe 0.10.14    # Pose/face backend
opencv-python 4.13   # Computer vision
torch 2.12.0         # PyTorch (MPS support)
torchvision          # Image transforms
transformers         # ViT emotion model
pillow               # Image handling
ur-rtde              # UR robot control (optional)
```

Install missing packages:

```bash
conda activate face
pip install torch torchvision transformers pillow opencv-python cvzone mediapipe
```

---

## 💡 Key Design Decisions

| Decision | Reasoning |
|---|---|
| **ViT over EfficientNet/DeepFace** | No protobuf conflicts; FER+ trained weights generalize better to real webcam faces |
| **Background thread for emotion** | Main camera loop never blocks; emotion updates asynchronously |
| **Confidence threshold** | Prevents low-confidence fear/disgust guesses from dominating display |
| **50-frame emotion interval** | Balances responsiveness with inference cost |
| **State machine** | Safe state transitions prevent erratic robot behavior |
| **EMA + velocity lasso** | Smooth tracking without lag or overshoot |
| **Plane mode** | Simplifies control for tabletop / constrained setups |

---

## 📄 License & Attribution

Uses:
- **cvzone** — Face & hand detection
- **MediaPipe** — Pose tracking backbone
- **trpakov/vit-face-expression** — ViT emotion model (Hugging Face)
- **OpenCV** — Image processing
- **ur-rtde** — Universal Robots communication

---

## ❓ Questions?

Add a print statement to trace execution:

```python
print(f"[Debug] State: {self.state_m.get()}, Face: {face.detected}, Emotion: {self._emotion_display}")
```
