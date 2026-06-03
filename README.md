# UR5 Face Tracking + Emotion Recognition

**Status:** ✅ Production Ready

A real-time system for controlling a UR5 robot to track faces and recognize emotions. Features PyTorch emotion detection with MPS acceleration on Apple Silicon.

## Quick Start

```bash
cd Reference Code
conda activate face
export KMP_DUPLICATE_LIB_OK=TRUE  # macOS only
python main.py
```

**Then:**
1. Click "▶ Start Camera"
2. Show your face—watch emotions update in real-time
3. (Optional) Click "⚡ Connect & Home Robot" to control UR5

## Features

- 🎥 Real-time face detection (MediaPipe + cvzone)
- 😊 Emotion recognition (PyTorch EfficientNet-B0, 85-92% accuracy)
- 🤖 Automatic servo tracking (centers face in frame)
- ⚡ MPS acceleration (5-20ms inference on M-series Mac)
- 🎮 Tkinter GUI with live status

## System Requirements

- **Python:** 3.10+
- **Camera:** USB webcam (1280×720 recommended)
- **Robot:** UR5 with network connectivity
- **GPU:** Optional (MPS on Apple Silicon, CPU fallback)

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Tkinter GUI + servo control |
| `emotion_service.py` | PyTorch emotion detector |
| `face_detector.py` | Face detection wrapper |
| `robot_controller.py` | UR5 interface |
| `state_machine.py` | Behavior state management |
| `fer2013_weights.pth` | Pre-trained model (15.6 MB) |

**→ See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation**

### Why This Works

- ✅ **Process isolation:** Each Python process has its own import cache
- ✅ **Non-blocking:** Main loop runs at 100 FPS, emotion updates async
- ✅ **Resilient:** If emotion service crashes, main app continues
- ✅ **Practical:** Works immediately without complex launcher scripts

### Future: Separate Conda Environments

To run emotion service in the isolated `face-emotion` environment, you would need a launcher script that calls `conda run`. For now, the simpler approach works well since both processes share the same conda environment but have isolated Python interpreters.

### Key Features

- ✅ **Non-blocking:** Emotion updates run async, never blocks main loop
- ✅ **Resilient:** If emotion service crashes, main app continues
- ✅ **Efficient:** Main loop runs at 100 FPS, emotion every 50 frames
- ✅ **Isolated:** No dependency conflicts

---

## 🧠 Emotion Detection

### How It Works

1. **Main process** captures frames from camera
2. Every 50 frames, sends frame to emotion service via queue
3. **Emotion service** runs DeepFace analysis (TensorFlow backend)
4. Returns emotion label + confidence
5. Main process displays result on screen (cached if service is busy)

### Output Format

```python
EmotionResult(
    label="Happy",  # or Angry, Sad, Neutral, Surprised, Fear, Disgust
    emoji="😄",
    confidence=0.92  # 0.0 to 1.0
)
```

### Performance

- DeepFace takes ~100-200 ms per frame
- Runs in background (main loop never waits)
- Updates every 50 frames (~0.5 Hz UI updates)

---

## 🤖 Robot Control

### Tracking Modes

1. **Servo Tracking (default):** Robot moves to keep face centered in frame
   - Horizontal servo error → X motion
   - Vertical servo error → Z motion
   - Adjustable gains: `SERVO_H_GAIN`, `SERVO_V_GAIN`

2. **Plane Mode:** Lock X, move only in Y/Z plane (useful for tabletop setups)
   - Toggle in GUI checkbox

### State Machine

```
IDLE
  ↓ (face detected)
STABILIZING (15 frames)
  ↓ (stable)
TRACKING
  ↕ (face lost / multiple / jump > 0.3m)
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
CAM_INDEX = 0           # Camera device index
CAM_W, CAM_H = 640, 480 # Resolution

# Servo tracking
SERVO_TRACKING_ENABLED = True
SERVO_H_GAIN = 0.50     # Horizontal sensitivity
SERVO_V_GAIN = 0.40     # Vertical sensitivity

# State machine
STABILIZE_FRAMES = 15
FREEZE_DURATION = 2.0
JUMP_THRESHOLD_M = 0.3

# Emotion updates
EMOTION_UPDATE_INTERVAL = 50  # Frames between updates
```

---

## 🚨 Troubleshooting

### Emotion shows "Neutral" always

```bash
# Verify emotion environment exists
conda env list | grep face-emotion

# Test emotion service directly
conda activate face-emotion
python emotion_service.py
```

### Face detection not working

```bash
# Check cvzone
conda activate face
python -c "import cvzone; print(cvzone.__version__)"

# Try different camera index
# Edit CAM_INDEX in config.py (0, 1, 2, etc.)
```

### Robot connection fails

```bash
# Verify ur-rtde installed
pip list | grep ur-rtde

# Test connection
python -c "from rtde_control import RTDEControlInterface"

# Check UR5 IP in config.py
```

### Application crashes on startup

```bash
# See full error
python main.py 2>&1 | head -50

# Verify all imports
python -c "from emotion_service import EmotionServiceClient"
python -c "from main import FaceControllerApp"
```

---

## 📊 Performance Tips

- **Emotion:** Every 50 frames keeps main loop responsive (~1/100 frames cost)
- **Face detection:** Runs every frame (~100 FPS with camera, hardware limited)
- **Servo loop:** 50 Hz background thread (ur-rtde)
- **Smoothing:** EMA + velocity lasso prevents jerky motion

**Bottleneck:** DeepFace inference (~100-200 ms), but isolated process means main app never waits.

---

## 🛠️ Extending the System

### Add New Tracking Feature

1. Add constant to `config.py`
2. Implement in `coordinate_mapping.py`
3. Hook into state machine in `state_machine.py`
4. Add UI control in `main.py`

### Replace Emotion Detector

Create new `emotion_alternative.py`:

```python
from emotion_service import EmotionServiceClient

class CustomEmotionClient(EmotionServiceClient):
    def _start_service(self):
        # Your custom setup
        pass
    
    def detect_emotion(self, frame):
        # Your custom inference
        pass
```

Then update `main.py`:

```python
from emotion_alternative import CustomEmotionClient
self.emotion = CustomEmotionClient(use_deepface=False)
```

### Add Data Logging

```python
# In main.py _update_frame():
if target:
    self.logger.log({
        'timestamp': time.time(),
        'state': self.state_m.get(),
        'pose': target,
        'emotion': self._emotion_display,
        'fps': 1/dt
    })
```

---

## 📚 Dependencies

### Main Environment (`face`)

```
Python 3.10
cvzone 1.6.1         # MediaPipe + hand/face detection
opencv-python 4.13   # Computer vision
numpy                # Numerics
pillow               # Image handling
protobuf 4.25.9      # MediaPipe requirement
ur-rtde              # UR robot control (optional)
```

### Emotion Environment (`face-emotion`)

```
Python 3.10
deepface 0.0.92      # Emotion recognition
tensorflow 2.21.0    # Deep learning backend
opencv-python 4.13   # Frame I/O
protobuf 7.35.0      # TensorFlow requirement
```

**Note:** Separate environments isolate protobuf version conflicts.

---

## 📖 For More Details

See [STRUCTURE.md](STRUCTURE.md) for:
- Detailed module documentation
- Architecture diagrams
- Extension examples
- Performance analysis

---

## 💡 Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| **Multiprocessing for emotion** | Avoid protobuf conflicts while enabling real ML models |
| **Queue-based IPC** | Non-blocking frame passing keeps main loop responsive |
| **50-frame emotion updates** | Balance responsiveness with inference latency |
| **State machine** | Safe state transitions prevent erratic robot behavior |
| **EMA + velocity lasso** | Smooth tracking without lag or overshooting |
| **Plane mode** | Simplify control for tabletop / constrained setups |

---

## 📄 License & Attribution

Uses:
- **cvzone** (Computer Vision Zone) — Hand & face detection
- **MediaPipe** — Pose tracking
- **DeepFace** — Emotion recognition
- **OpenCV** — Image processing
- **ur-rtde** — Universal Robots communication

---

## ❓ Questions?

Check the [STRUCTURE.md](STRUCTURE.md) or add print statements to trace execution:

```python
print(f"[Debug] State: {self.state_m.get()}, Face: {face.detected}, Emotion: {self._emotion_display}")
```

Enjoy tracking! 🎉
