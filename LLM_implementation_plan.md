# LLM Implementation Plan: UR5 Conversational Robot

## 🎯 Vision

Transform the UR5 face-tracking robot into a conversational companion that:
- **Listens** to user speech in real-time
- **Understands** emotional context from detected expressions
- **Responds** with emotionally-aware dialogue via LLM
- **Reacts** physically with scripted movements (curious tilts, thinking pauses, empathy gestures)

---

## 📋 Recommended Technology Stack

### 1. Speech-to-Text: Whisper (OpenAI)
**Choice:** Local, privacy-first implementation
- **Library:** `openai-whisper`
- **Model:** base (77M params, ~140MB)
- **Performance:** ~30 sec to transcribe 1 min audio
- **Advantage:** Runs locally, zero latency after transcription, free
- **Integration:** Non-blocking thread, queues text to LLM

### 2. Large Language Model: Mistral 7B via Ollama
**Choice:** Best conversational LLM for M3 hardware
- **Framework:** Ollama (local inference engine)
- **Model:** Mistral 7B (quantized, ~7GB VRAM)
- **Performance:** ~5-10 tokens/sec (5-10 sec for typical response)
- **Advantages:**
  - Superior conversation quality vs Llama 2
  - MPS-accelerated on M3
  - Fast enough for natural dialogue
  - Full privacy (no internet required)
  - Easy personality injection via system prompts

### 3. Text-to-Speech: pyttsx3 (Local) + Optional ElevenLabs
**Primary:** pyttsx3 (built-in macOS voices)
- **Library:** `pyttsx3`
- **Latency:** Immediate (no API calls)
- **Cost:** Free
- **Voices:** 2-3 built-in macOS voices (customize accent/speed)

**Optional Upgrade:** ElevenLabs API
- **Cost:** ~$5/month or $0.30/1M chars
- **Quality:** Natural, expressive voices
- **Setup:** API key + requests library

---

## 🏗️ System Architecture

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  FACE TRACKING (Main Thread - Real-time, 30fps, never blocks)     │
│                                                                   │
│  ├─ Camera capture (1280×720)                                    │
│  ├─ Face detection (MediaPipe)                                   │
│  ├─ Emotion classification (PyTorch EfficientNet-B0)             │
│  ├─ Servo tracking (centers face)                                │
│  └─ Emotion state → shared memory                                │
└──────────────────────────────────────────────────────────────────┘
    │
    │ Current emotion + user engagement
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   WHISPER    │ │   MISTRAL    │ │  PYTTSX3/    │ │   MOVEMENT   │
│ (Audio→Text) │ │   LLM        │ │ ELEVENLAB    │ │   SCRIPTS    │
│              │ │ (Emotion     │ │ (Text→Speech)│ │ (Servo Ctrl) │
│ Runs async   │ │  Context)    │ │              │ │              │
│ every 10-30s │ │              │ │ Parallel     │ │ Triggered by │
│              │ │ Emotional    │ │ with LLM     │ │ LLM response │
│ Input:       │ │ awareness    │ │              │ │              │
│ Audio stream │ │ reasoning    │ │ Output:      │ │ Movements:   │
│              │ │              │ │ Audio stream │ │ - Curious    │
│ Output:      │ │ Input:       │ │              │ │ - Thinking   │
│ Text string  │ │ Text +       │ │ Input:       │ │ - Empathy    │
│ + confidence │ │ Emotion      │ │ Text string  │ │ - Happy      │
│              │ │              │ │              │ │ - Sad        │
│ Thread:      │ │ Output:      │ │ Thread:      │ │ Thread:      │
│ audio_thread │ │ Speech text  │ │ voice_thread │ │ movement_q   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
    │                  │                  │                  │
    └──────────────────┴──────────────────┴──────────────────┘
                       │
                       ▼
                   Tkinter GUI
            (Display transcript + emotion)
```

### Execution Threads

```
Main Thread (Face Tracking)
├─ 30fps video loop (unblocked)
├─ Face detection
├─ Emotion update
└─ Servo control

Audio Thread (Whisper)
├─ Capture microphone (5-10 sec chunks)
├─ Transcribe with Whisper
└─ Queue text to LLM

LLM Thread (Mistral)
├─ Wait for transcribed text
├─ Build prompt with emotion context
├─ Query Ollama (5-10 sec blocking)
└─ Queue response for TTS + movement

Voice Thread (pyttsx3/ElevenLabs)
├─ Queue responses from LLM
├─ Synthesize speech
└─ Play audio

Movement Thread (Servo)
├─ Queue movement commands from LLM
├─ Execute scripted gestures
└─ Non-blocking servo commands
```

---

## 💬 Conversation Flow Example

### Interaction Sequence

```
[User speaks]
"I'm feeling lonely today"
        │
        ▼
[Whisper transcribes]
"I'm feeling lonely today"
        │
        ▼
[LLM receives context]
{
  "emotion": "sad",
  "user_input": "I'm feeling lonely today",
  "conversation_history": [...],
  "task": "respond with empathy"
}
        │
        ▼
[Mistral generates response]
"I hear that. Loneliness can be tough. What's been on your mind lately?"
        │
        ├─────────────────────────┬──────────────────────┐
        │                         │                      │
        ▼                         ▼                      ▼
   [pyttsx3 speaks]        [Movement trigger]   [Emotion context updates]
   "I hear that..."        - Head tilts toward  Emotion flow: sad→empathic
                           - Blink slower
                           - Pause before speech
        │
        ▼
[User hears response + sees empathetic movement]
[Loop continues with real-time face tracking]
```

### System Prompt (Personality)

```python
SYSTEM_PROMPT = """You are a friendly and empathetic UR5 collaborative robot 
engaged in a 'get to know you' conversation. Your goal is to learn about the 
person and build genuine connection.

Current emotional state: {emotion}
Detected confidence: {confidence}

Guidelines:
- Respond with warmth and genuine interest
- Mirror their emotional tone (empathy for sadness, enthusiasm for joy)
- Ask follow-up questions to deepen conversation
- Keep responses concise (1-2 sentences for natural speech)
- Never break character as a robot learning about humans
- Use simple, clear language

Conversation history:
{history}

Respond naturally as if continuing a real conversation.
"""
```

---

## ⚙️ Implementation Steps

### Phase 1: Audio Input (Week 1)

**Step 1.1: Install Whisper**
```bash
pip install openai-whisper sounddevice numpy
```

**Step 1.2: Create `audio_service.py`**
- Capture 10-second audio chunks from microphone
- Non-blocking (runs in separate thread)
- Queue transcribed text → LLM thread

**Step 1.3: Test**
```bash
python -c "import whisper; model = whisper.load_model('base')"
# Verify transcription accuracy
```

### Phase 2: LLM Integration (Week 2)

**Step 2.1: Install & Run Ollama**
```bash
# Download Ollama from ollama.ai
# Run: ollama pull mistral
# Verify: curl localhost:11434/api/generate
```

**Step 2.2: Create `llm_service.py`**
- Receive transcribed text from audio thread
- Build prompt with emotion context
- Query Ollama via REST API (requests library)
- Queue response → TTS thread

**Step 2.3: Test**
```python
import requests
response = requests.post('http://localhost:11434/api/generate', 
    json={"model": "mistral", "prompt": "Hello"})
```

### Phase 3: Voice Output (Week 2)

**Step 3.1: Install TTS**
```bash
pip install pyttsx3
```

**Step 3.2: Create `voice_service.py`**
- Receive text from LLM thread
- Synthesize speech with pyttsx3
- Play audio through speaker
- Non-blocking

**Step 3.3: Test**
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("Hello, I'm a robot")
engine.runAndWait()
```

### Phase 4: Movement Scripting (Week 3)

**Step 4.1: Create `movement_scripts.py`**
Define robot poses:
```python
MOVEMENTS = {
    "curious": {
        "head": 30,        # tilt degrees
        "duration": 0.5,   # seconds
        "joints": {...}    # servo angles
    },
    "thinking": {
        "head": 0,
        "pause": 2.0,      # pause for thinking
        "joints": {...}
    },
    "empathy": {
        "head": 15,
        "joints": {...}
    },
    "happy": {
        "head": -15,       # slight nod
        "joints": {...}
    }
}
```

**Step 4.2: Trigger from LLM**
- Detect intent from response (question → curious, pause → thinking)
- Queue movement command
- Execute parallel to speech

**Step 4.3: Test**
- Manual servo commands
- Verify movement safety

### Phase 5: Integration (Week 3)

**Step 5.1: Update `main.py`**
```python
# Add new threads
self.audio_thread = AudioThread()  # Whisper
self.llm_thread = LLMThread()      # Mistral
self.voice_thread = VoiceThread()  # pyttsx3
self.movement_thread = MovementThread()  # Servo

# Shared queues
self.transcript_queue = queue.Queue()
self.response_queue = queue.Queue()
self.movement_queue = queue.Queue()

# Emotion sharing
self.current_emotion = None
```

**Step 5.2: Create Conversation History**
```python
class ConversationManager:
    def __init__(self):
        self.history = []
    
    def add_turn(self, role, text, emotion):
        self.history.append({
            "role": role,      # "user" or "assistant"
            "text": text,
            "emotion": emotion,
            "timestamp": time.time()
        })
    
    def get_context(self, max_turns=5):
        # Format for LLM prompt
        return self.history[-max_turns:]
```

**Step 5.3: Test End-to-End**
- Speak to robot
- See transcript in GUI
- Hear response
- Watch movements

### Phase 6: Tuning & Polish (Week 4)

**Step 6.1: Optimize Performance**
- Tune Whisper confidence threshold
- Adjust pyttsx3 speech rate
- Fine-tune Mistral prompt for personality

**Step 6.2: Add Features**
- Conversation memory (context accumulation)
- Emotion-to-response mapping
- Custom voice profiles
- ElevenLabs integration (optional)

**Step 6.3: Safety & UX**
- Kill switches (stop audio, stop robot)
- Error handling (offline Ollama, microphone failure)
- Status indicators in GUI
- Conversation logging

---

## 💻 Hardware Performance Expectations (M3 18GB)

| Operation | Duration | Notes |
|-----------|----------|-------|
| Whisper (10 sec audio) | ~3-5 sec | Parallel with tracking |
| Mistral inference (50 tokens) | ~5-8 sec | MPS accelerated, blocking LLM thread only |
| pyttsx3 (10 sec response) | Instant | Parallel |
| Movement script execution | 0.5-2 sec | Parallel with tracking |
| **Total conversation loop** | ~6-10 sec | Face tracking continues @ 30fps |

**Memory breakdown:**
- Whisper model: ~140MB
- Mistral 7B (quantized): ~7GB
- PyTorch + face detection: ~1GB
- Remaining: ~10GB free (buffer)

**Conclusion:** Everything fits comfortably, no stuttering.

---

## 📁 New File Structure

```
Reference Code/
├── main.py                    # Updated with threads
├── audio_service.py          # ⭐ NEW: Whisper service
├── llm_service.py            # ⭐ NEW: Mistral service
├── voice_service.py          # ⭐ NEW: pyttsx3/ElevenLabs
├── movement_scripts.py       # ⭐ NEW: Robot gestures
├── conversation_manager.py   # ⭐ NEW: Dialogue history
├── emotion_service.py        # Existing (no changes)
├── face_detector.py          # Existing
├── robot_controller.py       # Existing
├── [...other existing files...]
├── config.py                 # Update with LLM params
└── LLM_implementation_plan.md # This file
```

---

## 🎛️ Configuration Parameters (config.py)

```python
# Audio Input
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
AUDIO_CHUNK_DURATION = 10  # seconds
AUDIO_CONFIDENCE_THRESHOLD = 0.5

# LLM (Mistral via Ollama)
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.7  # Creativity (0=deterministic, 1=creative)
LLM_MAX_TOKENS = 100   # Limit response length
LLM_CONTEXT_SIZE = 2000

# Voice Output
TTS_ENGINE = "pyttsx3"  # Or "elevenlab"
VOICE_SPEED = 150  # words per minute
VOICE_VOLUME = 1.0  # 0-1 scale

# Movement
MOVEMENT_ENABLE = True
MOVEMENT_SPEED = 0.5  # 0-1 scale (lower = slower, safer)

# Emotion Context
EMOTION_HISTORY_SIZE = 5  # turns to remember
EMOTION_THRESHOLD = 0.6  # min confidence to use in prompt
```

---

## 🔧 Dependencies to Add

```bash
# Core
pip install openai-whisper sounddevice ollama

# Voice (choose one or both)
pip install pyttsx3
# pip install elevenlabs  # Optional, requires API key

# Utilities
pip install requests numpy

# No new conda packages needed (everything pip-installable)
```

---

## 🚀 Execution Plan Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Audio Input | 3-4 days | Whisper working, transcripts in GUI |
| 2. LLM Integration | 3-4 days | Mistral responses to transcripts |
| 3. Voice Output | 2-3 days | Robot speaks responses |
| 4. Movement Scripts | 3-4 days | Gestures synchronized with speech |
| 5. Full Integration | 2-3 days | End-to-end conversation test |
| 6. Polish & Tuning | 3-5 days | Natural conversation, personality |
| **Total** | **2-3 weeks** | Conversational robot ready |

---

## 🎭 Example Conversations

### Conversation 1: Getting to Know You
```
Human: "Hi, I'm Max"
Robot [curious tilt]: "Nice to meet you, Max! What brings you here today?"