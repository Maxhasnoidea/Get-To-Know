UR5 Conversational Robot: Get to Know You
A real-time system that transforms a UR5 collaborative robot into an emotionally-aware conversational companion. The robot sees you, recognizes your emotions, listens to what you say, and engages in natural conversation while reacting with thoughtful movements.

Features:

🎥 Real-time face detection & emotion recognition (PyTorch, 85-92% accuracy)
🤖 Conversational AI with emotion context (Mistral 7B via Ollama)
🎤 Speech-to-text input (Whisper) + natural voice output (pyttsx3)
🦾 Emotional servo reactions (curious tilts, thinking pauses, empathy gestures)
💬 "Get to know you" dialog with memory (remembers what you said)
⚡ Real-time face tracking at 30fps (non-blocking AI threads)
Tech Stack:
Python 3.10 • PyTorch + MPS • MediaPipe • Whisper • Mistral 7B • Ollama • ur-rtde • Tkinter

Hardware:

M3/M4 Mac with 16GB+ RAM (or compatible ARM64 system)
UR5 collaborative robot arm (or simulated via URSim Docker)
USB webcam with microphone
