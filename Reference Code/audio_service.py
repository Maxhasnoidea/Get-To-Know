"""
Speech-to-Text using OpenAI Whisper (local, privacy-first).

Model: base (77M params, ~140MB)
Performance: ~3-5 sec transcription on M3 Pro (MPS accelerated)

Two background threads:
  - capture thread  : records fixed-duration audio chunks from microphone
  - transcribe thread: runs Whisper on each chunk, emits TranscriptResult

Usage:
    svc = SpeechToTextService()
    svc.start()

    while running:
        result = svc.get_transcript()   # non-blocking, returns None if nothing new
        if result:
            print(result.text)

    svc.stop()
"""

import threading
import queue
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Config (also settable from config.py) ────────────────────────────────────
WHISPER_MODEL_NAME   = "base"
AUDIO_CHUNK_DURATION = 10       # seconds per recorded chunk
SAMPLE_RATE          = 16000    # Hz — Whisper expects 16 kHz
SILENCE_THRESHOLD    = 0.01     # RMS below this → skip transcription (silence)
CONFIDENCE_THRESHOLD = 0.50     # no_speech_prob must be < (1 - threshold) to keep

# Audio input device — None = system default (MacBook mic)
# Run `python audio_service.py --list` to see available devices.
# Set to the device index of your webcam mic to use it instead.
# e.g. AUDIO_DEVICE = 1  →  EMEET SmartCam C960
AUDIO_DEVICE = None


@dataclass
class TranscriptResult:
    text: str
    confidence: float   # 1.0 − no_speech_prob from Whisper
    language: str


class SpeechToTextService:
    """Captures microphone audio in chunks and transcribes with Whisper async."""

    def __init__(self, model_name: str = WHISPER_MODEL_NAME,
                 chunk_duration: float = AUDIO_CHUNK_DURATION):
        self.chunk_duration  = chunk_duration
        self.ready           = False
        self._model          = None
        self._device         = "cpu"
        self._stop_event        = threading.Event()
        self._audio_q           = queue.Queue(maxsize=2)
        self._transcript_q      = queue.Queue()
        self._capture_thread    = None
        self._transcribe_thread = None
        self._level_stream      = None
        self.current_level      = 0.0   # live RMS, read by UI thread
        self.paused             = False  # set True while robot is speaking

        self._load_model(model_name)

    # ── Init ─────────────────────────────────────────────────────────────────

    def _load_model(self, model_name: str):
        try:
            import torch
            import whisper

            if torch.backends.mps.is_available():
                self._device = "mps"
            print(f"[STT] Loading Whisper '{model_name}' on {self._device.upper()}...")
            self._model = whisper.load_model(model_name, device=self._device)
            self.ready = True
            print("[STT] Whisper ready")
        except Exception as e:
            print(f"[STT] Failed to load Whisper: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start capture, transcription, and level-monitor threads."""
        if not self.ready:
            print("[STT] Cannot start — model not loaded")
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="stt-capture"
        )
        self._transcribe_thread = threading.Thread(
            target=self._transcribe_loop, daemon=True, name="stt-transcribe"
        )
        self._capture_thread.start()
        self._transcribe_thread.start()
        self._start_level_monitor()
        print(f"[STT] Started — recording {self.chunk_duration}s chunks")

    def stop(self):
        """Stop capture, transcription, and level monitor."""
        self._stop_event.set()
        if self._level_stream is not None:
            self._level_stream.stop()
            self._level_stream.close()
            self._level_stream = None
        self.current_level = 0.0
        print("[STT] Stopped")

    def get_transcript(self) -> Optional[TranscriptResult]:
        """Non-blocking poll. Returns next TranscriptResult or None."""
        try:
            return self._transcript_q.get_nowait()
        except queue.Empty:
            return None

    # ── Threads ───────────────────────────────────────────────────────────────

    def _start_level_monitor(self):
        """Open a continuous InputStream just for reading the mic level."""
        import sounddevice as sd

        def _callback(indata, _frames, _time, _status):
            rms = float(np.sqrt(np.mean(indata ** 2)))
            # Smooth with exponential moving average
            self.current_level = 0.6 * self.current_level + 0.4 * rms

        self._level_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=AUDIO_DEVICE,
            callback=_callback,
        )
        self._level_stream.start()

    def _capture_loop(self):
        """Record fixed-duration chunks and push to audio queue."""
        import sounddevice as sd

        print("[STT] Capture thread started")
        samples = int(self.chunk_duration * SAMPLE_RATE)

        while not self._stop_event.is_set():
            # Don't record while the robot is speaking — avoids feedback loop
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                audio = sd.rec(samples, samplerate=SAMPLE_RATE,
                               channels=1, dtype="float32",
                               device=AUDIO_DEVICE)
                sd.wait()
                if self._stop_event.is_set():
                    break
                chunk = audio.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < SILENCE_THRESHOLD:
                    print(f"[STT] Silence detected (rms={rms:.4f}), skipping")
                    continue
                try:
                    self._audio_q.put_nowait(chunk)
                except queue.Full:
                    # Transcriber is behind — drop oldest, keep latest
                    try:
                        self._audio_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._audio_q.put_nowait(chunk)
            except Exception as e:
                print(f"[STT] Capture error: {e}")
                break

        print("[STT] Capture thread stopped")

    def _transcribe_loop(self):
        """Pull audio chunks and run Whisper transcription."""
        print("[STT] Transcription thread started")

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                print("[STT] Transcribing chunk...")
                result = self._model.transcribe(
                    chunk,
                    language="en",
                    fp16=False,          # fp16 can be unstable on MPS
                    verbose=False,
                )

                text = result["text"].strip()
                segments = result.get("segments", [])

                # Average no_speech_prob across segments (0 = speech, 1 = silence)
                if segments:
                    no_speech = float(np.mean([s["no_speech_prob"] for s in segments]))
                else:
                    no_speech = 1.0

                confidence = 1.0 - no_speech
                language   = result.get("language", "en")

                print(f"[STT] '{text}' | confidence={confidence:.2f}")

                if text and confidence >= CONFIDENCE_THRESHOLD:
                    self._transcript_q.put(TranscriptResult(
                        text=text,
                        confidence=confidence,
                        language=language,
                    ))
                else:
                    print("[STT] Low confidence or empty — discarded")

            except Exception as e:
                print(f"[STT] Transcription error: {e}")

        print("[STT] Transcription thread stopped")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, time

    if "--list" in sys.argv:
        import sounddevice as sd
        print("=== Available audio input devices ===")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                marker = " ← DEFAULT" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch){marker}")
        print(f"\nSet AUDIO_DEVICE in audio_service.py to use a specific device.")
        sys.exit(0)

    print("=== Whisper Speech-to-Text Test ===")
    print(f"Device: {'system default' if AUDIO_DEVICE is None else AUDIO_DEVICE}")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    svc = SpeechToTextService(model_name="base", chunk_duration=5)
    svc.start()

    try:
        while True:
            result = svc.get_transcript()
            if result:
                print(f"\n>>> '{result.text}'")
                print(f"    confidence={result.confidence:.2f}  lang={result.language}\n")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        svc.stop()
