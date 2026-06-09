"""
Text-to-Speech using macOS built-in `say` command (subprocess).

Replaces pyttsx3 which silently stops producing audio after many calls
due to a known NSSpeechSynthesizer degradation bug on macOS.

`say` is native, always reliable, uses the same voices, and never degrades.

Usage:
    svc = VoiceService()
    svc.start()
    svc.speak("Hello, nice to meet you!")
    svc.stop()
"""

import subprocess
import threading
import queue

VOICE = "Samantha"   # macOS voice name  — run `say -v ?` to list all
RATE  = 175           # words per minute


class VoiceService:
    """Speaks text responses asynchronously using the macOS `say` command."""

    def __init__(self):
        self._queue      = queue.Queue()
        self._stop_event = threading.Event()
        self._thread     = None
        self._process    = None   # current `say` subprocess
        self.ready       = True
        self.speaking    = False
        print(f"[Voice] Ready — voice={VOICE}, rate={RATE} wpm")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="voice-worker"
        )
        self._thread.start()
        print("[Voice] Service started")

    def stop(self):
        self._stop_event.set()
        self._interrupt()
        self._queue.put(None)   # unblock worker
        print("[Voice] Stopped")

    def speak(self, text: str):
        """Queue text for speech. Non-blocking."""
        if not text:
            return
        self._queue.put(text)
        print(f"[Voice] Queued ({len(text)} chars)")

    def interrupt(self):
        """Stop current speech immediately and clear the queue."""
        self._interrupt()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        print("[Voice] Interrupted")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _interrupt(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process = None

    def _worker(self):
        print("[Voice] Worker thread started")
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if text is None:
                break

            print(f"[Voice] Speaking: '{text}'")
            self.speaking = True
            try:
                self._process = subprocess.Popen(
                    ["say", "-v", VOICE, "-r", str(RATE), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._process.wait()
            except Exception as e:
                print(f"[Voice] Error: {e}")
            finally:
                self.speaking = False
                self._process = None
            print("[Voice] Done speaking")

        print("[Voice] Worker thread stopped")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    svc = VoiceService()
    svc.start()

    lines = [
        "Hello! I am your UR5 robot companion.",
        "I am happy to meet you today.",
        "I can detect your emotions and respond with empathy.",
        "This is the fourth sentence — pyttsx3 would have broken by now.",
        "But the say command keeps going strong.",
    ]

    for line in lines:
        svc.speak(line)
        while svc.speaking or not svc._queue.empty():
            time.sleep(0.1)
        time.sleep(0.2)

    svc.stop()
