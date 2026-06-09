"""
LLM service using Mistral 7B via Ollama (local, privacy-first).

Receives transcribed text + current emotion, builds an emotionally-aware
prompt, and returns a short spoken response. Runs in a background thread.

Usage:
    svc = LLMService()
    svc.start()
    svc.submit(text="Hello, I'm Max", emotion="happy", confidence=0.82)
    response = svc.get_response()   # None until ready
"""

import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional

OLLAMA_URL  = "http://localhost:11434/api/generate"
LLM_MODEL   = "mistral"
TEMPERATURE = 0.7
MAX_TOKENS  = 120      # ~2 sentences, short enough for natural speech

SYSTEM_PROMPT = (
    "You are a friendly, curious UR5 robot having a 'get to know you' conversation. "
    "Keep every reply to 1-2 short sentences — it will be spoken aloud. "
    "Mirror the person's emotional tone. Ask one follow-up question to keep the conversation going. "
    "Never use markdown, bullet points, or special characters. "
    "Never break character — you are a robot that is genuinely curious about humans."
)


@dataclass
class LLMResponse:
    text: str
    emotion_used: str
    duration_sec: float


@dataclass
class _Turn:
    role: str   # "user" or "assistant"
    text: str


class LLMService:
    """Sends transcripts to Mistral and returns conversational responses async."""

    def __init__(self):
        self._input_q    = queue.Queue(maxsize=4)
        self._output_q   = queue.Queue()
        self._stop_event = threading.Event()
        self._thread     = None
        self._history: list[_Turn] = []
        self.ready       = False
        self._check_ollama()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _check_ollama(self):
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            if any(LLM_MODEL in m for m in models):
                self.ready = True
                print("[LLM] Mistral ready via Ollama")
            else:
                print("[LLM] Mistral not found — run: ollama pull mistral")
        except Exception as e:
            print(f"[LLM] Ollama not reachable: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if not self.ready:
            print("[LLM] Cannot start — Ollama/Mistral not available")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="llm-worker"
        )
        self._thread.start()
        print("[LLM] Service started")

    def stop(self):
        self._stop_event.set()
        print("[LLM] Stopped")

    def submit(self, text: str, emotion: str = "neutral", confidence: float = 0.0):
        """Queue a user turn. Non-blocking."""
        try:
            self._input_q.put_nowait({
                "text": text, "emotion": emotion, "confidence": confidence
            })
            print(f"[LLM] Received: '{text}'  emotion={emotion} ({confidence:.0%})")
        except queue.Full:
            print("[LLM] Queue full — input dropped")

    def get_response(self) -> Optional[LLMResponse]:
        """Non-blocking poll. Returns LLMResponse or None."""
        try:
            return self._output_q.get_nowait()
        except queue.Empty:
            return None

    def clear_history(self):
        self._history.clear()
        print("[LLM] Conversation history cleared")

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        import requests

        print("[LLM] Worker thread started")
        while not self._stop_event.is_set():
            try:
                item = self._input_q.get(timeout=1.0)
            except queue.Empty:
                continue

            text       = item["text"]
            emotion    = item["emotion"]
            confidence = item["confidence"]
            prompt     = self._build_prompt(text, emotion, confidence)

            print("[LLM] Querying Mistral...")
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model":       LLM_MODEL,
                        "prompt":      prompt,
                        "stream":      False,
                        "temperature": TEMPERATURE,
                        "options":     {"num_predict": MAX_TOKENS},
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                reply = resp.json().get("response", "").strip()
            except Exception as e:
                print(f"[LLM] Request error: {e}")
                continue

            duration = time.perf_counter() - t0
            print(f"[LLM] Response ({duration:.1f}s): '{reply}'")

            self._history.append(_Turn("user",      text))
            self._history.append(_Turn("assistant", reply))
            if len(self._history) > 20:
                self._history = self._history[-20:]

            self._output_q.put(LLMResponse(
                text=reply,
                emotion_used=emotion,
                duration_sec=duration,
            ))

        print("[LLM] Worker thread stopped")

    def _build_prompt(self, user_text: str, emotion: str, confidence: float) -> str:
        history_lines = []
        for turn in self._history[-6:]:
            prefix = "Human" if turn.role == "user" else "Robot"
            history_lines.append(f"{prefix}: {turn.text}")

        history_str = (
            "\n".join(history_lines)
            if history_lines
            else "This is the start of the conversation."
        )

        emotion_note = ""
        if confidence >= 0.45 and emotion != "neutral":
            emotion_note = (
                f"Note: the person currently appears to feel {emotion} "
                f"({confidence:.0%} confidence). Respond with appropriate empathy.\n\n"
            )

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"{emotion_note}"
            f"Conversation so far:\n{history_str}\n\n"
            f"Human: {user_text}\nRobot:"
        )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    svc = LLMService()
    if not svc.ready:
        print("Start Ollama first: ollama serve")
        raise SystemExit(1)

    svc.start()

    tests = [
        ("Hi, I'm Max!", "happy",   0.82),
        ("I've been feeling a bit lonely lately.", "sad", 0.71),
        ("What do you think about robots?", "neutral", 0.40),
    ]

    for text, emotion, conf in tests:
        print(f"\n--- Sending: '{text}' [{emotion}] ---")
        svc.submit(text=text, emotion=emotion, confidence=conf)

        # Wait for response
        for _ in range(60):
            time.sleep(0.5)
            result = svc.get_response()
            if result:
                print(f"\n=== ROBOT SAYS ===")
                print(f"  {result.text}")
                print(f"  (took {result.duration_sec:.1f}s, emotion={result.emotion_used})")
                break

    svc.stop()
