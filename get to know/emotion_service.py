"""
Emotion detection using ViT (Vision Transformer) fine-tuned on FER+.
Model: trpakov/vit-face-expression  — 7 emotions, ~72% real-world accuracy.

Inference runs in a background thread so the main camera loop is never blocked.
"""

import threading
import queue
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EmotionResult:
    label: str
    emoji: str
    confidence: float

    @property
    def display(self) -> str:
        return f"{self.emoji} {self.label.title()}"


EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "sad":      "😢",
    "surprise": "😮",
    "neutral":  "😐",
}

# Report an emotion only when confidence exceeds this; otherwise show neutral
CONFIDENCE_THRESHOLD = 0.35


class EmotionDetector:
    """
    ViT emotion detector running in a background thread.
    Main thread submits face crops; results are available immediately via .last_result.
    """

    def __init__(self):
        import sys
        import torch
        from transformers import pipeline

        print(f"[Emotion] Python {sys.version.split()[0]} | torch {torch.__version__}")

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[Emotion] Using {device.upper()} device")

        print("[Emotion] Loading ViT model (trpakov/vit-face-expression)...")
        try:
            self._pipe = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",
                device=device,
            )
        except Exception as e:
            print(f"[Emotion] MPS pipeline failed ({e}), falling back to CPU")
            self._pipe = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",
                device="cpu",
            )
        print("[Emotion] Model ready")

        self._queue = queue.Queue(maxsize=1)   # only latest frame matters
        self.last_result: Optional[EmotionResult] = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="emotion-worker")
        self._thread.start()

    def _worker(self):
        from PIL import Image
        print("[Emotion] Worker thread started")
        while True:
            face_rgb = self._queue.get()
            if face_rgb is None:
                print("[Emotion] Worker thread stopping")
                break
            try:
                pil_img = Image.fromarray(face_rgb)
                results = self._pipe(pil_img)
                top = results[0]
                label = top["label"]
                confidence = float(top["score"])

                if confidence < CONFIDENCE_THRESHOLD:
                    label = "neutral"
                    confidence = next(
                        (float(r["score"]) for r in results if r["label"] == "neutral"),
                        confidence,
                    )

                self.last_result = EmotionResult(
                    label=label,
                    emoji=EMOTION_EMOJI.get(label, "😐"),
                    confidence=confidence,
                )
                print(f"[Emotion] {self.last_result.display} ({self.last_result.confidence*100:.0f}%)")
            except Exception as e:
                print(f"[Emotion] Worker error: {e}")

    def submit(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Submit a face crop for async inference. Drops frame if worker is busy."""
        x, y, w, h = bbox
        face_bgr = frame_bgr[max(0, y):y + h, max(0, x):x + w]
        if face_bgr.size == 0:
            print("[Emotion] submit: empty crop, skipping")
            return
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        try:
            self._queue.put_nowait(face_rgb)
        except queue.Full:
            pass

    def shutdown(self):
        self._queue.put(None)


class EmotionServiceClient:
    """Public interface used by main.py."""

    def __init__(self, weights_path: Optional[str] = None, use_deepface: bool = False):
        self.use_deepface = False
        self.detector = None
        self._last_result = None
        self.ready = False

        try:
            self.detector = EmotionDetector()
            self.ready = True
            print("[Emotion] Service ready")
        except Exception as e:
            print(f"[Emotion] Failed to initialize: {e}")

    def detect_emotion_from_frame(self, frame: np.ndarray, bbox: tuple) -> Optional[EmotionResult]:
        """Submit a new frame and return the most recent completed result."""
        if self.detector is None:
            return self._last_result
        self.detector.submit(frame, bbox)
        result = self.detector.last_result
        if result is not None:
            self._last_result = result
        return self._last_result

    def detect_emotion(self, face_crop_or_bbox) -> Optional[EmotionResult]:
        return self._last_result

    @property
    def process(self):
        class _Stub:
            def is_alive(self): return True
        return _Stub()

    def shutdown(self):
        print("[Emotion] Service shut down")
        if self.detector is not None:
            self.detector.shutdown()
            del self.detector
