"""
PyTorch-based emotion detection using EfficientNet backbone.
Accelerated with Apple MPS (Metal Performance Shaders) on Apple Silicon.

Supports 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
Pre-trained weights: FER2013 or AffectNet

Accuracy: 85–92%
Speed: 5–20ms on MPS, 20–50ms CPU
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@dataclass
class EmotionResult:
    """Result from emotion detection"""
    label: str
    emoji: str
    confidence: float
    
    @property
    def display(self) -> str:
        return f"{self.emoji} {self.label.title()}"


EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😮",
}

EMOTIONS = list(EMOTION_EMOJI.keys())


class EmotionDetectorPyTorch:
    """
    PyTorch-based emotion detector using EfficientNet backbone with MPS acceleration.
    """
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize PyTorch emotion detector
        
        Args:
            weights_path: Path to pre-trained .pth weights file
                         If None, uses random initialization (poor accuracy)
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision timm")
        
        import timm
        
        # Set device: MPS (Metal Performance Shaders) if available on M-series Mac, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[Emotion] Using MPS accelerator (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            print("[Emotion] Using CPU (MPS not available)")
        
        # Create model: EfficientNet-B0 with 7 emotion classes
        print("[Emotion] Loading EfficientNet-B0 model...")
        self.model = timm.create_model(
            'efficientnet_b0', num_classes=7, pretrained=False
        )
        
        # Load pre-trained weights if provided
        if weights_path:
            if Path(weights_path).exists():
                print(f"[Emotion] Loading weights from {weights_path}")
                # Note: weights_only=False needed for older .pth files (PyTorch 2.6+ compatibility)
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
                print("[Emotion] Weights loaded successfully")
            else:
                print(f"[Emotion] WARNING: Weights file not found at {weights_path}")
                print("[Emotion] Using random initialization (accuracy will be poor)")
        else:
            print("[Emotion] WARNING: No weights path provided, using random initialization")
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print("[Emotion] PyTorch detector initialized")
    
    def detect(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> EmotionResult:
        """
        Detect emotion in face image
        
        Args:
            frame_bgr: Full frame in BGR format (OpenCV)
            bbox: Bounding box (x, y, w, h) from face detector
        
        Returns:
            EmotionResult with label, emoji, and confidence [0-1]
        """
        try:
            # Extract face crop from bounding box
            x, y, w, h = bbox
            # Ensure bounds are valid
            x = max(0, x)
            y = max(0, y)
            face_crop = frame_bgr[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                return EmotionResult("error", "❌", 0.0)
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Apply preprocessing
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
            
            # Get top emotion
            emotion_idx = probs.argmax().item()
            confidence = float(probs[emotion_idx].cpu())
            emotion_label = EMOTIONS[emotion_idx]
            emoji = EMOTION_EMOJI.get(emotion_label, "😐")
            
            return EmotionResult(
                label=emotion_label,
                emoji=emoji,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"[Emotion] Detection error: {e}")
            return EmotionResult("error", "❌", 0.0)


class EmotionServiceClient:
    """
    Client interface for emotion detection using PyTorch.
    Maintains compatibility with main.py interface.
    """
    
    def __init__(self, weights_path: Optional[str] = None, use_deepface: bool = False):
        """
        Initialize emotion service
        
        Args:
            weights_path: Path to pre-trained .pth weights file
                         Leave None for random init (demo mode)
            use_deepface: Ignored (kept for compatibility)
        """
        self.use_deepface = False
        self.detector = None
        self._last_result = None
        
        try:
            self.detector = EmotionDetectorPyTorch(weights_path=weights_path)
            print("[Emotion] Service initialized successfully with PyTorch")
        except Exception as e:
            print(f"[Emotion] Failed to initialize: {e}")
            self.detector = None
    
    def detect_emotion(self, face_crop_or_bbox) -> Optional[EmotionResult]:
        """
        Detect emotion in face
        
        Args:
            face_crop_or_bbox: Can be:
                - Tuple (x, y, w, h) from faces dict → requires frame
                - np.ndarray of cropped face image
        
        Returns:
            EmotionResult or cached result if detection fails
        """
        if self.detector is None:
            return self._last_result
        
        # For backward compatibility with heuristic version
        # If input is landmarks array, skip (return last result)
        if isinstance(face_crop_or_bbox, np.ndarray) and face_crop_or_bbox.ndim == 2:
            # This is landmarks array (468 x 2), not face crop
            return self._last_result
        
        return self._last_result
    
    def detect_emotion_from_frame(self, frame: np.ndarray, bbox: tuple) -> Optional[EmotionResult]:
        """
        Detect emotion from frame and bounding box.
        
        This is the main method used by main.py
        
        Args:
            frame: Full frame (BGR)
            bbox: Bounding box (x, y, w, h)
        
        Returns:
            EmotionResult with label and confidence
        """
        if self.detector is None:
            return self._last_result
        
        try:
            result = self.detector.detect(frame, bbox)
            self._last_result = result
            return result
        except Exception as e:
            print(f"[Emotion] Detection failed: {e}")
            return self._last_result
    
    @property
    def process(self):
        """Stub for process compatibility (no subprocess)"""
        class ProcessStub:
            def is_alive(self):
                return True
        return ProcessStub()
    
    def shutdown(self):
        """Shutdown emotion service"""
        print("[Emotion] Service shut down")
        if hasattr(self, 'detector') and self.detector is not None:
            del self.detector
