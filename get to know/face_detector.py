"""
Face detection wrapper
"""
from cvzone.FaceDetectionModule import FaceDetector
from models import FaceState
from config import FACE_DETECTION_CONFIDENCE, FACE_MAX_FACES


class FaceDetectorWrapper:
    """
    Wrapper around cvzone's FaceDetector to abstract detection logic.
    """

    def __init__(self):
        self.detector = FaceDetector(
            minDetectionCon=FACE_DETECTION_CONFIDENCE
        )

    def find_faces(self, frame, draw=False):
        """
        Detect faces in frame.
        
        NOTE: cvzone FaceDetector returns (frame, faces) unlike HandDetector which returns (hands, frame)
        
        Returns:
            faces: List of face arrays from cvzone
            frame: Annotated frame (if draw=True)
        """
        # cvzone FaceDetector returns (frame, faces) - NOTE REVERSED ORDER from HandDetector!
        frame, faces = self.detector.findFaces(frame, draw=draw)
        return faces, frame

    def parse_faces(self, faces, frame_width: int, frame_height: int) -> FaceState:
        """
        Parse cvzone face detection into FaceState.
        
        cvzone FaceDetector returns faces as list of dicts:
        [{'id': int, 'bbox': (x, y, w, h), 'score': [conf], 'center': (cx, cy)}, ...]
        
        Uses the bounding box center as the tracking point.
        """
        state = FaceState()
        # Check length properly
        state.count    = len(faces) if len(faces) > 0 else 0
        state.detected = (state.count >= 1)
        
        if state.detected:
            # cvzone returns face dict with 'bbox' key
            bbox = faces[0]['bbox']
            bx, by, bw, bh = bbox
            
            # Normalize center coordinates
            state.cx = (bx + bw / 2) / frame_width
            state.cy = (by + bh / 2) / frame_height
            
            # Face size as ratio of frame area
            state.face_size = (bw * bh) / (frame_width * frame_height)
            
            # Store bbox for reference
            state.bbox = (bx, by, bw, bh)
        
        return state
