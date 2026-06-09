"""
Data models for face tracking system
"""
from dataclasses import dataclass


@dataclass
class FaceState:
    """One frame's worth of face detection, normalized to [0,1]."""
    detected:   bool  = False
    count:      int   = 0
    cx:         float = 0.0    # normalized center x
    cy:         float = 0.0    # normalized center y
    face_size:  float = 0.0    # bbox area / frame area
    bbox:       tuple = (0, 0, 0, 0)  # (x, y, w, h) pixel coordinates
