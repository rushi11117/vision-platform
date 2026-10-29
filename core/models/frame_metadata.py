"""
core/models/frame_metadata.py
──────────────────────────────
Contextual info attached to every frame as it moves through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FrameMetadata:
    """
    Carries context alongside the raw frame numpy array.

    camera_name  : source camera label
    frame_index  : 0-based frame counter for this stream session
    timestamp    : datetime the frame was grabbed
    fps          : source FPS hint (0 = unknown / image)
    width        : frame width in pixels
    height       : frame height in pixels
    """

    camera_name: str
    frame_index: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    fps: float = 0.0
    width: int = 0
    height: int = 0
