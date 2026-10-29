"""
core/models/detection_result.py
────────────────────────────────
Single detection event — the universal data unit flowing through the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Tuple


@dataclass
class DetectionResult:
    """
    One detected object / event in one frame.

    Fields
    ──────
    camera_name  : which camera produced this frame
    detector     : which detector found this (e.g. "FaceDetector")
    label        : human-readable label (e.g. "face", "person", "motion")
    confidence   : 0.0 – 1.0 (1.0 if detector doesn't produce scores)
    timestamp    : when the frame was captured
    bounding_box : (x, y, w, h) in pixels — None for non-spatial events
    frame_index  : sequential frame number within this stream
    """

    camera_name: str
    detector: str
    label: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    frame_index: int = 0

    # filled in by the pipeline after creation
    id: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> DetectionResult:
        d = dict(d)
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        if d.get("bounding_box"):
            d["bounding_box"] = tuple(d["bounding_box"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
