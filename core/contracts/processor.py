"""
core/contracts/processor.py
────────────────────────────
Abstract base for all result processors.

Processors run after detection and handle display, saving, alerting, etc.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata


class Processor(ABC):
    """
    Handles detection results for one frame.

    Contract
    ────────
    • process() is called for every frame, even when detections is [].
    • Processors must be non-blocking; heavy I/O should use threads internally.
    • release() cleans up resources (open files, DB connections, windows).
    """

    @abstractmethod
    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        """Handle frame + detections. Must not raise (log instead)."""

    def release(self) -> None:
        """Optional cleanup. Override as needed."""
