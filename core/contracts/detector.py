"""
core/contracts/detector.py
───────────────────────────
Abstract base for all detection algorithms.

Add a new detector by subclassing Detector and implementing detect().
No existing code needs to change (Open/Closed Principle).
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.models.detection_result import DetectionResult


class Detector(ABC):
    """
    Runs one detection algorithm on a single frame.

    Contract
    ────────
    • detect(frame) receives a BGR uint8 numpy array.
    • Returns a list of DetectionResult (empty list if nothing found).
    • Must NOT modify the original frame in-place.
    • detector_name identifies this detector in logs and DB.
    """

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Unique name used in DB records and logs."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Analyse frame and return detections.
        Returns [] when nothing is detected.
        Raises DetectorError on hard failures.
        """
