"""
detectors/qr_detector.py
─────────────────────────
Detects QR codes and barcodes using OpenCV's built-in QRCodeDetector.
(Barcodes require opencv-contrib-python; we skip gracefully if unavailable.)
"""

import logging
from typing import List

import cv2
import numpy as np

from core.contracts.detector import Detector
from core.models.detection_result import DetectionResult

log = logging.getLogger(__name__)


class QRDetector(Detector):
    """Detects QR codes in a frame. Decodes their text payload."""

    def __init__(self):
        self._detector = cv2.QRCodeDetector()
        log.info("QRDetector initialised")

    @property
    def detector_name(self) -> str:
        return "QRDetector"

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        data, points, _ = self._detector.detectAndDecode(frame)
        if not data or points is None:
            return []

        pts = points[0].astype(int)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        w = pts[:, 0].max() - x
        h = pts[:, 1].max() - y

        log.debug("QRDetector decoded: %s", data)
        return [
            DetectionResult(
                camera_name="",
                detector=self.detector_name,
                label=f"qr:{data}",
                confidence=1.0,
                bounding_box=(int(x), int(y), int(w), int(h)),
            )
        ]


# ── Future detector template ─────────────────────────────────
# To add a new detector (e.g. FireDetector):
#
#   class FireDetector(Detector):
#       @property
#       def detector_name(self): return "FireDetector"
#       def detect(self, frame): ...  ← implement here
#
# Then register it in engines/detection_engine.py's DETECTOR_MAP.
# No other existing file needs to change.
