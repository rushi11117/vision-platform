"""
detectors/person_detector.py
─────────────────────────────
Detects pedestrians and persons using OpenCV HOG + Linear SVM.
"""

import logging
from typing import List

import cv2
import numpy as np

from config.settings import PersonDetectionConfig
from core.contracts.detector import Detector
from core.models.detection_result import DetectionResult

log = logging.getLogger(__name__)


class PersonDetector(Detector):
    """
    Histogram of Oriented Gradients (HOG) + Linear SVM person detector.
    """

    def __init__(self, cfg: PersonDetectionConfig | None = None):
        self._cfg = cfg or PersonDetectionConfig()
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        log.info("PersonDetector initialised")

    @property
    def detector_name(self) -> str:
        return "PersonDetector"

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if frame is None or frame.size == 0:
            return []

        # Run HOG detector
        rects, weights = self._hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
        )

        if len(rects) == 0:
            return []

        # Flatten weights to 1D to prevent indexing errors on both scalars and 2D arrays
        weights_flat = np.asarray(weights).ravel() if weights is not None and len(weights) > 0 else np.array([])

        results: List[DetectionResult] = []
        conf_threshold = float(getattr(self._cfg, "confidence_threshold", 0.0))
        frame_h, frame_w = frame.shape[:2]

        for i, (x, y, w, h) in enumerate(rects):
            # Safe scalar extraction
            if i < len(weights_flat):
                raw_score = float(weights_flat[i])
                # Convert raw SVM margin distance to a 0.0 - 1.0 confidence score
                confidence = float(1.0 / (1.0 + np.exp(-raw_score)))
            else:
                confidence = conf_threshold

            if conf_threshold > 0.0 and confidence < conf_threshold:
                continue

            # Clamp coordinates inside image boundary
            clamped_x = max(0, int(x))
            clamped_y = max(0, int(y))
            clamped_w = min(frame_w - clamped_x, max(0, int(w)))
            clamped_h = min(frame_h - clamped_y, max(0, int(h)))

            results.append(
                DetectionResult(
                    camera_name="",
                    detector=self.detector_name,
                    label="person",
                    confidence=round(confidence, 2),
                    bounding_box=(clamped_x, clamped_y, clamped_w, clamped_h),
                )
            )

        if results:
            log.info("PersonDetector found %d person(s)", len(results))

        return results