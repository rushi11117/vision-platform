"""
detectors/motion_detector.py
─────────────────────────────
Detects motion using frame differencing OR background subtraction (MOG2).

Config keys (config.yaml → detection.motion):
  threshold             : minimum contour area to count as motion (default 1000)
  blur_kernel           : Gaussian blur size for noise reduction (default 5)
  diff_threshold        : pixel brightness diff to mark as changed (default 25)
  background_subtractor : if true uses MOG2, else frame differencing
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from core.contracts.detector import Detector
from core.models.detection_result import DetectionResult
from config.settings import MotionDetectionConfig

log = logging.getLogger(__name__)


class MotionDetector(Detector):
    """Detects motion via frame differencing or MOG2 background subtraction."""

    def __init__(self, cfg: MotionDetectionConfig | None = None):
        self._cfg = cfg or MotionDetectionConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._in_motion = False  # for motion start/end event tracking

        if self._cfg.background_subtractor:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)
            log.info("MotionDetector using MOG2 background subtractor")
        else:
            self._bg_sub = None
            log.info("MotionDetector using frame differencing")

    @property
    def detector_name(self) -> str:
        return "MotionDetector"

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        k = self._cfg.blur_kernel
        if k % 2 == 0:
            k += 1  # kernel must be odd

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (k, k), 0
        )

        if self._bg_sub:
            mask = self._bg_sub.apply(gray)
        else:
            if self._prev_gray is None:
                self._prev_gray = gray
                return []
            diff = cv2.absdiff(self._prev_gray, gray)
            _, mask = cv2.threshold(diff, self._cfg.diff_threshold, 255, cv2.THRESH_BINARY)

        self._prev_gray = gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self._cfg.threshold:
                x, y, w, h = cv2.boundingRect(cnt)
                results.append(
                    DetectionResult(
                        camera_name="",
                        detector=self.detector_name,
                        label="motion",
                        confidence=min(area / (self._cfg.threshold * 10), 1.0),
                        bounding_box=(int(x), int(y), int(w), int(h)),
                    )
                )

        if results:
            log.debug("MotionDetector found %d motion region(s)", len(results))
        return results
