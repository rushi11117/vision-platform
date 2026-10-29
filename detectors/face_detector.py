"""
detectors/face_detector.py
───────────────────────────
Detects human faces using OpenCV YuNet (ONNX model).
Handles arbitrary pitch, roll, yaw, and dense crowd scenes.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config.settings import FaceDetectionConfig
from core.contracts.detector import Detector
from core.exceptions.custom_exceptions import ModelLoadError
from core.models.detection_result import DetectionResult

log = logging.getLogger(__name__)

# Resolve path directly to D:\Learning\Survell\vision_platform\models
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceDetector(Detector):
    """
    OpenCV YuNet Deep Learning Face Detector.
    """

    def __init__(
        self,
        cfg: FaceDetectionConfig | None = None,
        model_path: Path | str = DEFAULT_MODEL_PATH,
    ):
        self._cfg = cfg or FaceDetectionConfig()
        self._model_path = Path(model_path)

        # Download model automatically if missing
        if not self._model_path.is_file():
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading YuNet model to %s ...", self._model_path)
            try:
                urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, str(self._model_path))
            except Exception as e:
                raise ModelLoadError(f"Failed to auto-download YuNet model: {e}") from e

        # Initialize OpenCV FaceDetectorYN
        score_thresh = float(getattr(self._cfg, "confidence_threshold", 0.5))
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self._model_path),
                config="",
                input_size=(320, 320),
                score_threshold=score_thresh,
                nms_threshold=0.3,
                top_k=5000,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load YuNet model from {self._model_path}: {e}") from e

        log.info("FaceDetector initialised using %s", self._model_path.name)

    @property
    def detector_name(self) -> str:
        return "FaceDetector"

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))

        _, faces = self._detector.detect(frame)

        results: List[DetectionResult] = []
        if faces is not None:
            for face in faces:
                # YuNet format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
                x, y, box_w, box_h = face[0:4].astype(int)
                score = float(face[-1])

                # Clamp bounding boxes to valid frame dimensions
                clamped_x = max(0, int(x))
                clamped_y = max(0, int(y))
                clamped_w = min(w - clamped_x, max(0, int(box_w)))
                clamped_h = min(h - clamped_y, max(0, int(box_h)))

                if clamped_w > 0 and clamped_h > 0:
                    results.append(
                        DetectionResult(
                            camera_name="",
                            detector=self.detector_name,
                            label="face",
                            confidence=round(score, 2),
                            bounding_box=(clamped_x, clamped_y, clamped_w, clamped_h),
                        )
                    )

        if results:
            log.info("FaceDetector found %d face(s)", len(results))
        else:
            log.warning("FaceDetector found 0 face(s)")

        return results