"""
engines/detection_engine.py
────────────────────────────
Builds and runs the configured set of detectors on each frame.

Add a new detector by:
  1. Creating detectors/your_detector.py (subclass Detector)
  2. Adding an entry to DETECTOR_MAP below
  3. Listing its key in config.yaml → cameras[i].detectors
"""

import logging
from typing import Dict, List, Type

import numpy as np

from config.settings import PlatformConfig
from core.contracts.detector import Detector
from core.exceptions.custom_exceptions import DetectorError
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata

# ── Registry: name → class ───────────────────────────────────
from detectors.face_detector import FaceDetector
from detectors.motion_detector import MotionDetector
from detectors.person_detector import PersonDetector
from detectors.qr_detector import QRDetector

DETECTOR_MAP: Dict[str, Type[Detector]] = {
    "face": FaceDetector,
    "person": PersonDetector,
    "motion": MotionDetector,
    "qr": QRDetector,
    # "vehicle": VehicleDetector,   ← add future detectors here
    # "fire":    FireDetector,
    # "plate":   NumberPlateDetector,
}

log = logging.getLogger(__name__)


class DetectionEngine:
    """
    Runs multiple detectors on each frame.

    Design
    ──────
    • Detectors are built once at startup from the config.
    • Each detector's detect() is called in sequence (single thread per camera).
    • Detector failures are caught and logged; other detectors still run.
    """

    def __init__(self, detector_names: List[str], cfg: PlatformConfig):
        self._detectors: List[Detector] = []
        self._cfg = cfg

        for name in detector_names:
            if name not in DETECTOR_MAP:
                log.warning("Unknown detector '%s' — skipping", name)
                continue
            detector_class = DETECTOR_MAP[name]
            detector = self._build(name, detector_class, cfg)
            if detector:
                self._detectors.append(detector)

        log.info("DetectionEngine ready with: %s", [d.detector_name for d in self._detectors])

    def _build(self, name: str, cls: Type[Detector], cfg: PlatformConfig) -> Detector | None:
        """Instantiate a detector, passing its config section if accepted."""
        try:
            det_cfg = getattr(cfg.detection, name, None)
            if det_cfg is not None:
                try:
                    return cls(det_cfg)
                except TypeError:
                    return cls()      # detector doesn't accept config
            return cls()
        except Exception as exc:
            log.error("Failed to build detector '%s': %s", name, exc)
            return None

    def run(self, frame: np.ndarray, metadata: FrameMetadata) -> List[DetectionResult]:
        """Run all detectors; return merged results with camera_name filled in."""
        all_results: List[DetectionResult] = []

        for detector in self._detectors:
            try:
                results = detector.detect(frame)
                for r in results:
                    r.camera_name = metadata.camera_name
                    r.frame_index = metadata.frame_index
                    r.timestamp = metadata.timestamp
                all_results.extend(results)
            except DetectorError as exc:
                log.error("DetectorError in %s: %s", detector.detector_name, exc)
            except Exception as exc:
                log.exception("Unexpected error in %s: %s", detector.detector_name, exc)

        return all_results

    @property
    def active_detectors(self) -> List[str]:
        return [d.detector_name for d in self._detectors]
