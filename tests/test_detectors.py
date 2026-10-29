"""
tests/test_detectors.py
────────────────────────
Unit tests for FaceDetector, PersonDetector, MotionDetector.

Run with:  pytest tests/test_detectors.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from config.settings import FaceDetectionConfig, MotionDetectionConfig, PersonDetectionConfig
from core.models.detection_result import DetectionResult


# ── helpers ──────────────────────────────────────────────────

def blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def noisy_frame(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── FaceDetector tests ────────────────────────────────────────

class TestFaceDetector:
    def setup_method(self):
        from detectors.face_detector import FaceDetector
        self.detector = FaceDetector()

    def test_returns_list(self):
        result = self.detector.detect(blank_frame())
        assert isinstance(result, list)

    def test_blank_frame_no_detections(self):
        result = self.detector.detect(blank_frame())
        assert result == []

    def test_result_type(self):
        """Even on noisy data, results must be DetectionResult instances."""
        results = self.detector.detect(noisy_frame())
        for r in results:
            assert isinstance(r, DetectionResult)
            assert r.label == "face"
            assert r.detector == "FaceDetector"

    def test_bounding_box_format(self):
        results = self.detector.detect(noisy_frame())
        for r in results:
            if r.bounding_box:
                assert len(r.bounding_box) == 4
                assert all(isinstance(v, int) for v in r.bounding_box)

    def test_custom_config(self):
        from detectors.face_detector import FaceDetector
        cfg = FaceDetectionConfig(scale_factor=1.3, min_neighbors=8)
        det = FaceDetector(cfg)
        result = det.detect(blank_frame())
        assert isinstance(result, list)


# ── PersonDetector tests ──────────────────────────────────────

class TestPersonDetector:
    def setup_method(self):
        from detectors.person_detector import PersonDetector
        self.detector = PersonDetector()

    def test_returns_list(self):
        assert isinstance(self.detector.detect(blank_frame()), list)

    def test_blank_frame_no_detections(self):
        assert self.detector.detect(blank_frame()) == []

    def test_detector_name(self):
        assert self.detector.detector_name == "PersonDetector"


# ── MotionDetector tests ──────────────────────────────────────

class TestMotionDetector:
    def setup_method(self):
        from detectors.motion_detector import MotionDetector
        self.detector = MotionDetector()

    def test_first_frame_no_detections(self):
        """No previous frame → can't detect motion."""
        result = self.detector.detect(blank_frame())
        assert result == []

    def test_identical_frames_no_motion(self):
        frame = blank_frame()
        self.detector.detect(frame)
        result = self.detector.detect(frame)
        assert result == []

    def test_different_frames_detect_motion(self):
        from detectors.motion_detector import MotionDetector
        det = MotionDetector(MotionDetectionConfig(threshold=100))
        frame_a = blank_frame()
        frame_b = blank_frame()
        # draw a large white rectangle on second frame
        frame_b[50:150, 50:200] = 255
        det.detect(frame_a)
        result = det.detect(frame_b)
        assert len(result) > 0
        assert result[0].label == "motion"

    def test_mog2_mode(self):
        from detectors.motion_detector import MotionDetector
        det = MotionDetector(MotionDetectionConfig(background_subtractor=True))
        det.detect(blank_frame())
        result = det.detect(blank_frame())
        assert isinstance(result, list)

    def test_detector_name(self):
        assert self.detector.detector_name == "MotionDetector"
