"""
tests/test_pipeline.py
───────────────────────
Integration tests: Source → DetectionEngine → Processor pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from config.settings import load_config, PlatformConfig
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata
from core.contracts.processor import Processor


# ── Helpers ───────────────────────────────────────────────────

def blank_frame(h=100, w=100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_image_file(path: Path) -> None:
    cv2.imwrite(str(path), blank_frame())


class CapturingProcessor(Processor):
    """Test double: records every call to process()."""
    def __init__(self):
        self.calls: List[tuple] = []

    def process(self, frame, detections, metadata):
        self.calls.append((frame, detections, metadata))

    def release(self):
        pass


# ── Config fixture ────────────────────────────────────────────

@pytest.fixture
def platform_cfg(tmp_path) -> PlatformConfig:
    cfg = load_config.__wrapped__() if hasattr(load_config, "__wrapped__") else None
    # Build minimal config programmatically
    from config.settings import (
        PlatformConfig, CameraConfig, DetectionConfig, ProcessorsConfig,
        DatabaseConfig, StorageConfig, LoggingConfig, WorkersConfig, SnapshotConfig,
        DisplayProcessorConfig, ImageSaveConfig, VideoSaveConfig, AlertProcessorConfig,
    )
    cfg = PlatformConfig(
        cameras=[CameraConfig(name="TestCam", stream=str(tmp_path), enabled=True, detectors=["motion"])],
        storage=StorageConfig(
            base_path=str(tmp_path),
            images_path=str(tmp_path / "images"),
            videos_path=str(tmp_path / "videos"),
            snapshots_path=str(tmp_path / "snapshots"),
            logs_path=str(tmp_path / "logs"),
        ),
        database=DatabaseConfig(db_path=str(tmp_path / "test.db")),
    )
    # ensure dirs exist
    for d in ["images", "videos", "snapshots", "logs"]:
        (tmp_path / d).mkdir(exist_ok=True)
    return cfg


# ── DetectionEngine tests ─────────────────────────────────────

class TestDetectionEngine:
    def test_build_known_detectors(self, platform_cfg):
        from engines.detection_engine import DetectionEngine
        engine = DetectionEngine(["motion", "face"], platform_cfg)
        assert "MotionDetector" in engine.active_detectors
        assert "FaceDetector" in engine.active_detectors

    def test_unknown_detector_skipped(self, platform_cfg):
        from engines.detection_engine import DetectionEngine
        engine = DetectionEngine(["motion", "nonexistent"], platform_cfg)
        assert "MotionDetector" in engine.active_detectors

    def test_run_returns_list(self, platform_cfg):
        from engines.detection_engine import DetectionEngine
        engine = DetectionEngine(["motion"], platform_cfg)
        meta = FrameMetadata(camera_name="TestCam", frame_index=0)
        result = engine.run(blank_frame(), meta)
        assert isinstance(result, list)

    def test_camera_name_filled(self, platform_cfg):
        from engines.detection_engine import DetectionEngine
        from detectors.motion_detector import MotionDetector
        from config.settings import MotionDetectionConfig

        engine = DetectionEngine(["motion"], platform_cfg)
        meta = FrameMetadata(camera_name="Gate1", frame_index=5)
        # feed two different frames to trigger motion
        frame_a = blank_frame()
        frame_b = blank_frame()
        frame_b[10:50, 10:50] = 200
        engine.run(frame_a, meta)
        results = engine.run(frame_b, meta)
        for r in results:
            assert r.camera_name == "Gate1"


# ── VisionPipeline integration ────────────────────────────────

class TestVisionPipeline:
    def test_processor_called_per_frame(self, tmp_path, platform_cfg):
        from sources.image_source import ImageSource
        from engines.detection_engine import DetectionEngine
        from engines.pipeline_engine import VisionPipeline

        img_path = tmp_path / "frame.jpg"
        make_image_file(img_path)

        source = ImageSource(str(img_path), source_name="TestCam")
        engine = DetectionEngine(["motion"], platform_cfg)
        proc = CapturingProcessor()

        pipeline = VisionPipeline(source, engine, [proc])
        pipeline.run()

        assert len(proc.calls) == 1   # one image → one call
        _, _, meta = proc.calls[0]
        assert meta.camera_name == "TestCam"

    def test_processor_receives_metadata(self, tmp_path, platform_cfg):
        from sources.image_source import ImageSource
        from engines.detection_engine import DetectionEngine
        from engines.pipeline_engine import VisionPipeline

        img_path = tmp_path / "img.jpg"
        make_image_file(img_path)

        source = ImageSource(str(img_path), source_name="Cam1")
        engine = DetectionEngine([], platform_cfg)
        proc = CapturingProcessor()

        VisionPipeline(source, engine, [proc]).run()

        frame, detections, meta = proc.calls[0]
        assert meta.camera_name == "Cam1"
        assert meta.frame_index == 1
        assert meta.width == 100
        assert meta.height == 100


# ── Repository integration ────────────────────────────────────

class TestSQLiteRepository:
    def test_save_and_retrieve(self, tmp_path):
        from repositories.sqlite_repository import SQLiteRepository
        from datetime import datetime

        repo = SQLiteRepository(str(tmp_path / "test.db"))
        det = DetectionResult(
            camera_name="Gate1", detector="FaceDetector", label="face",
            confidence=0.95, timestamp=datetime(2026, 10, 21, 10, 0, 0),
            bounding_box=(10, 20, 50, 60),
        )
        repo.save(det)
        results = repo.get_recent(10)
        assert len(results) == 1
        r = results[0]
        assert r.camera_name == "Gate1"
        assert r.label == "face"
        assert r.bounding_box == (10, 20, 50, 60)
        repo.close()

    def test_multiple_saves(self, tmp_path):
        from repositories.sqlite_repository import SQLiteRepository

        repo = SQLiteRepository(str(tmp_path / "test2.db"))
        for i in range(5):
            repo.save(DetectionResult(camera_name="Cam", detector="D", label=f"label{i}"))
        assert len(repo.get_recent(10)) == 5
        repo.close()
