"""
workers/camera_worker.py
─────────────────────────
Manages one camera pipeline in its own thread.
The CameraManager runs N workers via ThreadPoolExecutor.

Each worker:
  1. Builds the source from config (webcam / RTSP / video / image)
  2. Builds the DetectionEngine for the camera's detector list
  3. Builds the configured processors (display, save, DB, alert)
  4. Runs the VisionPipeline until stop() or source exhausted
"""

import logging
from typing import List

from config.settings import CameraConfig, PlatformConfig
from core.contracts.processor import Processor
from core.contracts.repository import Repository
from engines.detection_engine import DetectionEngine
from engines.pipeline_engine import VisionPipeline
from processors.alert_processor import AlertProcessor
from processors.database_processor import DatabaseProcessor
from processors.display_processor import DisplayProcessor
from processors.image_save_processor import ImageSaveProcessor
from processors.video_save_processor import VideoSaveProcessor
from utils.source_factory import build_source

log = logging.getLogger(__name__)


def _build_processors(cam: CameraConfig, cfg: PlatformConfig, repository: Repository) -> List[Processor]:
    processors: List[Processor] = []
    pc = cfg.processors

    if pc.display.get("enabled", True) if isinstance(pc.display, dict) else pc.display.enabled:
        processors.append(DisplayProcessor(cfg=pc.display if not isinstance(pc.display, dict) else None))

    if pc.image_save.enabled:
        processors.append(ImageSaveProcessor(cfg=pc.image_save, snap_cfg=cfg.snapshot))

    if pc.video_save.enabled:
        processors.append(VideoSaveProcessor(cfg=pc.video_save, source_name=cam.name))

    if pc.database.get("enabled", True) if isinstance(pc.database, dict) else True:
        processors.append(DatabaseProcessor(repository=repository))

    if pc.alert.enabled:
        processors.append(AlertProcessor(cfg=pc.alert))

    return processors


class CameraWorker:
    """
    Manages the full pipeline for a single camera.
    Call run() inside a thread; call stop() to shut it down.
    """

    def __init__(self, cam_cfg: CameraConfig, platform_cfg: PlatformConfig, repository: Repository):
        self._cam_cfg = cam_cfg
        self._platform_cfg = platform_cfg
        self._repository = repository
        self._pipeline: VisionPipeline | None = None

    def run(self) -> None:
        log.info("CameraWorker starting: %s", self._cam_cfg.name)
        try:
            source = build_source(self._cam_cfg.stream, self._cam_cfg.name, self._platform_cfg)
            engine = DetectionEngine(self._cam_cfg.detectors, self._platform_cfg)
            processors = _build_processors(self._cam_cfg, self._platform_cfg, self._repository)
            self._pipeline = VisionPipeline(
                source=source,
                detection_engine=engine,
                processors=processors,
                workers_cfg=self._platform_cfg.workers,
            )
            self._pipeline.run()
        except Exception as exc:
            log.exception("CameraWorker [%s] fatal error: %s", self._cam_cfg.name, exc)

    def stop(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
