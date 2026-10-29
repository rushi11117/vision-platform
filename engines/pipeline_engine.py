"""
engines/pipeline_engine.py
───────────────────────────
Core processing pipeline: Source → DetectionEngine → Processors.

One pipeline instance = one camera/source.
Designed to run inside a CameraWorker thread.
"""

import logging
from datetime import datetime
from typing import List

import cv2

from core.contracts.frame_source import FrameSource
from core.contracts.processor import Processor
from core.exceptions.custom_exceptions import SourceError, ProcessorError
from core.models.frame_metadata import FrameMetadata
from config.settings import WorkersConfig
from engines.detection_engine import DetectionEngine

log = logging.getLogger(__name__)


class VisionPipeline:
    """
    Orchestrates the full detection pipeline for a single source.

    Flow:
      for each frame in source:
          metadata = FrameMetadata(camera, timestamp, …)
          detections = detection_engine.run(frame, metadata)
          for each processor:
              processor.process(frame, detections, metadata)

    Stopping:
      • Call pipeline.stop() from another thread.
      • Or a DisplayProcessor sets stop_requested = True on 'q' keypress.
    """

    def __init__(
        self,
        source: FrameSource,
        detection_engine: DetectionEngine,
        processors: List[Processor],
        workers_cfg: WorkersConfig | None = None,
    ):
        self._source = source
        self._engine = detection_engine
        self._processors = processors
        self._cfg = workers_cfg or WorkersConfig()
        self._running = False
        self._frame_count = 0

    def run(self) -> None:
        """Start the pipeline. Blocks until source is exhausted or stop() is called."""
        self._running = True
        log.info("Pipeline [%s] starting", self._source.name)

        try:
            for frame in self._source.frames():
                if not self._running:
                    break

                # frame skip (0 = no skip)
                self._frame_count += 1
                skip = self._cfg.frame_skip
                if skip > 0 and (self._frame_count % (skip + 1)) != 0:
                    continue

                h, w = frame.shape[:2]
                metadata = FrameMetadata(
                    camera_name=self._source.name,
                    frame_index=self._frame_count,
                    timestamp=datetime.now(),
                    width=w,
                    height=h,
                )

                detections = self._engine.run(frame, metadata)

                for processor in self._processors:
                    try:
                        processor.process(frame, detections, metadata)
                        # stop on 'q' from display
                        if hasattr(processor, "stop_requested") and processor.stop_requested:
                            self._running = False
                            break
                    except ProcessorError as exc:
                        log.error("Processor %s error: %s", type(processor).__name__, exc)
                    except Exception as exc:
                        log.exception("Unexpected processor error: %s", exc)

        except SourceError as exc:
            log.error("Pipeline [%s] source error: %s", self._source.name, exc)
        except Exception as exc:
            log.exception("Pipeline [%s] unexpected error: %s", self._source.name, exc)
        finally:
            self._release_all()
            log.info("Pipeline [%s] stopped after %d frames", self._source.name, self._frame_count)

    def stop(self) -> None:
        """Signal the pipeline to stop after the current frame."""
        self._running = False
        log.info("Pipeline [%s] stop requested", self._source.name)

    def _release_all(self) -> None:
        try:
            self._source.release()
        except Exception as exc:
            log.warning("Source release error: %s", exc)

        for proc in self._processors:
            try:
                proc.release()
            except Exception as exc:
                log.warning("Processor release error: %s", exc)
