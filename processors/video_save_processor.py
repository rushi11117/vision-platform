"""
processors/video_save_processor.py
────────────────────────────────────
Saves processed video to disk.

Modes:
  continuous : records everything, one file per session
  event      : starts recording on detection, stops after idle_seconds with no detection
"""

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from core.contracts.processor import Processor
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata
from config.settings import VideoSaveConfig

log = logging.getLogger(__name__)


class VideoSaveProcessor(Processor):
    """Records processed frames to an MP4/AVI file."""

    _IDLE_SECONDS = 5  # stop recording after this many idle seconds (event mode)

    def __init__(self, cfg: VideoSaveConfig | None = None, source_name: str = ""):
        self._cfg = cfg or VideoSaveConfig()
        self._source = source_name
        self._out_dir = Path(self._cfg.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_file: Optional[Path] = None
        self._last_detection_time: Optional[datetime] = None
        self._pre_buffer: Deque[Tuple[np.ndarray, FrameMetadata]] = deque(
            maxlen=int(self._cfg.fps * self._cfg.pre_buffer_seconds)
        )
        log.info(
            "VideoSaveProcessor mode=%s → %s", self._cfg.mode, self._out_dir
        )

    # ── Internal helpers ─────────────────────────────────────

    def _fourcc(self) -> int:
        return cv2.VideoWriter_fourcc(*self._cfg.codec)

    def _new_filepath(self, label: str = "video") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{self._source}_{label}.mp4"
        return self._out_dir / name

    def _start_writer(self, frame: np.ndarray, label: str = "event") -> None:
        h, w = frame.shape[:2]
        self._current_file = self._new_filepath(label)
        self._writer = cv2.VideoWriter(
            str(self._current_file),
            self._fourcc(),
            float(self._cfg.fps),
            (w, h),
        )
        log.info("VideoSaveProcessor started recording → %s", self._current_file.name)

    def _write(self, frame: np.ndarray) -> None:
        if self._writer and self._writer.isOpened():
            self._writer.write(frame)

    def _stop_writer(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
            log.info("VideoSaveProcessor stopped recording: %s", self._current_file)
            self._current_file = None

    # ── Processor interface ──────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        if self._cfg.mode == "continuous":
            self._handle_continuous(frame, metadata)
        else:
            self._handle_event(frame, detections, metadata)

    def _handle_continuous(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        if self._writer is None:
            self._start_writer(frame, "continuous")
        self._write(frame)

    def _handle_event(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        self._pre_buffer.append((frame.copy(), metadata))

        if detections:
            self._last_detection_time = metadata.timestamp
            if self._writer is None:
                label = detections[0].label if detections else "event"
                self._start_writer(frame, label)
                # flush pre-buffer
                for buf_frame, _ in self._pre_buffer:
                    self._write(buf_frame)
                self._pre_buffer.clear()
            self._write(frame)
        elif self._writer is not None:
            self._write(frame)
            if self._last_detection_time:
                elapsed = (metadata.timestamp - self._last_detection_time).total_seconds()
                if elapsed > self._IDLE_SECONDS:
                    self._stop_writer()

    def release(self) -> None:
        self._stop_writer()
