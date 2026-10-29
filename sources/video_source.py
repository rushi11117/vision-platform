"""
sources/video_source.py
────────────────────────
Yields frames from a video file (MP4, AVI, MOV, MKV, etc.).
"""

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from core.contracts.frame_source import FrameSource
from core.exceptions.custom_exceptions import CorruptVideoError

log = logging.getLogger(__name__)


class VideoSource(FrameSource):
    """Reads a video file and yields frames sequentially."""

    def __init__(self, path: str, source_name: str = "", frame_skip: int = 0):
        self._path = Path(path)
        self._name = source_name or self._path.stem
        self._frame_skip = max(0, frame_skip)   # yield every (skip+1)th frame
        self._cap: cv2.VideoCapture | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    def frames(self) -> Generator[np.ndarray, None, None]:
        if not self._path.exists():
            raise CorruptVideoError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise CorruptVideoError(f"Cannot open video: {self._path}")

        log.info(
            "VideoSource [%s] %.1f fps, %d frames",
            self._path.name, self.fps, self.total_frames,
        )

        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if self._frame_skip == 0 or idx % (self._frame_skip + 1) == 0:
                yield frame
            idx += 1

        log.info("VideoSource [%s] finished (%d frames read)", self._path.name, idx)

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
