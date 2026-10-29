"""
sources/image_source.py
────────────────────────
Yields a single frame from any supported image file format.
"""

import logging
from pathlib import Path
from typing import Generator

import numpy as np

from core.contracts.frame_source import FrameSource
from utils.image_utils import read_image

log = logging.getLogger(__name__)


class ImageSource(FrameSource):
    """Loads and yields a single frame from an image file."""

    def __init__(self, path: str | Path, source_name: str = "Image"):
        self._path = Path(path)
        self._name = source_name

        if not self._path.is_file():
            raise FileNotFoundError(f"Image not found: {self._path}")

    @property
    def name(self) -> str:
        """Required by FrameSource interface."""
        return self._name

    @property
    def source_name(self) -> str:
        """Alias for backwards compatibility."""
        return self._name

    def frames(self) -> Generator[np.ndarray, None, None]:
        log.info("ImageSource [%s] yielding 1 frame", self._path.name)

        frame = read_image(self._path)

        if frame is None:
            log.error("Failed to decode image: %s", self._path)
            return

        # Yield frame directly as a numpy array, not as a (success, frame) tuple
        yield frame

    def release(self) -> None:
        pass