"""
core/contracts/frame_source.py
───────────────────────────────
Abstract base for all video/image input sources.

Implement FrameSource to add any new source (RTSP, file, phone…).
"""

from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


class FrameSource(ABC):
    """
    Represents any source of image frames.

    Contract
    ────────
    • frames() must yield numpy arrays (BGR, uint8) one at a time.
    • release() must free any held resources (cameras, file handles).
    • name property identifies the source in logs and DB records.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name (used in logs and DB)."""

    @abstractmethod
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield frames one by one.
        Raises SourceError on unrecoverable failures.
        """

    @abstractmethod
    def release(self) -> None:
        """Release hardware/file resources."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
