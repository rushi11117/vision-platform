"""
sources/webcam_source.py
─────────────────────────
Yields live frames from a USB or built-in laptop webcam.
"""

import logging
import time
from typing import Generator

import cv2
import numpy as np

from core.contracts.frame_source import FrameSource
from core.exceptions.custom_exceptions import CameraDisconnectedError

log = logging.getLogger(__name__)


class WebcamSource(FrameSource):
    """Streams live frames from a local webcam (index-based)."""

    def __init__(
        self,
        index: int = 0,
        source_name: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 3.0,
    ):
        self._index = index
        self._name = source_name or f"Webcam-{index}"
        self._width = width
        self._height = height
        self._fps = fps
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise CameraDisconnectedError(f"Cannot open webcam index {self._index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        log.info("WebcamSource [%s] opened (index=%d)", self._name, self._index)
        return cap

    def frames(self) -> Generator[np.ndarray, None, None]:
        self._running = True
        attempts = 0

        while self._running:
            try:
                self._cap = self._open()
                attempts = 0
                while self._running:
                    ok, frame = self._cap.read()
                    if not ok:
                        raise CameraDisconnectedError(f"Read failed on {self._name}")
                    yield frame
            except CameraDisconnectedError as exc:
                attempts += 1
                if self._reconnect_attempts > 0 and attempts > self._reconnect_attempts:
                    log.error("WebcamSource [%s] max reconnects reached", self._name)
                    raise
                log.warning(
                    "WebcamSource [%s] disconnected (%s). Retry %d in %.1fs…",
                    self._name, exc, attempts, self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
            finally:
                if self._cap:
                    self._cap.release()
                    self._cap = None

    def stop(self) -> None:
        self._running = False

    def release(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
