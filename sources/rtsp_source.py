"""
sources/rtsp_source.py
───────────────────────
Yields frames from RTSP CCTV cameras, IP cameras, or Android phone streams.
Handles disconnects and reconnects automatically.

Works for:
  rtsp://admin:pass@192.168.1.100:554/live  (CCTV / NVR)
  http://192.168.1.101:8080/video            (DroidCam / IP Webcam)
  http://192.168.1.102/mjpg/video.mjpg       (IP camera MJPEG)
"""

import logging
import time
from typing import Generator

import cv2
import numpy as np

from core.contracts.frame_source import FrameSource
from core.exceptions.custom_exceptions import RTSPTimeoutError

log = logging.getLogger(__name__)


class RTSPSource(FrameSource):
    """Streams from any OpenCV-compatible network URL with auto-reconnect."""

    def __init__(
        self,
        url: str,
        source_name: str = "",
        reconnect_attempts: int = 10,
        reconnect_delay: float = 5.0,
        read_timeout_ms: int = 5000,
        frame_skip: int = 0,
    ):
        self._url = url
        self._name = source_name or url
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._read_timeout_ms = read_timeout_ms
        self._frame_skip = max(0, frame_skip)
        self._cap: cv2.VideoCapture | None = None
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._url)
        # Reduce initial buffering latency for live streams
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RTSPTimeoutError(f"Cannot connect to stream: {self._url}")
        log.info("RTSPSource [%s] connected", self._name)
        return cap

    def frames(self) -> Generator[np.ndarray, None, None]:
        self._running = True
        attempts = 0
        frame_idx = 0

        while self._running:
            try:
                self._cap = self._open()
                attempts = 0
                while self._running:
                    ok, frame = self._cap.read()
                    if not ok:
                        raise RTSPTimeoutError(f"Stream read failed: {self._name}")
                    if self._frame_skip == 0 or frame_idx % (self._frame_skip + 1) == 0:
                        yield frame
                    frame_idx += 1
            except RTSPTimeoutError as exc:
                attempts += 1
                if self._reconnect_attempts > 0 and attempts > self._reconnect_attempts:
                    log.error("RTSPSource [%s] max reconnects reached. Giving up.", self._name)
                    raise
                log.warning(
                    "RTSPSource [%s] lost (%s). Reconnecting %d/%d in %.1fs…",
                    self._name, exc, attempts, self._reconnect_attempts, self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
            finally:
                if self._cap:
                    self._cap.release()
                    self._cap = None

    def stop(self) -> None:
        self._running = False

    def release(self) -> None:
        self.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
