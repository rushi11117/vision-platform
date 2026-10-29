"""
utils/source_factory.py
────────────────────────
Factory function that constructs the correct FrameSource from any stream value.
Supports the full spectrum of OpenCV-decodable formats without third-party libraries.

Usage:
    from utils.source_factory import build_source
    source = build_source(stream="rtsp://...", name="Gate1", cfg=platform_cfg)
"""

import logging
import os
from pathlib import Path
from typing import Union

from config.settings import PlatformConfig
from core.contracts.frame_source import FrameSource

log = logging.getLogger(__name__)

# Complete set of native OpenCV-supported image formats
_IMAGE_EXTS = {
    # Standard web & everyday rasters
    ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib", ".webp",
    # High-depth, scientific, and archive formats
    ".tiff", ".tif", ".jp2",
    # Portable Anymap / Netpbm
    ".pbm", ".pgm", ".ppm", ".pnm", ".pxm",
    # Sun Raster and HDR formats
    ".sr", ".ras", ".hdr", ".pic", ".exr",
}

# Complete set of native OpenCV/FFmpeg-supported video containers
_VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v",
    ".webm", ".ts", ".mts", ".m2ts", ".3gp", ".mpg", ".mpeg",
    ".vob", ".ogv",
}

_STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def build_source(
    stream: Union[str, int],
    name: str,
    cfg: PlatformConfig,
) -> FrameSource:
    """Auto-detect the stream type and return the correct FrameSource."""
    w = cfg.workers

    # 1. Hardware webcam index (int or numeric string)
    if isinstance(stream, int) or (isinstance(stream, str) and str(stream).isdigit()):
        from sources.webcam_source import WebcamSource
        log.info("Source [%s]: webcam index=%s", name, stream)
        return WebcamSource(
            index=int(stream),
            source_name=name,
            reconnect_attempts=w.max_reconnect_attempts,
            reconnect_delay=float(w.reconnect_delay_seconds),
        )

    s = str(stream).strip()

    # 2. Network streams (RTSP, RTMP, HTTP/HTTPS IP camera feeds)
    if s.lower().startswith(_STREAM_SCHEMES):
        from sources.rtsp_source import RTSPSource
        log.info("Source [%s]: network stream (%s)", name, s.split("://")[0].upper())
        return RTSPSource(
            url=s,
            source_name=name,
            reconnect_attempts=w.max_reconnect_attempts,
            reconnect_delay=float(w.reconnect_delay_seconds),
            frame_skip=w.frame_skip,
        )

    # 3. Local file or directory paths
    path = Path(s)
    suffix = path.suffix.lower()

    if suffix in _IMAGE_EXTS:
        from sources.image_source import ImageSource
        log.info("Source [%s]: image file (%s)", name, suffix)
        return ImageSource(path=s, source_name=name)

    if suffix in _VIDEO_EXTS:
        from sources.video_source import VideoSource
        log.info("Source [%s]: video file (%s)", name, suffix)
        return VideoSource(path=s, source_name=name, frame_skip=w.frame_skip)

    if path.is_dir() or os.path.isdir(s):
        from sources.folder_source import FolderSource
        log.info("Source [%s]: image folder", name)
        return FolderSource(folder_path=s, source_name=name)

    # 4. Fallback: attempt generic URL/connection via RTSPSource
    from sources.rtsp_source import RTSPSource
    log.warning("Source [%s]: unrecognized format '%s', falling back to stream source", name, s)
    return RTSPSource(url=s, source_name=name)