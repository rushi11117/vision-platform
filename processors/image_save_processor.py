"""
processors/image_save_processor.py
────────────────────────────────────
Saves annotated snapshot images when detections occur.
Supports multi-format encoding (JPG, PNG, WEBP, TIFF, BMP) with thin borders.
"""

import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np

from config.settings import ImageSaveConfig, SnapshotConfig
from core.contracts.processor import Processor
from core.exceptions.custom_exceptions import DiskFullError
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata
from utils.image_utils import draw_detections, write_image

log = logging.getLogger(__name__)


class ImageSaveProcessor(Processor):
    """Saves annotated frames to disk when detections are present."""

    def __init__(
        self,
        cfg: ImageSaveConfig | None = None,
        snap_cfg: SnapshotConfig | None = None,
    ):
        self._cfg = cfg or ImageSaveConfig()
        self._snap = snap_cfg or SnapshotConfig()
        self._out_dir = Path(self._cfg.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        log.info("ImageSaveProcessor -> %s", self._out_dir)

    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        if self._cfg.save_on_detection and not detections:
            return

        self._check_disk()

        # Render 1px bounding boxes and confidence labels on a copy
        annotated_frame = draw_detections(
            frame=frame,
            detections=detections,
            draw_boxes=True,
            draw_labels=True,
            thickness=1,  # Half thickness
        )

        labels = {d.label for d in detections} or {"frame"}
        label_str = "_".join(sorted(labels))

        ts = metadata.timestamp.strftime(self._snap.timestamp_format)
        filename = (
            self._snap.filename_pattern
            .replace("{timestamp}", ts)
            .replace("{label}", label_str)
            .replace("{camera}", metadata.camera_name)
        )

        # Enforce extension from configuration (supports jpg, png, webp, tiff, etc.)
        target_fmt = self._cfg.format.lower().lstrip(".")
        suffix = f".{target_fmt}"
        if not filename.lower().endswith(suffix):
            filename = f"{Path(filename).stem}{suffix}"

        out_path = self._out_dir / filename

        # Memory buffer write (handles non-ASCII paths and cross-platform encoding)
        success = write_image(
            file_path=out_path,
            frame=annotated_frame,
            quality=self._cfg.quality,
        )

        if success:
            log.info("ImageSaveProcessor saved %s", out_path.name)
        else:
            log.error("ImageSaveProcessor failed to write %s", out_path)

    def _check_disk(self) -> None:
        usage = shutil.disk_usage(self._out_dir)
        if usage.free < 100 * 1024 * 1024:  # < 100 MB free
            raise DiskFullError(f"Low disk space: {usage.free // (1024 * 1024)} MB free")

    def release(self) -> None:
        pass