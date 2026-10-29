"""
sources/folder_source.py
─────────────────────────
Yields frames from every image file in a directory.
Supported extensions: jpg, jpeg, png, bmp, tiff, tif, webp.
"""

import logging
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np

from core.contracts.frame_source import FrameSource
from core.exceptions.custom_exceptions import SourceError

log = logging.getLogger(__name__)

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class FolderSource(FrameSource):
    """Reads all images in a folder and yields them one by one."""

    def __init__(self, folder_path: str, source_name: str = "", recursive: bool = False):
        self._folder = Path(folder_path)
        self._name = source_name or self._folder.name
        self._recursive = recursive
        self._files: List[Path] = []

    @property
    def name(self) -> str:
        return self._name

    def _collect_files(self) -> List[Path]:
        if not self._folder.exists():
            raise SourceError(f"Folder not found: {self._folder}")
        glob = self._folder.rglob if self._recursive else self._folder.glob
        files = sorted(p for p in glob("*") if p.suffix.lower() in _EXTENSIONS)
        log.info("FolderSource [%s] found %d images", self._name, len(files))
        return files

    def frames(self) -> Generator[np.ndarray, None, None]:
        self._files = self._collect_files()
        for img_path in self._files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                log.warning("Skipping unreadable image: %s", img_path)
                continue
            log.debug("FolderSource yielding %s", img_path.name)
            yield frame

    def release(self) -> None:
        pass
