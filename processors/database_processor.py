"""
processors/database_processor.py
──────────────────────────────────
Persists every DetectionResult to the configured database via the Repository abstraction.
"""

import logging
from typing import List

import numpy as np

from core.contracts.processor import Processor
from core.contracts.repository import Repository
from core.exceptions.custom_exceptions import RepositoryError
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata

log = logging.getLogger(__name__)


class DatabaseProcessor(Processor):
    """Saves detection events to the database."""

    def __init__(self, repository: Repository):
        self._repo = repository
        log.info("DatabaseProcessor initialised with %s", type(repository).__name__)

    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        for detection in detections:
            try:
                self._repo.save(detection)
            except RepositoryError as exc:
                log.error("DatabaseProcessor failed to save detection: %s", exc)
            except Exception as exc:
                log.exception("Unexpected DB error: %s", exc)

    def release(self) -> None:
        # Repository lifecycle is owned by StreamService, not by this processor.
        # Closing here would race with other camera workers still writing.
        pass
