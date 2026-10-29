"""
core/contracts/repository.py
──────────────────────────────
Abstract base for all data-persistence backends.

Swap SQLite ↔ PostgreSQL by switching the implementation — no other
code changes (Dependency Inversion Principle).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models.detection_result import DetectionResult


class Repository(ABC):
    """
    Persists and queries DetectionResult records.

    Contract
    ────────
    • save() writes one record.
    • get_recent() returns the N most recent records.
    • Implementations must be thread-safe (multiple workers share one repo).
    • close() releases DB connections.
    """

    @abstractmethod
    def save(self, result: DetectionResult) -> None:
        """Persist a single detection event."""

    @abstractmethod
    def get_recent(self, limit: int = 100) -> List[DetectionResult]:
        """Retrieve most-recent detection events."""

    @abstractmethod
    def close(self) -> None:
        """Close DB connection / release resources."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
