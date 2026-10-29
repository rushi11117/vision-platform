"""
repositories/sqlite_repository.py
───────────────────────────────────
Thread-safe SQLite implementation of the Repository contract.

Schema:
  detections(id, camera_name, detector, label, confidence,
             timestamp, bbox_x, bbox_y, bbox_w, bbox_h, frame_index)
"""

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.contracts.repository import Repository
from core.exceptions.custom_exceptions import DatabaseConnectionError, RepositoryError
from core.models.detection_result import DetectionResult

log = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name TEXT    NOT NULL,
    detector    TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 1.0,
    timestamp   TEXT    NOT NULL,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER,
    frame_index INTEGER DEFAULT 0
);
"""

_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);"

_INSERT = """
INSERT INTO detections
    (camera_name, detector, label, confidence, timestamp,
     bbox_x, bbox_y, bbox_w, bbox_h, frame_index)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT = """
SELECT id, camera_name, detector, label, confidence, timestamp,
       bbox_x, bbox_y, bbox_w, bbox_h, frame_index
FROM   detections
ORDER BY id DESC
LIMIT  ?;
"""


class SQLiteRepository(Repository):
    """Thread-safe SQLite storage for detection events."""

    def __init__(self, db_path: str = "./database/vision.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,   # we lock manually
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_IDX)
            self._conn.commit()
            log.info("SQLiteRepository connected: %s", self._db_path)
        except sqlite3.Error as exc:
            raise DatabaseConnectionError(f"SQLite connect failed: {exc}") from exc

    def save(self, result: DetectionResult) -> None:
        bbox = result.bounding_box or (None, None, None, None)
        row = (
            result.camera_name,
            result.detector,
            result.label,
            result.confidence,
            result.timestamp.isoformat(),
            bbox[0], bbox[1], bbox[2], bbox[3],
            result.frame_index,
        )
        with self._lock:
            if self._conn is None:
                raise RepositoryError("SQLiteRepository is closed")
            try:
                self._conn.execute(_INSERT, row)
                self._conn.commit()
            except sqlite3.Error as exc:
                raise RepositoryError(f"SQLite insert failed: {exc}") from exc

    def get_recent(self, limit: int = 100) -> List[DetectionResult]:
        with self._lock:
            try:
                rows = self._conn.execute(_SELECT_RECENT, (limit,)).fetchall()
            except sqlite3.Error as exc:
                raise RepositoryError(f"SQLite query failed: {exc}") from exc

        results = []
        for row in rows:
            (rid, cam, det, label, conf, ts, bx, by, bw, bh, fidx) = row
            bbox = (bx, by, bw, bh) if bx is not None else None
            results.append(
                DetectionResult(
                    id=rid,
                    camera_name=cam,
                    detector=det,
                    label=label,
                    confidence=conf,
                    timestamp=datetime.fromisoformat(ts),
                    bounding_box=bbox,
                    frame_index=fidx or 0,
                )
            )
        return results

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                log.info("SQLiteRepository connection closed")
