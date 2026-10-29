"""
repositories/postgres_repository.py
─────────────────────────────────────
PostgreSQL implementation of the Repository contract.

Requires: pip install psycopg2-binary

Configure in config.yaml:
  database:
    type: postgres
    host: localhost
    port: 5432
    name: vision_db
    user: vision_user
    password: ""   # or set env var DB_PASSWORD
"""

import logging
import os
import threading
from datetime import datetime
from typing import List, Optional

from core.contracts.repository import Repository
from core.exceptions.custom_exceptions import DatabaseConnectionError, RepositoryError
from core.models.detection_result import DetectionResult
from config.settings import DatabaseConfig

log = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    log.warning("psycopg2 not installed. PostgreSQL support unavailable.")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id          SERIAL PRIMARY KEY,
    camera_name VARCHAR(128) NOT NULL,
    detector    VARCHAR(128) NOT NULL,
    label       VARCHAR(64)  NOT NULL,
    confidence  FLOAT        NOT NULL DEFAULT 1.0,
    timestamp   TIMESTAMPTZ  NOT NULL,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER,
    frame_index INTEGER      DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pg_timestamp ON detections(timestamp);
"""

_INSERT = """
INSERT INTO detections
    (camera_name, detector, label, confidence, timestamp,
     bbox_x, bbox_y, bbox_w, bbox_h, frame_index)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

_SELECT_RECENT = """
SELECT id, camera_name, detector, label, confidence, timestamp,
       bbox_x, bbox_y, bbox_w, bbox_h, frame_index
FROM   detections
ORDER BY id DESC
LIMIT  %s;
"""


class PostgreSQLRepository(Repository):
    """Thread-safe PostgreSQL storage for detection events."""

    def __init__(self, cfg: DatabaseConfig):
        if not _PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2-binary is required for PostgreSQL. Run: pip install psycopg2-binary")
        self._cfg = cfg
        self._lock = threading.Lock()
        self._conn = None
        self._connect()

    def _connect(self) -> None:
        password = self._cfg.password or os.environ.get("DB_PASSWORD", "")
        try:
            self._conn = psycopg2.connect(
                host=self._cfg.host,
                port=self._cfg.port,
                dbname=self._cfg.name,
                user=self._cfg.user,
                password=password,
            )
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
            self._conn.commit()
            log.info("PostgreSQLRepository connected to %s:%s/%s", self._cfg.host, self._cfg.port, self._cfg.name)
        except Exception as exc:
            raise DatabaseConnectionError(f"PostgreSQL connect failed: {exc}") from exc

    def save(self, result: DetectionResult) -> None:
        bbox = result.bounding_box or (None, None, None, None)
        row = (
            result.camera_name, result.detector, result.label,
            result.confidence, result.timestamp,
            bbox[0], bbox[1], bbox[2], bbox[3],
            result.frame_index,
        )
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(_INSERT, row)
                self._conn.commit()
            except Exception as exc:
                self._conn.rollback()
                raise RepositoryError(f"PostgreSQL insert failed: {exc}") from exc

    def get_recent(self, limit: int = 100) -> List[DetectionResult]:
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(_SELECT_RECENT, (limit,))
                    rows = cur.fetchall()
            except Exception as exc:
                raise RepositoryError(f"PostgreSQL query failed: {exc}") from exc

        results = []
        for row in rows:
            (rid, cam, det, label, conf, ts, bx, by, bw, bh, fidx) = row
            bbox = (bx, by, bw, bh) if bx is not None else None
            results.append(DetectionResult(
                id=rid, camera_name=cam, detector=det, label=label,
                confidence=conf, timestamp=ts, bounding_box=bbox, frame_index=fidx or 0,
            ))
        return results

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                log.info("PostgreSQLRepository connection closed")
