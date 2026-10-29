"""
services/stream_service.py
───────────────────────────
Manages all camera workers using ThreadPoolExecutor.

Supports 1 → 50+ cameras.
Each camera runs in its own thread; a single shared Repository handles DB writes
(SQLiteRepository uses WAL mode + a threading.Lock for safety).
"""

import logging
import signal
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List

from config.settings import PlatformConfig
from core.contracts.repository import Repository
from workers.camera_worker import CameraWorker

log = logging.getLogger(__name__)


def _build_repository(cfg: PlatformConfig) -> Repository:
    db = cfg.database
    if db.type == "postgres":
        from repositories.postgres_repository import PostgreSQLRepository
        return PostgreSQLRepository(db)
    else:
        from repositories.sqlite_repository import SQLiteRepository
        return SQLiteRepository(db.db_path)


class StreamService:
    """
    Starts and supervises one CameraWorker per enabled camera.

    Usage:
        svc = StreamService(config)
        svc.start()   # blocks until all cameras finish or Ctrl+C
    """

    def __init__(self, cfg: PlatformConfig):
        self._cfg = cfg
        self._repository: Repository | None = None
        self._workers: List[CameraWorker] = []
        self._futures: Dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        enabled = [c for c in self._cfg.cameras if c.enabled]
        if not enabled:
            log.warning("No enabled cameras in config. Nothing to do.")
            return

        log.info("StreamService starting %d camera(s)", len(enabled))
        self._repository = _build_repository(self._cfg)

        max_w = max(self._cfg.workers.max_workers, len(enabled))
        self._executor = ThreadPoolExecutor(max_workers=max_w, thread_name_prefix="cam")

        # register Ctrl+C handler
        signal.signal(signal.SIGINT, self._handle_signal)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._handle_signal)

        for cam in enabled:
            worker = CameraWorker(cam, self._cfg, self._repository)
            self._workers.append(worker)
            future = self._executor.submit(worker.run)
            self._futures[cam.name] = future
            log.info("Submitted worker for camera: %s", cam.name)

        # wait for all workers to finish
        for future in as_completed(self._futures.values()):
            try:
                future.result()  # re-raises any exception from the worker thread
            except Exception as exc:
                log.error("Worker finished with error: %s", exc)

        self._shutdown()

    def stop_all(self) -> None:
        log.info("StreamService stopping all workers…")
        for worker in self._workers:
            try:
                worker.stop()
            except Exception as exc:
                log.warning("Error stopping worker: %s", exc)

    def _handle_signal(self, signum, frame) -> None:
        log.info("Signal %s received — shutting down", signum)
        self.stop_all()

    def _shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)   # wait for all threads to finish
        if self._repository:
            try:
                self._repository.close()
            except Exception as exc:
                log.warning("Repository close error: %s", exc)
        log.info("StreamService shut down cleanly")
