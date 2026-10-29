"""
services/health_service.py
───────────────────────────
Health checks for the platform: DB, disk space, camera reachability.
Used by the CLI and optionally a monitoring endpoint.

Usage:
    from services.health_service import HealthService
    hs = HealthService(cfg)
    report = hs.check_all()
    hs.print_report(report)
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from config.settings import PlatformConfig

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    latency_ms: float = 0.0


@dataclass
class HealthReport:
    timestamp: float
    all_ok: bool
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        if not result.ok:
            self.all_ok = False


class HealthService:
    """Runs health checks and returns a structured report."""

    def __init__(self, cfg: PlatformConfig):
        self._cfg = cfg

    def check_all(self) -> HealthReport:
        report = HealthReport(timestamp=time.time(), all_ok=True)
        report.add(self._check_database())
        report.add(self._check_disk_space())
        for cam in self._cfg.cameras:
            if cam.enabled:
                report.add(self._check_camera_reachable(cam.name, cam.stream))
        return report

    def _check_database(self) -> CheckResult:
        t0 = time.time()
        db_path = self._cfg.database.db_path
        try:
            if self._cfg.database.type == "sqlite":
                import sqlite3
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(db_path, timeout=3)
                conn.execute("SELECT 1")
                conn.close()
                return CheckResult(
                    name="database",
                    ok=True,
                    message=f"SQLite OK ({db_path})",
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )
            else:
                # PostgreSQL ping
                from repositories.postgres_repository import PostgreSQLRepository
                repo = PostgreSQLRepository(self._cfg.database)
                repo.close()
                return CheckResult(
                    name="database", ok=True, message="PostgreSQL OK",
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )
        except Exception as exc:
            return CheckResult(
                name="database", ok=False,
                message=f"DB error: {exc}",
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

    def _check_disk_space(self) -> CheckResult:
        t0 = time.time()
        try:
            base = Path(self._cfg.storage.base_path)
            base.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(base)
            free_gb = usage.free / (1024 ** 3)
            pct_used = (usage.used / usage.total) * 100
            ok = free_gb > 0.5  # warn if < 500 MB
            return CheckResult(
                name="disk_space",
                ok=ok,
                message=f"{free_gb:.1f} GB free ({pct_used:.0f}% used)" + ("" if ok else " ⚠️ LOW"),
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
        except Exception as exc:
            return CheckResult(name="disk_space", ok=False, message=str(exc))

    def _check_camera_reachable(self, name: str, stream) -> CheckResult:
        t0 = time.time()
        try:
            # For file/folder sources, just check path exists
            s = str(stream)
            if str(stream).isdigit() or isinstance(stream, int):
                # Don't open webcam during health check (would block)
                return CheckResult(
                    name=f"camera:{name}", ok=True,
                    message=f"Webcam index {stream} (not opened during health check)",
                )
            if Path(s).exists():
                return CheckResult(
                    name=f"camera:{name}", ok=True,
                    message=f"Path exists: {s}",
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )
            if s.startswith(("rtsp://", "http://")):
                # Quick TCP connect check
                import socket, urllib.parse
                parsed = urllib.parse.urlparse(s)
                host = parsed.hostname
                port = parsed.port or (554 if s.startswith("rtsp") else 80)
                sock = socket.create_connection((host, port), timeout=3)
                sock.close()
                return CheckResult(
                    name=f"camera:{name}", ok=True,
                    message=f"TCP reachable {host}:{port}",
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )
            return CheckResult(name=f"camera:{name}", ok=False, message=f"Path not found: {s}")
        except Exception as exc:
            return CheckResult(
                name=f"camera:{name}", ok=False,
                message=str(exc),
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

    def print_report(self, report: HealthReport | None = None) -> None:
        if report is None:
            report = self.check_all()
        import time as _time
        print(f"\n{'─'*50}")
        print(f"  Health Check  {_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(report.timestamp))}")
        print(f"{'─'*50}")
        for check in report.checks:
            icon = "✅" if check.ok else "❌"
            ms = f"  ({check.latency_ms}ms)" if check.latency_ms else ""
            print(f"  {icon}  {check.name:<25} {check.message}{ms}")
        print(f"{'─'*50}")
        overall = "✅ ALL OK" if report.all_ok else "❌ ISSUES FOUND"
        print(f"  Overall: {overall}\n")
