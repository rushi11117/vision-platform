"""
services/analytics_service.py
───────────────────────────────
Queries the detection database and returns structured analytics.

Used by the CLI report tool and the Streamlit dashboard.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class HourlyStat:
    hour: str          # "2026-10-21 14"
    camera: str
    label: str
    count: int


@dataclass
class CameraSummary:
    camera_name: str
    total_events: int
    by_label: Dict[str, int] = field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class AnalyticsReport:
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_events: int
    cameras: List[CameraSummary]
    hourly: List[HourlyStat]
    top_label: Optional[str] = None


class AnalyticsService:
    """
    Reads the SQLite detection database and produces analytics.

    All methods return plain Python dataclasses — no DB objects leak out.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_report(
        self,
        hours: int = 24,
        camera: Optional[str] = None,
    ) -> AnalyticsReport:
        """Generate a full analytics report for the last N hours."""
        now = datetime.now()
        period_start = now - timedelta(hours=hours)

        filters = ["timestamp >= ?"]
        params: list = [period_start.isoformat()]
        if camera:
            filters.append("camera_name = ?")
            params.append(camera)
        where = " AND ".join(filters)

        with self._conn() as conn:
            # ensure table exists
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='detections'"
            ).fetchone()
            if not table_exists:
                return AnalyticsReport(
                    generated_at=now, period_start=period_start, period_end=now,
                    total_events=0, cameras=[], hourly=[],
                )
            # total count
            total = conn.execute(
                f"SELECT COUNT(*) FROM detections WHERE {where}", params
            ).fetchone()[0]

            # per-camera breakdown
            cam_rows = conn.execute(
                f"""SELECT camera_name, label, COUNT(*) as cnt,
                           MIN(timestamp) as first_seen, MAX(timestamp) as last_seen
                    FROM detections WHERE {where}
                    GROUP BY camera_name, label
                    ORDER BY camera_name, cnt DESC""",
                params,
            ).fetchall()

            # hourly breakdown
            hourly_rows = conn.execute(
                f"""SELECT strftime('%Y-%m-%d %H', timestamp) as hour,
                           camera_name, label, COUNT(*) as cnt
                    FROM detections WHERE {where}
                    GROUP BY hour, camera_name, label
                    ORDER BY hour""",
                params,
            ).fetchall()

        # Build CameraSummary objects
        cam_map: Dict[str, CameraSummary] = {}
        for row in cam_rows:
            cam_name = row["camera_name"]
            if cam_name not in cam_map:
                cam_map[cam_name] = CameraSummary(
                    camera_name=cam_name,
                    total_events=0,
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )
            s = cam_map[cam_name]
            s.by_label[row["label"]] = row["cnt"]
            s.total_events += row["cnt"]
            ts_first = datetime.fromisoformat(row["first_seen"])
            ts_last = datetime.fromisoformat(row["last_seen"])
            if s.first_seen is None or ts_first < s.first_seen:
                s.first_seen = ts_first
            if s.last_seen is None or ts_last > s.last_seen:
                s.last_seen = ts_last

        hourly = [
            HourlyStat(
                hour=row["hour"],
                camera=row["camera_name"],
                label=row["label"],
                count=row["cnt"],
            )
            for row in hourly_rows
        ]

        # most common label
        label_totals: Dict[str, int] = {}
        for s in cam_map.values():
            for label, cnt in s.by_label.items():
                label_totals[label] = label_totals.get(label, 0) + cnt
        top_label = max(label_totals, key=label_totals.get) if label_totals else None

        return AnalyticsReport(
            generated_at=now,
            period_start=period_start,
            period_end=now,
            total_events=total,
            cameras=list(cam_map.values()),
            hourly=hourly,
            top_label=top_label,
        )

    def print_report(self, hours: int = 24, camera: Optional[str] = None) -> None:
        """Print a formatted analytics report to stdout."""
        if not Path(self._db_path).exists():
            print("No database found yet — run the platform first.")
            return

        report = self.get_report(hours=hours, camera=camera)
        print(f"\n{'='*60}")
        print(f"  Vision Platform Analytics Report")
        print(f"  Period: last {hours}h ({report.period_start:%Y-%m-%d %H:%M} → {report.period_end:%H:%M})")
        print(f"{'='*60}")
        print(f"  Total events: {report.total_events}")
        if report.top_label:
            print(f"  Most common:  {report.top_label}")
        print()

        for cam in report.cameras:
            print(f"  📷 {cam.camera_name}  ({cam.total_events} events)")
            for label, cnt in sorted(cam.by_label.items(), key=lambda x: -x[1]):
                bar = "█" * min(cnt, 40)
                print(f"      {label:<10} {cnt:>5}  {bar}")
            if cam.first_seen:
                print(f"      first: {cam.first_seen:%H:%M:%S}  last: {cam.last_seen:%H:%M:%S}")
            print()

        if report.hourly:
            print(f"  Hourly breakdown:")
            for h in report.hourly:
                print(f"      {h.hour}  {h.camera:<12} {h.label:<10} {h.count}")
        print(f"{'='*60}\n")
