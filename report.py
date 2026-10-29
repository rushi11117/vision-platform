"""
report.py
──────────
Command-line analytics report for the Vision Platform.

Usage:
  python report.py                        # last 24h, all cameras
  python report.py --hours 48             # last 48h
  python report.py --camera Gate1         # filter by camera
  python report.py --db ./database/vision.db
  python report.py --export report.json   # export to JSON
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import load_config
from services.analytics_service import AnalyticsService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision Platform Analytics Report")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--db", default=None, help="Override DB path")
    p.add_argument("--hours", type=int, default=24, help="Report period in hours")
    p.add_argument("--camera", default=None, help="Filter by camera name")
    p.add_argument("--export", default=None, help="Export JSON to this file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    db_path = args.db or cfg.database.db_path

    svc = AnalyticsService(db_path)

    if args.export:
        report = svc.get_report(hours=args.hours, camera=args.camera)
        data = {
            "generated_at": report.generated_at.isoformat(),
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "total_events": report.total_events,
            "top_label": report.top_label,
            "cameras": [
                {
                    "camera_name": c.camera_name,
                    "total_events": c.total_events,
                    "by_label": c.by_label,
                    "first_seen": c.first_seen.isoformat() if c.first_seen else None,
                    "last_seen": c.last_seen.isoformat() if c.last_seen else None,
                }
                for c in report.cameras
            ],
            "hourly": [
                {"hour": h.hour, "camera": h.camera, "label": h.label, "count": h.count}
                for h in report.hourly
            ],
        }
        with open(args.export, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"Report exported to {args.export}")
    else:
        svc.print_report(hours=args.hours, camera=args.camera)


if __name__ == "__main__":
    main()
