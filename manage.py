"""
manage.py
──────────
All-in-one management CLI for the Vision Platform.

Commands:
  python manage.py run        → start the platform (same as app.py)
  python manage.py report     → show detection analytics
  python manage.py health     → health check (DB, disk, cameras)
  python manage.py export     → export detections to JSON / CSV
  python manage.py purge      → delete detections older than N days
  python manage.py list-cameras → list configured cameras
  python manage.py dashboard  → launch Streamlit dashboard

All commands accept --config to point at a custom config.yaml.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_run(args):
    from config.settings import load_config, CameraConfig
    from config.logging_config import setup_logging
    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    if args.image:
        cfg.cameras = [CameraConfig(name="CLI-Image", stream=args.image,
                                    enabled=True, detectors=["face", "person", "motion"])]
    elif args.video:
        cfg.cameras = [CameraConfig(name="CLI-Video", stream=args.video,
                                    enabled=True, detectors=["face", "person", "motion"])]
    elif args.webcam is not None:
        cfg.cameras = [CameraConfig(name=f"Webcam-{args.webcam}", stream=args.webcam,
                                    enabled=True, detectors=["face", "person", "motion"])]
    from services.stream_service import StreamService
    StreamService(cfg).start()


def cmd_report(args):
    from config.settings import load_config
    cfg = load_config(args.config)
    from services.analytics_service import AnalyticsService
    svc = AnalyticsService(cfg.database.db_path)
    if args.json:
        report = svc.get_report(hours=args.hours, camera=args.camera)
        data = {
            "generated_at": report.generated_at.isoformat(),
            "total_events": report.total_events,
            "top_label": report.top_label,
            "cameras": [{"name": c.camera_name, "total": c.total_events,
                         "by_label": c.by_label} for c in report.cameras],
        }
        print(json.dumps(data, indent=2))
    else:
        svc.print_report(hours=args.hours, camera=args.camera)


def cmd_health(args):
    from config.settings import load_config
    cfg = load_config(args.config)
    from services.health_service import HealthService
    hs = HealthService(cfg)
    report = hs.check_all()
    hs.print_report(report)
    sys.exit(0 if report.all_ok else 1)


def cmd_export(args):
    from config.settings import load_config
    import sqlite3, csv
    cfg = load_config(args.config)
    db = cfg.database.db_path
    if not Path(db).exists():
        print("No database found."); return
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM detections ORDER BY timestamp DESC").fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        print("No detections to export yet."); return
    if args.format == "csv":
        out = args.output or "detections.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])
        print(f"Exported {len(rows)} rows → {out}")
    else:
        out = args.output or "detections.json"
        with open(out, "w") as f:
            json.dump([dict(r) for r in rows], f, indent=2)
        print(f"Exported {len(rows)} rows → {out}")


def cmd_purge(args):
    from config.settings import load_config
    import sqlite3
    cfg = load_config(args.config)
    db = cfg.database.db_path
    if not Path(db).exists():
        print("No database found."); return
    days = args.days
    with sqlite3.connect(db) as conn:
        try:
            cur = conn.execute(
                f"DELETE FROM detections WHERE timestamp < datetime('now', '-{days} days')"
            )
            conn.commit()
            print(f"Purged {cur.rowcount} detections older than {days} days.")
        except sqlite3.OperationalError:
            print("No detections table found yet — nothing to purge.")


def cmd_list_cameras(args):
    from config.settings import load_config
    cfg = load_config(args.config)
    print(f"\n{'─'*60}")
    print(f"  Configured cameras ({len(cfg.cameras)} total)")
    print(f"{'─'*60}")
    for cam in cfg.cameras:
        status = "✅ enabled" if cam.enabled else "⏸ disabled"
        print(f"  {status}  {cam.name:<15} stream={cam.stream}")
        print(f"            detectors={cam.detectors}")
    print(f"{'─'*60}\n")


def cmd_dashboard(args):
    import subprocess
    dash_path = str(Path(__file__).parent / "dashboards" / "streamlit_dashboard.py")
    print("Launching Streamlit dashboard…")
    print("Open http://localhost:8501 in your browser. Press Ctrl+C to stop.\n")
    subprocess.run(["streamlit", "run", dash_path], check=True)


# ── Argument parser ──────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vision Platform Management CLI")
    p.add_argument("--config", default="config/config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    # run
    r = sub.add_parser("run", help="Start the platform")
    r.add_argument("--image"); r.add_argument("--video"); r.add_argument("--webcam", type=int)

    # report
    rp = sub.add_parser("report", help="Show detection analytics")
    rp.add_argument("--hours", type=int, default=24)
    rp.add_argument("--camera", default=None)
    rp.add_argument("--json", action="store_true")

    # health
    sub.add_parser("health", help="Run health checks")

    # export
    ex = sub.add_parser("export", help="Export detections to JSON or CSV")
    ex.add_argument("--format", choices=["json", "csv"], default="json")
    ex.add_argument("--output", default=None)

    # purge
    pu = sub.add_parser("purge", help="Purge old detections")
    pu.add_argument("--days", type=int, default=30)

    # list-cameras
    sub.add_parser("list-cameras", help="List configured cameras")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    return p


COMMANDS = {
    "run": cmd_run,
    "report": cmd_report,
    "health": cmd_health,
    "export": cmd_export,
    "purge": cmd_purge,
    "list-cameras": cmd_list_cameras,
    "dashboard": cmd_dashboard,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
