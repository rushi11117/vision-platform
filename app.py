"""
app.py
───────
Vision Platform entry point.

Usage examples:
  python app.py                              # use config/config.yaml
  python app.py --config config/custom.yaml # alternate config
  python app.py --image path/to/photo.jpg   # run on a single image
  python app.py --video path/to/video.mp4   # run on a video file
  python app.py --webcam 0                  # webcam index 0
  python app.py --rtsp rtsp://192.168.1.1/live
"""

import argparse
import logging
import sys
from pathlib import Path

# ── path setup so modules resolve from project root ──────────
sys.path.insert(0, str(Path(__file__).parent))

from config.logging_config import setup_logging
from config.settings import CameraConfig, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision Platform")
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--image",  help="Run on a single image file")
    p.add_argument("--video",  help="Run on a video file")
    p.add_argument("--webcam", type=int, help="Webcam index (e.g. 0)")
    p.add_argument("--rtsp",   help="RTSP / HTTP stream URL")
    p.add_argument("--folder", help="Folder of images")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────
    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    log = logging.getLogger(__name__)
    log.info("Vision Platform starting")

    # ── CLI overrides: build a temporary camera entry ────────
    if args.image or args.video or args.webcam is not None or args.rtsp or args.folder:
        if args.image:
            stream, name = args.image, "CLI-Image"
        elif args.video:
            stream, name = args.video, "CLI-Video"
        elif args.webcam is not None:
            stream, name = args.webcam, f"CLI-Webcam-{args.webcam}"
        elif args.rtsp:
            stream, name = args.rtsp, "CLI-RTSP"
        else:
            stream, name = args.folder, "CLI-Folder"

        cfg.cameras = [
            CameraConfig(
                name=name,
                stream=stream,
                enabled=True,
                detectors=["face", "person", "motion"],
            )
        ]
        log.info("CLI override: using source '%s' -> %s", name, stream)

    # ── Launch ───────────────────────────────────────────────
    from services.stream_service import StreamService
    service = StreamService(cfg)

    try:
        service.start()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
        service.stop_all()

    log.info("Vision Platform stopped")


if __name__ == "__main__":
    main()
