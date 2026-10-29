"""
config/settings.py
──────────────────
Loads config.yaml and exposes typed dataclasses.
All other modules import from here — never parse YAML themselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Dataclasses ─────────────────────────────────────────────

@dataclass
class CameraConfig:
    name: str
    stream: str | int
    enabled: bool = True
    detectors: List[str] = field(default_factory=list)


@dataclass
class FaceDetectionConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: tuple = (30, 30)
    confidence_threshold: float = 0.6


@dataclass
class PersonDetectionConfig:
    win_stride: tuple = (8, 8)
    padding: tuple = (4, 4)
    scale: float = 1.05
    confidence_threshold: float = 0.5


@dataclass
class MotionDetectionConfig:
    threshold: int = 1000
    blur_kernel: int = 5
    diff_threshold: int = 25
    background_subtractor: bool = False


@dataclass
class DetectionConfig:
    face: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    person: PersonDetectionConfig = field(default_factory=PersonDetectionConfig)
    motion: MotionDetectionConfig = field(default_factory=MotionDetectionConfig)


@dataclass
class DisplayProcessorConfig:
    enabled: bool = True
    window_title: str = "Vision Platform"
    draw_boxes: bool = True
    draw_labels: bool = True


@dataclass
class ImageSaveConfig:
    enabled: bool = True
    output_dir: str = "./storage/snapshots"
    format: str = "jpg"
    quality: int = 85
    save_on_detection: bool = True


@dataclass
class VideoSaveConfig:
    enabled: bool = False
    output_dir: str = "./storage/videos"
    codec: str = "mp4v"
    fps: int = 20
    mode: str = "event"
    pre_buffer_seconds: int = 3


@dataclass
class EmailAlertConfig:
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender: str = ""
    password: str = ""
    recipients: List[str] = field(default_factory=list)


@dataclass
class WebhookAlertConfig:
    enabled: bool = False
    url: str = ""
    timeout_seconds: int = 5


@dataclass
class AlertProcessorConfig:
    enabled: bool = False
    cooldown_seconds: int = 60
    email: EmailAlertConfig = field(default_factory=EmailAlertConfig)
    webhook: WebhookAlertConfig = field(default_factory=WebhookAlertConfig)


@dataclass
class ProcessorsConfig:
    display: DisplayProcessorConfig = field(default_factory=DisplayProcessorConfig)
    image_save: ImageSaveConfig = field(default_factory=ImageSaveConfig)
    video_save: VideoSaveConfig = field(default_factory=VideoSaveConfig)
    database: Dict = field(default_factory=lambda: {"enabled": True})
    alert: AlertProcessorConfig = field(default_factory=AlertProcessorConfig)


@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    db_path: str = "./database/vision.db"
    host: str = "localhost"
    port: int = 5432
    name: str = "vision_db"
    user: str = "vision_user"
    password: str = ""


@dataclass
class StorageConfig:
    base_path: str = "./storage"
    images_path: str = "./storage/images"
    videos_path: str = "./storage/videos"
    snapshots_path: str = "./storage/snapshots"
    logs_path: str = "./storage/logs"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "./storage/logs/platform.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5


@dataclass
class WorkersConfig:
    max_workers: int = 10
    frame_skip: int = 0
    reconnect_delay_seconds: int = 5
    max_reconnect_attempts: int = 10


@dataclass
class SnapshotConfig:
    filename_pattern: str = "{timestamp}_{label}.jpg"
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class PlatformConfig:
    cameras: List[CameraConfig] = field(default_factory=list)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    processors: ProcessorsConfig = field(default_factory=ProcessorsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)


# ── Loader ──────────────────────────────────────────────────

def _merge(dataclass_instance, raw: dict):
    """Recursively populate a dataclass from a dict, ignoring unknown keys."""
    for key, val in raw.items():
        if hasattr(dataclass_instance, key):
            attr = getattr(dataclass_instance, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(val, dict):
                _merge(attr, val)
            else:
                setattr(dataclass_instance, key, val)
    return dataclass_instance


def load_config(path: str = "config/config.yaml") -> PlatformConfig:
    """
    Load and parse config.yaml into a PlatformConfig dataclass tree.

    Environment variables override YAML values:
      - ALERT_EMAIL_PASSWORD → processors.alert.email.password
      - DB_PASSWORD          → database.password
    """
    config_path = Path(path)
    raw: dict = {}
    if config_path.exists():
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = PlatformConfig()

    # cameras
    if "cameras" in raw:
        cfg.cameras = [
            CameraConfig(
                name=cam["name"],
                stream=cam["stream"],
                enabled=cam.get("enabled", True),
                detectors=cam.get("detectors", []),
            )
            for cam in raw["cameras"]
        ]

    if "detection" in raw:
        d = raw["detection"]
        if "face" in d:
            _merge(cfg.detection.face, d["face"])
        if "person" in d:
            _merge(cfg.detection.person, d["person"])
        if "motion" in d:
            _merge(cfg.detection.motion, d["motion"])

    if "processors" in raw:
        p = raw["processors"]
        if "display" in p:
            _merge(cfg.processors.display, p["display"])
        if "image_save" in p:
            _merge(cfg.processors.image_save, p["image_save"])
        if "video_save" in p:
            _merge(cfg.processors.video_save, p["video_save"])
        if "database" in p:
            cfg.processors.database = p["database"]
        if "alert" in p:
            a = p["alert"]
            _merge(cfg.processors.alert, {k: v for k, v in a.items() if k not in ("email", "webhook")})
            if "email" in a:
                _merge(cfg.processors.alert.email, a["email"])
            if "webhook" in a:
                _merge(cfg.processors.alert.webhook, a["webhook"])

    if "database" in raw:
        _merge(cfg.database, raw["database"])

    if "storage" in raw:
        _merge(cfg.storage, raw["storage"])

    if "logging" in raw:
        _merge(cfg.logging, raw["logging"])

    if "workers" in raw:
        _merge(cfg.workers, raw["workers"])

    if "snapshot" in raw:
        _merge(cfg.snapshot, raw["snapshot"])

    # env-var overrides
    email_pw = os.environ.get("ALERT_EMAIL_PASSWORD")
    if email_pw:
        cfg.processors.alert.email.password = email_pw

    db_pw = os.environ.get("DB_PASSWORD")
    if db_pw:
        cfg.database.password = db_pw

    # ensure storage dirs exist
    for attr in ["base_path", "images_path", "videos_path", "snapshots_path", "logs_path"]:
        Path(getattr(cfg.storage, attr)).mkdir(parents=True, exist_ok=True)
    Path(cfg.database.db_path).parent.mkdir(parents=True, exist_ok=True)

    return cfg
