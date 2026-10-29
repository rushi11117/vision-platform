"""
processors/alert_processor.py
──────────────────────────────
Sends alerts (email / webhook / SMS / WhatsApp) when detections occur.

Features:
  • Per-label cooldown to avoid alert floods
  • Pluggable alert backends (add new ones by implementing _send_*)
  • Non-blocking (each send runs in a daemon thread)
"""

import logging
import os
import smtplib
import threading
from datetime import datetime
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import numpy as np

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from core.contracts.processor import Processor
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata
from config.settings import AlertProcessorConfig

log = logging.getLogger(__name__)


class AlertProcessor(Processor):
    """Fires alerts when detections occur, with cooldown per label."""

    def __init__(self, cfg: AlertProcessorConfig | None = None):
        self._cfg = cfg or AlertProcessorConfig()
        self._last_alert: Dict[str, datetime] = {}  # label → last alert time
        log.info("AlertProcessor initialised (cooldown=%ds)", self._cfg.cooldown_seconds)

    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        for det in detections:
            if self._should_alert(det.label, metadata.timestamp):
                self._last_alert[det.label] = metadata.timestamp
                self._fire(det, metadata)

    def _should_alert(self, label: str, now: datetime) -> bool:
        last = self._last_alert.get(label)
        if last is None:
            return True
        elapsed = (now - last).total_seconds()
        return elapsed >= self._cfg.cooldown_seconds

    def _fire(self, det: DetectionResult, metadata: FrameMetadata) -> None:
        subject = f"[Vision] {det.label.upper()} detected on {metadata.camera_name}"
        body = (
            f"Detection event\n"
            f"  Camera   : {metadata.camera_name}\n"
            f"  Detector : {det.detector}\n"
            f"  Label    : {det.label}\n"
            f"  Confidence: {det.confidence:.2f}\n"
            f"  Time     : {metadata.timestamp}\n"
            f"  Box      : {det.bounding_box}\n"
        )

        if self._cfg.email.enabled:
            threading.Thread(
                target=self._send_email, args=(subject, body), daemon=True
            ).start()

        if self._cfg.webhook.enabled:
            threading.Thread(
                target=self._send_webhook, args=(det, metadata), daemon=True
            ).start()

    def _send_email(self, subject: str, body: str) -> None:
        cfg = self._cfg.email
        password = cfg.password or os.environ.get("ALERT_EMAIL_PASSWORD", "")
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = cfg.sender
            msg["To"] = ", ".join(cfg.recipients)

            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
                server.starttls()
                server.login(cfg.sender, password)
                server.sendmail(cfg.sender, cfg.recipients, msg.as_string())

            log.info("AlertProcessor email sent to %s", cfg.recipients)
        except Exception as exc:
            log.error("AlertProcessor email failed: %s", exc)

    def _send_webhook(self, det: DetectionResult, metadata: FrameMetadata) -> None:
        if not _HAS_REQUESTS:
            log.warning("'requests' package not installed — webhook alert skipped")
            return
        cfg = self._cfg.webhook
        try:
            payload = det.to_dict()
            resp = _requests.post(cfg.url, json=payload, timeout=cfg.timeout_seconds)
            resp.raise_for_status()
            log.info("AlertProcessor webhook sent (status %d)", resp.status_code)
        except Exception as exc:
            log.error("AlertProcessor webhook failed: %s", exc)

    def release(self) -> None:
        pass
