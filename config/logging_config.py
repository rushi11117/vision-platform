"""
config/logging_config.py
────────────────────────
Sets up rotating file + console logging from PlatformConfig.
Call setup_logging() once at startup; every module uses logging.getLogger(__name__).
"""

import logging
import logging.handlers
from pathlib import Path

from config.settings import LoggingConfig

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure root logger. Safe to call multiple times (idempotent)."""
    level = _LEVEL_MAP.get(cfg.level.upper(), logging.INFO)
    root = logging.getLogger()

    # avoid adding duplicate handlers on re-init
    if root.handlers:
        return

    root.setLevel(level)
    fmt = logging.Formatter(_FMT, datefmt=_DATE)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # rotating file handler
    if cfg.log_to_file:
        Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            cfg.log_file,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    logging.getLogger("vision_platform").info("Logging initialised at level %s", cfg.level)
