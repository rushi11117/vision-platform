"""utils/datetime_utils.py — datetime helpers used across the platform."""

from datetime import datetime


def now_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Current timestamp as a filename-safe string."""
    return datetime.now().strftime(fmt)


def iso_now() -> str:
    return datetime.now().isoformat()
