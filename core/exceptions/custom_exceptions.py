"""
core/exceptions/custom_exceptions.py
──────────────────────────────────────
All platform-specific exceptions.

Catching these lets callers distinguish recoverable vs. fatal errors
without depending on third-party exception types.
"""


class VisionPlatformError(Exception):
    """Root exception. Catch this to handle any platform error."""


# ── Source errors ────────────────────────────────────────────

class SourceError(VisionPlatformError):
    """Base for all frame-source failures."""


class CameraDisconnectedError(SourceError):
    """Camera went offline or USB was unplugged."""


class RTSPTimeoutError(SourceError):
    """RTSP stream did not respond within the timeout window."""


class CorruptVideoError(SourceError):
    """Video file is unreadable or truncated."""


class InvalidImageError(SourceError):
    """Image file could not be decoded."""


class SourceExhaustedError(SourceError):
    """Source has no more frames (end of file / folder)."""


# ── Detector errors ──────────────────────────────────────────

class DetectorError(VisionPlatformError):
    """A detector failed on a frame."""


class ModelLoadError(DetectorError):
    """Cascade / model weights file not found or corrupt."""


# ── Processor errors ─────────────────────────────────────────

class ProcessorError(VisionPlatformError):
    """A processor failed to handle a frame."""


class DiskFullError(ProcessorError):
    """No disk space left when saving image or video."""


# ── Repository errors ────────────────────────────────────────

class RepositoryError(VisionPlatformError):
    """Database operation failed."""


class DatabaseConnectionError(RepositoryError):
    """Could not connect to the database."""


# ── Alert errors ─────────────────────────────────────────────

class AlertError(VisionPlatformError):
    """Failed to send an alert."""


class ConfigurationError(VisionPlatformError):
    """config.yaml is missing a required key or has an invalid value."""
