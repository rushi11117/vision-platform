"""
sources/android_camera_source.py
──────────────────────────────────
Thin wrapper around RTSPSource for Android phone camera apps.

Supported apps:
  • IP Webcam (port 8080 by default) → http://<phone-ip>:8080/video
  • DroidCam  (port 4747 by default) → http://<phone-ip>:4747/video

Usage:
  source = AndroidCameraSource("192.168.1.55", port=8080, name="PhoneCam")
"""

from sources.rtsp_source import RTSPSource


class AndroidCameraSource(RTSPSource):
    """Connects to an Android camera streaming app over WiFi."""

    def __init__(
        self,
        host: str,
        port: int = 8080,
        path: str = "/video",
        source_name: str = "",
        **kwargs,
    ):
        url = f"http://{host}:{port}{path}"
        name = source_name or f"Android-{host}"
        super().__init__(url=url, source_name=name, **kwargs)
