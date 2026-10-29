"""
utils/image_utils.py
─────────────────────
Reusable OpenCV image helpers used across the platform.
Handles maximum native OpenCV formats without external dependencies.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.models.detection_result import DetectionResult

# Colour map: label -> BGR
COLOUR_MAP = {
    "face":   (0, 255, 0),    # Green
    "person": (255, 128, 0),  # Blue-ish orange
    "motion": (0, 0, 255),    # Red
    "qr":     (255, 255, 0),  # Cyan
}
DEFAULT_COLOUR = (200, 200, 200)


def read_image(file_path: str | Path) -> Optional[np.ndarray]:
    """
    Reads an image from disk safely on all platforms (including Windows non-ASCII paths).
    Supports all native OpenCV formats: JPG, PNG, WEBP, BMP, TIFF, JP2, PPM, HDR, EXR.
    """
    path = Path(file_path)
    if not path.is_file():
        return None

    try:
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_encode_params(fmt: str, quality: int = 85) -> List[int]:
    """Generates OpenCV compression parameters for various target formats."""
    f = fmt.lower().lstrip(".")
    if f in ("jpg", "jpeg", "jpe"):
        return [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif f == "png":
        # PNG compression range 0-9 (3 is fast with good compression)
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif f == "webp":
        return [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif f in ("tiff", "tif"):
        # LZW compression
        return [cv2.IMWRITE_TIFF_COMPRESSION, 5]
    elif f == "jp2":
        # JPEG 2000 compression
        return [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(quality * 10)]
    return []


def write_image(file_path: str | Path, frame: np.ndarray, quality: int = 85) -> bool:
    """
    Writes an image to disk via an in-memory buffer.
    Guarantees compatibility with non-ASCII and Windows paths.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if not ext:
        ext = ".jpg"
        path = path.with_suffix(ext)

    params = get_encode_params(ext, quality)
    success, buf = cv2.imencode(ext, frame, params)
    if not success:
        return False

    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def draw_detections(
    frame: np.ndarray,
    detections: List[DetectionResult],
    draw_boxes: bool = True,
    draw_labels: bool = True,
    thickness: int = 1,  # 1px thin border
) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of the frame."""
    out = frame.copy()
    box_th = max(1, int(thickness))

    for det in detections:
        if not det.bounding_box:
            continue
        x, y, w, h = det.bounding_box
        colour = COLOUR_MAP.get(det.label, DEFAULT_COLOUR)

        if draw_boxes:
            cv2.rectangle(
                out,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                colour,
                box_th,
            )

        if draw_labels:
            conf_str = f" {det.confidence:.2f}" if det.confidence is not None else ""
            text = f"{det.label}{conf_str}"
            cv2.putText(
                out,
                text,
                (int(x), max(16, int(y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,       # Fine text scale
                colour,
                1,          # Thickness must remain an integer
                cv2.LINE_AA,
            )
    return out


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Resize frame while keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def encode_frame(frame: np.ndarray, fmt: str = "jpg", quality: int = 85) -> bytes:
    """Encode a frame to bytes (for streaming or storage)."""
    params = get_encode_params(fmt, quality)
    _, buf = cv2.imencode(f".{fmt.lstrip('.')}", frame, params)
    return buf.tobytes()