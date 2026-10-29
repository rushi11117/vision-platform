"""
processors/display_processor.py
─────────────────────────────────
Shows live annotated video in an OpenCV window.
Press 'q' to stop; pipeline sets _stop flag accordingly.
"""

import logging
from typing import List

import cv2
import numpy as np

from core.contracts.processor import Processor
from core.models.detection_result import DetectionResult
from core.models.frame_metadata import FrameMetadata
from config.settings import DisplayProcessorConfig

log = logging.getLogger(__name__)

# colour per label (BGR)
_COLOURS = {
    "face":   (0, 255, 0),    # green
    "person": (255, 128, 0),  # blue-orange
    "motion": (0, 0, 255),    # red
}
_DEFAULT_COLOUR = (200, 200, 200)


class DisplayProcessor(Processor):
    """Draws bounding boxes + labels on the frame and shows it on screen."""

    def __init__(self, cfg: DisplayProcessorConfig | None = None):
        self._cfg = cfg or DisplayProcessorConfig()
        self.stop_requested = False
        log.info("DisplayProcessor initialised (window='%s')", self._cfg.window_title)

    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        metadata: FrameMetadata,
    ) -> None:
        display = frame.copy()

        if self._cfg.draw_boxes:
            for det in detections:
                if det.bounding_box:
                    x, y, w, h = det.bounding_box
                    colour = _COLOURS.get(det.label, _DEFAULT_COLOUR)
                    cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)

                    if self._cfg.draw_labels:
                        label_text = f"{det.label} {det.confidence:.2f}"
                        cv2.putText(
                            display, label_text, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
                        )

        # camera name overlay
        cv2.putText(
            display, metadata.camera_name, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        title = f"{self._cfg.window_title} - {metadata.camera_name}"
        cv2.imshow(title, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop_requested = True
            log.info("DisplayProcessor: 'q' pressed, stop requested")

    def release(self) -> None:
        cv2.destroyAllWindows()
