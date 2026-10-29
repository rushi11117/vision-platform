"""
tests/test_sources.py
──────────────────────
Unit tests for ImageSource, FolderSource, VideoSource.

Run with: pytest tests/test_sources.py -v
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pytest

from core.exceptions.custom_exceptions import InvalidImageError, CorruptVideoError, SourceError


def _make_image(path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = (0, 200, 0)
    cv2.imwrite(str(path), img)


def _make_video(path: Path, n_frames: int = 10) -> None:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(n_frames):
        out.write(np.zeros((64, 64, 3), dtype=np.uint8))
    out.release()


# ── ImageSource tests ─────────────────────────────────────────

class TestImageSource:
    def test_yields_one_frame(self, tmp_path):
        from sources.image_source import ImageSource
        img_path = tmp_path / "test.jpg"
        _make_image(img_path)
        src = ImageSource(str(img_path))
        frames = list(src.frames())
        assert len(frames) == 1
        assert isinstance(frames[0], np.ndarray)

    def test_missing_file_raises(self):
        from sources.image_source import ImageSource
        src = ImageSource("/nonexistent/image.jpg")
        with pytest.raises(InvalidImageError):
            list(src.frames())

    def test_name_from_filename(self, tmp_path):
        from sources.image_source import ImageSource
        img_path = tmp_path / "gate1.jpg"
        _make_image(img_path)
        src = ImageSource(str(img_path))
        assert src.name == "gate1"

    def test_custom_name(self, tmp_path):
        from sources.image_source import ImageSource
        img_path = tmp_path / "x.jpg"
        _make_image(img_path)
        src = ImageSource(str(img_path), source_name="MyCamera")
        assert src.name == "MyCamera"

    def test_release_safe(self, tmp_path):
        from sources.image_source import ImageSource
        img_path = tmp_path / "test.jpg"
        _make_image(img_path)
        src = ImageSource(str(img_path))
        src.release()  # should not raise


# ── FolderSource tests ────────────────────────────────────────

class TestFolderSource:
    def test_yields_all_images(self, tmp_path):
        from sources.folder_source import FolderSource
        for i in range(3):
            _make_image(tmp_path / f"img{i}.jpg")
        src = FolderSource(str(tmp_path))
        frames = list(src.frames())
        assert len(frames) == 3

    def test_skips_non_images(self, tmp_path):
        from sources.folder_source import FolderSource
        _make_image(tmp_path / "real.jpg")
        (tmp_path / "ignore.txt").write_text("text")
        frames = list(FolderSource(str(tmp_path)).frames())
        assert len(frames) == 1

    def test_missing_folder_raises(self):
        from sources.folder_source import FolderSource
        with pytest.raises(SourceError):
            list(FolderSource("/nonexistent/folder").frames())


# ── VideoSource tests ─────────────────────────────────────────

class TestVideoSource:
    def test_yields_frames(self, tmp_path):
        from sources.video_source import VideoSource
        vid_path = tmp_path / "test.mp4"
        _make_video(vid_path, n_frames=5)
        frames = list(VideoSource(str(vid_path)).frames())
        assert len(frames) == 5

    def test_frame_skip(self, tmp_path):
        from sources.video_source import VideoSource
        vid_path = tmp_path / "test.mp4"
        _make_video(vid_path, n_frames=10)
        frames = list(VideoSource(str(vid_path), frame_skip=1).frames())
        assert len(frames) == 5  # every 2nd frame

    def test_missing_file_raises(self):
        from sources.video_source import VideoSource
        with pytest.raises(CorruptVideoError):
            list(VideoSource("/nonexistent/video.mp4").frames())

    def test_release_after_exhaustion(self, tmp_path):
        from sources.video_source import VideoSource
        vid_path = tmp_path / "test.mp4"
        _make_video(vid_path, n_frames=3)
        src = VideoSource(str(vid_path))
        list(src.frames())
        src.release()  # should not raise
