from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from reel2map.config import Settings
from reel2map.models import Place, Platform, Post

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test", google_maps_api_key=None, cache_dir=tmp_path / "cache"
    )


@pytest.fixture
def post() -> Post:
    return Post(
        url="https://www.tiktok.com/@romefood/video/123",
        platform=Platform.TIKTOK,
        id="123",
        uploader="romefood",
        description="5 places you can't miss in Rome #rome #pizza",
        hashtags=["rome", "pizza"],
    )


@pytest.fixture
def place() -> Place:
    return Place(
        name="Da Enzo al 29",
        category="restaurant",
        city="Rome",
        country="Italy",
        address_hint="Trastevere",
        why="Classic carbonara.",
        evidence=["on_screen_text"],
        confidence=0.9,
        timestamp=4.0,
    )


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """An 8 second clip with three very different 'shots' and a tone on the audio track."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path / "clip.mp4"
    src = (
        "color=c=red:s=320x240:d=3[a];color=c=blue:s=320x240:d=3[b];"
        "testsrc=s=320x240:d=2[c];[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-filter_complex",
            src,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-map",
            "[v]",
            "-map",
            "0:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    return out
