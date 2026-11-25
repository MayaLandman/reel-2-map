"""Thin helpers around the ffmpeg binary.

A system ffmpeg is used when one is on PATH (or set with REEL2MAP_FFMPEG). Otherwise the
static build shipped in the `imageio-ffmpeg` wheel is used, so `pip install` is enough on
machines where installing ffmpeg is a hassle. Only `ffmpeg` is needed, not `ffprobe`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
AUDIO_RE = re.compile(r"Stream #\S+.*?: Audio:")


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = os.environ.get("REEL2MAP_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FFmpegError(
            "ffmpeg not found: install it (brew install ffmpeg / apt install ffmpeg) "
            "or `pip install imageio-ffmpeg`"
        ) from exc


def available() -> bool:
    try:
        ffmpeg_exe()
    except FFmpegError:
        return False
    return True


def require_ffmpeg() -> None:
    ffmpeg_exe()


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    if args and args[0] == "ffmpeg":
        args = [ffmpeg_exe(), *args[1:]]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed: {proc.stderr[-800:]}")
    return proc


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None
    has_audio: bool


def media_info(path: Path) -> MediaInfo:
    """Read duration and streams from `ffmpeg -i` (which exits non-zero by design)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if "Invalid data" in proc.stderr or "No such file" in proc.stderr:
        raise FFmpegError(f"cannot read {path.name}: {proc.stderr.strip()[-300:]}")
    match = DURATION_RE.search(proc.stderr)
    seconds = None
    if match:
        h, m, s = match.groups()
        seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return MediaInfo(duration=seconds, has_audio=bool(AUDIO_RE.search(proc.stderr)))


def has_audio(path: Path) -> bool:
    return media_info(path).has_audio


def duration(path: Path) -> float | None:
    return media_info(path).duration


def extract_audio(video: Path, out: Path) -> Path:
    """16 kHz mono WAV, the format Whisper expects."""
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out),
        ]
    )
    return out
