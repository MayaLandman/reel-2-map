"""Thin helpers around the ffmpeg / ffprobe binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FFmpegError(f"{binary} not found on PATH; install ffmpeg first")


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(args[:3])}... failed: {proc.stderr[-800:]}")
    return proc


def probe(path: Path) -> dict:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
    )
    return json.loads(proc.stdout or "{}")


def has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def duration(path: Path) -> float | None:
    value = probe(path).get("format", {}).get("duration")
    return float(value) if value else None


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
