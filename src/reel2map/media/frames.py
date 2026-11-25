"""Pick the frames worth reading.

Recommendation videos are mostly "one place per shot", with the name burnt into the
picture as an overlay. We want one good frame per shot, not one every N milliseconds.

1. ffmpeg's scene-change score picks frames where the picture changes a lot, plus at
   least one frame every `interval` seconds so slow pans with changing text are covered.
2. Near-duplicates (same shot, same overlay) are dropped by comparing small grayscale
   thumbnails and counting how many pixels changed strongly.
3. If there are still too many, keep an even spread across the video.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from reel2map.media.ffmpeg import require_ffmpeg, run
from reel2map.models import Frame

PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def signature(image: Image.Image, width: int = 64) -> bytes:
    """A small grayscale thumbnail (aspect kept): enough detail to notice a new text overlay."""
    height = max(16, round(width * image.height / image.width))
    return image.convert("L").resize((width, height), Image.Resampling.BOX).tobytes()


def changed_fraction(a: bytes, b: bytes, pixel_delta: int = 24) -> float:
    """Share of thumbnail pixels that changed noticeably between two frames.

    Compression noise moves pixels by a few levels; a new caption, a cut or a new
    product shot moves a block of pixels by a lot. Counting strongly changed pixels
    (instead of averaging) keeps small but important changes such as overlay text."""
    changed = sum(abs(x - y) > pixel_delta for x, y in zip(a, b, strict=True))
    return changed / len(a)


def dedupe(frames: Sequence[Frame], min_change: float = 0.004) -> list[Frame]:
    """Drop frames that look like the previously kept one."""
    kept: list[Frame] = []
    last: bytes | None = None
    for frame in frames:
        with Image.open(frame.path) as img:
            sig = signature(img)
        if last is not None and changed_fraction(sig, last) < min_change:
            frame.path.unlink(missing_ok=True)
        else:
            kept.append(frame)
            last = sig
    return kept


def spread(items: Sequence[Frame], limit: int) -> list[Frame]:
    """Evenly subsample, always keeping the first and the last item."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(i * step)] for i in range(limit)]


def extract_frames(
    video: Path,
    outdir: Path,
    *,
    scene_threshold: float = 0.3,
    interval: float = 2.0,
    max_frames: int = 60,
    width: int = 720,
) -> list[Frame]:
    require_ffmpeg()
    outdir.mkdir(parents=True, exist_ok=True)
    select = f"select='eq(n\\,0)+gt(scene\\,{scene_threshold})+gte(t-prev_selected_t\\,{interval})'"
    vf = f"{select},scale='min({width}\\,iw)':-2,showinfo"
    proc = run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            vf,
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            str(outdir / "frame_%04d.jpg"),
        ]
    )
    times = [float(t) for t in PTS_RE.findall(proc.stderr)]
    paths = sorted(outdir.glob("frame_*.jpg"))
    frames = [
        Frame(path=p, timestamp=times[i] if i < len(times) else None) for i, p in enumerate(paths)
    ]
    return spread(dedupe(frames), max_frames)


def frames_from_images(images: Sequence[Path]) -> list[Frame]:
    """Photo carousels: every slide is already a frame."""
    return [Frame(path=p, timestamp=None) for p in images]
