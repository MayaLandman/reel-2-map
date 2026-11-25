from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from reel2map.media import ffmpeg
from reel2map.media.frames import changed_fraction, dedupe, extract_frames, signature, spread
from reel2map.media.ocr import novelty_order
from reel2map.models import Frame
from tests.conftest import needs_ffmpeg


def _img(path: Path, color: tuple[int, int, int], stripe: bool = False) -> Path:
    img = Image.new("RGB", (64, 64), color)
    if stripe:
        for x in range(0, 64, 8):
            for y in range(64):
                img.putpixel((x, y), (255, 255, 255))
    img.save(path)
    return path


def test_signature_ignores_noise_but_sees_text(tmp_path: Path) -> None:
    from PIL import ImageDraw

    base = Image.new("RGB", (360, 640), (40, 60, 80))
    noisy = base.copy()
    noisy.putpixel((10, 10), (45, 62, 83))
    titled = base.copy()
    ImageDraw.Draw(titled).rectangle((40, 280, 320, 330), fill=(255, 255, 255))
    assert changed_fraction(signature(base), signature(noisy)) == 0
    assert changed_fraction(signature(base), signature(titled)) > 0.03


def test_dedupe_drops_near_duplicates(tmp_path: Path) -> None:
    frames = [
        Frame(path=_img(tmp_path / "1.jpg", (10, 10, 10), stripe=True), timestamp=0),
        Frame(path=_img(tmp_path / "2.jpg", (12, 10, 10), stripe=True), timestamp=1),
        Frame(path=_img(tmp_path / "3.jpg", (200, 30, 30)), timestamp=2),
    ]
    kept = dedupe(frames)
    assert [f.timestamp for f in kept] == [0, 2]
    assert not (tmp_path / "2.jpg").exists()


def test_spread_keeps_first_and_last() -> None:
    items = [Frame(path=Path(f"{i}.jpg"), timestamp=i) for i in range(10)]
    picked = spread(items, 4)
    assert len(picked) == 4
    assert picked[0].timestamp == 0 and picked[-1].timestamp == 9
    assert spread(items, 20) == items


def test_novelty_prefers_frames_with_new_text() -> None:
    frames = [Frame(path=Path(f"{i}.jpg"), timestamp=i) for i in range(6)]
    frames[1].ocr_text = "Pizzeria Remo Testaccio"
    frames[2].ocr_text = "Pizzeria Remo Testaccio"  # repeat, adds nothing
    frames[4].ocr_text = "Bar del Fico | spritz"
    chosen = novelty_order(frames, 3)
    ts = [f.timestamp for f in chosen]
    assert 1 in ts and 4 in ts and 2 not in ts
    assert ts == sorted(ts)


@needs_ffmpeg
def test_extract_frames_finds_each_shot(sample_video: Path, tmp_path: Path) -> None:
    frames = extract_frames(sample_video, tmp_path / "frames", interval=10, max_frames=20)
    # red, blue and the test pattern: one frame each, near-duplicates removed
    assert 3 <= len(frames) <= 5
    assert frames[0].timestamp == 0
    assert all(f.path.exists() for f in frames)


@needs_ffmpeg
def test_audio_helpers(sample_video: Path, tmp_path: Path) -> None:
    assert ffmpeg.has_audio(sample_video)
    wav = ffmpeg.extract_audio(sample_video, tmp_path / "a.wav")
    assert wav.stat().st_size > 1000
    assert 7 < (ffmpeg.duration(sample_video) or 0) < 9


@needs_ffmpeg
def test_listicle_frames_and_ocr(tmp_path: Path) -> None:
    """A fake 'top 3' reel: one title card per place on the same background."""
    pytest.importorskip("rapidocr_onnxruntime")
    import subprocess

    from reel2map.media.ocr import ocr_frames

    cards = [("ROSCIOLI", 0, 3), ("PIZZARIUM", 3, 6), ("GIOLITTI", 6, 9)]
    vf = ",".join(
        f"drawtext=text='{name}':fontsize=64:fontcolor=white:x=(w-tw)/2:y=600"
        f":enable='between(t,{a},{b - 0.01})'"
        for name, a, b in cards
    )
    video = tmp_path / "listicle.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x334455:s=720x1280:d=9",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    frames = ocr_frames(extract_frames(video, tmp_path / "frames", interval=2.0))
    text = " ".join(f.ocr_text for f in frames).upper()
    for name, _, _ in cards:
        assert name in text
    assert len(frames) <= 6  # repeated title cards were deduplicated


@needs_ffmpeg
def test_load_wav_reads_ffmpeg_output(sample_video: Path, tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from reel2map.media.transcribe import load_wav

    samples = load_wav(ffmpeg.extract_audio(sample_video, tmp_path / "a.wav"))
    assert 7.5 * 16000 < len(samples) < 8.5 * 16000
    assert 0.1 < float(abs(samples).max()) <= 1.0
