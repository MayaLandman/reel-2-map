"""Read on-screen text (place names, prices, "#3", addresses) from frames.

Uses RapidOCR (PaddleOCR models on ONNX Runtime): no system packages, runs on CPU.
Optional: install with `pip install reel2map[ocr]`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from reel2map.models import Frame

log = logging.getLogger(__name__)


class OCRUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise OCRUnavailable("rapidocr is not installed; run: pip install 'reel2map[ocr]'") from exc
    return RapidOCR()


def read_text(frame: Frame, min_confidence: float = 0.6) -> str:
    result, _ = _engine()(str(frame.path))
    if not result:
        return ""
    # Each item is (box, text, score). Sort top-to-bottom, then left-to-right.
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
    words = [text.strip() for _, text, score in lines if float(score) >= min_confidence]
    return " | ".join(w for w in words if len(w) > 1)


def ocr_frames(frames: Sequence[Frame]) -> list[Frame]:
    for frame in frames:
        try:
            frame.ocr_text = read_text(frame)
        except OCRUnavailable:
            raise
        except Exception as exc:  # pragma: no cover - defensive, one bad frame
            log.warning("OCR failed on %s: %s", frame.path.name, exc)
    return list(frames)


def novelty_order(frames: Sequence[Frame], limit: int) -> list[Frame]:
    """Choose which frames to send to the LLM as images.

    Frames whose on-screen text adds new words come first, then the rest evenly. The
    result is returned in video order so the model sees the story as it was told.
    """
    if len(frames) <= limit:
        return list(frames)
    seen: set[str] = set()
    novel: list[tuple[int, int]] = []
    for i, frame in enumerate(frames):
        words = {w.lower() for w in frame.ocr_text.replace("|", " ").split() if len(w) > 2}
        new = len(words - seen)
        seen |= words
        if new:
            novel.append((new, i))
    chosen = {i for _, i in sorted(novel, key=lambda x: (-x[0], x[1]))[:limit]}
    rest = [i for i in range(len(frames)) if i not in chosen]
    free = limit - len(chosen)
    if free > 0 and rest:
        step = len(rest) / free
        chosen |= {rest[int(k * step)] for k in range(free)}
    return [f for i, f in enumerate(frames) if i in chosen]
