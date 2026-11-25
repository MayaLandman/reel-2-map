"""Turn the scraped evidence into a structured list of places with Claude.

The model gets text (caption, transcript, OCR) for the whole post, plus the frames as
images. Long posts are split into several calls over consecutive groups of frames, so a
20-place listicle is not squeezed into one context; the per-call results are merged.
Structured output is enforced through a single forced tool call.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import unicodedata
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from reel2map.extract.prompts import REPORT_TOOL, SYSTEM_PROMPT, build_evidence_text
from reel2map.models import Frame, Place, Post, Transcript

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English",
    "he": "Hebrew",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


class Extractor(Protocol):
    def extract(
        self, post: Post, transcript: Transcript | None, frames: Sequence[Frame]
    ) -> Extraction: ...


@dataclass
class Extraction:
    summary: str = ""
    places: list[Place] = field(default_factory=list)


def encode_image(frame: Frame, max_side: int = 768) -> dict[str, Any]:
    with Image.open(frame.path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode(),
        },
    }


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(the|ristorante|trattoria|osteria|pizzeria|bar|cafe|caffe)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text) or name.strip().lower()


def merge_places(groups: Sequence[Sequence[Place]]) -> list[Place]:
    """Merge places reported by several calls, keeping the most confident version."""
    merged: dict[str, Place] = {}
    for group in groups:
        for place in group:
            k = _key(place.name)
            current = merged.get(k)
            if current is None:
                merged[k] = place.model_copy(deep=True)
                continue
            best = place if place.confidence > current.confidence else current
            other = current if best is place else place
            combined = best.model_copy(deep=True)
            for attr in ("city", "country", "address_hint", "timestamp"):
                if getattr(combined, attr) is None:
                    setattr(combined, attr, getattr(other, attr))
            if len(other.why) > len(combined.why):
                combined.why = other.why
            combined.evidence = sorted(set(combined.evidence) | set(other.evidence))
            merged[k] = combined
    return sorted(merged.values(), key=lambda p: (p.timestamp is None, p.timestamp or 0))


class ClaudeExtractor:
    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-sonnet-5-5",
        *,
        max_images: int = 20,
        language: str = "en",
        client: Any | None = None,
        max_parallel: int = 3,
    ) -> None:
        if client is None:
            import anthropic

            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic(api_key=api_key)
        self.client = client
        self.model = model
        self.max_images = max_images
        self.language = LANGUAGE_NAMES.get(language, language)
        self.max_parallel = max_parallel

    def _call(self, text: str, images: Sequence[Frame]) -> Extraction:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for frame in images:
            label = f"{frame.timestamp:.1f}s" if frame.timestamp is not None else frame.path.stem
            content.append({"type": "text", "text": f"Frame at {label}:"})
            content.append(encode_image(frame))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=SYSTEM_PROMPT.format(language=self.language),
            tools=[REPORT_TOOL],
            tool_choice={"type": "tool", "name": REPORT_TOOL["name"]},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                data = block.input or {}
                places = []
                for raw in data.get("places", []):
                    try:
                        places.append(Place.model_validate(raw))
                    except ValueError as exc:
                        log.warning("dropping malformed place %r: %s", raw, exc)
                return Extraction(summary=data.get("summary", ""), places=places)
        log.warning("model returned no tool call (stop_reason=%s)", response.stop_reason)
        return Extraction()

    def extract(
        self, post: Post, transcript: Transcript | None, frames: Sequence[Frame]
    ) -> Extraction:
        from reel2map.media.ocr import novelty_order

        groups = _image_groups(frames, self.max_images)
        if len(groups) <= 1:
            images = novelty_order(frames, self.max_images)
            return self._call(build_evidence_text(post, transcript, frames), images)

        def run(i: int) -> Extraction:
            part = (
                f"This is part {i + 1} of {len(groups)}: the images below are a consecutive "
                "slice of the video. Report the places visible or mentioned in this slice, "
                "using the full transcript and OCR for context."
            )
            images = novelty_order(groups[i], self.max_images)
            return self._call(build_evidence_text(post, transcript, frames, part=part), images)

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            results = list(pool.map(run, range(len(groups))))
        summary = next((r.summary for r in results if r.summary), "")
        return Extraction(summary=summary, places=merge_places([r.places for r in results]))


def _image_groups(frames: Sequence[Frame], per_call: int, max_calls: int = 4) -> list[list[Frame]]:
    """Split frames into consecutive groups when there are too many for one call."""
    if len(frames) <= per_call * 1.5:
        return [list(frames)]
    calls = min(max_calls, -(-len(frames) // per_call))
    size = -(-len(frames) // calls)
    return [list(frames[i : i + size]) for i in range(0, len(frames), size)]
