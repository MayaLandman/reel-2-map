"""Prompt and tool schema for place extraction."""

from __future__ import annotations

from collections.abc import Sequence

from reel2map.models import Frame, Post, Transcript

SYSTEM_PROMPT = """\
You read short-form travel and food videos (TikTok, Instagram Reels, YouTube Shorts) and \
list every real-world place they recommend, so each one can be pinned on a map.

You receive the evidence that was scraped from one post:
- the caption, hashtags, location tag and top comments,
- a timestamped speech transcript,
- on-screen text read by OCR from each sampled frame,
- a selection of the frames themselves.

Rules:
- List EVERY place the post recommends. Listicles often have 10, 20 or more; do not stop early.
- A place is a specific venue or landmark someone could navigate to: a restaurant, cafe, \
bar, bakery, gelateria, shop, hotel, market, museum, viewpoint, park, street or square. \
Not a dish, a brand in general, or a whole city.
- Use the exact spelling shown on screen or on the storefront when you can see it. OCR \
and speech recognition make mistakes: cross-check them against each other and the images.
- Background mentions do not count (a chain store visible behind someone, a product).
- If the post names a branch or a street, put it in address_hint.
- Infer city and country from the caption, hashtags, location tag, language, signage and \
landmarks. Leave them empty rather than guessing wildly.
- confidence: 0.9+ when the name is clearly written or said; 0.5-0.8 when inferred from \
partial evidence; below 0.5 when you are unsure it is even a real place.
- evidence lists which sources support the place.
- timestamp: the second in the video where the place is shown, if known.
- why: one or two concrete sentences on why it is recommended (the dish, the view, the \
price), written in {language}.
Report the result by calling the report_places tool exactly once.
"""

PLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "category": {
            "type": "string",
            "description": "restaurant, pizzeria, cafe, bar, bakery, gelato, dessert, "
            "street_food, market, shop, hotel, attraction, museum, viewpoint, park, other",
        },
        "city": {"type": ["string", "null"]},
        "country": {"type": ["string", "null"]},
        "address_hint": {"type": ["string", "null"]},
        "why": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "caption",
                    "hashtag",
                    "location_tag",
                    "speech",
                    "on_screen_text",
                    "visual",
                    "comment",
                ],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "timestamp": {"type": ["number", "null"]},
    },
    "required": ["name", "category", "why", "evidence", "confidence"],
}

REPORT_TOOL = {
    "name": "report_places",
    "description": "Report every place recommended in the post.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One sentence: what the post is about.",
            },
            "places": {"type": "array", "items": PLACE_SCHEMA},
        },
        "required": ["summary", "places"],
    },
}


def build_evidence_text(
    post: Post, transcript: Transcript | None, frames: Sequence[Frame], *, part: str = ""
) -> str:
    lines = [f"POST: {post.url} ({post.platform.value})"]
    if part:
        lines.append(part)
    if post.uploader:
        lines.append(f"Author: {post.uploader}")
    if post.title:
        lines.append(f"Title: {post.title}")
    lines.append(f"Caption: {post.description.strip() or '(none)'}")
    if post.hashtags:
        lines.append("Hashtags: " + " ".join(f"#{t}" for t in post.hashtags))
    if post.location_tag:
        lines.append(f"Location tag: {post.location_tag}")
    if post.duration:
        lines.append(f"Duration: {post.duration:.0f}s")
    if post.comments:
        lines.append("Top comments:")
        lines += [f"- {c[:300]}" for c in post.comments]

    lines.append("")
    if transcript and transcript.segments:
        lines.append(f"SPEECH TRANSCRIPT (language: {transcript.language or 'unknown'}):")
        lines.append(transcript.timestamped())
    else:
        lines.append("SPEECH TRANSCRIPT: (no speech detected or transcription disabled)")

    lines.append("")
    ocr_lines = [f"[{_ts(f)}] {f.ocr_text}" for f in frames if f.ocr_text.strip()]
    if ocr_lines:
        lines.append("ON-SCREEN TEXT BY FRAME (OCR):")
        lines += ocr_lines
    else:
        lines.append("ON-SCREEN TEXT: (none found or OCR disabled)")
    return "\n".join(lines)


def _ts(frame: Frame) -> str:
    return f"{frame.timestamp:6.1f}s" if frame.timestamp is not None else frame.path.stem
