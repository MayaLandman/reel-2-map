"""Free, fully local extraction with an open model served by Ollama (https://ollama.com).

No API key and no cost: the model runs on your own machine. Small vision models are
weaker than Claude at reading frames, so this backend leans on the OCR text and the
transcript and only sends a handful of images. Ollama's structured outputs (`format`
with a JSON schema) keep the reply machine-readable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from reel2map.extract.claude_code import parse_json_object
from reel2map.extract.llm import LANGUAGE_NAMES, Extraction, encode_image
from reel2map.extract.prompts import REPORT_TOOL, SYSTEM_PROMPT, build_evidence_text
from reel2map.models import Frame, Place, Post, Transcript

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def ollama_running(base_url: str = DEFAULT_URL, client: httpx.Client | None = None) -> bool:
    try:
        resp = (client or httpx).get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


class OllamaExtractor:
    def __init__(
        self,
        model: str = "gemma3:4b",
        *,
        base_url: str = DEFAULT_URL,
        max_images: int = 6,
        language: str = "en",
        client: httpx.Client | None = None,
        timeout: float = 600,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_images = max_images
        self.language = LANGUAGE_NAMES.get(language, language)
        self.client = client or httpx.Client(timeout=timeout)

    def extract(
        self, post: Post, transcript: Transcript | None, frames: Sequence[Frame]
    ) -> Extraction:
        from reel2map.media.ocr import novelty_order

        images = novelty_order(frames, self.max_images) if self.max_images else []
        system = SYSTEM_PROMPT.format(language=self.language).replace(
            "Report the result by calling the report_places tool exactly once.",
            "Answer with one JSON object that matches the requested schema.",
        )
        body = {
            "model": self.model,
            "stream": False,
            "format": REPORT_TOOL["input_schema"],
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": build_evidence_text(post, transcript, frames)
                    + (
                        f"\n\nThe {len(images)} attached images are frames from the video, "
                        "in order."
                        if images
                        else ""
                    ),
                    "images": [encode_image(f)["source"]["data"] for f in images],
                },
            ],
        }
        try:
            resp = self.client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Ollama is not running at {self.base_url}. Install it from https://ollama.com, "
                f"then run: ollama pull {self.model}"
            ) from exc
        if resp.status_code == 404:
            raise OllamaError(
                f"model {self.model!r} is not installed; run: ollama pull {self.model}"
            )
        resp.raise_for_status()
        data = parse_json_object(resp.json().get("message", {}).get("content", ""))

        places = []
        for raw in data.get("places", []):
            try:
                places.append(Place.model_validate(raw))
            except ValueError as exc:
                log.warning("dropping malformed place %r: %s", raw, exc)
        return Extraction(summary=data.get("summary", ""), places=places)
