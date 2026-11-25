"""Extraction through the local Claude Code CLI (`claude -p`), using a Claude plan login.

This lets people who have Claude Code signed in with a Claude subscription run
reel2map without creating an API key: usage counts against their plan's limits.
Claude Code reads the frame images itself with its Read tool, and is asked to answer
with a single JSON object that follows the same schema as the API backend.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from reel2map.extract.llm import LANGUAGE_NAMES, Extraction
from reel2map.extract.prompts import REPORT_TOOL, SYSTEM_PROMPT, build_evidence_text
from reel2map.models import Frame, Place, Post, Transcript

log = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class ClaudeCodeError(RuntimeError):
    pass


def claude_cli() -> str | None:
    return shutil.which("claude")


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply, tolerating ``` fences or a stray sentence."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else (JSON_BLOCK_RE.search(text) or [None])[0]
    if not candidate:
        raise ClaudeCodeError(f"no JSON object in Claude Code reply: {text[:300]!r}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ClaudeCodeError(f"invalid JSON from Claude Code: {exc}") from exc


class ClaudeCodeExtractor:
    def __init__(
        self,
        model: str = "sonnet",
        *,
        max_images: int = 20,
        language: str = "en",
        cli: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: int = 900,
    ) -> None:
        cli = cli or claude_cli()
        if not cli:
            raise ValueError(
                "Claude Code CLI not found. Install it and sign in with `claude`, "
                "or set ANTHROPIC_API_KEY to use the API instead."
            )
        self.cli = cli
        self.model = model
        self.max_images = max_images
        self.language = LANGUAGE_NAMES.get(language, language)
        self.runner = runner
        self.timeout = timeout

    def build_prompt(
        self,
        post: Post,
        transcript: Transcript | None,
        frames: Sequence[Frame],
        images: Sequence[Frame],
    ) -> str:
        instructions = SYSTEM_PROMPT.format(language=self.language).replace(
            "Report the result by calling the report_places tool exactly once.",
            "Read every frame image listed below with the Read tool before answering.",
        )
        listing = "\n".join(
            f"- {f.path.resolve()}"
            + (f"  (at {f.timestamp:.1f}s)" if f.timestamp is not None else "")
            for f in images
        )
        schema = json.dumps(REPORT_TOOL["input_schema"], ensure_ascii=False)
        return (
            f"{instructions}\n\n{build_evidence_text(post, transcript, frames)}\n\n"
            f"FRAME IMAGES:\n{listing or '(none)'}\n\n"
            "Answer with ONLY one JSON object (no prose, no code fence) that matches this "
            f"JSON schema:\n{schema}\n"
        )

    def extract(
        self, post: Post, transcript: Transcript | None, frames: Sequence[Frame]
    ) -> Extraction:
        from reel2map.media.ocr import novelty_order

        images = novelty_order(frames, self.max_images)
        dirs = sorted({str(f.path.resolve().parent) for f in images})
        cmd = [
            self.cli,
            "-p",
            self.build_prompt(post, transcript, frames, images),
            "--output-format",
            "json",
            "--model",
            self.model,
            "--allowedTools",
            "Read",
        ]
        for d in dirs:
            cmd += ["--add-dir", d]
        proc = self.runner(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=dirs[0] if dirs else None,
            check=False,
        )
        if proc.returncode != 0:
            raise ClaudeCodeError(f"claude -p failed: {(proc.stderr or proc.stdout)[-500:]}")
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise ClaudeCodeError(f"Claude Code error: {envelope.get('result')}")
        data = parse_json_object(envelope.get("result", ""))

        places = []
        for raw in data.get("places", []):
            try:
                places.append(Place.model_validate(raw))
            except ValueError as exc:
                log.warning("dropping malformed place %r: %s", raw, exc)
        return Extraction(summary=data.get("summary", ""), places=places)
