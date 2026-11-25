from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from reel2map.extract import claude_code
from reel2map.extract.claude_code import ClaudeCodeError, ClaudeCodeExtractor, parse_json_object
from reel2map.extract.llm import ClaudeExtractor
from reel2map.models import Frame
from reel2map.pipeline import make_extractor


def _frames(tmp_path: Path, n: int = 3) -> list[Frame]:
    out = []
    for i in range(n):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (16, 16), (i * 40, 0, 0)).save(p)
        out.append(Frame(path=p, timestamp=float(i), ocr_text=f"PLACE {i}"))
    return out


def _runner(reply: str, calls: list, returncode: int = 0):
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        envelope = {"type": "result", "is_error": False, "result": reply}
        return subprocess.CompletedProcess(cmd, returncode, json.dumps(envelope), "")

    return run


def test_parse_json_object_tolerates_fences_and_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"places": []} hope it helps') == {"places": []}
    with pytest.raises(ClaudeCodeError):
        parse_json_object("sorry, no places")


def test_claude_code_extractor_builds_command_and_parses(tmp_path: Path, post) -> None:
    calls: list = []
    reply = json.dumps(
        {
            "summary": "Rome eats",
            "places": [
                {
                    "name": "Roscioli",
                    "category": "restaurant",
                    "why": "carbonara",
                    "evidence": ["on_screen_text"],
                    "confidence": 0.9,
                }
            ],
        }
    )
    ex = ClaudeCodeExtractor(cli="/usr/bin/claude", runner=_runner(reply, calls), language="he")
    result = ex.extract(post, None, _frames(tmp_path))

    assert [p.name for p in result.places] == ["Roscioli"] and result.summary == "Rome eats"
    cmd, _ = calls[0]
    assert cmd[:2] == ["/usr/bin/claude", "-p"]
    assert cmd[cmd.index("--allowedTools") :][:2] == ["--allowedTools", "Read"]
    assert str(tmp_path.resolve()) in cmd
    prompt = cmd[2]
    assert "Hebrew" in prompt and "frame_2.jpg" in prompt and "PLACE 1" in prompt
    assert "report_places tool" not in prompt


def test_claude_code_error_is_reported(tmp_path: Path, post) -> None:
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "not logged in")

    ex = ClaudeCodeExtractor(cli="claude", runner=run)
    with pytest.raises(ClaudeCodeError, match="not logged in"):
        ex.extract(post, None, _frames(tmp_path))


def test_backend_selection(settings, monkeypatch) -> None:
    monkeypatch.setattr(claude_code, "claude_cli", lambda: "/usr/local/bin/claude")
    assert isinstance(make_extractor(settings), ClaudeExtractor)  # key set -> API
    settings.anthropic_api_key = None
    assert isinstance(make_extractor(settings), ClaudeCodeExtractor)  # no key -> plan
    settings.backend = "api"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_extractor(settings)
    monkeypatch.setattr(claude_code, "claude_cli", lambda: None)
    settings.backend = "claude-code"
    with pytest.raises(ValueError, match="Claude Code CLI not found"):
        make_extractor(settings)
