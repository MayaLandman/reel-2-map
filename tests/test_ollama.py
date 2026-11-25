from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from reel2map.extract import claude_code, ollama
from reel2map.extract.ollama import OllamaError, OllamaExtractor
from reel2map.models import Frame
from reel2map.pipeline import make_extractor


def _frames(tmp_path: Path, n: int = 10) -> list[Frame]:
    out = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (16, 16), (i * 20, 0, 0)).save(p)
        out.append(Frame(path=p, timestamp=float(i), ocr_text=f"spot{i}"))
    return out


def test_ollama_request_and_parse(tmp_path: Path, post) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        reply = {
            "summary": "Rome",
            "places": [
                {
                    "name": "Giolitti",
                    "category": "gelato",
                    "why": "old school gelato",
                    "evidence": ["on_screen_text"],
                    "confidence": 0.8,
                }
            ],
        }
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": json.dumps(reply)}}
        )

    ex = OllamaExtractor(
        "gemma3:4b",
        max_images=3,
        language="he",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = ex.extract(post, None, _frames(tmp_path))

    assert [p.name for p in result.places] == ["Giolitti"]
    body = seen["body"]
    assert body["model"] == "gemma3:4b" and body["stream"] is False
    assert body["format"]["required"] == ["summary", "places"]  # structured output schema
    assert len(body["messages"][1]["images"]) == 3
    assert "Hebrew" in body["messages"][0]["content"]
    assert "spot7" in body["messages"][1]["content"]  # OCR of every frame is still sent


def test_ollama_missing_model_message(tmp_path: Path, post) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    ex = OllamaExtractor("qwen2.5vl:7b", client=client)
    with pytest.raises(OllamaError, match=r"ollama pull qwen2\.5vl:7b"):
        ex.extract(post, None, _frames(tmp_path, 2))


def test_ollama_not_running_message(tmp_path: Path, post) -> None:
    def handler(request):
        raise httpx.ConnectError("refused")

    ex = OllamaExtractor(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(OllamaError, match="not running"):
        ex.extract(post, None, _frames(tmp_path, 2))


def test_auto_falls_back_to_ollama(settings, monkeypatch) -> None:
    settings.anthropic_api_key = None
    monkeypatch.setattr(claude_code, "claude_cli", lambda: None)
    monkeypatch.setattr(ollama, "ollama_running", lambda url: True)
    assert isinstance(make_extractor(settings), OllamaExtractor)

    monkeypatch.setattr(ollama, "ollama_running", lambda url: False)
    with pytest.raises(ValueError, match="No LLM backend available"):
        make_extractor(settings)
