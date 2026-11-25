from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from reel2map.extract.llm import ClaudeExtractor, _image_groups, merge_places
from reel2map.extract.prompts import build_evidence_text
from reel2map.models import Frame, Place, Transcript, TranscriptSegment


class FakeMessages:
    def __init__(self, places_per_call):
        self.calls = []
        self.places_per_call = places_per_call

    def create(self, **kwargs):
        self.calls.append(kwargs)
        places = self.places_per_call[min(len(self.calls) - 1, len(self.places_per_call) - 1)]
        block = SimpleNamespace(type="tool_use", input={"summary": "Rome food", "places": places})
        return SimpleNamespace(content=[block], stop_reason="tool_use")


def _client(places_per_call):
    return SimpleNamespace(messages=FakeMessages(places_per_call))


def _frames(tmp_path: Path, n: int) -> list[Frame]:
    frames = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (32, 32), (i * 3 % 255, 0, 0)).save(p)
        frames.append(Frame(path=p, timestamp=float(i)))
    return frames


def test_evidence_text_contains_every_source(post) -> None:
    transcript = Transcript(
        language="en",
        segments=[TranscriptSegment(start=1.0, end=3.0, text="first stop is Roscioli")],
    )
    frames = [Frame(path=Path("a.jpg"), timestamp=2.0, ocr_text="ROSCIOLI | Via dei Giubbonari")]
    post.comments = ["what's the name of #2?"]
    text = build_evidence_text(post, transcript, frames)
    for needle in ("#rome", "first stop is Roscioli", "Via dei Giubbonari", "#2?", "romefood"):
        assert needle in text


def test_single_call_parses_tool_output(tmp_path: Path, post) -> None:
    client = _client(
        [
            [
                {
                    "name": "Roscioli",
                    "category": "restaurant",
                    "city": "Rome",
                    "why": "carbonara",
                    "evidence": ["speech", "on_screen_text"],
                    "confidence": 0.95,
                    "timestamp": 2,
                },
                {"name": "broken", "confidence": 7},  # invalid: dropped, not fatal
            ]
        ]
    )
    ex = ClaudeExtractor(None, client=client, max_images=5, language="he")
    result = ex.extract(post, None, _frames(tmp_path, 4))
    assert result.summary == "Rome food"
    assert [p.name for p in result.places] == ["Roscioli"]
    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "report_places"}
    assert "Hebrew" in call["system"]
    images = [c for c in call["messages"][0]["content"] if c["type"] == "image"]
    assert len(images) == 4


def test_long_videos_are_split_and_merged(tmp_path: Path, post) -> None:
    a = {
        "name": "Pizzeria Remo",
        "category": "pizzeria",
        "why": "thin crust",
        "evidence": ["visual"],
        "confidence": 0.6,
        "timestamp": 3,
    }
    b = {
        "name": "Remo",
        "category": "pizzeria",
        "city": "Rome",
        "why": "the thinnest crust in town",
        "evidence": ["on_screen_text"],
        "confidence": 0.9,
        "timestamp": 3,
    }
    c = {
        "name": "Giolitti",
        "category": "gelato",
        "why": "since 1900",
        "evidence": ["speech"],
        "confidence": 0.8,
        "timestamp": 50,
    }
    client = _client([[a], [b, c], [c]])
    ex = ClaudeExtractor(None, client=client, max_images=10)
    result = ex.extract(post, None, _frames(tmp_path, 40))
    assert len(client.messages.calls) == 4
    names = [p.name for p in result.places]
    assert names == ["Remo", "Giolitti"]
    remo = result.places[0]
    assert remo.city == "Rome" and set(remo.evidence) == {"visual", "on_screen_text"}


def test_image_groups() -> None:
    frames = [Frame(path=Path(f"{i}.jpg")) for i in range(25)]
    assert len(_image_groups(frames, 20)) == 1
    groups = _image_groups([Frame(path=Path(f"{i}.jpg")) for i in range(100)], 20)
    assert len(groups) == 4 and sum(map(len, groups)) == 100


def test_merge_keeps_unique_places(place) -> None:
    other = Place(
        name="Trattoria Da Enzo al 29",
        category="restaurant",
        why="x",
        confidence=0.5,
        address_hint=None,
        city=None,
    )
    merged = merge_places([[place], [other]])
    assert len(merged) == 1 and merged[0].address_hint == "Trastevere"
