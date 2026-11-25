from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from reel2map.api.app import create_app
from reel2map.extract.llm import Extraction
from reel2map.models import GeoMatch, Place, Transcript, TranscriptSegment
from reel2map.pipeline import Options, Pipeline
from tests.conftest import needs_ffmpeg


class FakeExtractor:
    def __init__(self):
        self.seen = None

    def extract(self, post, transcript, frames):
        self.seen = (post, transcript, frames)
        return Extraction(
            summary="3 Rome spots",
            places=[
                Place(
                    name="Roscioli",
                    category="restaurant",
                    city="Rome",
                    why="carbonara",
                    confidence=0.95,
                    timestamp=1,
                ),
                Place(name="maybe a bar", category="bar", why="?", confidence=0.2),
            ],
        )


class FakeGeocoder:
    def locate(self, place):
        return GeoMatch(
            provider="nominatim",
            name=place.name,
            lat=41.89,
            lon=12.47,
            maps_url="https://example.com",
        )


def _fake_download(video: Path):
    def downloader(url, workdir, settings, with_comments=False):
        from reel2map.models import Platform, Post

        return Post(
            url=url, platform=Platform.TIKTOK, id="1", description="#rome", video_path=video
        )

    return downloader


@needs_ffmpeg
def test_pipeline_end_to_end_with_fakes(sample_video, settings) -> None:
    extractor = FakeExtractor()
    pipe = Pipeline(
        settings,
        downloader=_fake_download(sample_video),
        transcriber=lambda audio, model: Transcript(
            language="en", segments=[TranscriptSegment(start=0, end=1, text="go to Roscioli")]
        ),
        ocr=lambda frames: frames,
        extractor=extractor,
        geocoder=FakeGeocoder(),
    )
    stages = []
    result = pipe.run("https://www.tiktok.com/@a/video/1", Options(), progress=stages.append)

    assert [p.name for p in result.places] == ["Roscioli"]  # low-confidence one filtered
    assert result.places[0].geo and result.places[0].geo.lat == 41.89
    assert result.transcript and "Roscioli" in result.transcript.text
    assert result.frames and result.post.video_path is None  # temp paths scrubbed
    assert stages == ["download", "transcribe + frames + ocr", "extract", "geocode"]
    _, transcript, frames = extractor.seen
    assert transcript is not None and len(frames) >= 3


@needs_ffmpeg
def test_pipeline_degrades_when_transcription_fails(sample_video, settings) -> None:
    def broken(audio, model):
        raise RuntimeError("no whisper")

    pipe = Pipeline(
        settings,
        downloader=_fake_download(sample_video),
        transcriber=broken,
        ocr=lambda f: f,
        extractor=FakeExtractor(),
        geocoder=FakeGeocoder(),
    )
    result = pipe.run("https://x.com/v", Options(geocode=False))
    assert result.transcript is None
    assert any("transcription skipped" in w for w in result.warnings)
    assert result.places[0].geo is None


def test_api_job_lifecycle(settings) -> None:
    class InstantPipeline:
        def run(self, url, options, progress=None):
            from reel2map.models import ExtractionResult, Platform, Post, ResolvedPlace

            progress and progress("extract")
            return ExtractionResult(
                post=Post(url=url, platform=Platform.TIKTOK, id="1"),
                places=[
                    ResolvedPlace(name="Roscioli", category="restaurant", why="x", confidence=0.9)
                ],
            )

    client = TestClient(create_app(pipeline_factory=lambda lang: InstantPipeline()))
    assert client.get("/api/health").json()["status"] == "ok"
    assert "reel2map" in client.get("/").text

    job = client.post("/api/jobs", json={"url": "https://www.tiktok.com/@a/video/1"}).json()
    for _ in range(50):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] == "done":
            break
        time.sleep(0.05)
    assert state["status"] == "done"
    assert state["result"]["places"][0]["name"] == "Roscioli"

    csv = client.get(f"/api/jobs/{job['id']}/export", params={"format": "csv"})
    assert csv.status_code == 200 and "Roscioli" in csv.text
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs", json={"url": "not a url"}).status_code == 422


def test_pipeline_fails_fast_without_api_key(settings) -> None:
    import pytest

    called = []
    settings.anthropic_api_key = None
    settings.backend = "api"
    pipe = Pipeline(settings, downloader=lambda *a, **k: called.append(a))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        pipe.run("https://www.tiktok.com/@a/video/1")
    assert called == []  # nothing was downloaded
