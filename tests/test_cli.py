from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar

from typer.testing import CliRunner

from reel2map import cli
from reel2map.models import ExtractionResult, GeoMatch, Platform, Post, ResolvedPlace

runner = CliRunner()


class FakePipeline:
    calls: ClassVar[list[str]] = []

    def __init__(self, settings):
        self.settings = settings

    def run(self, url, options, workdir=None, progress=None):
        FakePipeline.calls.append(url)
        if "broken" in url:
            raise RuntimeError("private video")
        return ExtractionResult(
            post=Post(url=url, platform=Platform.TIKTOK, id="1"),
            summary="Rome",
            places=[
                ResolvedPlace(
                    name="Roscioli",
                    category="restaurant",
                    why="carbonara",
                    confidence=0.9,
                    geo=GeoMatch(
                        provider="nominatim",
                        name="Roscioli",
                        lat=41.89,
                        lon=12.47,
                        maps_url="https://maps.example/r",
                    ),
                )
            ],
        )


def test_extract_reads_url_file_and_skips_failures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    FakePipeline.calls = []
    urls = tmp_path / "urls.txt"
    urls.write_text(
        "# my trip\nhttps://www.tiktok.com/@a/video/1\nhttps://www.tiktok.com/@a/broken/2\n"
    )
    out = tmp_path / "places.csv"

    result = runner.invoke(cli.app, ["extract", str(urls), "-o", str(out), "--lang", "he"])

    assert result.exit_code == 0, result.output
    assert len(FakePipeline.calls) == 2
    rows = list(csv.DictReader(out.open()))
    assert [r["Name"] for r in rows] == ["Roscioli"]


def test_extract_fails_when_nothing_worked(monkeypatch) -> None:
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    result = runner.invoke(cli.app, ["extract", "https://www.tiktok.com/@a/broken/1"])
    assert result.exit_code == 1


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0 and result.output.strip()
