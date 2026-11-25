"""Orchestrates the stages: download -> transcribe + frames/OCR -> LLM -> geocode.

Every stage is injected, so tests (and alternative backends) can swap any of them.
Optional stages degrade gracefully: a missing extra or a failed transcription becomes a
warning on the result instead of a crash.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from reel2map.config import Settings
from reel2map.extract.llm import ClaudeExtractor, Extractor
from reel2map.geo.geocoder import Geocoder, make_geocoder
from reel2map.media import ffmpeg
from reel2map.media.frames import extract_frames, frames_from_images
from reel2map.models import (
    ExtractionResult,
    Frame,
    Post,
    ResolvedPlace,
    StageTiming,
    Transcript,
)
from reel2map.scraper.download import download_post

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class Options:
    transcribe: bool = True
    ocr: bool = True
    geocode: bool = True
    comments: bool = False
    min_confidence: float = 0.35


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        downloader: Callable[..., Post] = download_post,
        transcriber: Callable[[Path, str], Transcript] | None = None,
        ocr: Callable[[list[Frame]], list[Frame]] | None = None,
        extractor: Extractor | None = None,
        geocoder: Geocoder | None = None,
    ) -> None:
        self.settings = settings
        self.downloader = downloader
        self._transcriber = transcriber
        self._ocr = ocr
        self._extractor = extractor
        self._geocoder = geocoder

    # Lazy defaults: heavy models are only loaded when a stage actually runs.
    @property
    def transcriber(self) -> Callable[[Path, str], Transcript]:
        if self._transcriber is None:
            from reel2map.media.transcribe import transcribe

            self._transcriber = transcribe
        return self._transcriber

    @property
    def ocr(self) -> Callable[[list[Frame]], list[Frame]]:
        if self._ocr is None:
            from reel2map.media.ocr import ocr_frames

            self._ocr = ocr_frames
        return self._ocr

    @property
    def extractor(self) -> Extractor:
        if self._extractor is None:
            self._extractor = make_extractor(self.settings)
        return self._extractor

    @property
    def geocoder(self) -> Geocoder:
        if self._geocoder is None:
            self._geocoder = make_geocoder(self.settings)
        return self._geocoder

    def run(
        self,
        url: str,
        options: Options | None = None,
        *,
        workdir: Path | None = None,
        progress: Progress | None = None,
    ) -> ExtractionResult:
        options = options or Options()
        say = progress or (lambda msg: log.info(msg))
        timings: list[StageTiming] = []
        warnings: list[str] = []

        @contextmanager
        def stage(name: str) -> Iterator[None]:
            say(name)
            start = time.perf_counter()
            try:
                yield
            finally:
                timings.append(
                    StageTiming(stage=name, seconds=round(time.perf_counter() - start, 2))
                )

        # Fail fast on configuration errors (e.g. a missing API key) before spending a
        # minute on downloading and transcribing.
        extractor = self.extractor

        with _workdir(workdir) as wd:
            with stage("download"):
                post = self.downloader(
                    url, wd / "media", self.settings, with_comments=options.comments
                )

            transcript, frames = self._analyse_media(post, wd, options, stage, warnings)

            with stage("extract"):
                extraction = extractor.extract(post, transcript, frames)

            places = [
                ResolvedPlace(**p.model_dump())
                for p in extraction.places
                if p.confidence >= options.min_confidence
            ]
            if options.geocode and places:
                with stage("geocode"):
                    for place in places:
                        try:
                            place.geo = self.geocoder.locate(place)
                        except Exception as exc:
                            warnings.append(f"geocoding failed for {place.name}: {exc}")

            # Paths point into a temp dir that is about to be deleted.
            if workdir is None:
                post = post.model_copy(update={"video_path": None, "image_paths": []})
                frames = [f.model_copy(update={"path": Path(f.path.name)}) for f in frames]

        return ExtractionResult(
            post=post,
            transcript=transcript,
            frames=frames,
            summary=extraction.summary,
            places=places,
            timings=timings,
            warnings=warnings,
        )

    def _analyse_media(self, post, wd, options, stage, warnings):
        """Speech and pictures are independent, so run them side by side."""

        def speech() -> Transcript | None:
            if not (options.transcribe and post.video_path):
                return None
            try:
                if not ffmpeg.has_audio(post.video_path):
                    return None
                audio = ffmpeg.extract_audio(post.video_path, wd / "audio.wav")
                return self.transcriber(audio, self.settings.whisper_model)
            except Exception as exc:
                warnings.append(f"transcription skipped: {exc}")
                return None

        def pictures() -> list[Frame]:
            s = self.settings
            if post.video_path:
                frames = extract_frames(
                    post.video_path,
                    wd / "frames",
                    scene_threshold=s.scene_threshold,
                    interval=s.frame_interval,
                    max_frames=s.max_frames,
                )
            else:
                frames = frames_from_images(post.image_paths)
            if options.ocr and frames:
                try:
                    frames = self.ocr(frames)
                except Exception as exc:
                    warnings.append(f"OCR skipped: {exc}")
            return frames

        with stage("transcribe + frames + ocr"), ThreadPoolExecutor(max_workers=2) as pool:
            t = pool.submit(speech)
            f = pool.submit(pictures)
            return t.result(), f.result()


def make_extractor(settings: Settings) -> Extractor:
    """Pick the LLM backend: the API when a key is set, else a signed-in Claude Code."""
    from reel2map.extract.claude_code import ClaudeCodeExtractor, claude_cli

    s = settings
    backend = s.backend
    if backend == "auto":
        backend = "api" if s.anthropic_api_key or not claude_cli() else "claude-code"
    if backend == "claude-code":
        return ClaudeCodeExtractor(
            s.claude_code_model, max_images=s.max_images, language=s.output_language
        )
    if backend != "api":
        raise ValueError(f"unknown backend {backend!r}; use api, claude-code or auto")
    if not s.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to .env, or install Claude Code and sign in "
            "with your Claude plan to use --backend claude-code."
        )
    return ClaudeExtractor(
        s.anthropic_api_key, s.model, max_images=s.max_images, language=s.output_language
    )


@contextmanager
def _workdir(path: Path | None) -> Iterator[Path]:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="reel2map_") as tmp:
        yield Path(tmp)
