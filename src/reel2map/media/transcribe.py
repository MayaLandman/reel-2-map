"""Speech-to-text with faster-whisper (CTranslate2 port of OpenAI Whisper).

Optional: install with `pip install reel2map[transcribe]`. Without it the pipeline
still works from captions, on-screen text and the frames themselves.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from reel2map.models import Transcript, TranscriptSegment

log = logging.getLogger(__name__)


class TranscriptionUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_model(name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise TranscriptionUnavailable(
            "faster-whisper is not installed; run: pip install 'reel2map[transcribe]'"
        ) from exc
    # int8 on CPU is fast enough for 1-3 minute clips and needs no GPU.
    return WhisperModel(name, device="auto", compute_type="int8")


def transcribe(audio: Path, model_name: str = "small") -> Transcript:
    model = _load_model(model_name)
    # VAD skips the music-only parts that make Whisper hallucinate lyrics.
    segments, info = model.transcribe(str(audio), vad_filter=True, beam_size=5)
    result = Transcript(
        language=info.language,
        segments=[
            TranscriptSegment(start=s.start, end=s.end, text=s.text)
            for s in segments
            if s.text.strip() and s.no_speech_prob < 0.8
        ],
    )
    log.info("transcribed %d segments (%s)", len(result.segments), result.language)
    return result
