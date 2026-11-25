"""Typed data passed between pipeline stages."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Platform(StrEnum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    OTHER = "other"


class Post(BaseModel):
    """Everything scraped from the post itself, before any media analysis."""

    url: str
    platform: Platform
    id: str
    uploader: str | None = None
    title: str | None = None
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    location_tag: str | None = None
    duration: float | None = None
    comments: list[str] = Field(default_factory=list)
    video_path: Path | None = None
    image_paths: list[Path] = Field(default_factory=list)

    @property
    def is_slideshow(self) -> bool:
        return self.video_path is None and bool(self.image_paths)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def timestamped(self) -> str:
        return "\n".join(f"[{s.start:6.1f}s] {s.text.strip()}" for s in self.segments)


class Frame(BaseModel):
    path: Path
    timestamp: float | None = None
    ocr_text: str = ""


Evidence = Literal[
    "caption", "hashtag", "location_tag", "speech", "on_screen_text", "visual", "comment"
]


class Place(BaseModel):
    """A place recommended in the post, as understood by the LLM."""

    name: str
    category: str = Field(
        description="e.g. restaurant, cafe, bar, bakery, gelato, attraction, shop"
    )
    city: str | None = None
    country: str | None = None
    country_code: str | None = Field(default=None, description="ISO 3166-1 alpha-2, e.g. IT")
    city_local: str | None = Field(default=None, description="City name in the output language")
    address_hint: str | None = Field(default=None, description="Street, neighborhood or landmark")
    why: str = Field(default="", description="One or two sentences: why it is recommended")
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: float | None = Field(
        default=None, description="Second in the video where it appears"
    )


class GeoMatch(BaseModel):
    provider: Literal["google", "nominatim"]
    name: str
    address: str | None = None
    lat: float
    lon: float
    maps_url: str
    place_id: str | None = None
    rating: float | None = None


class ResolvedPlace(Place):
    geo: GeoMatch | None = None

    @property
    def maps_url(self) -> str:
        from urllib.parse import quote_plus

        if self.geo:
            return self.geo.maps_url
        query = ", ".join(x for x in (self.name, self.address_hint, self.city, self.country) if x)
        return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


class StageTiming(BaseModel):
    stage: str
    seconds: float


class ExtractionResult(BaseModel):
    post: Post
    transcript: Transcript | None = None
    frames: list[Frame] = Field(default_factory=list)
    summary: str = ""
    places: list[ResolvedPlace] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
