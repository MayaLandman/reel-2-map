"""Runtime configuration, read from environment variables or a local `.env` file."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REEL2MAP_", env_file=".env", extra="ignore")

    # --- API keys (read without the prefix too, so standard env names work) ---
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "REEL2MAP_ANTHROPIC_API_KEY"),
    )
    google_maps_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_MAPS_API_KEY", "REEL2MAP_GOOGLE_MAPS_API_KEY"),
    )

    # --- LLM ---
    model: str = "claude-sonnet-5-5"
    max_images: int = Field(default=20, description="Frames sent to the LLM as images.")
    output_language: str = Field(default="en", description="Language for the 'why' field.")

    # --- Media processing ---
    whisper_model: str = "small"
    scene_threshold: float = 0.30
    frame_interval: float = Field(default=2.0, description="Max seconds between sampled frames.")
    max_frames: int = Field(default=60, description="Frames kept for OCR after dedupe.")
    max_video_seconds: int = 900

    # --- Downloading ---
    cookies_file: Path | None = None
    cookies_from_browser: str | None = None
    max_comments: int = 30

    # --- Geocoding ---
    user_agent: str = "reel2map/0.1 (https://github.com/MayaLandman/reel-2-map)"
    nominatim_url: str = "https://nominatim.openstreetmap.org"

    # --- Misc ---
    cache_dir: Path = Path.home() / ".cache" / "reel2map"


def get_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
