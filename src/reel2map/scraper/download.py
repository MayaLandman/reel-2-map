"""Download a post (video or photo carousel) plus its metadata with yt-dlp.

yt-dlp does the heavy lifting: it knows how to talk to TikTok, Instagram, YouTube Shorts
and ~1800 other sites. This module wraps it so the rest of the pipeline gets a typed
`Post` with local file paths, caption, hashtags, location tag and (optionally) comments.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from reel2map.config import Settings
from reel2map.models import Platform, Post

log = logging.getLogger(__name__)

HASHTAG_RE = re.compile(r"#([\w֐-׿]+)", re.UNICODE)
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class DownloadError(RuntimeError):
    """Raised when a post cannot be downloaded."""


def detect_platform(url: str) -> Platform:
    host = (urlparse(url).hostname or "").lower()
    if "tiktok" in host:
        return Platform.TIKTOK
    if "instagram" in host:
        return Platform.INSTAGRAM
    if "youtube" in host or host == "youtu.be":
        return Platform.YOUTUBE
    return Platform.OTHER


def parse_hashtags(*texts: str | None) -> list[str]:
    seen: dict[str, None] = {}
    for text in texts:
        for tag in HASHTAG_RE.findall(text or ""):
            seen.setdefault(tag.lower(), None)
    return list(seen)


def _ydl_options(settings: Settings, workdir: Path, with_comments: bool) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "outtmpl": str(workdir / "%(id)s_%(autonumber)02d.%(ext)s"),
        # 720p is plenty for OCR and keeps downloads small.
        "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "match_filter": _duration_filter(settings.max_video_seconds),
        "getcomments": with_comments,
        "extractor_args": {
            "tiktok": {"max_comments": [str(settings.max_comments)]},
            "youtube": {"max_comments": [str(settings.max_comments), "0", "0"]},
        },
    }
    if settings.cookies_file:
        opts["cookiefile"] = str(settings.cookies_file)
    if settings.cookies_from_browser:
        opts["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    return opts


def _duration_filter(max_seconds: int):
    def _filter(info: dict[str, Any], *, incomplete: bool = False) -> str | None:
        duration = info.get("duration")
        if duration and duration > max_seconds:
            return f"video is {duration:.0f}s, longer than the {max_seconds}s limit"
        return None

    return _filter


def _collect_files(info: dict[str, Any]) -> list[Path]:
    """All files yt-dlp actually wrote, for a single video or every carousel entry."""
    entries = info.get("entries") or [info]
    files: list[Path] = []
    for entry in entries:
        if not entry:
            continue
        for dl in entry.get("requested_downloads") or []:
            path = dl.get("filepath") or dl.get("_filename")
            if path and Path(path).exists():
                files.append(Path(path))
    return files


def _fetch_images(urls: list[str], workdir: Path, post_id: str) -> list[Path]:
    """Carousel photos are often exposed only as thumbnails; fetch them directly."""
    paths: list[Path] = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for i, url in enumerate(urls, start=1):
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:  # pragma: no cover - network
                log.warning("could not fetch image %s: %s", url, exc)
                continue
            path = workdir / f"{post_id}_img{i:02d}.jpg"
            path.write_bytes(resp.content)
            paths.append(path)
    return paths


def _comments(info: dict[str, Any], limit: int) -> list[str]:
    comments = info.get("comments") or []
    # Pinned and most-liked comments are where people ask "where is this?" and get answers.
    comments = sorted(comments, key=lambda c: (not c.get("is_pinned"), -(c.get("like_count") or 0)))
    return [c["text"].strip() for c in comments[:limit] if c.get("text")]


def _alternate_urls(url: str) -> list[str]:
    """TikTok photo posts are served by the regular video endpoint as well."""
    if "tiktok.com" in url and "/photo/" in url:
        return [url.replace("/photo/", "/video/")]
    return []


def download_post(
    url: str, workdir: Path, settings: Settings, *, with_comments: bool = False
) -> Post:
    import yt_dlp  # imported lazily so the rest of the package imports fast

    workdir.mkdir(parents=True, exist_ok=True)
    opts = _ydl_options(settings, workdir, with_comments)

    info: dict[str, Any] | None = None
    errors: list[str] = []
    for candidate in [url, *_alternate_urls(url)]:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                raw = ydl.extract_info(candidate, download=True)
                info = ydl.sanitize_info(raw) if raw else None
            if info:
                break
        except yt_dlp.utils.DownloadError as exc:
            errors.append(str(exc))
    if not info:
        hint = ""
        if detect_platform(url) is Platform.INSTAGRAM:
            hint = " Instagram usually needs cookies: set REEL2MAP_COOKIES_FROM_BROWSER=chrome."
        raise DownloadError(f"could not download {url}: {'; '.join(errors)}{hint}")

    files = _collect_files(info)
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]

    entries = [e for e in (info.get("entries") or []) if e]
    if not videos and not images:
        # Photo carousel: no video stream, but every entry has an image URL.
        thumbs = [e.get("url") or e.get("thumbnail") for e in entries] or [info.get("thumbnail")]
        images = _fetch_images([t for t in thumbs if t], workdir, str(info.get("id", "post")))

    description = info.get("description") or ""
    title = info.get("title")
    location = info.get("location") or (entries[0].get("location") if entries else None)

    return Post(
        url=url,
        platform=detect_platform(url),
        id=str(info.get("id") or "post"),
        uploader=info.get("uploader") or info.get("channel"),
        title=title if title and title != description else None,
        description=description,
        hashtags=parse_hashtags(description, " ".join(f"#{t}" for t in info.get("tags") or [])),
        location_tag=location,
        duration=info.get("duration"),
        comments=_comments(info, settings.max_comments) if with_comments else [],
        video_path=videos[0] if videos else None,
        image_paths=images,
    )
