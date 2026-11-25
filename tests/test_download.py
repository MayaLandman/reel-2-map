from __future__ import annotations

from pathlib import Path

import pytest

from reel2map.models import Platform
from reel2map.scraper import download
from reel2map.scraper.download import DownloadError, detect_platform, parse_hashtags


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        ("https://www.tiktok.com/@a/video/1", Platform.TIKTOK),
        ("https://vm.tiktok.com/ZMabc/", Platform.TIKTOK),
        ("https://www.instagram.com/reel/Cxyz/", Platform.INSTAGRAM),
        ("https://youtu.be/abc", Platform.YOUTUBE),
        ("https://www.youtube.com/shorts/abc", Platform.YOUTUBE),
        ("https://vimeo.com/1", Platform.OTHER),
    ],
)
def test_detect_platform(url: str, platform: Platform) -> None:
    assert detect_platform(url) is platform


def test_parse_hashtags_dedupes_and_keeps_unicode() -> None:
    assert parse_hashtags("Best #Rome eats #pizza #rome", "#רומא") == ["rome", "pizza", "רומא"]


def test_tiktok_photo_alternate_url() -> None:
    assert download._alternate_urls("https://www.tiktok.com/@a/photo/9") == [
        "https://www.tiktok.com/@a/video/9"
    ]
    assert download._alternate_urls("https://www.tiktok.com/@a/video/9") == []


def test_comments_sorted_pinned_then_likes() -> None:
    info = {
        "comments": [
            {"text": "nice", "like_count": 5},
            {"text": "where is #3?", "like_count": 50},
            {"text": "it's Roscioli!", "is_pinned": True, "like_count": 1},
        ]
    }
    assert download._comments(info, 2) == ["it's Roscioli!", "where is #3?"]


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: writes a file and returns an info dict."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        out = Path(self.opts["outtmpl"].replace("%(id)s_%(autonumber)02d.%(ext)s", "v1_01.mp4"))
        out.write_bytes(b"\x00")
        return {
            "id": "v1",
            "uploader": "chef",
            "title": "Rome guide",
            "description": "Top 3 in Rome #rome #food",
            "tags": ["travel"],
            "duration": 42.0,
            "location": "Rome, Italy",
            "requested_downloads": [{"filepath": str(out)}],
        }

    def sanitize_info(self, info):
        return info


def test_download_post_maps_metadata(monkeypatch, tmp_path: Path, settings) -> None:
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    post = download.download_post("https://www.tiktok.com/@chef/video/1", tmp_path, settings)
    assert post.id == "v1"
    assert post.platform is Platform.TIKTOK
    assert post.hashtags == ["rome", "food", "travel"]
    assert post.location_tag == "Rome, Italy"
    assert post.video_path and post.video_path.name == "v1_01.mp4"
    assert not post.is_slideshow


def test_download_post_reports_instagram_hint(monkeypatch, tmp_path: Path, settings) -> None:
    import yt_dlp

    class Failing(FakeYDL):
        def extract_info(self, url, download=True):
            raise yt_dlp.utils.DownloadError("login required")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", Failing)
    with pytest.raises(DownloadError, match="cookies"):
        download.download_post("https://www.instagram.com/reel/x/", tmp_path, settings)


def test_resolve_short_url_follows_redirect() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "vt.tiktok.com":
            return httpx.Response(
                301, headers={"Location": "https://www.tiktok.com/@chef/photo/99?_r=1&u=x"}
            )
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    url = download.resolve_short_url("https://vt.tiktok.com/ZSabc/", client)
    assert url == "https://www.tiktok.com/@chef/photo/99"
    assert download._alternate_urls(url) == ["https://www.tiktok.com/@chef/video/99"]
    assert download.resolve_short_url("https://www.tiktok.com/@a/video/1") == (
        "https://www.tiktok.com/@a/video/1"
    )


def test_error_message_is_clean_and_has_hint(monkeypatch, tmp_path: Path, settings) -> None:
    import yt_dlp

    class Forbidden(FakeYDL):
        def extract_info(self, url, download=True):
            raise yt_dlp.utils.DownloadError(
                "\x1b[0;31mERROR:\x1b[0m [TikTok] 1: Unable to download webpage: HTTP Error 403"
            )

    monkeypatch.setattr(yt_dlp, "YoutubeDL", Forbidden)
    with pytest.raises(DownloadError) as info:
        download.download_post("https://www.tiktok.com/@a/video/1", tmp_path, settings)
    message = str(info.value)
    assert "\x1b" not in message and "cookies-from-browser" in message
