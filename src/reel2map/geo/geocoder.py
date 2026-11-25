"""Find each extracted place on the map.

Two interchangeable backends behind one small interface:
- GooglePlacesGeocoder: Places API (New) Text Search. Best for businesses; needs a key.
- NominatimGeocoder: OpenStreetMap. Free, no key, good for landmarks and well-known venues.
  Respects the public instance's usage policy (1 request/second, identifying User-Agent).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol
from urllib.parse import quote_plus

import httpx

from reel2map.config import Settings
from reel2map.models import GeoMatch, Place

log = logging.getLogger(__name__)


class Geocoder(Protocol):
    def locate(self, place: Place) -> GeoMatch | None: ...


def _queries(place: Place) -> list[str]:
    """Most specific first, then progressively looser."""
    parts = [
        [place.name, place.address_hint, place.city, place.country],
        [place.name, place.city, place.country],
        [place.name, place.country],
    ]
    seen: list[str] = []
    for p in parts:
        q = ", ".join(x for x in p if x)
        if q and q not in seen:
            seen.append(q)
    return seen


class NominatimGeocoder:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.base = settings.nominatim_url.rstrip("/")
        self.client = client or httpx.Client(
            timeout=20, headers={"User-Agent": settings.user_agent}
        )
        self._lock = threading.Lock()
        self._last = 0.0

    def _throttle(self) -> None:
        with self._lock:
            wait = 1.0 - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def locate(self, place: Place) -> GeoMatch | None:
        for query in _queries(place):
            self._throttle()
            resp = self.client.get(
                f"{self.base}/search",
                params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 0},
            )
            resp.raise_for_status()
            hits = resp.json()
            if hits:
                hit = hits[0]
                lat, lon = float(hit["lat"]), float(hit["lon"])
                return GeoMatch(
                    provider="nominatim",
                    name=hit.get("name") or place.name,
                    address=hit.get("display_name"),
                    lat=lat,
                    lon=lon,
                    maps_url=(
                        "https://www.google.com/maps/search/?api=1&query="
                        + quote_plus(f"{place.name}, {hit.get('display_name', '')}")
                    ),
                    place_id=f"osm:{hit.get('osm_type')}/{hit.get('osm_id')}",
                )
        return None


class GooglePlacesGeocoder:
    URL = "https://places.googleapis.com/v1/places:searchText"
    FIELDS = ",".join(
        [
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
            "places.googleMapsUri",
            "places.rating",
        ]
    )

    def __init__(self, api_key: str, language: str = "en", client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=20)
        self.headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": self.FIELDS}
        self.language = language

    def locate(self, place: Place) -> GeoMatch | None:
        for query in _queries(place):
            resp = self.client.post(
                self.URL,
                headers=self.headers,
                json={"textQuery": query, "languageCode": self.language, "pageSize": 1},
            )
            resp.raise_for_status()
            hits = resp.json().get("places") or []
            if hits:
                hit = hits[0]
                loc = hit.get("location") or {}
                return GeoMatch(
                    provider="google",
                    name=(hit.get("displayName") or {}).get("text") or place.name,
                    address=hit.get("formattedAddress"),
                    lat=loc.get("latitude", 0.0),
                    lon=loc.get("longitude", 0.0),
                    maps_url=hit.get("googleMapsUri")
                    or f"https://www.google.com/maps/place/?q=place_id:{hit['id']}",
                    place_id=hit.get("id"),
                    rating=hit.get("rating"),
                )
        return None


def make_geocoder(settings: Settings) -> Geocoder:
    if settings.google_maps_api_key:
        return GooglePlacesGeocoder(settings.google_maps_api_key, settings.output_language)
    return NominatimGeocoder(settings)
