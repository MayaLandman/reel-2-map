from __future__ import annotations

import csv
import io
import json

import httpx

from reel2map.export import export
from reel2map.geo.geocoder import GooglePlacesGeocoder, NominatimGeocoder, _queries, make_geocoder
from reel2map.models import ExtractionResult, GeoMatch, ResolvedPlace


def test_queries_go_from_specific_to_loose(place) -> None:
    assert _queries(place) == [
        "Da Enzo al 29, Trastevere, Rome, Italy",
        "Da Enzo al 29, Rome, Italy",
        "Da Enzo al 29, Italy",
    ]


def test_nominatim_falls_back_to_looser_query(settings, place, monkeypatch) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["q"])
        if len(seen) == 1:
            return httpx.Response(200, json=[])
        return httpx.Response(
            200,
            json=[
                {
                    "lat": "41.888",
                    "lon": "12.47",
                    "name": "Da Enzo",
                    "display_name": "Via dei Vascellari 29, Roma",
                    "osm_type": "node",
                    "osm_id": 1,
                }
            ],
        )

    geo = NominatimGeocoder(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(geo, "_throttle", lambda: None)
    match = geo.locate(place)
    assert match and match.provider == "nominatim"
    assert (match.lat, match.lon) == (41.888, 12.47)
    assert len(seen) == 2


def test_google_places_parses_response(place) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Goog-Api-Key"] == "k"
        assert json.loads(request.content)["textQuery"].startswith("Da Enzo")
        return httpx.Response(
            200,
            json={
                "places": [
                    {
                        "id": "ChIJ123",
                        "displayName": {"text": "Da Enzo al 29"},
                        "formattedAddress": "Via dei Vascellari, 29, Roma",
                        "rating": 4.6,
                        "location": {"latitude": 41.888, "longitude": 12.477},
                        "googleMapsUri": "https://maps.google.com/?cid=1",
                    }
                ]
            },
        )

    geo = GooglePlacesGeocoder("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    match = geo.locate(place)
    assert match and match.place_id == "ChIJ123" and match.rating == 4.6


def test_make_geocoder_prefers_google(settings) -> None:
    assert isinstance(make_geocoder(settings), NominatimGeocoder)
    settings.google_maps_api_key = "k"
    assert isinstance(make_geocoder(settings), GooglePlacesGeocoder)


def _result(post, place) -> ExtractionResult:
    located = ResolvedPlace(
        **place.model_dump(),
        geo=GeoMatch(
            provider="google",
            name="Da Enzo al 29",
            address="Via dei Vascellari 29",
            lat=41.888,
            lon=12.477,
            maps_url="https://maps.google.com/?cid=1",
        ),
    )
    unlocated = ResolvedPlace(
        name="Mystery bakery", category="bakery", why="croissant", confidence=0.4
    )
    return ExtractionResult(post=post, places=[located, unlocated])


def test_csv_export(post, place) -> None:
    rows = list(csv.DictReader(io.StringIO(export([_result(post, place)], "csv"))))
    assert rows[0]["Name"] == "Da Enzo al 29" and rows[0]["Latitude"] == "41.888"
    assert rows[1]["Latitude"] == "" and "google.com/maps/search" in rows[1]["Google Maps"]


def test_geojson_and_kml_skip_unlocated(post, place) -> None:
    gj = json.loads(export([_result(post, place)], "geojson"))
    assert len(gj["features"]) == 1
    assert gj["features"][0]["geometry"]["coordinates"] == [12.477, 41.888]
    kml = export([_result(post, place)], "kml")
    assert kml.count("<Placemark>") == 1 and "12.477,41.888,0" in kml


def test_json_roundtrip(post, place) -> None:
    data = json.loads(export([_result(post, place)], "json"))
    assert ExtractionResult.model_validate(data[0]).places[0].geo.lat == 41.888
