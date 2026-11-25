"""Write results in formats map tools can import.

- CSV: import into Google My Maps (Create map -> Import), columns detected automatically.
- KML: Google Earth, Google My Maps, most GIS tools.
- GeoJSON: web maps (Leaflet, Mapbox, geojson.io).
- JSON: the full result, including transcript and evidence.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from xml.sax.saxutils import escape

from reel2map.models import ExtractionResult, ResolvedPlace

FORMATS = ("csv", "json", "geojson", "kml")


def _rows(results: Sequence[ExtractionResult]) -> list[tuple[ResolvedPlace, str]]:
    return [(p, r.post.url) for r in results for p in r.places]


def to_csv(results: Sequence[ExtractionResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Name",
            "Category",
            "Latitude",
            "Longitude",
            "Address",
            "City",
            "Country",
            "Why",
            "Confidence",
            "Google Maps",
            "Source",
        ]
    )
    for place, source in _rows(results):
        geo = place.geo
        writer.writerow(
            [
                place.name,
                place.category,
                geo.lat if geo else "",
                geo.lon if geo else "",
                (geo.address if geo else None) or place.address_hint or "",
                place.city or "",
                place.country or "",
                place.why,
                f"{place.confidence:.2f}",
                place.maps_url,
                source,
            ]
        )
    return buf.getvalue()


def to_geojson(results: Sequence[ExtractionResult]) -> str:
    features = []
    for place, source in _rows(results):
        if not place.geo:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [place.geo.lon, place.geo.lat]},
                "properties": {
                    "name": place.name,
                    "category": place.category,
                    "why": place.why,
                    "address": place.geo.address,
                    "maps_url": place.maps_url,
                    "source": source,
                    "confidence": place.confidence,
                },
            }
        )
    return json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2
    )


def to_kml(results: Sequence[ExtractionResult], name: str = "reel2map") -> str:
    marks = []
    for place, source in _rows(results):
        if not place.geo:
            continue
        desc = f"{place.why}\n\nSource: {source}\n{place.maps_url}"
        marks.append(
            "    <Placemark>\n"
            f"      <name>{escape(place.name)}</name>\n"
            f"      <description>{escape(desc)}</description>\n"
            f'      <ExtendedData><Data name="category"><value>{escape(place.category)}'
            "</value></Data></ExtendedData>\n"
            f"      <Point><coordinates>{place.geo.lon},{place.geo.lat},0</coordinates></Point>\n"
            "    </Placemark>"
        )
    body = "\n".join(marks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n'
        f"    <name>{escape(name)}</name>\n{body}\n  </Document>\n</kml>\n"
    )


def to_json(results: Sequence[ExtractionResult]) -> str:
    return json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False, indent=2)


def export(results: Sequence[ExtractionResult], fmt: str) -> str:
    match fmt:
        case "csv":
            return to_csv(results)
        case "json":
            return to_json(results)
        case "geojson":
            return to_geojson(results)
        case "kml":
            return to_kml(results)
    raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
