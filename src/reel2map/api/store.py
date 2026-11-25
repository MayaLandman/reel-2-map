"""Saved places ("my lists") for the web app, kept in a small JSON file.

A single-user local app does not need a database: writes are rare, the file stays small,
and it is easy to inspect or back up. Writes go to a temp file first and are then
renamed over the old one, so a crash never leaves a half-written file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field


class SavedPlace(BaseModel):
    id: str
    name: str
    category: str = "other"
    city: str | None = None
    country: str | None = None
    country_code: str | None = None
    city_local: str | None = None
    address: str | None = None
    why: str = ""
    maps_url: str
    lat: float | None = None
    lon: float | None = None
    sources: list[str] = Field(default_factory=list)
    in_google: bool = False
    saved_at: float = Field(default_factory=time.time)


class NewPlace(BaseModel):
    name: str
    category: str = "other"
    city: str | None = None
    country: str | None = None
    country_code: str | None = None
    city_local: str | None = None
    address: str | None = None
    why: str = ""
    maps_url: str
    lat: float | None = None
    lon: float | None = None
    source: str | None = None


def place_id(name: str, city: str | None) -> str:
    def norm(text: str | None) -> str:
        text = unicodedata.normalize("NFKC", text or "").casefold()
        return re.sub(r"\W+", "", text)

    return hashlib.sha1(f"{norm(name)}|{norm(city)}".encode()).hexdigest()[:12]


class PlaceStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, SavedPlace]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return {p["id"]: SavedPlace.model_validate(p) for p in data}

    def _write(self, places: dict[str, SavedPlace]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            [p.model_dump() for p in places.values()], ensure_ascii=False, indent=1
        )
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, self.path)

    def all(self) -> list[SavedPlace]:
        with self._lock:
            return sorted(self._read().values(), key=lambda p: p.saved_at, reverse=True)

    def add(self, new: list[NewPlace]) -> list[SavedPlace]:
        with self._lock:
            places = self._read()
            added = []
            for item in new:
                pid = place_id(item.name, item.city)
                current = places.get(pid)
                if current:
                    if item.source and item.source not in current.sources:
                        current.sources.append(item.source)
                    current.why = current.why or item.why
                    added.append(current)
                    continue
                data = item.model_dump(exclude={"source"})
                saved = SavedPlace(id=pid, sources=[item.source] if item.source else [], **data)
                places[pid] = saved
                added.append(saved)
            self._write(places)
            return added

    def update(self, pid: str, **fields) -> SavedPlace | None:
        with self._lock:
            places = self._read()
            if pid not in places:
                return None
            places[pid] = places[pid].model_copy(update=fields)
            self._write(places)
            return places[pid]

    def delete(self, pid: str) -> bool:
        with self._lock:
            places = self._read()
            if places.pop(pid, None) is None:
                return False
            self._write(places)
            return True
