"""REST API + a small web page.

Extraction takes 20-90 seconds, so it runs as a background job:
    POST /api/jobs            {"url": "..."}            -> {"id": "...", "status": "queued"}
    GET  /api/jobs/{id}                                  -> status, current stage, result
    GET  /api/jobs/{id}/export?format=csv|kml|geojson|json
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl

from reel2map import __version__
from reel2map.config import get_settings
from reel2map.export import export
from reel2map.models import ExtractionResult
from reel2map.pipeline import Options, Pipeline

STATIC = Path(__file__).parent / "static"
MEDIA_TYPES = {
    "csv": "text/csv",
    "kml": "application/vnd.google-earth.kml+xml",
    "geojson": "application/geo+json",
    "json": "application/json",
}


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobRequest(BaseModel):
    url: HttpUrl
    language: str = "en"
    transcribe: bool = True
    ocr: bool = True
    geocode: bool = True
    comments: bool = False


class Job(BaseModel):
    id: str
    url: str
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: ExtractionResult | None = None


class JobStore:
    """In-memory store. Swap for Redis/Postgres + a real queue to scale out."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update=fields)


def create_app(pipeline_factory=None, max_workers: int = 2) -> FastAPI:
    api = FastAPI(
        title="reel2map",
        version=__version__,
        description="Turn recommendation videos into a map of real places.",
    )
    store = JobStore()
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def default_factory(language: str) -> Pipeline:
        return Pipeline(get_settings(output_language=language))

    factory = pipeline_factory or default_factory

    def work(job_id: str, req: JobRequest) -> None:
        store.update(job_id, status=JobStatus.RUNNING)
        try:
            result = factory(req.language).run(
                str(req.url),
                Options(
                    transcribe=req.transcribe,
                    ocr=req.ocr,
                    geocode=req.geocode,
                    comments=req.comments,
                ),
                progress=lambda stage: store.update(job_id, stage=stage),
            )
            store.update(job_id, status=JobStatus.DONE, stage=None, result=result)
        except Exception as exc:
            store.update(job_id, status=JobStatus.FAILED, error=str(exc))

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @api.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @api.post("/api/jobs", status_code=202)
    def create_job(req: JobRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], url=str(req.url))
        store.add(job)
        executor.submit(work, job.id, req)
        return job

    @api.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @api.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str, format: Literal["csv", "kml", "geojson", "json"] = Query("csv")):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if not job.result:
            raise HTTPException(409, f"job is {job.status}")
        return PlainTextResponse(
            export([job.result], format),
            media_type=MEDIA_TYPES[format],
            headers={"Content-Disposition": f'attachment; filename="places-{job_id}.{format}"'},
        )

    return api


app = create_app()
