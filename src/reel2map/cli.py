"""Command-line interface.

reel2map extract https://www.tiktok.com/@user/video/123 -o rome.csv
reel2map extract urls.txt --format kml -o trip.kml --lang he
reel2map serve
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from reel2map import __version__
from reel2map.config import get_settings
from reel2map.export import FORMATS, export
from reel2map.models import ExtractionResult
from reel2map.pipeline import Options, Pipeline

app = typer.Typer(add_completion=False, help="Turn recommendation videos into a map of places.")
console = Console(stderr=True)


def _read_urls(sources: list[str]) -> list[str]:
    urls: list[str] = []
    for src in sources:
        path = Path(src)
        if not src.startswith("http") and path.is_file():
            urls += [
                ln.strip()
                for ln in path.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
        else:
            urls.append(src)
    return urls


def _print_table(result: ExtractionResult) -> None:
    table = Table(title=result.summary or result.post.url, show_lines=False)
    for col in ("#", "Place", "Category", "City", "Conf.", "Found on map", "Why"):
        table.add_column(col, overflow="fold")
    for i, p in enumerate(result.places, 1):
        table.add_row(
            str(i),
            p.name,
            p.category,
            p.city or "",
            f"{p.confidence:.2f}",
            (p.geo.address or "yes")[:40] if p.geo else "[yellow]no[/]",
            p.why[:80],
        )
    console.print(table)
    for w in result.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    console.print(
        "[dim]" + "  ".join(f"{t.stage}: {t.seconds:.1f}s" for t in result.timings) + "[/]"
    )


@app.command()
def extract(
    sources: Annotated[
        list[str], typer.Argument(help="Post URLs, or text files with one URL per line.")
    ],
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Write results here.")
    ] = None,
    fmt: Annotated[str, typer.Option("-f", "--format", help=f"One of: {', '.join(FORMATS)}")] = "",
    lang: Annotated[str, typer.Option(help="Language for the 'why' text (en, he, ...).")] = "en",
    transcribe: Annotated[bool, typer.Option(help="Transcribe speech with Whisper.")] = True,
    ocr: Annotated[bool, typer.Option(help="Read on-screen text.")] = True,
    geocode: Annotated[bool, typer.Option(help="Look each place up on the map.")] = True,
    comments: Annotated[bool, typer.Option(help="Also scrape top comments.")] = False,
    cookies_from_browser: Annotated[
        str | None, typer.Option(help="Browser to take cookies from (needed for Instagram).")
    ] = None,
    keep_media: Annotated[
        Path | None, typer.Option(help="Keep downloaded video and frames in this folder.")
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Extract the places recommended in one or more posts."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    overrides: dict = {"output_language": lang}
    if cookies_from_browser:
        overrides["cookies_from_browser"] = cookies_from_browser
    pipeline = Pipeline(get_settings(**overrides))
    options = Options(transcribe=transcribe, ocr=ocr, geocode=geocode, comments=comments)

    results: list[ExtractionResult] = []
    for url in _read_urls(sources):
        workdir = keep_media / url.rstrip("/").split("/")[-1] if keep_media else None
        with console.status(f"[bold]{url}") as status:
            try:
                result = pipeline.run(
                    url,
                    options,
                    workdir=workdir,
                    progress=lambda s, u=url: status.update(f"[bold]{u}[/] · {s}"),
                )
            except Exception as exc:
                console.print(f"[red]failed:[/] {url}: {exc}")
                continue
        _print_table(result)
        results.append(result)

    if not results:
        raise typer.Exit(1)
    fmt = fmt or (output.suffix.lstrip(".") if output else "json")
    text = export(results, fmt)
    if output:
        output.write_text(text, encoding="utf-8")
        total = sum(len(r.places) for r in results)
        console.print(f"[green]saved {total} places to {output}[/]")
    else:
        typer.echo(text)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Run the web app and REST API."""
    import uvicorn

    uvicorn.run("reel2map.api.app:app", host=host, port=port, reload=reload)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
