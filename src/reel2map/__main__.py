"""Allow `python -m reel2map ...`, which also works when the venv path has spaces."""

from reel2map.cli import app

app(prog_name="reel2map")
