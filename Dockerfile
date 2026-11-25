FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[all]"

# Download the Whisper model at build time so the first request is not slow.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', compute_type='int8')"

RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "reel2map.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
