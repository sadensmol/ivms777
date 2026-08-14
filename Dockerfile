FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libheif1 libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev --frozen --no-install-project || uv sync --no-dev --no-install-project

# Flat layout: each top-level package lives at the repo root.
COPY config.py ./
COPY albums ./albums
COPY db ./db
COPY embedding ./embedding
COPY inference ./inference
COPY ingest ./ingest
COPY search ./search
COPY storage ./storage
COPY web ./web
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 IVMS777_DATA_DIR=/data
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "web.app:app_factory", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
