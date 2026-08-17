# LegalAId backend — one image, run as different services via the command.
#
# The same image serves the gateway and each split-out service because they share one codebase
# and one set of Pydantic contracts. That is deliberate: building separate images per service
# would duplicate ~40 MB of identical dependencies and let the contracts drift apart.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Build tooling is needed for faiss/numpy wheels on slim, then removed to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches independently of source changes.
COPY pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip \
    && pip install \
        "fastapi==0.120.4" \
        "uvicorn[standard]==0.38.0" \
        "pydantic==2.12.4" \
        "pydantic-settings==2.13.1" \
        "sqlalchemy==2.0.44" \
        "httpx==0.28.1" \
        "faiss-cpu==1.13.2" \
        "numpy==2.4.2" \
        "pymupdf==1.26.5" \
        "pypdf==6.7.1" \
        "python-docx==1.2.0" \
        "python-multipart==0.0.20" \
        "pyyaml==6.0.2" \
        "structlog==25.4.0" \
        "aiofiles==25.1.0" \
        "pyarrow==22.0.0" \
    && apt-get purge -y --auto-remove build-essential

COPY src /app/src

# Writable volumes for the database, uploads and FAISS indexes.
RUN mkdir -p /data/uploads /data/indexes /data/knowledge_sources

# Run as a non-root user: this container accepts file uploads, so it should not be root.
RUN useradd --create-home --uid 10001 legalaid \
    && chown -R legalaid:legalaid /app /data
USER legalaid

EXPOSE 8000

# The image serves several services on different ports with different health paths, so the check
# is defined per service in docker-compose.yml rather than baked in here. A single hardcoded
# URL would mark llm/embedding/retrieval unhealthy even when they are fine.

CMD ["uvicorn", "legalaid.services.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
