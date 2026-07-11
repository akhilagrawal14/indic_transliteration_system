## ----------------------------------------------------------
## Stage 1: build -- install all Python deps in an isolated layer
## ----------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

## ----------------------------------------------------------
## Stage 2: runtime -- lean image with only what we need
## ----------------------------------------------------------
FROM python:3.10-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY server/ /app/server/
COPY requirements/requirements.txt /app/

# Create directories for mounted volumes
RUN mkdir -p /app/models /app/server/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-4}"]