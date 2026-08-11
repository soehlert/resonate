FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies required for audio processing and tagging
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    beets \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary into image
COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /app

# Copy project files and install editable package
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install --system --no-cache -e .

COPY . .

ENTRYPOINT ["python", "-m", "resonate.main"]
