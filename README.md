# Resonate Music Metadata Engine

Resonate is an intelligent music metadata enrichment engine designed to analyze your local music collection, map audio features and Last.fm tags to normalized moods, write metadata tags via Beets, and sync mood playlists directly into Plex.

## Key Features

- **13 Standardized Mood Categories**: Normalizes diverse tag descriptions into 13 high-level moods: `chill`, `energetic`, `upbeat`, `melancholic`, `dark`, `aggressive`, `happy`, `groovy`, `romantic`, `nostalgic`, `trippy`, `soulful`, and `moody`.
- **Zero-Key Last.fm Web Scraping Fallback**: Automatically falls back to scraping Last.fm track tags if an API key is not provided in `config.yaml`.
- **Essentia TensorFlow Audio Analysis**: Extracts acoustic attributes and Discogs EffNet mood predictions directly from audio files.
- **Beets Integration**: Writes clean metadata tags back into audio files using Beets.
- **Plex Sync**: Pushes analyzed mood metadata and creates dynamic mood playlists in Plex.
- **Docker Support**: Containerized runtime for reproducible execution across environments.

## Quick Start

### 1. Initial Setup
Run the setup wizard to generate configuration files and verify connections:
```bash
./resonate setup
```

### 2. Analyze Library
Run analysis on your music library:
```bash
./resonate analyze
```

## Configuration

Edit `config.yaml` to specify settings for Plex, Last.fm, Essentia, Beets, and mapping options:

```yaml
moods:
  - chill
  - energetic
  - upbeat
  - melancholic
  - dark
  - aggressive
  - happy
  - groovy
  - romantic
  - nostalgic
  - trippy
  - soulful
  - moody

plex:
  url: "http://localhost:32400"
  token: "YOUR_PLEX_TOKEN"
  library_name: "Music"

lastfm:
  api_key: ""
  api_secret: ""

mapping:
  threshold: 0.45
  model_name: "all-MiniLM-L6-v2"
```

## Docker Usage

Build and run using Docker Compose:

```bash
docker compose build
docker compose run --rm resonate python -m resonate.main setup
docker compose run --rm resonate python -m resonate.main analyze
```

Or execute directly with the included `./resonate` CLI wrapper script.
