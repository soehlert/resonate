# Resonate Music Metadata Engine

Resonate is an intelligent music metadata enrichment engine designed to analyze your local music collection, map audio features and community tags to normalized moods and genres, perform BPM tempo detection, write metadata tags via Mutagen directly (or Beets), and sync tags directly to Plex.

## Key Features

- **Automated Genre Classification**: Maps raw tags to 14 standard primary genres: `Rock`, `Pop`, `Indie`, `Hip-Hop`, `Electronic`, `Jazz`, `Blues`, `Country`, `Folk`, `R&B`, `Metal`, `Punk`, `Reggae`, and `Latin`.
- **Sub-Genre & Style Mapping**: Resolves granular sub-genres: `Indie Rock`, `Synthpop`, `Downtempo`, `Lo-Fi`, `Motown`, `Shoegaze`, `Garage Rock`, `Post-Rock`, `Classic Rock`, `Acoustic Rock`, `House`, and `Techno`.
- **9 Standardized Mood Categories**: Normalizes diverse tag descriptions into 9 high-level moods: `Party`, `Chill Hang`, `Energetic`, `Groovy`, `Acoustic`, `Electronic`, `Melancholic`, `Lively`, and `Relaxed`.
- **BPM Audio Analysis**: Estimates exact tempo (BPM) from the local audio file waveform using `librosa`.
- **Direct File Tagging (Mutagen)**: Writes standardized metadata tags directly into FLAC, MP3, and M4A/MP4 files (e.g., `TCON` / `genre` for Genre, `TMOO` / `mood` for Mood, `TBPM` / `tmpo` for BPM).
- **Multi-Source Metadata Enrichment**: Retrieves tags from Last.fm (Track, Album, and Artist top tags), MusicBrainz (fully public API), and Discogs (authenticated database search).
- **Dual Plex Syncing**: Updates track genres, moods, and BPM directly in the Plex Media Server library, locking updated fields.
- **Selective Flag Processing**: Filter execution by feature (`--genre`, `--subgenre`, `--mood`, `--bpm`). If no specific feature flag is set, all are processed by default.
- **Terminal Progress Tracker**: Employs a rich terminal progress bar tracking processed count, percentage completion, processing speed, and real-time remaining time (ETA).

---

## Quick Start

### 1. Initial Setup
Run the setup wizard to generate configuration files and verify connections:
```bash
./resonate setup
```

### 2. Run Enrichment
Enrich your entire music library (runs Genre, Sub-genre, Mood, and BPM analysis, and updates Plex and files):
```bash
./resonate analyze --write-id3 --sync-plex
```

### 3. Check State Database Status
Check how many tracks have been processed in the local SQLite tracker:
```bash
./resonate status
```

---

## CLI Options Reference

The `analyze` command accepts the following options:

| Flag / Option | Description |
|---|---|
| `--genre` | Process and classify primary genres. |
| `--subgenre` | Process and classify sub-genres/styles. |
| `--mood` | Process and map mood categories. |
| `--bpm` | Analyze local audio file to estimate BPM. |
| `--write-id3` | Write resolved tags directly to local files using Mutagen. |
| `--sync-plex` | Sync resolved tags directly to Plex Media Server. |
| `--overwrite-tags` | Overwrite existing tags on local files or Plex (default is to only write if tags are empty). |
| `--overwrite` | Re-process tracks that are already marked as processed in the local SQLite DB. |
| `--artist`, `-a` | Filter tracks to a specific artist name. |
| `--limit`, `-l` | Limit the number of tracks processed in this run. |
| `--verbose`, `-v` | Show detailed track-by-track logs instead of the progress bar. |
| `--config`, `-c` | Specify path to a custom configuration file (default is `config.yaml`). |

### Examples

* **Dry run (All features)**:
  ```bash
  ./resonate analyze --dry-run --artist "Foo Fighters" --verbose
  ```
* **BPM Analysis only (Write to local files)**:
  ```bash
  ./resonate analyze --bpm --write-id3 --limit 50
  ```
* **Genre/Sub-Genre only (Sync to Plex)**:
  ```bash
  ./resonate analyze --genre --subgenre --sync-plex --limit 100
  ```

---

## Configuration

Edit `config.yaml` to specify settings for Plex, Last.fm, Discogs, MusicBrainz, and processing options:

```yaml
plex:
  url: "http://localhost:32400"
  token: "YOUR_PLEX_TOKEN"
  library_name: "Music"

lastfm:
  api_key: "YOUR_LASTFM_KEY"
  api_secret: "YOUR_LASTFM_SECRET"

discogs:
  api_token: "YOUR_DISCOGS_PERSONAL_TOKEN"  # Optional

musicbrainz:
  enabled: true

mutagen:
  enabled: true

mapping:
  threshold: 0.45
  model_name: "all-MiniLM-L6-v2"
  target_moods:
    - Party
    - Chill Hang
    - Energetic
    - Groovy
    - Acoustic
    - Electronic
    - Melancholic
    - Lively
    - Relaxed

processing:
  batch_size: 100
  dry_run: false
  path_map_source: "/data/music"
  path_map_target: "/music"
```

---

## Docker Usage

Build and run using Docker Compose:

```bash
docker compose build
docker compose run --rm resonate python -m resonate.main setup
docker compose run --rm resonate python -m resonate.main analyze --write-id3 --sync-plex
```

Or execute directly with the included `./resonate` CLI wrapper script.
