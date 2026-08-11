# Resonate Music Metadata Engine

Resonate is an intelligent music metadata enrichment engine designed to analyze your local music collection, map audio features and community tags to normalized moods and genres, perform BPM tempo detection, write metadata tags via Mutagen directly (or Beets), and sync tags directly to Plex.

## Key Features

- **Primary Genre Classification**: Classifies tracks into standard primary genres (like Rock, Pop, Indie, Jazz, etc.) based on community tags.
- **Sub-Genre & Style Mapping**: Maps granular community sub-genres and styles for detailed library filtering.
- **Mood Mapping**: Normalizes raw tag descriptions into standardized high-level mood categories (like Chill Hang, Energetic, Melancholic, etc.).
- **BPM Audio Analysis**: Estimates exact tempo (BPM) from the local audio file waveform using `librosa`.
- **Direct File Tagging**: Writes standardized metadata tags directly into FLAC, MP3, and M4A/MP4 files using Mutagen.
- **Multi-Source Tag Enrichment**: Combines track, album, and artist tags from Last.fm, MusicBrainz, and Discogs.
- **Plex Integration**: Syncs resolved genres, moods, and BPM values directly back to your Plex library.

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
| `--overwrite-tags` | Force overwrite metadata fields (genre, mood, BPM) on disk or Plex even if they already contain values. (By default, Resonate only writes to empty metadata fields). |
| `--overwrite` | Force re-process tracks that are already marked as completed in Resonate's local SQLite database. (By default, Resonate skips tracks it already successfully processed in previous runs). |
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
