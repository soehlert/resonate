# Resonate Music Metadata Engine

Resonate is an intelligent music metadata enrichment engine designed to analyze your local music collection, map audio features and community tags to normalized moods and genres, perform BPM tempo detection, write metadata tags via Mutagen directly (or Beets), and sync tags directly to Plex.

All processing runs containerized in a local Docker environment, with seamless command execution provided by an included `./resonate` wrapper script.

## Key Features

- **Primary Genre Classification**: Classifies tracks into standard primary genres (like Rock, Pop, Indie, Jazz, etc.) based on community tags.
- **Sub-Genre & Style Mapping**: Maps granular community sub-genres and styles for detailed library filtering.
- **Mood Mapping**: Normalizes raw tag descriptions into standardized high-level mood categories (like Chill Hang, Energetic, Melancholic, etc.).
- **Lyrics Sentiment & Mood Analysis**: Retrieves synced/unsynced lyrics (from local tags or keyless LRCLIB API), analyzes semantic mood themes and sentiment valence, and guards against inappropriate mood assignments (e.g., preventing dark lyrics from getting tagged Happy).
- **BPM Audio Analysis**: Estimates exact tempo (BPM) offline directly from the local audio file waveform using `librosa`.
- **Direct File Tagging**: Writes standardized metadata tags directly into FLAC, MP3, and M4A/MP4 files using Mutagen.
- **Multi-Source Tag Enrichment**: Combines track, album, and artist tags from Last.fm, MusicBrainz, and Discogs.
- **Plex Integration**: Syncs resolved genres, moods, and BPM values directly back to your Plex library.

---

## Running inside Docker

Resonate is fully containerized. You can run all commands in two ways:
1. **Via the `./resonate` wrapper script (Recommended)**: A convenient shell script that automatically boots the Docker Compose container and runs the command inside it.
2. **Via Docker Compose directly**: Manual execution of the container using `docker compose`.

---

## Quick Start

### 1. Initial Setup
Run the interactive setup wizard to generate `config.yaml` automatically (no manual file creation required):
```bash
./resonate setup
# Or: docker compose run --rm resonate python -m resonate.main setup
```

> [!NOTE]
> You **do not** need to manually create `config.yaml` from scratch. The `./resonate setup` command will prompt you for all required settings, test your connections, and generate the configuration file for you.

### 2. Run Enrichment
Enrich your entire music library (runs Genre, Sub-genre, Mood, and BPM analysis, and updates Plex and files):
```bash
./resonate analyze --write-id3 --sync-plex
# Or: docker compose run --rm resonate python -m resonate.main analyze --write-id3 --sync-plex
```

### 3. Check State Database Status
Check how many tracks have been processed in the local SQLite tracker:
```bash
./resonate status
# Or: docker compose run --rm resonate python -m resonate.main status
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

## Configuration Guide (`config.yaml`)

> [!TIP]
> Use `./resonate setup` to configure these options interactively rather than creating this file by hand.

Below is an annotated explanation of each configuration section generated in `config.yaml`:

```yaml
plex:
  url: "http://localhost:32400"        # The HTTP URL of your Plex Media Server (e.g. http://192.168.1.100:32400)
  token: "YOUR_PLEX_TOKEN"             # Your Plex X-Plex-Token used for authentication
  library_name: "Music"               # The exact name of your music library section in Plex

lastfm:
  api_key: ""                          # Optional: Last.fm API key (falls back to web scraping if left empty)
  api_secret: ""                       # Optional: Last.fm API secret

discogs:
  api_token: ""                        # Optional: Discogs personal access token (skips Discogs lookup if left empty)

musicbrainz:
  enabled: true                        # Enable canonical tag/genre lookups via MusicBrainz (no API key required)

mutagen:
  enabled: true                        # Enable direct ID3 / FLAC / MP4 file tag writing via Mutagen

mapping:
  threshold: 0.45                      # Minimum cosine similarity score (0.0 - 1.0) required to match a tag
  model_name: "all-MiniLM-L6-v2"       # SentenceTransformers embedding model used for vector tag matching
  target_moods:                        # Curated list of target mood categories to match against
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
  batch_size: 100                      # Number of tracks to process per database transaction batch
  dry_run: false                       # If true, performs analysis without saving to DB or writing metadata
  path_map_source: "/data/music"       # SOURCE PATH: The file path prefix as reported by your Plex Server API
  path_map_target: "/music"            # TARGET PATH: The file path prefix where music is mounted inside Resonate's container

lyrics:
  enabled: true                        # Enable lyrics retrieval and sentiment/mood analysis
  weight: 0.15                         # Auxiliary weight (0.0 - 1.0) applied to lyrical mood candidates
  prefer_embedded: true                # Prefer local embedded ID3/FLAC/MP4/sidecar lyrics before querying LRCLIB
  lrclib_url: "https://lrclib.net"     # Keyless LRCLIB public API endpoint
```

### Path Mapping Explained (`path_map_source` vs `path_map_target`)

When Resonate queries Plex for tracks, Plex returns the file path as indexed on the Plex server machine (e.g., `/data/music/Foo Fighters/Breakout.mp3`).

Because Resonate runs in its own Docker container, its local mount point for that same music folder might be different (e.g., `/music/Foo Fighters/Breakout.mp3`).

* **`path_map_source`**: The folder prefix Plex reports to Resonate over the API (e.g., `/data/music` or `/media/synology/music`).
* **`path_map_target`**: The folder prefix where your music is mounted inside Resonate's container (e.g., `/music`).

Resonate swaps `path_map_source` $\rightarrow$ `path_map_target` on incoming Plex paths so it can locate, analyze, and tag the audio files locally. 

*(If both Plex and Resonate see the music at the exact same directory path, you can leave both mapping fields empty: `""`).*

---

## Docker Build Command

To rebuild the local image after modifying the source code:

```bash
docker compose build
```
