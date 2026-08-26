# Implementation Plan: Incorporate Lyrics into Mood Detection Pipeline

## Goal
Integrate lyrics fetching (local embedded tags, sidecar files, and keyless LRCLIB API) with semantic mood and valence scoring into Resonate's mood detection pipeline as an auxiliary, weighted signal (e.g., 15–20% weight), letting the existing unified conflict resolution rules (`resolve_mood_conflicts`) naturally mediate between acoustic vibes and lyrical moods without hardcoded booleans.

## Assumptions
- LRCLIB (`https://lrclib.net`) remains free, public, and requires no API key or user authentication.
- Audio files may have embedded lyrics in ID3 (`USLT`), Vorbis/FLAC (`LYRICS`), MP4 (`©lyr`), or sidecar `.lrc` / `.txt` files.
- `sentence-transformers` (`all-MiniLM-L6-v2`) is available in memory for semantic embeddings.
- Track processing should never crash or block if lyrics are unavailable or if external network requests fail.

## Plan

### Step 1: Configuration and Data Models
- **Files**:
  - `src/resonate/config.py`
  - `src/resonate/models.py`
  - `config.yaml`
- **Change**:
  - Add `LyricsConfig` Pydantic model (`enabled: bool = True`, `weight: float = 0.15`, `prefer_embedded: bool = True`, `lrclib_url: str = "https://lrclib.net"`).
  - Add `lyrics: LyricsConfig` field to `Settings` in `src/resonate/config.py`.
  - Add `LyricsAnalysisResult` model to `src/resonate/models.py` (`lyrics_text: str | None`, `source: str`, `valence_score: float`, `mood_scores: dict[str, float]`).
  - Update default `config.yaml` with the `lyrics` configuration block.
- **Verify**:
  - `uv run pytest tests/test_config.py`

### Step 2: SQLite Lyrics Caching in StateManager
- **Files**:
  - `src/resonate/utils/state.py`
  - `tests/test_modules.py`
- **Change**:
  - In `src/resonate/utils/state.py`, add `CREATE TABLE IF NOT EXISTS track_lyrics` schema with columns (`artist`, `title`, `lyrics_text`, `source`, `fetched_at`).
  - Add methods `get_cached_lyrics(artist: str, title: str) -> dict | None` and `save_cached_lyrics(artist: str, title: str, lyrics_text: str, source: str) -> None`.
  - Add unit tests for SQLite lyrics caching in `tests/test_modules.py`.
- **Verify**:
  - `uv run pytest tests/test_modules.py`

### Step 3: Multi-Tier Lyrics Fetcher Module
- **Files**:
  - `src/resonate/modules/lyrics.py`
  - `src/resonate/modules/__init__.py`
  - `tests/test_lyrics.py`
- **Change**:
  - Create `LyricsFetcher` in `src/resonate/modules/lyrics.py`.
  - Implement `extract_embedded_lyrics(file_path: str) -> str | None` reading ID3 `USLT` / `TXXX:LYRICS`, Vorbis `LYRICS` / `UNSYNCEDLYRICS`, MP4 `©lyr`, and checking for sidecar `.lrc` / `.txt` files with matching basenames.
  - Implement `fetch_lrclib_lyrics(artist: str, title: str, album: str | None = None, duration: int | None = None) -> str | None` using `requests.get` with clean timeouts and status code handling.
  - Implement `get_lyrics(...)` coordinating cache check -> embedded extraction -> LRCLIB fetch -> SQLite caching.
  - Create `tests/test_lyrics.py` testing embedded extraction, LRCLIB API client (mocked), and fallback handling.
- **Verify**:
  - `uv run pytest tests/test_lyrics.py`

### Step 4: Semantic Mood & Valence Scorer for Lyrics
- **Files**:
  - `src/resonate/modules/lyrics.py`
  - `tests/test_lyrics.py`
- **Change**:
  - Implement `analyze_lyrics(lyrics_text: str, tag_mapper: TagMapper) -> LyricsAnalysisResult`:
    - Clean and preprocess lyrics (strip timestamp tags `[00:12.34]` from synced lyrics, drop headers).
    - Compute continuous valence score (lexicon-based polarity detection for negative vs positive lyrical sentiment).
    - Extract continuous semantic mood confidence scores via `tag_mapper` embeddings against target lyrical moods (`melancholic`, `dark`, `romantic`, `party`, `energetic`, `calm`, `happy`).
  - Add comprehensive unit tests in `tests/test_lyrics.py` for lyrical mood scoring.
- **Verify**:
  - `uv run pytest tests/test_lyrics.py`

### Step 5: Pipeline Integration and Natural Conflict Resolution
- **Files**:
  - `src/resonate/main.py`
  - `src/resonate/modules/tag_mapper.py`
  - `tests/test_enrichment_mapping.py`
- **Change**:
  - In `src/resonate/main.py`, instantiate `LyricsFetcher` and incorporate Phase 2.7 (Lyrics Analysis):
    - When `do_mood` and `settings.lyrics.enabled`, retrieve lyrics and analyze semantic mood scores.
    - Blend lyrical mood scores with the configured weight (`settings.lyrics.weight`) into the candidate mood pool.
    - Feed the unified mood candidates into the existing `resolve_mood_conflicts()` pipeline (which naturally resolves mutually exclusive mood pairs like `Happy` vs `Dark` based on continuous confidence without ad-hoc booleans).
    - Add rich CLI logging in verbose mode showing lyrics source (`embedded`, `lrclib`, `cached`) and top detected lyrical moods.
- **Verify**:
  - `uv run pytest tests/test_enrichment_mapping.py tests/test_tag_mapper.py`

### Step 6: Setup Wizard and Documentation
- **Files**:
  - `src/resonate/wizard.py`
  - `README.md`
  - `docs/backlog.md`
- **Change**:
  - Update interactive setup wizard in `src/resonate/wizard.py` to optionally configure lyrics enrichment.
  - Update `README.md` with documentation on lyrics configuration and LRCLIB integration.
  - Update `docs/backlog.md` tracking the completed feature.
- **Verify**:
  - `uv run pytest`
  - `uv run ruff check src/ tests/`

## Risks & mitigations
- **Risk**: External LRCLIB API downtime or slow responses during batch runs.
  - **Mitigation**: 5-second connection timeout, fallback to embedded/cache, and graceful continuation without blocking processing.
- **Risk**: False positives from synced lyric timestamps.
  - **Mitigation**: Clean regex parser to strip `[mm:ss.xx]` tags before NLP analysis.

## Rollback plan
- If lyrics enrichment causes issues, set `lyrics.enabled: false` in `config.yaml` or use CLI flag to disable lyrics analysis.
- Revert commits on the feature branch via `git revert` or switch branch.
