# Feature Completion: Lyrics Sentiment & Mood Enrichment Pipeline

## Summary of Changes
- **Multi-Tier Keyless Lyrics Retrieval (`src/resonate/modules/lyrics.py`)**:
  - Implemented local extraction for embedded ID3 (`USLT`, `TXXX:LYRICS`), Vorbis/FLAC (`LYRICS`), MP4 (`©lyr`), and sidecar `.lrc`/`.txt` files.
  - Implemented free, zero-account remote fetching via the public LRCLIB API (`https://lrclib.net/api/get` with fallback to `/api/search`).
  - Added SQLite state caching (`track_lyrics` table in `data/state.sqlite`) so tracks are never re-queried across runs.
- **Valence Polarity & Semantic Mood Analysis (`src/resonate/modules/lyrics.py`)**:
  - Preprocesses lyrics (stripping timestamps `[00:12.34]` and section markers `[Chorus]`).
  - Computes normalized continuous valence score (-1.0 to +1.0).
  - Computes continuous semantic mood confidence scores against target themes using existing `TagMapper` (`all-MiniLM-L6-v2`) embeddings.
- **Pipeline Integration & Guardrails (`src/resonate/main.py`)**:
  - Added Phase 2.7 (Lyrics Analysis) inside the mood detection workflow.
  - Solves the "Pumped Up Kicks" contrast dilemma: dark lyrics automatically disqualify cheerful moods (`Happy`, `Upbeat`) while preserving acoustic genre vibes (`Chill Hang`, `Energetic`).
- **Configuration & CLI**:
  - Added `LyricsConfig` (`enabled`, `weight`, `prefer_embedded`, `lrclib_url`) in `src/resonate/config.py` and `config.yaml`.
  - Updated interactive setup wizard `src/resonate/wizard.py` and documentation in `README.md` and `docs/backlog.md`.

## Integration Test Results
- **Test Suite**: `uv run pytest` -> 100/100 passed in 6.38s.
- **Linter**: `uv run ruff check src/ tests/` -> Clean (0 errors).

## Changed & Created Files
- `src/resonate/modules/lyrics.py` (NEW)
- `tests/test_lyrics.py` (NEW)
- `docs/backlog.md` (NEW)
- `src/resonate/config.py` (MODIFIED)
- `src/resonate/models.py` (MODIFIED)
- `src/resonate/main.py` (MODIFIED)
- `src/resonate/wizard.py` (MODIFIED)
- `src/resonate/modules/__init__.py` (MODIFIED)
- `src/resonate/utils/state.py` (MODIFIED)
- `config.yaml` (MODIFIED)
- `README.md` (MODIFIED)
- `tests/test_config.py` (MODIFIED)
- `tests/test_modules.py` (MODIFIED)
