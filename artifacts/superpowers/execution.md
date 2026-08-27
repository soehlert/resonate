## Step 1: Test Suite Hardening, Error Paths & Safe 5.0s Rate Limiting
- **Branch**: `refactor/step-1-test-hardening`
- **Files changed**:
  - `src/resonate/modules/external_metadata.py`
  - `tests/test_external_metadata.py`
  - `tests/test_lyrics.py`
- **What changed**:
  - Configured safe 5.0-second rate limiting on MusicBrainz API calls with exponential backoff on 429/503 responses.
  - Added unit tests for MusicBrainz 429 recovery, 503 max-retry limits, and corrupt/malformed JSON handling.
  - Added unit tests for Discogs 429 rate limit and timeout exceptions.
  - Added unit tests for LRCLIB 404, 500 server errors, and network timeouts.
- **Verification**: `uv run pytest tests/test_external_metadata.py tests/test_lyrics.py`, `uv run pytest`, `uv run ruff check src tests`
- **Result**: PASS (145 passed in 18.07s; All ruff checks passed)


