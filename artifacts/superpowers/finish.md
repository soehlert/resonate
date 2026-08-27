# Finish Summary: Fix Genre/Sub-Genre Match Count Reporting

## Review Pass
- **Blocker**: None
- **Major**: None
- **Minor**: None
- **Nit**: None

## Summary of Changes
- [`src/resonate/main.py`](file:///Users/soehlert/projects/personal/resonate/src/resonate/main.py):
  - Removed intermediate increments of `genre_matches_count` in tag consensus matching and Essentia fallback/override blocks.
  - Removed intermediate increments of `subgenre_matches_count` in tag and audio blocks.
  - Added per-track single increments for `genre_matches_count` and `subgenre_matches_count` after full genre and subgenre resolution and sanity guards complete.
- [`tests/test_enrichment_mapping.py`](file:///Users/soehlert/projects/personal/resonate/tests/test_enrichment_mapping.py):
  - Added unit regression test `test_genre_and_subgenre_match_accounting` to verify that tracks evaluated by multiple enrichment sources increment counts at most once per track.

## Verification Commands & Results
- `uv run pytest`: 136 passed in 14.81s.
- `uv run ruff check src tests`: All checks passed.
- `uv run ruff format src/resonate/main.py tests/test_enrichment_mapping.py`: Formatted cleanly.

## Follow-ups / Manual Validation
- Running `uv run resonate analyze` in future runs will report `Genres Mapped` $\le$ `Total Processed`.
