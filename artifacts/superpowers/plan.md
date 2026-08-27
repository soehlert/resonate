### Goal
Fix `genre_matches_count` and `subgenre_matches_count` double-counting in `analyze` summary statistics so metrics accurately reflect the number of tracks enriched.

### Assumptions
- Each processed track should increment `genre_matches_count` at most once (if a primary genre was resolved).
- Each processed track should increment `subgenre_matches_count` at most once (if subgenres were resolved).
- Output metadata behavior and file/Plex writing remain unchanged.

### Plan
1. Fix summary counter increments in CLI analyze flow
   - Files: `src/resonate/main.py`
   - Change:
     - Remove intermediate increments of `genre_matches_count` at lines 883 and 910.
     - Remove intermediate increments of `subgenre_matches_count` at lines 915 and 949.
     - Add singular increments for `genre_matches_count` and `subgenre_matches_count` after all genre/subgenre resolution logic completes for a track.
   - Verify: `uv run pytest tests/test_enrichment_mapping.py`

2. Add regression unit test for analyze count tracking
   - Files: `tests/test_enrichment_mapping.py`
   - Change:
     - Add test verifying that when both text tags and audio waveform predictions provide genres, the track counts reflect 1 match per enriched track.
   - Verify: `uv run pytest` and `uv run ruff check .`

### Risks & mitigations
- Risk: Omitting counts for tracks that only matched via Essentia or only via tags.
- Mitigation: Check `if mapped_genre:` and `if mapped_subgenres:` after all fallback, override, and promotion rules execute.

### Rollback plan
- Revert changes via `git checkout -- src/resonate/main.py tests/test_enrichment_mapping.py`.
