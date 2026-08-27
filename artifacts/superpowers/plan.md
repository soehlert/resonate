# Implementation Plan: Modular Architecture, Test Hardening & Pluggable Providers (<500 lines/file)

## Goal
Refactor Resonate's monolithic `src/resonate/main.py` (~1,900 lines) into a clean, modular, test-first architecture where **every single file is strictly under 400 lines** and functions adhere to Single Responsibility Principle (SRP). Introduce Pydantic v2 domain models, a pluggable metadata provider layer, centralized album caching, multi-threaded fetching, and a 5.0-second safe MusicBrainz rate limiter to eliminate 429 errors and drastically accelerate full-library enrichment.

## Assumptions
* Work begins on an umbrella feature branch `refactor/modular-providers-and-cli`. Each step executes in its own isolated sub-branch (e.g. `refactor/step-1-test-hardening`) and will **STOP** for explicit user approval before proceeding to the next step.
* We will **NEVER merge into `main`** without explicit user instruction.
* All test commands will strictly use `uv run pytest`.
* All external API tests remain 100% mocked with zero network calls.
* Existing CLI flags (`--write-id3`, `--write-plex`, `--genre`, `--bpm`, etc.) and configuration in `config.yaml` remain 100% backward compatible.
* Wizard setup merges existing YAML configs and never touches the SQLite database.

## Plan

### Step 1: Test Suite Hardening, Error Paths & Safe 5.0s Rate Limiting
* **Branch**: `refactor/step-1-test-hardening` (branched from `refactor/modular-providers-and-cli`)
* **Files**:
  - `src/resonate/modules/external_metadata.py` (Modify: update MusicBrainz rate limit to 5.0s with exponential backoff on 503/429)
  - `tests/test_external_metadata.py` (Modify: add comprehensive error-path tests for 429, 503, JSON decode errors, timeouts)
  - `tests/test_lyrics.py` (Modify: add error paths for LRCLIB 404, 500, network timeouts)
* **Change**:
  - Increase MusicBrainz default request interval from 3.5s to 5.0s to completely eliminate 429/503 rate-limit bans.
  - Implement robust error-path test coverage for all external fetchers before touching core architecture.
* **Verify**:
  `uv run pytest tests/test_external_metadata.py tests/test_lyrics.py`

---

### Step 2: Pydantic v2 Domain Models
* **Branch**: `refactor/step-2-pydantic-models` (branched from `refactor/modular-providers-and-cli`)
* **Files**:
  - `src/resonate/models.py` (Modify: add domain models)
  - `tests/test_models.py` (New, ~100 lines)
* **Detailed Models & Responsibilities**:
  - `ProviderResult`: Standardized data contract returned by any metadata provider (track tags, album tags, artist tags, canonical artist alias, status).
  - `TrackEnrichmentResult`: Structured output contract of the enrichment pipeline for a track (primary genre, subgenres, moods, BPM, lyrics valence, tag write status).
  - `TaxonomyDecision`: Explicit audit model recording why a genre was promoted (e.g. Rock -> Punk) or filtered for clear verbose logging and debugging.
  - `ProviderConfig`: Typed configuration options for external providers.
* **Verify**:
  `uv run pytest tests/test_models.py`

---

### Step 3: Pluggable Provider Architecture & Unified Album Caching
* **Branch**: `refactor/step-3-pluggable-providers` (branched from `refactor/modular-providers-and-cli`)
* **Files & File Responsibilities**:
  - `src/resonate/providers/base.py` (New, ~80 lines):
    - *Purpose*: Defines the abstract `BaseMetadataProvider` interface.
    - *Functions*: `fetch_track_tags()`, `fetch_album_tags()`, `fetch_artist_tags()`.
  - `src/resonate/providers/manager.py` (New, ~200 lines):
    - *Purpose*: Central coordinator managing all active providers.
    - *Functions*: `get_tags_for_track()`, `_query_providers_concurrently()` (ThreadPoolExecutor), `_get_cached_album_tags()`, `_save_album_tags_cache()`.
  - `src/resonate/providers/lastfm.py` (New, ~150 lines):
    - *Purpose*: Isolated Last.fm API and fallback scraper.
    - *Functions*: `fetch_track_tags()`, `fetch_album_tags()`, `fetch_artist_tags()`.
  - `src/resonate/providers/musicbrainz.py` (New, ~180 lines):
    - *Purpose*: MusicBrainz API client with safe 5.0s rate limiting.
    - *Functions*: `fetch_recording_tags()`, `fetch_release_tags()`, `resolve_canonical_artist()`.
  - `src/resonate/providers/discogs.py` (New, ~120 lines):
    - *Purpose*: Discogs release and style lookup client.
    - *Functions*: `fetch_release_tags()`, `fetch_artist_tags()`.
  - `tests/test_providers.py` (New, ~150 lines):
    - *Purpose*: Unit tests for provider implementations, thread-safe concurrent queries, and album cache hits.
* **Verify**:
  `uv run pytest tests/test_providers.py tests/test_external_metadata.py`

---

### Step 4: Taxonomy, Genre Promotion & Mood Rules Engine
* **Branch**: `refactor/step-4-taxonomy-engine` (branched from `refactor/modular-providers-and-cli`)
* **Files**:
  - `src/resonate/engine/__init__.py` (New)
  - `src/resonate/engine/taxonomy.py` (New, ~220 lines)
  - `src/resonate/engine/mood_rules.py` (New, ~200 lines)
  - `tests/test_taxonomy_engine.py` (New, ~150 lines)
* **Change**:
  - Extract genre keyword sets, tag validation predicates (`is_valid_subgenre_tag`, `is_valid_mood_tag`), family hierarchies (e.g. Rock -> Punk / Metal promotions), and cross-genre sanity filters.
  - Extract acoustic mood heuristics, BPM-grounded validation, lyrics valence guards, and conflict resolution.
* **Verify**:
  `uv run pytest tests/test_taxonomy_rules.py tests/test_taxonomy_engine.py`

---

### Step 5: Core Pipeline Engine & Reporting
* **Branch**: `refactor/step-5-pipeline-engine` (branched from `refactor/modular-providers-and-cli`)
* **Files**:
  - `src/resonate/engine/pipeline.py` (New, ~280 lines)
  - `src/resonate/cli/reporting.py` (New, ~120 lines)
  - `tests/test_pipeline.py` (New, ~120 lines)
* **Change**:
  - Implement `EnrichmentPipeline` class orchestrating `ProviderManager`, `TagMapper`, `EssentiaAnalyzer`, `BpmDetector`, `LyricsFetcher`, and tag writers, returning `TrackEnrichmentResult`.
  - Extract Rich transformation tables, progress bars, and batch reporting into `cli/reporting.py`.
* **Verify**:
  `uv run pytest tests/test_pipeline.py tests/test_enrichment_mapping.py`

---

### Step 6: Decompose CLI Commands & Slim Down `main.py`
* **Branch**: `refactor/step-6-cli-decomposition` (branched from `refactor/modular-providers-and-cli`)
* **Files**:
  - `src/resonate/cli/__init__.py` (New)
  - `src/resonate/cli/analyze.py` (New, ~220 lines)
  - `src/resonate/cli/clean.py` (New, ~140 lines)
  - `src/resonate/cli/check.py` (New, ~120 lines)
  - `src/resonate/cli/status.py` (New, ~60 lines)
  - `src/resonate/cli/setup.py` (New, ~80 lines)
  - `src/resonate/main.py` (Modify: reduce from 1,863 lines to ~80 lines)
* **Change**:
  - Move CLI command definitions into dedicated modules under `src/resonate/cli/`.
  - Refactor `src/resonate/main.py` into a lean Typer router and application entrypoint.
* **Verify**:
  `uv run pytest`
  `uv run ruff check src tests`

---

### Follow-up Step (Post-Refactor): Modularize `test_taxonomy_rules.py`
* After the main refactor is approved and stable on the base branch, split `test_taxonomy_rules.py` (1,458 lines) into smaller, logically scoped test files:
  - `tests/test_taxonomy_stemming.py` (~250 lines)
  - `tests/test_taxonomy_promotions.py` (~300 lines)
  - `tests/test_mood_heuristics.py` (~250 lines)
  - `tests/test_taxonomy_edge_cases.py` (~350 lines)

---

## Risks & mitigations
* **Risk: MusicBrainz 429 Rate Limiting**
  - *Mitigation*: Rate limit increased to a safe 5.0 seconds per query, combined with centralized SQLite album-level caching so an entire album is queried only once.
* **Risk: Subgenre / Mood Rule Drift**
  - *Mitigation*: The 1,458-line regression suite in `tests/test_taxonomy_rules.py` will run continuously against `engine/taxonomy.py` to ensure 100% fidelity.
* **Risk: Scope Bleed Across Steps**
  - *Mitigation*: Each step runs in its own sub-branch and pauses for user confirmation before advancing.

## Rollback plan
If any step introduces blockers:
1. Discard the specific step's sub-branch: `git checkout refactor/modular-providers-and-cli && git branch -D refactor/step-X-...`.
2. Existing codebase on `main` is completely untouched.
