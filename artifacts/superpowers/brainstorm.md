# Brainstorm: Redacted (RED) Integration & Plugin Architecture for Resonate

## Goal
Explore the feasibility, design, and architecture for integrating Redacted (`redacted.ch` / Gazelle API) as a metadata provider and/or extensible plugin in Resonate to leverage its curated community music tags, release groups, and edition metadata.

## Constraints
* **Gazelle API Rate Limits**: Redacted strictly enforces rate limits (max 5 requests per 10 seconds / 2.0s delay per request) with penalties (API key revocation / temporary IP bans) for abuse.
* **Authentication**: Requires a valid Redacted user API token (`Authorization: <api_key>`).
* **Tag Granularity**: Redacted organizes tags at the Artist and Torrent Group (Release/Album) levels; individual track-level tags are rarely defined independently.
* **Architecture Consistency**: Any integration should seamlessly cooperate with existing metadata providers (Last.fm, MusicBrainz, Discogs, LRCLIB) and feed into Resonate's `TagMapper` semantic embedding pipeline.
* **Offline / Mocking Testing**: All tests must mock external Redacted API endpoints without real network calls, following project testing guidelines.

## Known context
* Resonate's current enrichment pipeline aggregates metadata from multiple sources in `src/resonate/main.py` using fetchers in `src/resonate/modules/external_metadata.py` and `src/resonate/modules/lastfm.py`.
* Resonate utilizes a local SQLite database (`state_mgr`) for tracking processed tracks and caching artist alias mappings.
* Tags collected from external sources are mapped via SentenceTransformers (`all-MiniLM-L6-v2`) in `TagMapper` to standardized primary genres, subgenres, and moods.
* Configuration is managed via Pydantic models in `src/resonate/config.py` and loaded from `config.yaml`.

## Risks
* **Account Risk / Rate Limit Violations**: Excessive or unthrottled requests during batch library analysis could get user accounts or API keys flagged or rate-limited.
* **Search Matching Ambiguity**: Matching local track metadata to the correct Redacted Torrent Group (e.g. standard edition vs. deluxe, remix albums, live bootlegs) requires robust release title cleaning and fuzzy matching.
* **Cold Cache Latency**: Sequential rate-limited API calls (2s per album query) will slow down initial full-library analysis if not properly cached at the album/artist level in SQLite.

## Options (2–4)

### Option 1: Native External Metadata Fetcher (`RedactedFetcher`)
* **Description**: Build a dedicated `RedactedFetcher` module in `src/resonate/modules/redacted.py` (or within `external_metadata.py`), configured in `config.yaml` alongside `lastfm`, `discogs`, and `musicbrainz`.
* **Details**:
  - Implements `get_artist_tags(artist)` and `get_release_tags(artist, album)`.
  - Integrates directly into `main.py`'s existing tag collection pipeline.
  - Queries `ajax.php?action=browse` and `ajax.php?action=torrentgroup` with strict 2-second rate limiting and SQLite response caching.
* **Pros**: Simple, zero overhead, immediately compatible with existing setup wizard, CLI flags, and tag aggregation.
* **Cons**: Tightly coupled to Resonate core; doesn't establish a generic third-party plugin interface for future niche trackers or community sources.

### Option 2: Generic Pluggable Provider System with Redacted Reference Plugin
* **Description**: Refactor Resonate's metadata layer into a unified `BaseMetadataProvider` plugin interface (using Python entry points or dynamic class discovery), implementing `RedactedProvider` as a first-class plugin.
* **Details**:
  - Defines an abstract base class `MetadataProvider` with standard hooks: `fetch_track_tags()`, `fetch_album_tags()`, `fetch_artist_tags()`, and `priority`.
  - Providers (Last.fm, MusicBrainz, Discogs, Redacted, Bandcamp, RateYourMusic) become modular plugins loaded dynamically based on `config.yaml`.
* **Pros**: Clean architecture, enables easy community plugin creation, unifies caching and rate-limiting policies across all providers.
* **Cons**: Requires refactoring the metadata fetching architecture in `main.py` and `config.py` before adding Redacted support.

### Option 3: Standalone Beets Plugin with Resonate Tag Sync
* **Description**: Leverage the existing ecosystem (e.g. `beets-redacted` / Gazelle plugin for Beets) and build a bridge / Beets plugin that imports Redacted metadata into Beets, which Resonate reads.
* **Details**:
  - Beets handles Redacted API querying and embeds raw genre/tag metadata into ID3/FLAC tags.
  - Resonate reads the embedded tags and applies its semantic `TagMapper`, lyrics mood validation, and Plex synchronization.
* **Pros**: Offloads Redacted API maintenance and tracker matching logic to existing Beets tooling.
* **Cons**: Requires users to run Beets alongside Resonate; poor standalone Docker experience for users who only use Resonate.

## Recommendation
**Adopt Option 1 (Native `RedactedFetcher`) first with a lightweight provider interface**, paving the way for Option 2:
1. Implement `RedactedFetcher` with strict rate limiting (Gazelle 2-second token bucket), proper headers (`User-Agent: Resonate/1.0`), and artist/album level SQLite caching.
2. Add a `redacted` configuration block in `config.yaml` (`api_key`, `base_url`, `rate_limit_seconds`, `enabled`).
3. Connect Redacted tags to `main.py`'s tag aggregation pipeline so high-quality Redacted tags feed directly into genre, sub-genre, and mood mapping.

## Acceptance criteria
1. **Configuration**: `config.yaml` and `src/resonate/config.py` support a `redacted` section with API token and custom base URL.
2. **API Client**: `RedactedFetcher` successfully queries `ajax.php?action=browse` and `ajax.php?action=torrentgroup` to extract release-level and artist-level tags.
3. **Rate Limiting & Safety**: Enforces a minimum 2.0-second delay between requests and gracefully handles HTTP 429 / authentication errors.
4. **Caching**: Query results are cached in SQLite by `artist + album` to eliminate redundant API calls across multiple tracks on the same album.
5. **Tag Integration**: Retrieved Redacted tags are aggregated in `main.py` and processed through `TagMapper` (genre, subgenre, mood).
6. **Testing**: Comprehensive unit tests covering API success responses, alias handling, rate limiting, and HTTP error scenarios with 100% mocked requests.
