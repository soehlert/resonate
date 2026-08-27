"""Metadata provider manager with multi-threaded querying and SQLite album caching."""

import html
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from resonate.modules.external_metadata import ARTIST_ALIASES
from resonate.providers.base import BaseMetadataProvider
from resonate.utils.state import StateManager

logger = logging.getLogger(__name__)


class ProviderManager:
    """Orchestrates active metadata providers with thread pooling and album-level caching."""

    def __init__(
        self,
        providers: list[BaseMetadataProvider],
        state_manager: StateManager | None = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize ProviderManager with active providers and optional state database."""
        self.providers = [p for p in providers if getattr(p, "enabled", True)]
        self.state_manager = state_manager
        self.max_workers = max_workers
        self._session_album_cache: dict[tuple[str, str], list[str]] = {}

    def get_provider(self, name: str) -> BaseMetadataProvider | None:
        """Retrieve a registered provider by name."""
        for p in self.providers:
            if p.name.lower() == name.lower():
                return p
        return None

    def resolve_artist_alias(self, raw_artist: str) -> str:
        """Resolve canonical artist name from SQLite cache or provider discovery."""
        if not raw_artist:
            return raw_artist

        # 1. Check SQLite state cache
        if self.state_manager:
            cached = self.state_manager.get_cached_artist_alias(raw_artist)
            if cached:
                return cached

        # 2. Query enabled providers for canonical alias discovery
        for provider in self.providers:
            discovered = provider.resolve_canonical_artist(raw_artist)
            if discovered and discovered.lower() != raw_artist.lower():
                clean_raw = raw_artist.lower().strip()
                if clean_raw not in ARTIST_ALIASES:
                    ARTIST_ALIASES[clean_raw] = []
                if discovered not in ARTIST_ALIASES[clean_raw]:
                    ARTIST_ALIASES[clean_raw].append(discovered)

                if self.state_manager:
                    self.state_manager.save_cached_artist_alias(
                        raw_artist, discovered, source=provider.name
                    )
                return discovered

        return raw_artist

    def fetch_album_tags(self, artist: str, album: str) -> list[str]:
        """Fetch and consolidate album tags across providers with memory and SQLite caching."""
        if not artist or not album:
            return []

        cache_key = (artist.strip().lower(), album.strip().lower())
        if cache_key in self._session_album_cache:
            return self._session_album_cache[cache_key]

        if self.state_manager:
            db_cached = self.state_manager.get_cached_album_tags(artist, album)
            if db_cached is not None:
                self._session_album_cache[cache_key] = db_cached
                return db_cached

        # Parallel query to enabled providers
        album_tags: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(p.fetch_album_tags, artist, album): p.name
                for p in self.providers
            }
            for future in as_completed(futures):
                p_name = futures[future]
                try:
                    res = future.result()
                    if res:
                        album_tags.extend(res)
                except Exception as err:
                    logger.debug(f"Provider '{p_name}' album tag fetch failed: {err}")

        # Deduplicate preserving case
        seen: set[str] = set()
        deduped: list[str] = []
        for t in album_tags:
            clean = html.unescape(t).strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)

        self._session_album_cache[cache_key] = deduped
        if self.state_manager and deduped:
            self.state_manager.save_cached_album_tags(artist, album, deduped)

        return deduped

    def fetch_track_tags(
        self, artist: str, title: str, album: str | None = None
    ) -> list[str]:
        """Fetch track-level tags concurrently across active providers."""
        if not artist or not title:
            return []

        track_tags: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(p.fetch_track_tags, artist, title, album): p.name
                for p in self.providers
            }
            for future in as_completed(futures):
                p_name = futures[future]
                try:
                    res = future.result()
                    if res:
                        track_tags.extend(res)
                except Exception as err:
                    logger.debug(f"Provider '{p_name}' track tag fetch failed: {err}")

        seen: set[str] = set()
        deduped: list[str] = []
        for t in track_tags:
            clean = html.unescape(t).strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)
        return deduped

    def fetch_artist_fallback_tags(self, artist: str) -> list[str]:
        """Fetch unverified artist-level tags when no track or album tags exist."""
        if not artist:
            return []

        artist_tags: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(p.fetch_artist_tags, artist): p.name
                for p in self.providers
            }
            for future in as_completed(futures):
                p_name = futures[future]
                try:
                    res = future.result()
                    if res:
                        artist_tags.extend(res)
                except Exception as err:
                    logger.debug(f"Provider '{p_name}' artist tag fetch failed: {err}")

        seen: set[str] = set()
        deduped: list[str] = []
        for t in artist_tags:
            clean = html.unescape(t).strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)
        return deduped

    def get_tags_for_track(
        self, artist: str, title: str, album: str | None = None
    ) -> tuple[list[str], list[str], bool, str]:
        """Consolidate metadata tags for a track with verified and fallback logic.

        Returns:
            tuple of (raw_tags, track_specific_tags, has_verified_tags, resolved_artist)
        """
        resolved_artist = self.resolve_artist_alias(artist)

        # 1. Fetch Track-level tags (concurrent)
        track_tags = self.fetch_track_tags(resolved_artist, title, album=album)

        # 2. Fetch Album-level tags (cached)
        album_tags = self.fetch_album_tags(resolved_artist, album) if album else []

        verified_tags = track_tags + album_tags

        # 3. If no verified tags found, retry alias resolution if not already cached
        if not verified_tags and resolved_artist == artist:
            discovered = self.resolve_artist_alias(artist)
            if discovered != artist:
                resolved_artist = discovered
                track_tags = self.fetch_track_tags(resolved_artist, title, album=album)
                if album:
                    album_tags = self.fetch_album_tags(resolved_artist, album)
                verified_tags = track_tags + album_tags

        # 4. Fallback to artist-level tags ONLY if no verified track/album tags found
        artist_tags = []
        if not verified_tags:
            artist_tags = self.fetch_artist_fallback_tags(resolved_artist)

        raw_tags = list(verified_tags) if verified_tags else list(artist_tags)
        has_verified = bool(verified_tags)

        return raw_tags, track_tags, has_verified, resolved_artist
