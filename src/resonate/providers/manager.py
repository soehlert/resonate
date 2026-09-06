"""Metadata provider manager with multi-threaded querying and SQLite album caching."""

import html
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from resonate.modules.external_metadata import ARTIST_ALIASES
from resonate.providers.base import BaseMetadataProvider
from resonate.utils.state import StateManager

logger = logging.getLogger(__name__)

COMPILATION_ARTIST_NAMES: set[str] = {
    "various artists",
    "various",
    "va",
    "soundtrack",
    "soundtracks",
    "original soundtrack",
    "ost",
    "compilation",
}


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
        self._cache_lock = threading.Lock()

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

        clean_raw = raw_artist.lower().strip()
        if clean_raw in COMPILATION_ARTIST_NAMES:
            return raw_artist
        with self._cache_lock:
            if clean_raw in ARTIST_ALIASES and ARTIST_ALIASES[clean_raw]:
                return ARTIST_ALIASES[clean_raw][0]

        # 2. Check SQLite state cache
        if self.state_manager:
            cached = self.state_manager.get_cached_artist_alias(raw_artist)
            if cached:
                return cached

        # 3. Query enabled providers for canonical alias discovery
        for provider in self.providers:
            discovered = provider.resolve_canonical_artist(raw_artist)
            if discovered:
                canonical = discovered.strip()
                with self._cache_lock:
                    if clean_raw not in ARTIST_ALIASES:
                        ARTIST_ALIASES[clean_raw] = []
                    if canonical not in ARTIST_ALIASES[clean_raw]:
                        ARTIST_ALIASES[clean_raw].append(canonical)

                if self.state_manager:
                    self.state_manager.save_cached_artist_alias(
                        raw_artist, canonical, source=provider.name
                    )
                return canonical

        # Cache negative result in SQLite so we never query MusicBrainz again for this artist
        if self.state_manager:
            self.state_manager.save_cached_artist_alias(raw_artist, raw_artist, source="none")
        return raw_artist

    def _execute_provider_fetch(
        self,
        providers: list[BaseMetadataProvider],
        func_name: str,
        *args: Any,
    ) -> list[str]:
        """Execute a fetch function across a list of providers concurrently."""
        if not providers:
            return []
        tags: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(providers), self.max_workers)) as executor:
            future_to_provider = [
                (p.name, executor.submit(getattr(p, func_name), *args)) for p in providers
            ]
            for p_name, future in future_to_provider:
                try:
                    res = future.result()
                    if res:
                        tags.extend(res)
                except Exception as err:
                    logger.debug(f"Provider '{p_name}' {func_name} failed: {err}")
        return tags

    def fetch_album_tags(self, artist: str, album: str) -> list[str]:
        """Fetch and consolidate album tags across providers with memory and SQLite caching."""
        if not artist or not album:
            return []

        cache_key = (artist.strip().lower(), album.strip().lower())
        with self._cache_lock:
            if cache_key in self._session_album_cache:
                return self._session_album_cache[cache_key]

        if self.state_manager:
            db_cached = self.state_manager.get_cached_album_tags(artist, album)
            if db_cached is not None:
                with self._cache_lock:
                    self._session_album_cache[cache_key] = db_cached
                return db_cached

        primary_providers = [p for p in self.providers if p.name.lower() != "musicbrainz"]
        fallback_providers = [p for p in self.providers if p.name.lower() == "musicbrainz"]

        album_tags = self._execute_provider_fetch(
            primary_providers, "fetch_album_tags", artist, album
        )
        if not album_tags and fallback_providers:
            album_tags = self._execute_provider_fetch(
                fallback_providers, "fetch_album_tags", artist, album
            )

        # Deduplicate preserving case and priority
        seen: set[str] = set()
        deduped: list[str] = []
        for t in album_tags:
            clean = html.unescape(t).strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)

        with self._cache_lock:
            self._session_album_cache[cache_key] = deduped
        if self.state_manager and deduped:
            self.state_manager.save_cached_album_tags(artist, album, deduped)

        return deduped

    def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
        """Fetch track-level tags with primary-then-fallback provider strategy."""
        if not artist or not title:
            return []

        primary_providers = [p for p in self.providers if p.name.lower() != "musicbrainz"]
        fallback_providers = [p for p in self.providers if p.name.lower() == "musicbrainz"]

        track_tags = self._execute_provider_fetch(
            primary_providers, "fetch_track_tags", artist, title, album
        )
        if not track_tags and fallback_providers:
            track_tags = self._execute_provider_fetch(
                fallback_providers, "fetch_track_tags", artist, title, album
            )

        seen: set[str] = set()
        deduped: list[str] = []
        for t in track_tags:
            clean = html.unescape(t).strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)
        return deduped

    def fetch_artist_fallback_tags(self, artist: str) -> list[str]:
        """Fetch unverified artist-level tags in order when no track/album tags exist."""
        if not artist:
            return []

        clean_art = artist.lower().strip()
        if clean_art in COMPILATION_ARTIST_NAMES:
            return []

        primary_providers = [p for p in self.providers if p.name.lower() != "musicbrainz"]
        fallback_providers = [p for p in self.providers if p.name.lower() == "musicbrainz"]

        artist_tags = self._execute_provider_fetch(primary_providers, "fetch_artist_tags", artist)
        if not artist_tags and fallback_providers:
            artist_tags = self._execute_provider_fetch(
                fallback_providers, "fetch_artist_tags", artist
            )

        seen: set[str] = set()
        deduped: list[str] = []
        for t in artist_tags:
            clean = html.unescape(t).strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)
        return deduped

    def get_tags_for_track(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        album_artist: str | None = None,
    ) -> tuple[list[str], list[str], bool, str]:
        """Consolidate metadata tags for a track with verified and fallback logic.

        Returns:
            tuple of (raw_tags, track_specific_tags, has_verified_tags, resolved_artist)
        """
        # Check if alias is already cached/known without hitting network
        resolved_artist = artist
        clean_raw = artist.lower().strip()
        if clean_raw in ARTIST_ALIASES and ARTIST_ALIASES[clean_raw]:
            resolved_artist = ARTIST_ALIASES[clean_raw][0]
        elif self.state_manager:
            cached = self.state_manager.get_cached_artist_alias(artist)
            if cached:
                resolved_artist = cached

        # 1. Fetch Track-level tags
        track_tags = self.fetch_track_tags(resolved_artist, title, album=album)

        # 2. Fetch Album-level tags (cached)
        album_tags = self.fetch_album_tags(resolved_artist, album) if album else []
        if (
            not album_tags
            and album
            and album_artist
            and album_artist.strip().lower() != resolved_artist.strip().lower()
        ):
            album_tags = self.fetch_album_tags(album_artist.strip(), album)

        verified_tags = track_tags + album_tags

        # 3. If no verified tags found, trigger provider alias discovery (skip generic compilations)
        if not verified_tags and clean_raw not in COMPILATION_ARTIST_NAMES:
            discovered = self.resolve_artist_alias(artist)
            if discovered != resolved_artist:
                resolved_artist = discovered
                track_tags = self.fetch_track_tags(resolved_artist, title, album=album)
                if album:
                    album_tags = self.fetch_album_tags(resolved_artist, album)
                    if (
                        not album_tags
                        and album_artist
                        and album_artist.strip().lower() != resolved_artist.strip().lower()
                    ):
                        album_tags = self.fetch_album_tags(album_artist.strip(), album)
                verified_tags = track_tags + album_tags

        # 4. Fallback to artist-level tags ONLY if no verified track/album tags found
        artist_tags = []
        if not verified_tags and clean_raw not in COMPILATION_ARTIST_NAMES:
            artist_tags = self.fetch_artist_fallback_tags(resolved_artist)

        raw_tags = list(verified_tags) if verified_tags else list(artist_tags)
        has_verified = bool(verified_tags)

        return raw_tags, track_tags, has_verified, resolved_artist
