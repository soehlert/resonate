"""Metadata provider manager with sequential querying and SQLite album caching."""

import html
import logging
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
    """Orchestrates active metadata providers with caching and alias resolution."""

    def __init__(
        self,
        providers: list[BaseMetadataProvider],
        state_manager: StateManager | None = None,
    ) -> None:
        """Initialize ProviderManager with active providers and optional state database."""
        self.providers = [p for p in providers if getattr(p, "enabled", True)]
        self.state_manager = state_manager
        self._session_album_cache: dict[tuple[str, str], list[str]] = {}
        self._session_track_cache: dict[tuple[str, str], list[str]] = {}

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
        """Execute a fetch function across a list of providers sequentially."""
        if not providers:
            return []
        tags: list[str] = []
        for p in providers:
            try:
                fn = getattr(p, func_name)
                res = fn(*args)
                if res:
                    tags.extend(res)
            except Exception as err:
                logger.debug(f"Provider '{p.name}' {func_name} failed: {err}")
        return tags

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

        self._session_album_cache[cache_key] = deduped
        if self.state_manager and deduped:
            self.state_manager.save_cached_album_tags(artist, album, deduped)

        return deduped

    def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
        """Fetch track-level tags with memory and SQLite caching."""
        if not artist or not title:
            return []

        cache_key = (artist.strip().lower(), title.strip().lower())
        if cache_key in self._session_track_cache:
            return self._session_track_cache[cache_key]

        if self.state_manager:
            db_cached = self.state_manager.get_cached_track_tags(artist, title)
            if db_cached is not None:
                self._session_track_cache[cache_key] = db_cached
                return db_cached

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

        self._session_track_cache[cache_key] = deduped
        if self.state_manager:
            self.state_manager.save_cached_track_tags(artist, title, deduped)

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
        # 1. ALWAYS query the tagged artist name first (do not pre-emptively swap with aliases!)
        resolved_artist = artist
        clean_raw = artist.lower().strip()

        # Fetch Track-level tags first
        track_tags = self.fetch_track_tags(resolved_artist, title, album=album)

        album_tags: list[str] = []
        # STRICT TRACK-TAG PRIORITY: If track tags exist, DO NOT fetch or blend album tags!
        if track_tags:
            verified_tags = list(track_tags)
        else:
            # Emergency Fallback 1: Query album-level tags ONLY when zero track tags exist anywhere
            if album:
                album_tags = self.fetch_album_tags(resolved_artist, album)
                if (
                    not album_tags
                    and album_artist
                    and album_artist.strip().lower() != resolved_artist.strip().lower()
                ):
                    album_tags = self.fetch_album_tags(album_artist.strip(), album)
            verified_tags = list(album_tags)

        # 2. Only if NO verified tags found for original name, test alias candidates
        if not verified_tags and clean_raw not in COMPILATION_ARTIST_NAMES:
            alias_candidates: list[str] = []
            if clean_raw in ARTIST_ALIASES:
                alias_candidates.extend(ARTIST_ALIASES[clean_raw])
            if self.state_manager:
                cached = self.state_manager.get_cached_artist_alias(artist)
                if cached and cached.lower() != clean_raw and cached not in alias_candidates:
                    alias_candidates.append(cached)

            for cand in alias_candidates:
                cand_track_tags = self.fetch_track_tags(cand, title, album=album)
                if cand_track_tags:
                    resolved_artist = cand
                    track_tags = cand_track_tags
                    verified_tags = list(cand_track_tags)
                    album_tags = []
                    break
                # Only check album tags for alias if alias track tags are also empty
                cand_album_tags = self.fetch_album_tags(cand, album) if album else []
                if (
                    not cand_album_tags
                    and album
                    and album_artist
                    and album_artist.strip().lower() != cand.strip().lower()
                ):
                    cand_album_tags = self.fetch_album_tags(album_artist.strip(), album)
                if cand_album_tags:
                    resolved_artist = cand
                    album_tags = cand_album_tags
                    verified_tags = list(cand_album_tags)
                    break

            # 3. If still no verified tags, query provider alias discovery
            if not verified_tags:
                discovered = self.resolve_artist_alias(artist)
                if (
                    discovered
                    and discovered.lower() != clean_raw
                    and discovered not in alias_candidates
                ):
                    resolved_artist = discovered
                    disc_track_tags = self.fetch_track_tags(resolved_artist, title, album=album)
                    if disc_track_tags:
                        track_tags = disc_track_tags
                        verified_tags = list(disc_track_tags)
                        album_tags = []
                    elif album:
                        disc_album_tags = self.fetch_album_tags(resolved_artist, album)
                        if (
                            not disc_album_tags
                            and album_artist
                            and album_artist.strip().lower() != resolved_artist.strip().lower()
                        ):
                            disc_album_tags = self.fetch_album_tags(album_artist.strip(), album)
                        if disc_album_tags:
                            album_tags = disc_album_tags
                            verified_tags = list(disc_album_tags)

        # 4. Fallback to artist-level tags ONLY if no verified track/album tags found
        artist_tags = []
        if not verified_tags and clean_raw not in COMPILATION_ARTIST_NAMES:
            artist_tags = self.fetch_artist_fallback_tags(resolved_artist)

        raw_tags = list(verified_tags) if verified_tags else list(artist_tags)
        has_verified = bool(track_tags)

        return raw_tags, track_tags, has_verified, resolved_artist
