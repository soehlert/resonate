"""Discogs metadata provider implementing BaseMetadataProvider."""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from resonate.modules.external_metadata import (
    clean_retailer_noise,
    uncensor_title,
)
from resonate.providers.base import BaseMetadataProvider

logger = logging.getLogger(__name__)


class DiscogsProvider(BaseMetadataProvider):
    """Fetch release genres and styles from the Discogs API."""

    name: str = "discogs"

    def __init__(self, api_token: str | None = None, enabled: bool = True) -> None:
        """Initialize DiscogsProvider with optional API token."""
        self.api_token = api_token
        self.enabled = enabled and bool(api_token)
        self.headers = {"User-Agent": "Resonate/1.0.0"}
        if self.api_token:
            self.headers["Authorization"] = f"Discogs token={self.api_token}"

    def fetch_track_tags(
        self, artist: str, title: str, album: str | None = None
    ) -> list[str]:
        """Track-level tag query (falls back to album release lookup if album provided)."""
        if not self.enabled or not artist:
            return []
        if album:
            return self.fetch_album_tags(artist, album)
        return self._search_release(artist, uncensor_title(title))

    def fetch_album_tags(self, artist: str, album: str) -> list[str]:
        """Search Discogs for release album genres and styles."""
        if not self.enabled or not artist or not album:
            return []
        cleaned = clean_retailer_noise(album) or album
        return self._search_release(artist, cleaned)

    def fetch_artist_tags(self, artist: str) -> list[str]:
        """Artist-level tag query (not natively supported by Discogs release search)."""
        return []

    def _search_release(self, artist: str, release_title: str) -> list[str]:
        """Internal helper to query Discogs release search endpoint."""
        query = f"{artist} - {release_title}"
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.discogs.com/database/search?q={encoded_query}&type=release"

        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status != 200:
                    return []
                data = json.loads(response.read().decode("utf-8"))

            results = data.get("results", [])
            if not results:
                return []

            first_result = results[0]
            genres = first_result.get("genre", [])
            styles = first_result.get("style", [])

            all_tags: list[str] = []
            for g in genres:
                if isinstance(g, str) and g.strip():
                    all_tags.append(g.strip())
            for s in styles:
                if isinstance(s, str) and s.strip():
                    all_tags.append(s.strip())

            return list(set(all_tags))
        except Exception as err:
            logger.warning(f"Discogs API query failed for '{artist} - {release_title}': {err}")
            return []
