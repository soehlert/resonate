"""External metadata fetchers for MusicBrainz and Discogs integrations."""

import json
import logging
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


class MusicBrainzFetcher:
    """Fetch tags and genres from the MusicBrainz API."""

    def __init__(self) -> None:
        """Initialize MusicBrainzFetcher."""
        self.headers = {"User-Agent": "Resonate/0.1.0 ( mailto:resonate@example.com )"}
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce MusicBrainz API rate limit of 1 request per second."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def get_recording_tags(self, artist: str, title: str) -> list[str]:
        """Search for a recording on MusicBrainz and return its tags and genres."""
        self._rate_limit()

        # Clean query terms
        clean_artist = artist.replace('"', '\\"')
        clean_title = title.replace('"', '\\"')
        query = f'artist:"{clean_artist}" AND recording:"{clean_title}"'
        encoded_query = urllib.parse.quote(query)
        url = f"https://musicbrainz.org/ws/2/recording/?query={encoded_query}&fmt=json"

        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return []
                data = json.loads(response.read().decode("utf-8"))

            recordings = data.get("recordings", [])
            if not recordings:
                return []

            # Extract tags/genres from the first matching recording
            tags: list[str] = []
            rec = recordings[0]

            # 1. Grab tags
            for t in rec.get("tags", []):
                name = t.get("name")
                if name:
                    tags.append(name)

            # 2. Grab genres
            for g in rec.get("genres", []):
                name = g.get("name")
                if name:
                    tags.append(name)

            return list(set(tags))
        except Exception as err:
            logger.warning(f"MusicBrainz API query failed for '{artist} - {title}': {err}")
            return []


class DiscogsFetcher:
    """Fetch release genres and styles from the Discogs API."""

    def __init__(self, api_token: str | None = None) -> None:
        """Initialize DiscogsFetcher with optional API token."""
        self.api_token = api_token
        self.headers = {"User-Agent": "Resonate/0.1.0"}
        if self.api_token:
            self.headers["Authorization"] = f"Discogs token={self.api_token}"

    def get_release_genres(self, artist: str, title: str) -> list[str]:
        """Search Discogs for a release and return its genres and styles."""
        if not self.api_token:
            logger.debug("Discogs API token not configured. Skipping Discogs lookup.")
            return []

        # Discogs search URL
        query = f"{artist} - {title}"
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.discogs.com/database/search?q={encoded_query}&type=release"

        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return []
                data = json.loads(response.read().decode("utf-8"))

            results = data.get("results", [])
            if not results:
                return []

            first_result = results[0]
            genres = first_result.get("genre", [])
            styles = first_result.get("style", [])

            # Ensure return is flat list of strings
            all_tags = []
            for g in genres:
                if isinstance(g, str):
                    all_tags.append(g)
            for s in styles:
                if isinstance(s, str):
                    all_tags.append(s)

            return list(set(all_tags))
        except Exception as err:
            logger.warning(f"Discogs API query failed for '{artist} - {title}': {err}")
            return []
