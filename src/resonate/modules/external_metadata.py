"""External metadata fetchers for MusicBrainz and Discogs integrations."""

import json
import logging
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


def artist_matches(expected: str, candidate: str) -> bool:
    """Verify that candidate artist name matches expected artist (preventing Ye matching Yes)."""
    exp = expected.lower().strip()
    cand = candidate.lower().strip()
    if not exp or not cand:
        return True
    if exp == cand:
        return True
    if exp.startswith("the ") and exp[4:] == cand:
        return True
    if cand.startswith("the ") and cand[4:] == exp:
        return True
    for sep in [" feat", " ft.", " with ", " & ", " and ", " / ", ", ", " x ", " vs ", " vs. "]:
        if cand.startswith(f"{exp}{sep}"):
            return True
    return False


class MusicBrainzFetcher:
    """Fetch tags and genres from the MusicBrainz API."""

    def __init__(self) -> None:
        """Initialize MusicBrainzFetcher."""
        self.headers = {"User-Agent": "Resonate/1.0.0 (https://github.com/soehlert/resonate)"}
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce MusicBrainz API rate limit of 3 seconds per request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        self._last_request_time = time.time()

    def get_recording_tags(self, artist: str, title: str) -> list[str]:
        """Search for a recording on MusicBrainz and return its tags and genres."""
        self._rate_limit()

        # Clean query terms
        clean_artist = artist.replace('"', '\\"')
        clean_title = title.replace('"', '\\"')
        query = f'artist:"{clean_artist}" AND (recording:"{clean_title}" OR track:"{clean_title}")'
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

            # Find matching recording whose artist credit aligns with query artist
            matching_rec = None
            for r in recordings:
                artist_credits = r.get("artist-credit", [])
                if not artist_credits:
                    matching_rec = r
                    break
                artist_credit_name = "".join(
                    ac.get("name", "") for ac in artist_credits if isinstance(ac, dict)
                )
                if not artist_credit_name:
                    first_ac = artist_credits[0]
                    if isinstance(first_ac, dict):
                        artist_credit_name = first_ac.get("artist", {}).get("name", "")
                if not artist_credit_name or artist_matches(artist, artist_credit_name):
                    matching_rec = r
                    break

            if not matching_rec:
                logger.debug(
                    f"No MusicBrainz recording matched artist '{artist}' for title '{title}'"
                )
                return []

            rec = matching_rec
            tags: list[str] = []

            # 1. Recording level tags & genres
            for t in rec.get("tags", []):
                if name := t.get("name"):
                    tags.append(name)
            for g in rec.get("genres", []):
                if name := g.get("name"):
                    tags.append(name)

            # 2. Artist level tags & genres
            for ac in rec.get("artist-credit", []):
                artist_obj = ac.get("artist", {})
                for t in artist_obj.get("tags", []):
                    if name := t.get("name"):
                        tags.append(name)
                for g in artist_obj.get("genres", []):
                    if name := g.get("name"):
                        tags.append(name)

            # 3. Release group tags & genres
            for rel in rec.get("releases", []):
                rg = rel.get("release-group", {})
                for t in rg.get("tags", []):
                    if name := t.get("name"):
                        tags.append(name)
                for g in rg.get("genres", []):
                    if name := g.get("name"):
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
