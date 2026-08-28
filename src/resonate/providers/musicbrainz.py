"""MusicBrainz metadata provider with 5.0s rate limiting and alias resolution."""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from resonate.modules.external_metadata import (
    artist_matches,
    clean_retailer_noise,
    uncensor_title,
)
from resonate.providers.base import BaseMetadataProvider

logger = logging.getLogger(__name__)


class MusicBrainzProvider(BaseMetadataProvider):
    """Fetch tags, genres, and artist aliases from MusicBrainz API."""

    name: str = "musicbrainz"

    def __init__(
        self,
        enabled: bool = True,
        rate_limit_delay: float = 2.5,
        user_agent: str = "Resonate/1.0.0 (https://github.com/soehlert/resonate)",
    ) -> None:
        """Initialize MusicBrainzProvider with rate limit delay and custom User-Agent."""
        self.enabled = enabled
        self.rate_limit_delay = rate_limit_delay
        self.headers = {"User-Agent": user_agent}
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce MusicBrainz API rate limit delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def resolve_canonical_artist(self, artist: str) -> str | None:
        """Query MusicBrainz artist search to resolve aliases/rebrands to canonical name."""
        if not self.enabled or not artist:
            return None

        clean_artist = artist.replace('"', '\"')
        query = f'artist:"{clean_artist}"'
        encoded_query = urllib.parse.quote(query)
        url = f"https://musicbrainz.org/ws/2/artist/?query={encoded_query}&fmt=json"

        self._rate_limit()
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return None
                data = json.loads(response.read().decode("utf-8"))
                artists = data.get("artists", [])
                if not artists:
                    return None
                top_match = artists[0]
                score = int(top_match.get("score", 0))
                canonical_name = top_match.get("name", "")
                if score >= 90 and canonical_name:
                    if canonical_name.lower() != artist.lower():
                        return str(canonical_name)
                    aliases = [
                        a for a in top_match.get("aliases", [])
                        if isinstance(a, dict) and a.get("name")
                    ]
                    for a in aliases:
                        a_name = a.get("name", "")
                        if (
                            a_name.lower() != artist.lower()
                            and a.get("type") == "Artist name"
                            and a.get("primary") is True
                        ):
                            return str(a_name)
                    for a in aliases:
                        a_name = a.get("name", "")
                        if a_name.lower() != artist.lower() and a.get("type") == "Artist name":
                            return str(a_name)
        except Exception as err:
            logger.debug(f"MusicBrainz artist alias query failed for '{artist}': {err}")
        return None

    def fetch_track_tags(
        self, artist: str, title: str, album: str | None = None
    ) -> list[str]:
        """Search for a recording on MusicBrainz and return its tags and genres."""
        if not self.enabled or not artist or not title:
            return []

        cleaned_album = clean_retailer_noise(album) if album else None
        uncensored_title = uncensor_title(title)
        effective_title = uncensored_title or title
        effective_album = cleaned_album or album

        return self._fetch_recording_tags_for_artist(
            artist, effective_title, album=effective_album, expected_artist=artist
        )

    def fetch_album_tags(self, artist: str, album: str) -> list[str]:
        """Search for a release group on MusicBrainz and return its tags and genres."""
        if not self.enabled or not artist or not album:
            return []

        cleaned_album = clean_retailer_noise(album) or album
        clean_artist = artist.replace('"', '\"')
        clean_album = cleaned_album.replace('"', '\"')
        query = f'artist:"{clean_artist}" AND releasegroup:"{clean_album}"'
        encoded = urllib.parse.quote(query)
        url = f"https://musicbrainz.org/ws/2/release-group/?query={encoded}&fmt=json"

        data = self._execute_query(url, f"release-group {artist} - {album}")
        if not data:
            return []

        tags: list[str] = []
        for rg in data.get("release-groups", []):
            artist_credits = rg.get("artist-credit", [])
            rg_artist = "".join(
                ac.get("name", "") for ac in artist_credits if isinstance(ac, dict)
            )
            if rg_artist and not artist_matches(artist, rg_artist):
                continue
            for t in rg.get("tags", []):
                if name := t.get("name"):
                    tags.append(name)
            for g in rg.get("genres", []):
                if name := g.get("name"):
                    tags.append(name)
        return list(set(tags))

    def fetch_artist_tags(self, artist: str) -> list[str]:
        """Fetch artist tags directly from MusicBrainz."""
        if not self.enabled or not artist:
            return []

        clean_artist = artist.replace('"', '\"')
        query = f'artist:"{clean_artist}"'
        encoded = urllib.parse.quote(query)
        url = f"https://musicbrainz.org/ws/2/artist/?query={encoded}&fmt=json"

        data = self._execute_query(url, f"artist {artist}")
        if not data:
            return []

        tags: list[str] = []
        for art in data.get("artists", []):
            name = art.get("name", "")
            if name and not artist_matches(artist, name):
                continue
            for t in art.get("tags", []):
                if t_name := t.get("name"):
                    tags.append(t_name)
            for g in art.get("genres", []):
                if g_name := g.get("name"):
                    tags.append(g_name)
        return list(set(tags))

    def _execute_query(self, url_or_query: str, log_context: str) -> dict | None:
        """Execute a MusicBrainz search query with retries and 5.0s rate limiting."""
        if url_or_query.startswith("http"):
            url = url_or_query
        else:
            encoded_query = urllib.parse.quote(url_or_query)
            url = f"https://musicbrainz.org/ws/2/recording/?query={encoded_query}&fmt=json"

        data = None
        max_retries = 2
        for attempt in range(max_retries + 1):
            self._rate_limit()
            req = urllib.request.Request(url, headers=self.headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    if response.status != 200:
                        return None
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as http_err:
                if http_err.code in (429, 503) and attempt < max_retries:
                    retry_delay = 5.0 * (attempt + 1)
                    logger.info(
                        f"MusicBrainz {http_err.code} for '{log_context}', "
                        f"retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay)
                    continue
                logger.warning(f"MusicBrainz API query failed for '{log_context}': {http_err}")
                return None
            except Exception as err:
                logger.warning(f"MusicBrainz API query failed for '{log_context}': {err}")
                return None
        return data

    def _fetch_recording_tags_for_artist(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        expected_artist: str | None = None,
    ) -> list[str]:
        """Query MusicBrainz for a specific artist name, title, and optional album."""
        target_artist = expected_artist or artist
        clean_artist = artist.replace('"', '\"')
        clean_title = title.replace('"', '\"')

        data = None
        if album and album.strip():
            clean_album = album.strip().replace('"', '\"')
            query = (
                f'artist:"{clean_artist}" AND release:"{clean_album}" '
                f'AND (recording:"{clean_title}" OR track:"{clean_title}")'
            )
            data = self._execute_query(query, f"{artist} - {album} - {title}")

        if not data or not data.get("recordings"):
            query = (
                f'artist:"{clean_artist}" AND (recording:"{clean_title}" OR track:"{clean_title}")'
            )
            data = self._execute_query(query, f"{artist} - {title}")

        if not data:
            return []

        try:
            recordings = data.get("recordings", [])
            if not recordings:
                return []

            matching_rec = None
            for r in recordings:
                artist_credits = r.get("artist-credit", [])
                artist_credit_name = "".join(
                    ac.get("name", "") for ac in artist_credits if isinstance(ac, dict)
                )
                if not artist_credit_name and artist_credits:
                    first_ac = artist_credits[0]
                    if isinstance(first_ac, dict):
                        artist_credit_name = first_ac.get("artist", {}).get("name", "")
                if artist_credit_name and not artist_matches(target_artist, artist_credit_name):
                    continue

                if album and album.strip():
                    releases = r.get("releases", [])
                    if any(
                        album.strip().lower() in rel.get("title", "").lower()
                        for rel in releases
                        if isinstance(rel, dict)
                    ):
                        matching_rec = r
                        break

                if matching_rec is None:
                    matching_rec = r

            if not matching_rec:
                return []

            rec = matching_rec
            tags: list[str] = []

            for t in rec.get("tags", []):
                if name := t.get("name"):
                    tags.append(name)
            for g in rec.get("genres", []):
                if name := g.get("name"):
                    tags.append(name)

            for ac in rec.get("artist-credit", []):
                artist_obj = ac.get("artist", {})
                for t in artist_obj.get("tags", []):
                    if name := t.get("name"):
                        tags.append(name)
                for g in artist_obj.get("genres", []):
                    if name := g.get("name"):
                        tags.append(name)

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
