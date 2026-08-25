"""Last.fm metadata fetcher with API integration and web scraping fallback."""

import html
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

import pylast

from resonate.modules.external_metadata import artist_matches

logger = logging.getLogger(__name__)


class LastFmFetcher:
    """Fetch track tags from Last.fm API or public web pages."""

    def __init__(self, api_key: str | None = None, api_secret: str | None = None) -> None:
        """Initialize LastFmFetcher with optional API key and secret."""
        self.api_key = api_key
        self.api_secret = api_secret
        self._cache: dict[tuple[str, str], list[str]] = {}
        self._network: Any = None

    def _get_network(self) -> Any:
        """Lazy load LastFMNetwork if api_key is present."""
        if self._network is None and self.api_key:
            try:
                self._network = pylast.LastFMNetwork(
                    api_key=self.api_key, api_secret=self.api_secret or ""
                )
            except Exception as err:
                logger.warning(f"Failed to initialize pylast network: {err}")
                self._network = None
        return self._network

    def get_track_tags(self, artist: str, title: str) -> list[str]:
        """Fetch track tags using pylast API or fallback to scraping web page."""
        cache_key = (artist.strip().lower(), title.strip().lower())
        if cache_key in self._cache:
            return self._cache[cache_key]

        tags: list[str] = []

        if self.api_key:
            tags = self._fetch_via_api(artist, title)

        if not tags:
            tags = self._fetch_via_scraping(artist, title)

        self._cache[cache_key] = tags
        return tags

    def _fetch_via_api(self, artist: str, title: str) -> list[str]:
        """Query top tags via pylast API."""
        try:
            network = self._get_network()
            if network is None:
                return []
            track = network.get_track(artist, title)
            track_artist = track.get_artist()
            track_artist_name = track_artist.get_name() if track_artist else ""
            if track_artist_name and not artist_matches(artist, track_artist_name):
                logger.warning(
                    f"Last.fm artist mismatch for '{artist} - {title}': got '{track_artist_name}'"
                )
                return []

            top_tags = track.get_top_tags(limit=10)
            result: list[str] = []
            for item in top_tags:
                tag_obj = getattr(item, "item", item)
                tag_name = getattr(tag_obj, "name", None)
                if tag_name is None and hasattr(tag_obj, "get_name"):
                    tag_name = tag_obj.get_name()
                if isinstance(tag_name, str) and tag_name:
                    result.append(tag_name)
            return result
        except Exception as err:
            logger.warning(f"pylast API query failed for '{artist} - {title}': {err}")
            return []

    def _fetch_via_scraping(self, artist: str, title: str) -> list[str]:
        """Scrape track tags from Last.fm web page."""
        encoded_artist = urllib.parse.quote_plus(artist)
        encoded_title = urllib.parse.quote_plus(title)
        url = f"https://www.last.fm/music/{encoded_artist}/_/{encoded_title}/+tags"
        return self._scrape_url_tags(url)

    def _scrape_url_tags(self, url: str) -> list[str]:
        """Helper to scrape tags from a specific Last.fm URL."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return []
                html_content = response.read().decode("utf-8", errors="ignore")

            raw_tags = re.findall(r'/tag/([^"/?#]+)', html_content)
            unique_tags: list[str] = []
            for t in raw_tags:
                decoded = html.unescape(urllib.parse.unquote(t).replace("+", " ")).strip()
                if decoded and decoded.lower() not in [x.lower() for x in unique_tags]:
                    unique_tags.append(decoded)
            return unique_tags
        except Exception as err:
            logger.debug(f"Failed to scrape tags from URL '{url}': {err}")
            return []

    def get_album_tags(self, artist: str, album: str) -> list[str]:
        """Fetch album tags using pylast API or fallback to scraping web page."""
        cache_key = (artist.strip().lower(), f"album:{album.strip().lower()}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        tags: list[str] = []

        if self.api_key:
            try:
                network = self._get_network()
                if network:
                    album_obj = network.get_album(artist, album)
                    album_artist = album_obj.get_artist()
                    album_artist_name = album_artist.get_name() if album_artist else ""
                    if album_artist_name and not artist_matches(artist, album_artist_name):
                        logger.warning(
                            f"Last.fm artist mismatch for album '{artist} - {album}': "
                            f"got '{album_artist_name}'"
                        )
                        return []

                    top_tags = album_obj.get_top_tags(limit=10)
                    for item in top_tags:
                        tag_obj = getattr(item, "item", item)
                        tag_name = getattr(tag_obj, "name", None)
                        if tag_name is None and hasattr(tag_obj, "get_name"):
                            tag_name = tag_obj.get_name()
                        if isinstance(tag_name, str) and tag_name:
                            tags.append(tag_name)
            except Exception as err:
                logger.warning(f"pylast API query failed for album '{artist} - {album}': {err}")

        if not tags:
            encoded_artist = urllib.parse.quote_plus(artist)
            encoded_album = urllib.parse.quote_plus(album)
            url = f"https://www.last.fm/music/{encoded_artist}/{encoded_album}/+tags"
            tags = self._scrape_url_tags(url)

        self._cache[cache_key] = tags
        return tags

    def get_artist_tags(self, artist: str) -> list[str]:
        """Fetch artist tags using pylast API or fallback to scraping web page."""
        cache_key = (artist.strip().lower(), "artist:tags")
        if cache_key in self._cache:
            return self._cache[cache_key]

        tags: list[str] = []

        if self.api_key:
            try:
                network = self._get_network()
                if network:
                    artist_obj = network.get_artist(artist)
                    artist_name = artist_obj.get_name() if artist_obj else ""
                    if artist_name and not artist_matches(artist, artist_name):
                        logger.warning(
                            f"Last.fm artist mismatch for artist '{artist}': got '{artist_name}'"
                        )
                        return []

                    top_tags = artist_obj.get_top_tags(limit=10)
                    for item in top_tags:
                        tag_obj = getattr(item, "item", item)
                        tag_name = getattr(tag_obj, "name", None)
                        if tag_name is None and hasattr(tag_obj, "get_name"):
                            tag_name = tag_obj.get_name()
                        if isinstance(tag_name, str) and tag_name:
                            tags.append(tag_name)
            except Exception as err:
                logger.warning(f"pylast API query failed for artist '{artist}': {err}")

        if not tags:
            encoded_artist = urllib.parse.quote_plus(artist)
            url = f"https://www.last.fm/music/{encoded_artist}/+tags"
            tags = self._scrape_url_tags(url)

        self._cache[cache_key] = tags
        return tags
