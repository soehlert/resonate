"""Last.fm metadata fetcher with API integration and web scraping fallback."""

import html
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

import pylast

from resonate.modules.external_metadata import (
    artist_matches,
    clean_retailer_noise,
    get_artist_aliases,
    uncensor_title,
)

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

        uncensored_title = uncensor_title(title)
        title_variants = [title]
        if uncensored_title and uncensored_title.lower() != title.lower():
            title_variants.append(uncensored_title)

        tags: list[str] = []
        for art in get_artist_aliases(artist):
            for tit in title_variants:
                if self.api_key:
                    tags = self._fetch_via_api(art, tit, expected_artist=artist)

                if not tags:
                    tags = self._fetch_via_scraping(art, tit, expected_artist=artist)

                if tags:
                    break
            if tags:
                break

        self._cache[cache_key] = tags
        return tags

    def _fetch_via_api(
        self, artist: str, title: str, expected_artist: str | None = None
    ) -> list[str]:
        """Query top tags via pylast API."""
        target_artist = expected_artist or artist
        try:
            network = self._get_network()
            if network is None:
                return []
            track = network.get_track(artist, title)
            track_artist = track.get_artist()
            track_artist_name = track_artist.get_name() if track_artist else ""
            if track_artist_name and not artist_matches(target_artist, track_artist_name):
                logger.warning(
                    f"Last.fm artist mismatch for '{target_artist} - {title}': "
                    f"got '{track_artist_name}'"
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

    def _fetch_via_scraping(
        self, artist: str, title: str, expected_artist: str | None = None
    ) -> list[str]:
        """Scrape track tags from Last.fm web page."""
        encoded_artist = urllib.parse.quote_plus(artist)
        encoded_title = urllib.parse.quote_plus(title)
        url = f"https://www.last.fm/music/{encoded_artist}/_/{encoded_title}/+tags"
        return self._scrape_url_tags(url, expected_artist=expected_artist or artist)

    def _scrape_url_tags(self, url: str, expected_artist: str | None = None) -> list[str]:
        """Helper to scrape tags from a specific Last.fm URL with redirect artist validation."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return []
                raw_geturl = getattr(response, "geturl", None)
                final_url = (
                    raw_geturl()
                    if callable(raw_geturl) and isinstance(raw_geturl(), str)
                    else url
                )
                if expected_artist and isinstance(final_url, str):
                    # Validate that Last.fm did not redirect to an unrelated artist (e.g. Ye -> Yes)
                    match = re.search(r"/music/([^/_#?]+)", final_url)
                    if match:
                        scraped_artist = urllib.parse.unquote_plus(match.group(1)).strip()
                        if not artist_matches(expected_artist, scraped_artist):
                            logger.warning(
                                f"Last.fm redirected artist mismatch for '{expected_artist}': "
                                f"got '{scraped_artist}' from '{final_url}'"
                            )
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

        cleaned_album = clean_retailer_noise(album) if album else None
        album_variants: list[str] = []
        if cleaned_album and cleaned_album.lower() != album.lower():
            album_variants.append(cleaned_album)
        album_variants.append(album)

        tags: list[str] = []
        for art in get_artist_aliases(artist):
            for alb in album_variants:
                if self.api_key:
                    try:
                        network = self._get_network()
                        if network:
                            album_obj = network.get_album(art, alb)
                            album_artist = album_obj.get_artist()
                            album_artist_name = (
                                album_artist.get_name() if album_artist else ""
                            )
                            if album_artist_name and not artist_matches(artist, album_artist_name):
                                logger.warning(
                                    f"Last.fm artist mismatch for album '{artist} - {alb}': "
                                    f"got '{album_artist_name}'"
                                )
                                continue

                            top_tags = album_obj.get_top_tags(limit=10)
                            for item in top_tags:
                                tag_obj = getattr(item, "item", item)
                                tag_name = getattr(tag_obj, "name", None)
                                if tag_name is None and hasattr(tag_obj, "get_name"):
                                    tag_name = tag_obj.get_name()
                                if isinstance(tag_name, str) and tag_name:
                                    tags.append(tag_name)
                            if tags:
                                break
                    except Exception as err:
                        logger.warning(
                            f"pylast API query failed for album '{art} - {alb}': {err}"
                        )

                if not tags:
                    encoded_artist = urllib.parse.quote_plus(art)
                    encoded_album = urllib.parse.quote_plus(alb)
                    url = f"https://www.last.fm/music/{encoded_artist}/{encoded_album}/+tags"
                    tags = self._scrape_url_tags(url, expected_artist=artist)
                    if tags:
                        break
            if tags:
                break

        self._cache[cache_key] = tags
        return tags

    def get_artist_tags(self, artist: str) -> list[str]:
        """Fetch artist tags using pylast API or fallback to scraping web page."""
        cache_key = (artist.strip().lower(), "artist:tags")
        if cache_key in self._cache:
            return self._cache[cache_key]

        tags: list[str] = []
        for art in get_artist_aliases(artist):
            if self.api_key:
                try:
                    network = self._get_network()
                    if network:
                        artist_obj = network.get_artist(art)
                        artist_name = artist_obj.get_name() if artist_obj else ""
                        if artist_name and not artist_matches(artist, artist_name):
                            logger.warning(
                                f"Last.fm artist mismatch for artist '{artist}': "
                                f"got '{artist_name}'"
                            )
                            continue

                        top_tags = artist_obj.get_top_tags(limit=10)
                        for item in top_tags:
                            tag_obj = getattr(item, "item", item)
                            tag_name = getattr(tag_obj, "name", None)
                            if tag_name is None and hasattr(tag_obj, "get_name"):
                                tag_name = tag_obj.get_name()
                            if isinstance(tag_name, str) and tag_name:
                                tags.append(tag_name)
                        if tags:
                            break
                except Exception as err:
                    logger.warning(f"pylast API query failed for artist '{art}': {err}")

            if not tags:
                encoded_artist = urllib.parse.quote_plus(art)
                url = f"https://www.last.fm/music/{encoded_artist}/+tags"
                tags = self._scrape_url_tags(url, expected_artist=artist)
                if tags:
                    break

        self._cache[cache_key] = tags
        return tags
