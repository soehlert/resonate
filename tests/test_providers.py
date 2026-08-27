"""Unit tests for pluggable metadata providers and ProviderManager."""

import json
from unittest.mock import MagicMock, patch

from resonate.providers.base import BaseMetadataProvider
from resonate.providers.discogs import DiscogsProvider
from resonate.providers.lastfm import LastFmProvider
from resonate.providers.manager import ProviderManager
from resonate.providers.musicbrainz import MusicBrainzProvider
from resonate.utils.state import StateManager


class DummyProvider(BaseMetadataProvider):
    """Dummy provider for interface testing."""

    name = "dummy"

    def fetch_track_tags(
        self, artist: str, title: str, album: str | None = None
    ) -> list[str]:
        return ["dummy-track-tag"]

    def fetch_album_tags(self, artist: str, album: str) -> list[str]:
        return ["dummy-album-tag"]

    def fetch_artist_tags(self, artist: str) -> list[str]:
        return ["dummy-artist-tag"]


def test_base_provider_query_all() -> None:
    """Test BaseMetadataProvider query_all standard payload generation."""
    provider = DummyProvider()
    result = provider.query_all("Radiohead", "Creep", album="Pablo Honey")
    assert result.provider_name == "dummy"
    assert result.track_tags == ["dummy-track-tag"]
    assert result.album_tags == ["dummy-album-tag"]
    assert result.artist_tags == ["dummy-artist-tag"]
    assert result.all_tags == ["dummy-track-tag", "dummy-album-tag", "dummy-artist-tag"]


def test_base_provider_disabled() -> None:
    """Test disabled provider returns empty result."""
    provider = DummyProvider()
    provider.enabled = False
    result = provider.query_all("Radiohead", "Creep")
    assert result.status == "disabled"
    assert result.all_tags == []


@patch("urllib.request.urlopen")
def test_musicbrainz_provider_fetch_track_and_album(mock_urlopen) -> None:
    """Test MusicBrainzProvider recording and release group queries."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_data = {
        "recordings": [
            {
                "id": "rec-1",
                "title": "Paranoid Android",
                "artist-credit": [{"name": "Radiohead", "artist": {"name": "Radiohead"}}],
                "releases": [
                    {
                        "title": "OK Computer",
                        "release-group": {"tags": [{"name": "art rock"}]},
                    }
                ],
                "tags": [{"name": "alternative rock"}],
                "genres": [],
            }
        ]
    }
    mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    mb = MusicBrainzProvider(rate_limit_delay=0.0)
    tags = mb.fetch_track_tags("Radiohead", "Paranoid Android", album="OK Computer")
    assert "alternative rock" in tags
    assert "art rock" in tags


@patch("urllib.request.urlopen")
def test_discogs_provider_fetch_album_tags(mock_urlopen) -> None:
    """Test DiscogsProvider release tags query."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_data = {"results": [{"genre": ["Rock"], "style": ["Art Rock", "Alternative Rock"]}]}
    mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    discogs = DiscogsProvider(api_token="valid-token")
    tags = discogs.fetch_album_tags("Radiohead", "OK Computer")
    assert "Rock" in tags
    assert "Art Rock" in tags
    assert "Alternative Rock" in tags


@patch("urllib.request.urlopen")
def test_lastfm_provider_fetch_track_and_album(mock_urlopen) -> None:
    """Test LastFmProvider tag scraping with mocked HTML."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.geturl.return_value = "https://www.last.fm/music/Radiohead/_/Creep/+tags"
    mock_resp.read.return_value = (
        b'<html><div class="header-new-crumb"><a href="/music/Radiohead">Radiohead</a></div>'
        b'<a href="/tag/alternative+rock">alternative rock</a>'
        b'<a href="/tag/90s">90s</a></html>'
    )
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    lastfm = LastFmProvider()
    tags = lastfm.fetch_track_tags("Radiohead", "Creep")
    assert "alternative rock" in tags
    assert "90s" in tags


def test_provider_manager_concurrent_fetch_and_album_caching(tmp_path) -> None:
    """Test ProviderManager executes providers concurrently and caches album queries in SQLite."""
    db_path = tmp_path / "test_state.sqlite"
    state_mgr = StateManager(sqlite_path=str(db_path))

    class ProviderA(BaseMetadataProvider):
        name = "provider_a"
        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            return ["indie rock"]
        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            return ["90s", "alternative"]
        def fetch_artist_tags(self, artist: str) -> list[str]:
            return ["rock"]

    class ProviderB(BaseMetadataProvider):
        name = "provider_b"
        call_count = 0
        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            return ["experimental"]
        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            self.call_count += 1
            return ["art rock"]
        def fetch_artist_tags(self, artist: str) -> list[str]:
            return ["oxford"]

    p_a = ProviderA()
    p_b = ProviderB()
    manager = ProviderManager(providers=[p_a, p_b], state_manager=state_mgr)

    # First Track: Hits providers, saves album tags to DB
    raw_tags, track_tags, has_verified, resolved_art = manager.get_tags_for_track(
        "Radiohead", "Airbag", album="OK Computer"
    )
    assert has_verified is True
    assert resolved_art == "Radiohead"
    assert "indie rock" in raw_tags
    assert "experimental" in raw_tags
    assert "alternative" in raw_tags
    assert "art rock" in raw_tags
    assert p_b.call_count == 1

    # Second Track (Same Album): Album tags MUST hit SQLite cache without calling provider B again
    raw_tags_2, track_tags_2, has_ver_2, _ = manager.get_tags_for_track(
        "Radiohead", "Paranoid Android", album="OK Computer"
    )
    assert has_ver_2 is True
    assert "art rock" in raw_tags_2
    assert p_b.call_count == 1  # Verify Provider B was NOT called again for album query!


def test_provider_manager_artist_fallback() -> None:
    """Test ProviderManager falls back to artist-level tags when no track/album tags exist."""
    class EmptyProvider(BaseMetadataProvider):
        name = "empty"
        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            return []
        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            return []
        def fetch_artist_tags(self, artist: str) -> list[str]:
            return ["ambient", "electronic"]

    manager = ProviderManager(providers=[EmptyProvider()])
    raw_tags, track_tags, has_verified, _ = manager.get_tags_for_track(
        "Aphex Twin", "Unknown Song"
    )
    assert has_verified is False
    assert raw_tags == ["ambient", "electronic"]
    assert track_tags == []
