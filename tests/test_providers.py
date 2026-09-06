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

    def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
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
    raw_tags, track_tags, has_verified, _ = manager.get_tags_for_track("Aphex Twin", "Unknown Song")
    assert has_verified is False
    assert raw_tags == ["ambient", "electronic"]
    assert track_tags == []


def test_provider_manager_fallback_tiering() -> None:
    """Verify primary providers bypass MusicBrainz when tags are found, and fallback when empty."""

    class PrimaryProv(BaseMetadataProvider):
        name = "lastfm"

        def __init__(self, return_tags: bool) -> None:
            self.return_tags = return_tags

        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            return ["primary-track-tag"] if self.return_tags else []

        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            return ["primary-album-tag"] if self.return_tags else []

        def fetch_artist_tags(self, artist: str) -> list[str]:
            return ["primary-artist-tag"] if self.return_tags else []

    class FallbackProv(BaseMetadataProvider):
        name = "musicbrainz"

        def __init__(self) -> None:
            self.call_count = 0

        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            self.call_count += 1
            return ["fallback-track-tag"]

        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            self.call_count += 1
            return ["fallback-album-tag"]

        def fetch_artist_tags(self, artist: str) -> list[str]:
            self.call_count += 1
            return ["fallback-artist-tag"]

    # Case 1: Primary provider succeeds -> MusicBrainz is never called
    primary_success = PrimaryProv(return_tags=True)
    fallback_1 = FallbackProv()
    mgr_1 = ProviderManager(providers=[primary_success, fallback_1])
    tags = mgr_1.fetch_track_tags("Eminem", "Lose Yourself")
    assert tags == ["primary-track-tag"]
    assert fallback_1.call_count == 0

    # Case 2: Primary provider returns empty -> MusicBrainz fallback is invoked
    primary_empty = PrimaryProv(return_tags=False)
    fallback_2 = FallbackProv()
    mgr_2 = ProviderManager(providers=[primary_empty, fallback_2])
    tags_fallback = mgr_2.fetch_track_tags("Obscure Artist", "Rare Song")
    assert tags_fallback == ["fallback-track-tag"]
    assert fallback_2.call_count == 1


def test_provider_manager_compilation_protection() -> None:
    """Verify compilation artists are exempted from alias resolution & fallback."""
    mock_provider = MagicMock(spec=BaseMetadataProvider)
    mock_provider.name = "mock"
    mock_provider.enabled = True
    mock_provider.resolve_canonical_artist.return_value = "Some Other Artist"
    mock_provider.fetch_artist_tags.return_value = ["soundtrack-tag"]

    mgr = ProviderManager(providers=[mock_provider])

    # Alias resolution must NOT change Various Artists or call providers
    assert mgr.resolve_artist_alias("Various Artists") == "Various Artists"
    assert mgr.resolve_artist_alias("Soundtrack") == "Soundtrack"
    assert mgr.resolve_artist_alias("VA") == "VA"
    assert mock_provider.resolve_canonical_artist.call_count == 0

    # Artist-level fallback tags must be empty for compilations
    assert mgr.fetch_artist_fallback_tags("Various Artists") == []
    assert mgr.fetch_artist_fallback_tags("soundtrack") == []
    assert mock_provider.fetch_artist_tags.call_count == 0


def test_provider_manager_album_artist_fallback() -> None:
    """Verify album tags fall back to album_artist when track artist yields no album tags."""

    class CompilationProvider(BaseMetadataProvider):
        name = "mock_provider"

        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            if artist == "Eminem" and title == "Lose Yourself":
                return ["rap", "hip hop"]
            return []

        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            if artist == "Various Artists" and "8 Mile" in album:
                return ["soundtrack", "ost"]
            return []

        def fetch_artist_tags(self, artist: str) -> list[str]:
            return []

    mgr = ProviderManager(providers=[CompilationProvider()])
    raw_tags, track_tags, has_verified, _ = mgr.get_tags_for_track(
        artist="Eminem",
        title="Lose Yourself",
        album="8 Mile: Music From and Inspired by the Motion Picture",
        album_artist="Various Artists",
    )

    assert has_verified is True
    assert "rap" in track_tags
    assert "hip hop" in track_tags
    assert "soundtrack" in raw_tags
    assert "ost" in raw_tags


def test_provider_manager_original_artist_queried_first() -> None:
    """Verify original tagged artist is queried first and not replaced by alias if tags exist."""
    queries: list[str] = []

    class AliasTestProvider(BaseMetadataProvider):
        name = "alias_mock"

        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            queries.append(artist)
            if artist == "Snoop Dogg":
                return ["hip hop", "west coast rap"]
            return []

        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            return []

        def fetch_artist_tags(self, artist: str) -> list[str]:
            return []

    mgr = ProviderManager(providers=[AliasTestProvider()])
    raw_tags, track_tags, has_verified, resolved_art = mgr.get_tags_for_track(
        artist="Snoop Dogg",
        title="Drop It Like It's Hot",
    )

    assert has_verified is True
    assert resolved_art == "Snoop Dogg"
    assert "hip hop" in track_tags
    # Ensure Snoop Lion was NEVER queried
    assert "snoop lion" not in [q.lower() for q in queries]


def test_provider_manager_alias_fallback_when_original_empty() -> None:
    """Verify alias is queried as secondary fallback when original artist has 0 verified tags."""
    queries: list[str] = []

    class AliasFallbackProvider(BaseMetadataProvider):
        name = "alias_fallback_mock"

        def fetch_track_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
            queries.append(artist)
            if artist.lower() == "snoop dogg":
                return ["hip hop"]
            return []

        def fetch_album_tags(self, artist: str, album: str) -> list[str]:
            return []

        def fetch_artist_tags(self, artist: str) -> list[str]:
            return []

    mgr = ProviderManager(providers=[AliasFallbackProvider()])
    raw_tags, track_tags, has_verified, resolved_art = mgr.get_tags_for_track(
        artist="Snoop Lion",
        title="Smoke The Weed",
    )

    assert has_verified is True
    assert resolved_art == "snoop dogg"
    assert "hip hop" in track_tags
    # Verify Snoop Lion was queried first, then Snoop Dogg
    assert queries[0] == "Snoop Lion"
    assert any("snoop dogg" == q.lower() for q in queries[1:])
