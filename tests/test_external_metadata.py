"""Unit tests for external API fetchers (Last.fm, MusicBrainz, Discogs) using mocks."""

import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from resonate.modules.external_metadata import (
    DiscogsFetcher,
    MusicBrainzFetcher,
    artist_matches,
)
from resonate.modules.lastfm import LastFmFetcher


def test_artist_matches_compound_band_names() -> None:
    """Test artist_matches with articles, compound names, and mismatches."""
    # Direct match & case insensitivity
    assert artist_matches("Radiohead", "radiohead") is True

    # Leading article 'The'
    assert artist_matches("The Beatles", "Beatles") is True
    assert artist_matches("Beatles", "The Beatles") is True

    # Compound band name 'Jay & Americans' vs 'Jay & The Americans'
    assert artist_matches("Jay & Americans", "Jay & The Americans") is True
    assert artist_matches("Jay & The Americans", "Jay & Americans") is True
    assert artist_matches("Huey Lewis & News", "Huey Lewis & The News") is True
    assert artist_matches("Echo & Bunnymen", "Echo & The Bunnymen") is True

    # Anti-mismatch protection (e.g. Ye must NOT match Yes)
    assert artist_matches("Ye", "Yes") is False
    assert artist_matches("Yes", "Ye") is False
    assert artist_matches("The Who", "The Weeknd") is False


def test_get_artist_aliases_variants() -> None:
    """Test get_artist_aliases generates dictionary and grammatical variants."""
    from resonate.modules.external_metadata import get_artist_aliases

    # Dictionary alias
    ye_aliases = get_artist_aliases("Ye")
    assert "kanye west" in ye_aliases or "Kanye West" in [a.lower() for a in ye_aliases]

    # Compound band name
    jay_aliases = get_artist_aliases("Jay & Americans")
    assert "Jay & The Americans" in jay_aliases

    # Leading The
    beatles_aliases = get_artist_aliases("Beatles")
    assert "The Beatles" in beatles_aliases

    cure_aliases = get_artist_aliases("The Cure")
    assert "Cure" in cure_aliases



# --- Last.fm Tests ---


@patch("urllib.request.urlopen")
def test_lastfm_fetcher_scraping_fallbacks(mock_urlopen):
    """Verify that LastFmFetcher correctly scrapes album and artist tags on fallback."""
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = (
        b"<html><body>"
        b'<a href="/tag/indie+rock">indie rock</a>'
        b'<a href="/tag/post-punk">post-punk</a>'
        b"</body></html>"
    )
    mock_urlopen.return_value.__enter__.return_value = mock_response

    fetcher = LastFmFetcher(api_key=None)

    # 1. Test album scraping
    album_tags = fetcher.get_album_tags("Interpol", "Turn On the Bright Lights")
    assert "indie rock" in album_tags
    assert "post-punk" in album_tags

    # 2. Test artist scraping
    artist_tags = fetcher.get_artist_tags("Interpol")
    assert "indie rock" in artist_tags
    assert "post-punk" in artist_tags


# --- MusicBrainz Tests ---


@patch("urllib.request.urlopen")
def test_musicbrainz_fetcher_happy_path(mock_urlopen):
    """Verify that MusicBrainzFetcher retrieves and parses recording tags correctly."""
    mock_response = MagicMock()
    mock_response.status = 200

    mock_data = {
        "recordings": [
            {
                "id": "rec-123",
                "title": "Obstacle 1",
                "tags": [{"name": "post-punk", "count": 5}],
                "genres": [{"name": "indie rock", "disambiguation": ""}],
            }
        ]
    }
    mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    fetcher = MusicBrainzFetcher()
    tags = fetcher.get_recording_tags("Interpol", "Obstacle 1")

    assert "post-punk" in tags
    assert "indie rock" in tags
    assert len(tags) == 2


@patch("urllib.request.urlopen")
def test_musicbrainz_fetcher_error_path(mock_urlopen):
    """Verify that MusicBrainzFetcher handles API exceptions gracefully."""
    mock_urlopen.side_effect = urllib.error.URLError("HTTP Error 503 Service Unavailable")

    fetcher = MusicBrainzFetcher()
    tags = fetcher.get_recording_tags("Interpol", "Obstacle 1")
    assert tags == []


@patch("urllib.request.urlopen")
def test_musicbrainz_resolve_canonical_artist(mock_urlopen):
    """Verify that MusicBrainzFetcher resolves alias to canonical artist name."""
    mock_response = MagicMock()
    mock_response.status = 200

    mock_data = {
        "artists": [
            {
                "id": "mb-artist-123",
                "name": "Kanye West",
                "score": 100,
                "aliases": [
                    {"name": "Ye", "type": "Legal name"},
                    {"name": "Yeezy", "type": "Search hint"},
                ],
            }
        ]
    }
    mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    fetcher = MusicBrainzFetcher()
    canonical = fetcher.resolve_canonical_artist("Ye")
    assert canonical == "Kanye West"

    # Same name should return None (not an alias rebrand)
    same_name = fetcher.resolve_canonical_artist("Kanye West")
    assert same_name is None


# --- Discogs Tests ---


def test_discogs_fetcher_no_token():
    """Verify that DiscogsFetcher returns empty list if token is missing."""
    fetcher = DiscogsFetcher(api_token=None)
    assert fetcher.get_release_genres("Interpol", "Obstacle 1") == []


@patch("urllib.request.urlopen")
def test_discogs_fetcher_happy_path(mock_urlopen):
    """Verify that DiscogsFetcher searches and returns genre and style lists."""
    mock_response = MagicMock()
    mock_response.status = 200

    mock_data = {"results": [{"genre": ["Rock"], "style": ["Post-Punk", "Indie Rock"]}]}
    mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    fetcher = DiscogsFetcher(api_token="fake-token")
    tags = fetcher.get_release_genres("Interpol", "Obstacle 1")

    assert "Rock" in tags
    assert "Post-Punk" in tags
    assert "Indie Rock" in tags
    assert len(tags) == 3


@patch("urllib.request.urlopen")
def test_musicbrainz_fetcher_with_album_disambiguation(mock_urlopen):
    """Verify that MusicBrainzFetcher queries with release and matches the right album."""
    mock_response = MagicMock()
    mock_response.status = 200

    mock_data = {
        "recordings": [
            {
                "id": "rec-electronic-1",
                "title": "Interlude",
                "artist-credit": [{"name": "Sleepwalkers", "artist": {"name": "Sleepwalkers"}}],
                "releases": [
                    {
                        "title": "Digital Sunrise",
                        "release-group": {"genres": [{"name": "ambient"}, {"name": "electronic"}]},
                    }
                ],
                "tags": [{"name": "downtempo"}],
                "genres": [],
            }
        ]
    }
    mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    fetcher = MusicBrainzFetcher()
    tags = fetcher.get_recording_tags("Sleepwalkers", "Interlude", album="Digital Sunrise")

    # Verify query included release filter
    called_req = mock_urlopen.call_args[0][0]
    has_release = (
        "release%3A%22Digital+Sunrise%22" in called_req.full_url
        or "release%3A%22Digital%20Sunrise%22" in called_req.full_url
    )
    assert has_release
    assert "ambient" in tags
    assert "electronic" in tags
    assert "downtempo" in tags


@patch("urllib.request.urlopen")
def test_discogs_fetcher_with_album(mock_urlopen):
    """Verify that DiscogsFetcher prioritizes album title in search query."""
    mock_response = MagicMock()
    mock_response.status = 200

    mock_data = {"results": [{"genre": ["Electronic"], "style": ["Ambient", "IDM"]}]}
    mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    fetcher = DiscogsFetcher(api_token="fake-token")
    tags = fetcher.get_release_genres("Sleepwalkers", "Interlude", album="Digital Sunrise")

    called_req = mock_urlopen.call_args[0][0]
    has_album_query = (
        "Sleepwalkers+-+Digital+Sunrise" in called_req.full_url
        or "Sleepwalkers%20-%20Digital%20Sunrise" in called_req.full_url
    )
    assert has_album_query
    assert "Electronic" in tags
    assert "Ambient" in tags
    assert "IDM" in tags

