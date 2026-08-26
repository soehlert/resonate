"""Pytest unit tests for Resonate processing pipeline modules."""

from unittest.mock import MagicMock, patch

from resonate.modules.beets import BeetsTagger
from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.lastfm import LastFmFetcher
from resonate.modules.plex import PlexSync
from resonate.modules.tag_mapper import TagMapper
from resonate.utils.state import StateManager


def test_tag_mapper_match_tags() -> None:
    """Test TagMapper match_tags with mocked SentenceTransformer."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, convert_to_tensor=False: [
        [1.0, 0.0] if "chill" in t or "ambient" in t else [0.0, 1.0] for t in texts
    ]

    mapper = TagMapper(
        target_moods=["chill", "energetic"],
        model_name="all-MiniLM-L6-v2",
        model=mock_model,
    )
    best_mood, _, _, score = mapper.match_tags(["ambient"], threshold=0.45)
    assert best_mood == "chill"
    assert score >= 0.45

    mood, _, _, low_score = mapper.match_tags([], threshold=0.45)
    assert mood is None
    assert low_score == 0.0


def test_lastfm_fetcher_caching_and_scraping() -> None:
    """Test LastFmFetcher caching behavior and fallback scraping."""
    fetcher = LastFmFetcher(api_key=None)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = (
            b'<html><a href="/tag/chillout">chillout</a><a href="/tag/ambient">ambient</a></html>'
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        tags1 = fetcher.get_track_tags("Artist", "Track")
        assert "chillout" in tags1
        assert "ambient" in tags1

        # Second call should return cached result without urlopen call
        tags2 = fetcher.get_track_tags("Artist", "Track")
        assert tags2 == tags1
        assert mock_urlopen.call_count == 1


def test_essentia_analyzer_missing_files() -> None:
    """Test EssentiaAnalyzer handling missing model and audio files."""
    analyzer = EssentiaAnalyzer(models_dir="/nonexistent", model_filename="missing.pb")
    moods, score, top = analyzer.analyze_waveform("/nonexistent/song.mp3", ["chill"])
    assert moods == []
    assert score == 0.0
    assert top == []

    genre, subgenres = analyzer.analyze_genre_waveform("/nonexistent/song.mp3")
    assert genre is None
    assert subgenres == []



def test_beets_tagger_dry_run_and_missing() -> None:
    """Test BeetsTagger dry run mode and missing file handling."""
    tagger = BeetsTagger(enabled=True)

    # Missing file returns False
    assert tagger.update_file_mood("/nonexistent/file.mp3", "chill") is False

    # Disabled tagger returns False
    disabled_tagger = BeetsTagger(enabled=False)
    assert disabled_tagger.update_file_mood("/nonexistent/file.mp3", "chill") is False


def test_plex_sync_mock() -> None:
    """Test PlexSync connection handling and dry run update."""
    plex = PlexSync(url="http://localhost:32400", token="fake-token")

    with patch("resonate.modules.plex.PlexServer") as mock_server_cls:
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        mock_library = MagicMock()
        mock_server.library.section.return_value = mock_library

        # Mock track items
        track1 = MagicMock()
        track1.ratingKey = "1"
        track1.title = "Fire Fly"
        track1.grandparentTitle = "Childish Gambino"
        track1.parentTitle = "Camp"
        track1.moods = []
        track1.media = []

        track2 = MagicMock()
        track2.ratingKey = "2"
        track2.title = "Redbone"
        track2.grandparentTitle = "Childish Gambino"
        track2.parentTitle = "Awaken, My Love!"
        track2.moods = []
        track2.media = []

        mock_library.searchTracks.return_value = [track1, track2]

        assert plex.connect() is True
        assert plex.update_track_mood("123", "chill", dry_run=True) is True

        # Test track_title filter
        filtered_tracks = plex.fetch_audio_tracks(artist="Childish Gambino", track_title="Fire Fly")
        assert len(filtered_tracks) == 1
        assert filtered_tracks[0].title == "Fire Fly"



def test_state_manager_lyrics_cache(tmp_path) -> None:
    """Test StateManager caching and retrieval of lyrics."""
    db_path = tmp_path / "test_state.sqlite"
    state = StateManager(sqlite_path=str(db_path))

    # Initially missing
    assert state.get_cached_lyrics("Foster the People", "Pumped Up Kicks") is None

    # Save lyrics
    lyrics_sample = "Robert's got a quick hand / He'll look around the room..."
    state.save_cached_lyrics("Foster the People", "Pumped Up Kicks", lyrics_sample, "lrclib")

    # Case-insensitive / whitespace-insensitive retrieval
    cached = state.get_cached_lyrics("foster the people ", " pumped up kicks")
    assert cached is not None
    assert cached["lyrics_text"] == lyrics_sample
    assert cached["source"] == "lrclib"

    # Blank/missing parameters
    assert state.get_cached_lyrics("", "Song") is None
    assert state.get_cached_lyrics("Artist", "") is None


def test_state_manager_artist_alias_cache(tmp_path) -> None:
    """Test StateManager caching and retrieval of artist aliases."""
    db_path = tmp_path / "test_state.sqlite"
    state = StateManager(sqlite_path=str(db_path))

    # Initially missing
    assert state.get_cached_artist_alias("Ye") is None

    # Save alias
    state.save_cached_artist_alias("Ye", "Kanye West", "musicbrainz")

    # Case-insensitive / whitespace-insensitive retrieval
    assert state.get_cached_artist_alias("ye") == "Kanye West"
    assert state.get_cached_artist_alias("  YE  ") == "Kanye West"

    # Blank parameters
    assert state.get_cached_artist_alias("") is None


