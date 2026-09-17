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
    assert analyzer.load_audio("/nonexistent/song.mp3") is None
    assert analyzer.extract_embeddings(file_path="/nonexistent/song.mp3") is None
    moods, score, top = analyzer.predict_moods(None, ["chill"])
    assert moods == []
    assert score == 0.0
    assert top == []

    genre, subgenres = analyzer.analyze_genre_waveform("/nonexistent/song.mp3")
    assert genre is None
    assert subgenres == []


def test_essentia_analyzer_predictor_caching(tmp_path) -> None:
    """Verify EssentiaAnalyzer caches compiled predictors and metadata across calls."""
    import numpy as np

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    emb_model = models_dir / "discogs-effnet-bs64-1.pb"
    emb_model.write_bytes(b"x" * 20000)
    head_model = models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    head_model.write_bytes(b"y" * 20000)
    json_meta = models_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
    json_meta.write_text(
        '{"classes": ["energetic", "dark", "happy"], "schema": '
        '{"inputs": [{"name": "input_1"}], "outputs": [{"name": "output_1", '
        '"output_purpose": "predictions"}]}}',
        encoding="utf-8",
    )
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"dummy")

    analyzer = EssentiaAnalyzer(
        models_dir=str(models_dir),
        model_filename="mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    )

    mock_emb_inst = MagicMock(return_value=[[0.1, 0.2]])
    mock_head_inst = MagicMock(return_value=np.array([[0.8, 0.1, 0.1]]))

    mock_es = MagicMock()
    mock_es.TensorflowPredictEffnetDiscogs.return_value = mock_emb_inst
    mock_es.TensorflowPredict2D.return_value = mock_head_inst

    mock_pkg = MagicMock()
    mock_pkg.standard = mock_es

    dummy_audio = np.zeros(16000, dtype=np.float32)

    with patch.dict("sys.modules", {"essentia": mock_pkg, "essentia.standard": mock_es}):
        # Call 1: compiles models and caches
        embs1 = analyzer.extract_embeddings(audio=dummy_audio)
        moods1, score1, _ = analyzer.predict_moods(embs1, ["Energetic"])
        assert mock_es.TensorflowPredictEffnetDiscogs.call_count == 1
        assert mock_es.TensorflowPredict2D.call_count == 1

        # Call 2: must reuse cached predictor instances without re-compiling!
        embs2 = analyzer.extract_embeddings(audio=dummy_audio)
        moods2, score2, _ = analyzer.predict_moods(embs2, ["Energetic"])
        assert mock_es.TensorflowPredictEffnetDiscogs.call_count == 1
        assert mock_es.TensorflowPredict2D.call_count == 1
        assert moods2 == moods1


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


def test_plex_compilation_track_artist() -> None:
    """Verify PlexSync prioritizes track-specific originalTitle on compilation albums."""
    plex = PlexSync(url="http://localhost:32400", token="fake-token")

    with patch("resonate.modules.plex.PlexServer") as mock_server_cls:
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server
        mock_library = MagicMock()
        mock_server.library.section.return_value = mock_library

        # Compilation track: Album artist is Various Artists, track artist is Eminem
        comp_track = MagicMock()
        comp_track.ratingKey = "4328"
        comp_track.title = "Lose Yourself"
        comp_track.originalTitle = "Eminem"
        comp_track.grandparentTitle = "Various Artists"
        comp_track.parentTitle = "8 Mile: Music From and Inspired by the Motion Picture"
        comp_track.moods = []
        comp_track.media = []

        mock_library.searchTracks.return_value = [comp_track]

        # 1. Fetch tracks without filter -> artist is Eminem, album_artist is Various Artists
        tracks = plex.fetch_audio_tracks()
        assert len(tracks) == 1
        assert tracks[0].artist == "Eminem"
        assert tracks[0].album_artist == "Various Artists"
        assert tracks[0].title == "Lose Yourself"
        assert tracks[0].album == "8 Mile: Music From and Inspired by the Motion Picture"

        # 2. Filter by track artist "Eminem" -> matches
        filtered_by_track = plex.fetch_audio_tracks(artist="Eminem")
        assert len(filtered_by_track) == 1
        assert filtered_by_track[0].artist == "Eminem"

        # 3. Filter by album artist "Various Artists" -> also matches
        filtered_by_album_artist = plex.fetch_audio_tracks(artist="Various Artists")
        assert len(filtered_by_album_artist) == 1
        assert filtered_by_album_artist[0].artist == "Eminem"


def test_state_manager_self_healing_compilation_aliases(tmp_path) -> None:
    """Verify StateManager automatically purges contaminated compilation aliases on init."""
    db_path = tmp_path / "test_state_healing.sqlite"

    # Manually seed a database with contaminated rows
    state = StateManager(sqlite_path=str(db_path))
    state.save_cached_artist_alias("Ye", "Kanye West", "musicbrainz")
    with state._get_connection() as conn:
        conn.execute(
            "INSERT INTO artist_aliases (raw_artist, canonical_artist, source) VALUES (?, ?, ?)",
            ("Various Artists", "Разни изведувачи", "musicbrainz"),
        )
        conn.commit()

    # Verify both exist before re-init
    assert state.get_cached_artist_alias("Ye") == "Kanye West"
    assert state.get_cached_artist_alias("Various Artists") == "Разни изведувачи"

    # Re-initialize StateManager (simulating startup)
    reloaded_state = StateManager(sqlite_path=str(db_path))

    # Legitimate alias remains, contaminated compilation alias is purged
    assert reloaded_state.get_cached_artist_alias("Ye") == "Kanye West"
    assert reloaded_state.get_cached_artist_alias("Various Artists") is None


def test_state_manager_batch_operations(tmp_path) -> None:
    """Verify StateManager handles sequential batch reads and writes without errors."""
    db_path = tmp_path / "test_batch_state.sqlite"
    state = StateManager(sqlite_path=str(db_path))

    for i in range(25):
        artist = f"Artist_{i}"
        title = f"Title_{i}"
        state.save_cached_track_tags(artist, title, [f"tag_{i}", "rock"])
        tags = state.get_cached_track_tags(artist, title)
        assert tags == [f"tag_{i}", "rock"]
        state.save_cached_lyrics(artist, title, f"Lyrics for {title}", "lrclib")
        lyric_data = state.get_cached_lyrics(artist, title)
        assert lyric_data is not None
        assert lyric_data["source"] == "lrclib"

    state.close()


def test_plex_fetch_mood_anchor_playlists() -> None:
    """Verify PlexSync auto-discovers resonate_* playlists and maps canonical moods."""
    plex = PlexSync(url="http://localhost:32400", token="fake-token")

    mock_track = MagicMock()
    mock_track.ratingKey = 1234
    mock_track.title = "Texas Sun"
    mock_track.originalTitle = "Khruangbin"
    mock_track.grandparentTitle = "Khruangbin"
    mock_track.parentTitle = "Texas Sun EP"
    mock_track.moods = []
    mock_part = MagicMock()
    mock_part.file = "/data/music/Khruangbin/Texas_Sun.flac"
    mock_media = MagicMock()
    mock_media.parts = [mock_part]
    mock_track.media = [mock_media]

    mock_pl1 = MagicMock()
    mock_pl1.title = "resonate_chill_hang"
    mock_pl1.items.return_value = [mock_track]

    mock_pl2 = MagicMock()
    mock_pl2.title = "Other Playlist"
    mock_pl2.items.return_value = []

    mock_server = MagicMock()
    mock_server.playlists.return_value = [mock_pl1, mock_pl2]

    with patch("resonate.modules.plex.PlexServer", return_value=mock_server):
        res = plex.fetch_mood_anchor_playlists(
            prefix="resonate_",
            path_map_source="/data/music",
            path_map_target="/music",
        )

        assert "Chill Hang" in res
        assert len(res["Chill Hang"]) == 1
        item = res["Chill Hang"][0]
        assert item.title == "Texas Sun"
        assert item.artist == "Khruangbin"
        assert item.file_path == "/music/Khruangbin/Texas_Sun.flac"
        assert "Other Playlist" not in res
