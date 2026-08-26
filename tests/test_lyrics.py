"""Pytest unit tests for LyricsFetcher and lyric sentiment/mood analysis."""

from unittest.mock import MagicMock, patch

from resonate.modules.lyrics import (
    LyricsFetcher,
    calculate_valence_score,
    clean_lyrics_text,
)
from resonate.modules.tag_mapper import TagMapper
from resonate.utils.state import StateManager


def test_clean_lyrics_text() -> None:
    """Test cleaning synced timestamps and metadata headers from lyrics text."""
    raw = """[00:12.34] [Verse 1]
[00:15.00]All the other kids with the pumped up kicks
[00:18.50]You better run, better run, outrun my gun
[00:22.00] [Chorus]
[00:25.00]All the other kids with the pumped up kicks
(Outro)
"""
    cleaned = clean_lyrics_text(raw)
    assert "[00:12.34]" not in cleaned
    assert "[Verse 1]" not in cleaned
    assert "[Chorus]" not in cleaned
    assert "(Outro)" not in cleaned
    assert "All the other kids with the pumped up kicks" in cleaned
    assert "outrun my gun" in cleaned


def test_calculate_valence_score() -> None:
    """Test valence scoring on positive, negative, and neutral texts."""
    pos_text = (
        "I love this wonderful sunny day full of joy, happiness and peace, smiling all the way!"
    )
    neg_text = "There is only death, pain, bleeding, bullets and suicide in the dark grave."
    neutral_text = "The bus arrives at four o'clock on the corner of the avenue."

    assert calculate_valence_score(pos_text) > 0.5
    assert calculate_valence_score(neg_text) < -0.5
    assert calculate_valence_score(neutral_text) == 0.0
    assert calculate_valence_score("") == 0.0


def test_lyrics_fetcher_sidecar_file(tmp_path) -> None:
    """Test reading lyrics from sidecar .lrc or .txt files."""
    audio_path = tmp_path / "song.flac"
    audio_path.write_bytes(b"dummy")
    lrc_path = tmp_path / "song.lrc"
    lrc_path.write_text("[00:01.00]Hello world\n[00:05.00]Sidecar lyrics", encoding="utf-8")

    fetcher = LyricsFetcher(prefer_embedded=True)
    extracted = fetcher.extract_embedded_lyrics(str(audio_path))
    assert extracted is not None
    assert "Sidecar lyrics" in extracted


def test_fetch_lrclib_api_get_and_search() -> None:
    """Test fetching from LRCLIB API with mocked responses."""
    fetcher = LyricsFetcher(lrclib_url="https://lrclib.net")

    # 1. Exact match via /api/get
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "plainLyrics": "All the other kids with the pumped up kicks...",
            "syncedLyrics": "[00:10.00]All the other kids...",
        }
        mock_get.return_value = mock_resp

        lyrics = fetcher.fetch_lrclib_lyrics("Foster the People", "Pumped Up Kicks")
        assert lyrics == "All the other kids with the pumped up kicks..."
        assert mock_get.call_count == 1

    # 2. Search fallback
    with patch("requests.get") as mock_get:
        mock_get_fail = MagicMock()
        mock_get_fail.status_code = 404
        mock_search_success = MagicMock()
        mock_search_success.status_code = 200
        mock_search_success.json.return_value = [
            {"plainLyrics": "Found via search fallback lyrics..."}
        ]
        mock_get.side_effect = [mock_get_fail, mock_search_success]

        lyrics = fetcher.fetch_lrclib_lyrics("Some Artist", "Some Song")
        assert lyrics == "Found via search fallback lyrics..."


def test_lyrics_fetcher_orchestrator_caching(tmp_path) -> None:
    """Test coordinating cache -> LRCLIB and state persistence."""
    db_path = tmp_path / "state.sqlite"
    state = StateManager(sqlite_path=str(db_path))
    fetcher = LyricsFetcher(state_manager=state, prefer_embedded=False)

    with patch.object(fetcher, "fetch_lrclib_lyrics") as mock_fetch:
        mock_fetch.return_value = "Test lyrics line 1\nTest lyrics line 2"

        lyrics, source = fetcher.get_lyrics("Artist X", "Song Y")
        assert lyrics == "Test lyrics line 1\nTest lyrics line 2"
        assert source == "lrclib"
        assert mock_fetch.call_count == 1

        # Second call should come from SQLite cache
        lyrics2, source2 = fetcher.get_lyrics("Artist X", "Song Y")
        assert lyrics2 == lyrics
        assert source2 == "cached:lrclib"
        # mock_fetch should not be called again
        assert mock_fetch.call_count == 1


def test_analyze_lyrics_mood_scoring() -> None:
    """Test semantic mood and valence analysis on dark vs joyful lyrics."""
    fetcher = LyricsFetcher()

    mock_model = MagicMock()

    def mock_encode(texts, convert_to_tensor=False):
        res = []
        for t in texts:
            t_low = t.lower()
            dark_words = ["gun", "kill", "die", "death", "dark", "bleeding", "bullet"]
            if any(w in t_low for w in dark_words):
                res.append([0.9, 0.1, 0.0])
            elif any(w in t_low for w in ["happy", "joy", "sun", "peace", "celebrate"]):
                res.append([0.0, 0.9, 0.1])
            else:
                res.append([0.3, 0.3, 0.3])
        return res

    mock_model.encode.side_effect = mock_encode

    mapper = TagMapper(target_moods=["Dark", "Happy", "Chill Hang"], model=mock_model)

    dark_lyrics = (
        "Better run, outrun my gun, faster than my bullet... "
        "He'll look around the room, he won't tell you his plan."
    )
    result_dark = fetcher.analyze_lyrics(dark_lyrics, source="lrclib", tag_mapper=mapper)
    assert result_dark.valence_score < 0
    assert result_dark.mood_scores.get("Dark", 0.0) > 0.4
    assert result_dark.mood_scores.get("Happy", 0.0) < 0.2

    happy_lyrics = "Walking on sunshine, having a wonderful joyful day full of smiles and love!"
    result_happy = fetcher.analyze_lyrics(happy_lyrics, source="embedded", tag_mapper=mapper)
    assert result_happy.valence_score > 0
    assert result_happy.mood_scores.get("Happy", 0.0) > 0.4


def test_lyrics_mood_contrast_pumped_up_kicks() -> None:
    """Test that dark lyrics knock out Happy/Upbeat without destroying Chill Hang."""
    fetcher = LyricsFetcher()

    mock_model = MagicMock()

    def mock_encode(texts, convert_to_tensor=False):
        res = []
        for t in texts:
            dark_words = ["gun", "kill", "die", "death", "dark"]
            if any(w in t.lower() for w in dark_words):
                res.append([0.9, 0.1, 0.0])
            else:
                res.append([0.2, 0.2, 0.2])
        return res

    mock_model.encode.side_effect = mock_encode
    mapper = TagMapper(target_moods=["Dark", "Happy", "Chill Hang", "Upbeat"], model=mock_model)

    pumped_up_lyrics = (
        "Robert's got a quick hand... All the other kids with the pumped up kicks "
        "you better run, outrun my gun, faster than my bullet."
    )
    analysis = fetcher.analyze_lyrics(pumped_up_lyrics, source="lrclib", tag_mapper=mapper)

    # Initial candidate moods from genre/lastfm
    combined_moods = ["Chill Hang", "Upbeat"]

    # Apply lyrical negative valence / darkness filter
    if analysis.valence_score < -0.30 or analysis.mood_scores.get("Dark", 0.0) >= 0.65:
        combined_moods = [m for m in combined_moods if m.lower() not in {"happy", "upbeat"}]

    # Upbeat was disqualified by dark lyrics, but Chill Hang survived!
    assert "Upbeat" not in combined_moods
    assert "Happy" not in combined_moods
    assert "Chill Hang" in combined_moods

