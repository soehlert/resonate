"""Unit tests for MutagenTagger using mocks to avoid writing to real files."""

from unittest.mock import MagicMock, patch

from resonate.modules.mutagen import MutagenTagger


@patch("os.path.exists")
def test_mutagen_tagger_disabled_or_not_found(mock_exists):
    """Verify tagger returns False if disabled or file does not exist."""
    # Disabled tagger
    tagger = MutagenTagger(enabled=False)
    assert not tagger.update_file_tags("/fake/song.mp3", genres=["Rock"])

    # File not found
    tagger = MutagenTagger(enabled=True)
    mock_exists.return_value = False
    assert not tagger.update_file_tags("/non/existent.mp3", genres=["Rock"])


@patch("os.path.exists")
@patch("mutagen.id3.ID3")
def test_update_mp3_happy_path(mock_id3_cls, mock_exists):
    """Verify that MP3 ID3 tags are correctly set on a file."""
    mock_exists.return_value = True

    mock_audio = MagicMock()
    mock_audio.get.return_value = None  # No existing tags
    mock_id3_cls.return_value = mock_audio

    tagger = MutagenTagger(enabled=True)
    success = tagger.update_file_tags(
        "/fake/song.mp3",
        genres=["Rock", "Grunge"],
        moods=["Energetic"],
        bpm=120,
        overwrite_tags=False,
    )

    assert success
    mock_audio.save.assert_called_once_with("/fake/song.mp3")

    # Assert tag assignments
    assert "TCON" in mock_audio.__setitem__.call_args_list[0][0][0]
    assert "TMOO" in mock_audio.__setitem__.call_args_list[1][0][0]
    assert "TXXX:MOOD" in mock_audio.__setitem__.call_args_list[2][0][0]
    assert "TBPM" in mock_audio.__setitem__.call_args_list[3][0][0]


@patch("os.path.exists")
@patch("mutagen.flac.FLAC")
def test_update_flac_happy_path(mock_flac_cls, mock_exists):
    """Verify that FLAC comments are correctly set on a file."""
    mock_exists.return_value = True

    mock_audio = MagicMock()
    mock_audio.get.return_value = None
    mock_flac_cls.return_value = mock_audio

    tagger = MutagenTagger(enabled=True)
    success = tagger.update_file_tags(
        "/fake/song.flac",
        genres=["Electronic", "House"],
        moods=["Chill Hang"],
        bpm=125,
        overwrite_tags=False,
    )

    assert success
    mock_audio.save.assert_called_once()
    mock_audio.__setitem__.assert_any_call("genre", ["Electronic", "House"])
    mock_audio.__setitem__.assert_any_call("mood", ["Chill Hang"])
    mock_audio.__setitem__.assert_any_call("bpm", ["125"])


@patch("os.path.exists")
@patch("mutagen.mp4.MP4")
def test_update_mp4_happy_path(mock_mp4_cls, mock_exists):
    """Verify that MP4 metadata atoms are correctly set on a file."""
    mock_exists.return_value = True

    mock_audio = MagicMock()
    mock_audio.get.return_value = None
    mock_mp4_cls.return_value = mock_audio

    tagger = MutagenTagger(enabled=True)
    success = tagger.update_file_tags(
        "/fake/song.m4a", genres=["Pop"], moods=["Lively"], bpm=110, overwrite_tags=False
    )

    assert success
    mock_audio.save.assert_called_once()
    mock_audio.__setitem__.assert_any_call("\xa9gen", ["Pop"])
    mock_audio.__setitem__.assert_any_call("----:com.apple.iTunes:mood", [b"Lively"])
    mock_audio.__setitem__.assert_any_call("tmpo", [110])


@patch("os.path.exists")
@patch("mutagen.flac.FLAC")
def test_mutagen_overwrite_behavior(mock_flac_cls, mock_exists):
    """Verify that MutagenTagger respects the overwrite_tags flag."""
    mock_exists.return_value = True

    # Existing tags are present
    mock_audio = MagicMock()
    mock_audio.get.side_effect = lambda key, default=None: ["ExistingValue"]
    mock_flac_cls.return_value = mock_audio

    tagger = MutagenTagger(enabled=True)

    # Do not overwrite
    success_no_overwrite = tagger.update_file_tags(
        "/fake/song.flac", genres=["NewGenre"], overwrite_tags=False
    )
    assert not success_no_overwrite
    mock_audio.save.assert_not_called()

    # Overwrite
    success_overwrite = tagger.update_file_tags(
        "/fake/song.flac", genres=["NewGenre"], overwrite_tags=True
    )
    assert success_overwrite
    mock_audio.save.assert_called_once()
    mock_audio.__setitem__.assert_any_call("genre", ["NewGenre"])
