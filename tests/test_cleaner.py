"""Unit tests for TagCleaner module and clean CLI command."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from resonate.main import app
from resonate.modules.cleaner import (
    TagCleaner,
    clean_whitespace_and_artifacts,
    normalize_track_number,
)

runner = CliRunner()


def test_normalize_track_number() -> None:
    """Test track number formatting, leading zero stripping, and disc extraction."""
    assert normalize_track_number("01") == ("1", None)
    assert normalize_track_number("09") == ("9", None)
    assert normalize_track_number("01/12") == ("1/12", None)
    assert normalize_track_number("1/0") == ("1", None)
    assert normalize_track_number("01/00") == ("1", None)
    assert normalize_track_number("1-04") == ("4", "1")
    assert normalize_track_number("2-12") == ("12", "2")
    assert normalize_track_number("12/12") == ("12/12", None)
    assert normalize_track_number("") == (None, None)
    assert normalize_track_number(None) == (None, None)


def test_clean_whitespace_and_artifacts() -> None:
    """Test whitespace collapse, null character removal, and dangling bracket stripping."""
    assert clean_whitespace_and_artifacts("  Album   Title  ") == "Album Title"
    assert clean_whitespace_and_artifacts("Track ()") == "Track"
    assert clean_whitespace_and_artifacts("Track [ ]") == "Track"
    assert clean_whitespace_and_artifacts("Track\x00") == "Track"
    assert clean_whitespace_and_artifacts(None) is None


@patch("mutagen.File")
def test_tag_cleaner_clean_file_dry_run(mock_mutagen_file, tmp_path) -> None:
    """Test cleaning tags on a single file in dry-run mode."""
    test_file = tmp_path / "01 - Hatef--k.mp3"
    test_file.write_bytes(b"dummy audio content")

    mock_audio = {
        "album": ["Stir The Blood (Best Buy Exclusive)"],
        "title": ["Hatef--k"],
        "tracknumber": ["01/12"],
        "artist": ["The Bravery  "],
    }
    mock_audio_obj = MagicMock()
    mock_audio_obj.__getitem__.side_effect = mock_audio.__getitem__
    mock_audio_obj.__setitem__.side_effect = mock_audio.__setitem__
    mock_audio_obj.__contains__.side_effect = mock_audio.__contains__
    mock_mutagen_file.return_value = mock_audio_obj

    cleaner = TagCleaner()
    result = cleaner.clean_file(str(test_file), dry_run=True)

    assert result.changed is True
    assert len(result.changes) == 4
    # Dry run must not save to disk
    assert mock_audio_obj.save.call_count == 0

    change_dict = {c.field: c.new_value for c in result.changes}
    assert change_dict["Album"] == "Stir The Blood"
    assert change_dict["Title"] == "Hatefuck"
    assert change_dict["Track"] == "1/12"
    assert change_dict["Artist"] == "The Bravery"


@patch("mutagen.File")
def test_tag_cleaner_clean_file_live_save(mock_mutagen_file, tmp_path) -> None:
    """Test cleaning tags on a single file with live save."""
    test_file = tmp_path / "song.flac"
    test_file.write_bytes(b"dummy audio content")

    mock_audio = {
        "album": ["Nevermind [Target Exclusive]"],
        "title": ["Smells Like Teen Spirit"],
        "tracknumber": ["1-01"],
    }
    mock_audio_obj = MagicMock()
    mock_audio_obj.__getitem__.side_effect = mock_audio.__getitem__
    mock_audio_obj.__setitem__.side_effect = mock_audio.__setitem__
    mock_audio_obj.__contains__.side_effect = mock_audio.__contains__
    mock_mutagen_file.return_value = mock_audio_obj

    cleaner = TagCleaner()
    result = cleaner.clean_file(str(test_file), dry_run=False)

    assert result.changed is True
    assert mock_audio_obj.save.call_count == 1

    change_dict = {c.field: c.new_value for c in result.changes}
    assert change_dict["Album"] == "Nevermind"
    assert change_dict["Track"] == "1"
    assert change_dict["Disc"] == "1"


def test_clean_cli_command_dry_run(tmp_path) -> None:
    """Test CLI 'resonate clean' command execution."""
    test_dir = tmp_path / "album_dir"
    test_dir.mkdir()
    song_file = test_dir / "01 - Hatef--k.mp3"
    song_file.write_bytes(b"dummy")

    with patch("mutagen.File") as mock_mutagen_file:
        mock_audio = {
            "album": ["Stir The Blood (Best Buy Exclusive)"],
            "title": ["Hatef--k"],
            "tracknumber": ["01"],
        }
        mock_audio_obj = MagicMock()
        mock_audio_obj.__getitem__.side_effect = mock_audio.__getitem__
        mock_audio_obj.__setitem__.side_effect = mock_audio.__setitem__
        mock_audio_obj.__contains__.side_effect = mock_audio.__contains__
        mock_mutagen_file.return_value = mock_audio_obj

        result = runner.invoke(app, ["clean", str(test_dir), "--dry-run"])
        assert result.exit_code == 0
        assert "Tag Cleanup Summary" in result.stdout
        assert "Stir The Blood" in result.stdout
        assert "Hatefuck" in result.stdout
        assert "DRY-RUN ACTIVE" in result.stdout
