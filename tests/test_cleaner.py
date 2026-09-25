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


def test_clean_cli_command_selective_flags(tmp_path) -> None:
    """Test CLI 'resonate clean' selective rule isolation (e.g. only --whitespace)."""
    test_dir = tmp_path / "album_dir2"
    test_dir.mkdir()
    song_file = test_dir / "01 - Hatef--k.mp3"
    song_file.write_bytes(b"dummy")

    with patch("mutagen.File") as mock_mutagen_file:
        mock_audio = {
            "album": ["Stir The Blood (Best Buy Exclusive)"],
            "title": ["Hatef--k"],
            "tracknumber": ["01"],
            "artist": ["The Bravery  "],
        }
        mock_audio_obj = MagicMock()
        mock_audio_obj.__getitem__.side_effect = mock_audio.__getitem__
        mock_audio_obj.__setitem__.side_effect = mock_audio.__setitem__
        mock_audio_obj.__contains__.side_effect = mock_audio.__contains__
        mock_mutagen_file.return_value = mock_audio_obj

        # 1. Test running ONLY --whitespace (clean artist whitespace, but NOT uncensor)
        res_ws = runner.invoke(app, ["clean", str(test_dir), "--whitespace", "--dry-run"])
        assert res_ws.exit_code == 0
        assert "Active rules: whitespace" in res_ws.stdout
        assert "The Bravery" in res_ws.stdout
        assert "Hatefuck" not in res_ws.stdout

        # 2. Test running ONLY --uncensor (uncensor title, but NOT clean track or whitespace)
        res_un = runner.invoke(app, ["clean", str(test_dir), "--uncensor", "--dry-run"])
        assert res_un.exit_code == 0
        assert "Active rules: uncensor" in res_un.stdout
        assert "Hatefuck" in res_un.stdout

        # 3. Test running with --no-uncensor (clean retailer noise, but NOT uncensor)
        res_no = runner.invoke(app, ["clean", str(test_dir), "--no-uncensor", "--dry-run"])
        assert res_no.exit_code == 0
        assert "Active rules: retailer-tags, track-numbers, whitespace" in res_no.stdout
        assert "Stir The Blood" in res_no.stdout
        assert "Hatefuck" not in res_no.stdout


def test_inspect_file_tags_and_check_cli(tmp_path) -> None:
    """Test inspect_file_tags, inspect_path, and CLI 'resonate check'."""
    from resonate.modules.cleaner import inspect_file_tags

    test_dir = tmp_path / "inspect_album"
    test_dir.mkdir()
    song_file = test_dir / "01 - Hatefuck.mp3"
    song_file.write_bytes(b"dummy")

    with patch("mutagen.File") as mock_mutagen_file:
        mock_audio = {
            "album": ["Stir The Blood"],
            "title": ["Hatefuck"],
            "artist": ["The Bravery"],
            "tracknumber": ["04"],
            "genre": ["Indie Rock", "Post-Punk"],
            "mood": ["Energetic"],
            "bpm": ["128"],
        }
        mock_audio_obj = MagicMock()
        mock_audio_obj.__getitem__.side_effect = mock_audio.__getitem__
        mock_audio_obj.get.side_effect = mock_audio.get
        mock_audio_obj.keys.return_value = ["TALB", "TIT2", "TPE1", "TCON", "TBPM"]
        mock_mutagen_file.return_value = mock_audio_obj

        # 1. Test programmatic inspection
        res = inspect_file_tags(str(song_file))
        assert res.error is None
        assert res.tags["album"] == "Stir The Blood"
        assert res.tags["title"] == "Hatefuck"
        assert res.tags["artist"] == "The Bravery"

        # 2. Test tag filter (e.g. only album)
        res_filter = inspect_file_tags(str(song_file), tag_filters=["album"])
        assert "album" in res_filter.tags
        assert "title" not in res_filter.tags

        # 3. Test CLI default table
        cli_res = runner.invoke(app, ["check", str(test_dir)])
        assert cli_res.exit_code == 0
        assert "Metadata Tags" in cli_res.stdout
        assert "Bravery" in cli_res.stdout
        assert "Hatefuck" in cli_res.stdout

        # 4. Test CLI with --tag album
        cli_res_tag = runner.invoke(app, ["check", str(test_dir), "--tag", "album"])
        assert cli_res_tag.exit_code == 0
        assert "Tag Check (album)" in cli_res_tag.stdout
        assert "Stir The Blood" in cli_res_tag.stdout

        # 5. Test CLI with --raw
        cli_res_raw = runner.invoke(app, ["check", str(test_dir), "--raw"])
        assert cli_res_raw.exit_code == 0
        assert "Raw Tags" in cli_res_raw.stdout


def test_sanitize_filename_component_and_format_audio_filename() -> None:
    """Test filename component sanitization and '{track} - {title}.ext' formatting."""
    from resonate.modules.cleaner import format_audio_filename, sanitize_filename_component

    # 1. Sanitization
    assert sanitize_filename_component("Hatefuck") == "Hatefuck"
    assert sanitize_filename_component("Song: Part 1") == "Song - Part 1"
    assert sanitize_filename_component("AC/DC - Highway") == "AC-DC - Highway"
    assert sanitize_filename_component('Track "Name"?') == "Track 'Name'"
    assert sanitize_filename_component("Track\x00") == "Track"

    # 2. Filename formatting with NO leading zeros and hyphen separator
    assert format_audio_filename(track="04", title="Hatefuck", ext=".mp3") == "4 - Hatefuck.mp3"
    assert format_audio_filename(track="01/12", title="Adored", ext=".mp3") == "1 - Adored.mp3"
    assert (
        format_audio_filename(track="10/11", title="Jack-O'-Lantern Man", ext=".mp3")
        == "10 - Jack-O'-Lantern Man.mp3"
    )
    assert format_audio_filename(track=None, title="Hatefuck", ext=".mp3") == "Hatefuck.mp3"
    assert (
        format_audio_filename(track="04", title="Hatefuck", ext=".mp3", disc="2")
        == "2-4 - Hatefuck.mp3"
    )


def test_clean_cli_command_rename_files(tmp_path) -> None:
    """Test CLI 'resonate clean --rename-files' renames files on disk."""
    test_dir = tmp_path / "rename_album"
    test_dir.mkdir()
    song_file = test_dir / "04 Hatef--k.mp3"
    song_file.write_bytes(b"dummy audio content")

    with patch("mutagen.File") as mock_mutagen_file:
        mock_audio = {
            "album": ["Stir The Blood (Best Buy Exclusive)"],
            "title": ["Hatef--k"],
            "tracknumber": ["04"],
        }
        mock_audio_obj = MagicMock()
        mock_audio_obj.__getitem__.side_effect = mock_audio.__getitem__
        mock_audio_obj.__setitem__.side_effect = mock_audio.__setitem__
        mock_audio_obj.__contains__.side_effect = mock_audio.__contains__
        mock_audio_obj.get.side_effect = mock_audio.get
        mock_mutagen_file.return_value = mock_audio_obj

        # 1. Dry run should preview renaming without renaming the file on disk
        res_dry = runner.invoke(app, ["clean", str(test_dir), "--rename-files", "--dry-run"])
        assert res_dry.exit_code == 0
        assert "4 - Hatefuck.mp3" in res_dry.stdout
        assert song_file.exists()

        # 2. Live run should rename the file on disk
        res_live = runner.invoke(app, ["clean", str(test_dir), "--rename-files"])
        assert res_live.exit_code == 0
        assert "4 - Hatefuck.mp3" in res_live.stdout
        renamed_file = test_dir / "4 - Hatefuck.mp3"
        assert renamed_file.exists()
        assert not song_file.exists()


def test_parse_audio_path_fallback() -> None:
    """Test path and filename metadata parsing across multiple conventions."""
    from resonate.modules.cleaner import parse_audio_path_fallback

    # 1. Artist - Track - Title
    res1 = parse_audio_path_fallback("/music/The Bristles/The Bristles - 01 - Fall In.mp3")
    assert res1 == {"disc": None, "track": "1", "title": "Fall In"}

    # 2. Track - Title
    res2 = parse_audio_path_fallback("/music/02 - San Francisco Drive.mp3")
    assert res2 == {"disc": None, "track": "2", "title": "San Francisco Drive"}

    # 3. Track. Title and Track Title
    res3 = parse_audio_path_fallback("/music/03. Chapters.mp3")
    assert res3 == {"disc": None, "track": "3", "title": "Chapters"}

    res4 = parse_audio_path_fallback("/music/04 On & On.mp3")
    assert res4 == {"disc": None, "track": "4", "title": "On & On"}

    # 4. Disc-Track - Title
    res5 = parse_audio_path_fallback("/music/1-05 - Galen Cisco.mp3")
    assert res5 == {"disc": "1", "track": "5", "title": "Galen Cisco"}

    # 5. Parent directory disc detection
    res6 = parse_audio_path_fallback("/music/Album/Disc 2/06 - End Of 23.mp3")
    assert res6 == {"disc": "2", "track": "6", "title": "End Of 23"}

    # 6. Title only
    res7 = parse_audio_path_fallback("/music/Fill in the Blanks.mp3")
    assert res7 == {"disc": None, "track": None, "title": "Fill in the Blanks"}

    # 7. Numerical title (e.g. 1984)
    res8 = parse_audio_path_fallback("/music/01 - 1984.mp3")
    assert res8 == {"disc": None, "track": "1", "title": "1984"}

    # 8. Bare track number without title
    res9 = parse_audio_path_fallback("/music/01.mp3")
    assert res9 == {"disc": None, "track": "1", "title": None}


def test_format_audio_filename_no_untitled() -> None:
    """Test that format_audio_filename returns None instead of 'Untitled' when title is absent."""
    from resonate.modules.cleaner import format_audio_filename

    assert format_audio_filename(track="01", title=None, ext=".mp3") is None
    assert format_audio_filename(track="01", title="", ext=".mp3") is None
    assert format_audio_filename(track="01", title="   ", ext=".mp3") is None
    assert format_audio_filename(track=None, title=None, ext=".mp3") is None


def test_clean_file_path_fallback_when_metadata_missing(tmp_path) -> None:
    """Test that TagCleaner uses path fallback when audio tags are missing."""
    test_dir = tmp_path / "the_bristles"
    test_dir.mkdir()
    song_file = test_dir / "The Bristles - 01 - Fall In.mp3"
    song_file.write_bytes(b"dummy")

    with patch("mutagen.File") as mock_mutagen_file:
        # Simulate mutagen returning empty tags (no title, no tracknumber)
        mock_audio_obj = MagicMock()
        mock_audio_obj.__contains__.return_value = False
        mock_audio_obj.get.return_value = None
        mock_mutagen_file.return_value = mock_audio_obj

        cleaner = TagCleaner(rename_files=True)
        result = cleaner.clean_file(str(song_file), dry_run=True)

        assert result.changed is True
        assert len(result.changes) == 1
        assert result.changes[0].field == "Filename"
        assert result.changes[0].old_value == "The Bristles - 01 - Fall In.mp3"
        assert result.changes[0].new_value == "1 - Fall In.mp3"


def test_clean_file_collision_guard_existing_and_batch(tmp_path) -> None:
    """Test collision guard preventing overwriting existing files or in-batch conflicts."""
    test_dir = tmp_path / "collision_dir"
    test_dir.mkdir()

    # Pre-existing target file on disk
    existing_target = test_dir / "1 - Fall In.mp3"
    existing_target.write_bytes(b"existing content")

    source_file = test_dir / "The Bristles - 01 - Fall In.mp3"
    source_file.write_bytes(b"source content")

    with patch("mutagen.File") as mock_mutagen_file:
        mock_audio_obj = MagicMock()
        mock_audio_obj.__contains__.return_value = False
        mock_audio_obj.get.return_value = None
        mock_mutagen_file.return_value = mock_audio_obj

        cleaner = TagCleaner(rename_files=True)

        # 1. Existing file on disk prevents rename
        res_existing = cleaner.clean_file(str(source_file), dry_run=True)
        assert res_existing.changed is False
        assert len(res_existing.changes) == 0

        # 2. In-batch collision prevents second file claiming same target
        claimed: set[str] = set()
        file_a = test_dir / "TrackA.mp3"
        file_b = test_dir / "TrackB.mp3"
        file_a.write_bytes(b"a")
        file_b.write_bytes(b"b")

        mock_audio_custom = {
            "title": ["Same Title"],
            "tracknumber": ["1"],
        }
        mock_audio_obj.__getitem__.side_effect = mock_audio_custom.__getitem__
        mock_audio_obj.__setitem__.side_effect = mock_audio_custom.__setitem__
        mock_audio_obj.__contains__.side_effect = mock_audio_custom.__contains__
        mock_audio_obj.get.side_effect = mock_audio_custom.get

        res_a = cleaner.clean_file(str(file_a), dry_run=True, claimed_targets=claimed)
        assert res_a.changed is True
        fn_changes_a = [c for c in res_a.changes if c.field == "Filename"]
        assert len(fn_changes_a) == 1
        assert fn_changes_a[0].new_value == "1 - Same Title.mp3"

        # Second file with same target should be blocked from clashing
        res_b = cleaner.clean_file(str(file_b), dry_run=True, claimed_targets=claimed)
        fn_changes_b = [c for c in res_b.changes if c.field == "Filename"]
        assert len(fn_changes_b) == 0


def test_clean_cli_path_fallback_user_scenario(tmp_path) -> None:
    """Test CLI clean --rename-files correctly infers filenames from paths without 'Untitled'."""
    album_dir = tmp_path / "The Bristles"
    album_dir.mkdir()

    f1 = album_dir / "The Bristles - 01 - Fall In.mp3"
    f2 = album_dir / "The Bristles - 02 - San Francisco Drive.mp3"
    f3 = album_dir / "Fill in the Blanks.mp3"
    f1.write_bytes(b"dummy1")
    f2.write_bytes(b"dummy2")
    f3.write_bytes(b"dummy3")

    with patch("mutagen.File") as mock_mutagen_file:

        def mock_file_loader(path, easy=True):
            obj = MagicMock()
            if "Fill in the Blanks" in path:
                tags = {"title": ["Fill in the Blanks"], "tracknumber": ["25"]}
            else:
                # Bristles tracks have empty/missing tags
                tags = {}
            obj.__contains__.side_effect = tags.__contains__
            obj.__getitem__.side_effect = tags.__getitem__
            obj.__setitem__.side_effect = tags.__setitem__
            obj.get.side_effect = tags.get
            return obj

        mock_mutagen_file.side_effect = mock_file_loader

        res = runner.invoke(app, ["clean", str(album_dir), "--rename-files", "--dry-run"])
        assert res.exit_code == 0
        assert "Untitled" not in res.stdout
        assert "1 - Fall In.mp3" in res.stdout
        assert "San Francisco" in res.stdout
        assert "Drive.mp3" in res.stdout
        assert "Blanks.mp3" in res.stdout
