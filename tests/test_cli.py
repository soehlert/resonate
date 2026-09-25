import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

from resonate.main import app
from resonate.models import TrackItem

runner = CliRunner()


def test_cli_help() -> None:
    """Test top-level CLI entrypoint lists available subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "check" in result.output
    assert "clean" in result.output
    assert "setup" in result.output
    assert "status" in result.output
    assert "tune" in result.output


def test_analyze_cmd_sequential_execution(tmp_path: Path) -> None:
    """Test analyze execution sequentially processes all tracks."""
    from unittest.mock import MagicMock, patch

    from resonate.models import TrackEnrichmentResult, TrackItem

    db_path = tmp_path / "test_state.sqlite"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
plex:
  url: "http://mockplex:32400"
  token: "test"
  library_name: "Music"
database:
  sqlite_path: "{db_path}"
processing:
  batch_size: 10
  dry_run: true
""",
        encoding="utf-8",
    )

    track1_file = tmp_path / "1.mp3"
    track2_file = tmp_path / "2.mp3"
    track1_file.touch()
    track2_file.touch()

    track1 = TrackItem(
        rating_key="1",
        title="Track 1",
        artist="Artist A",
        file_path=str(track1_file),
    )
    track2 = TrackItem(
        rating_key="2",
        title="Track 2",
        artist="Artist B",
        file_path=str(track2_file),
    )

    enrich_res = TrackEnrichmentResult(
        rating_key="1",
        title="Track 1",
        artist="Artist A",
        primary_genre="Rock",
        subgenres=["Punk Rock"],
        moods=["Energetic"],
        bpm=120,
    )

    with (
        patch("resonate.cli.analyze.PlexSync") as mock_plex_cls,
        patch("resonate.cli.analyze.EnrichmentPipeline") as mock_pipe_cls,
    ):
        mock_plex = MagicMock()
        mock_plex.fetch_audio_tracks.return_value = [track1, track2]
        mock_plex_cls.return_value = mock_plex

        mock_pipe = MagicMock()
        mock_pipe.enrich_track.return_value = enrich_res
        mock_pipe_cls.return_value = mock_pipe

        result = runner.invoke(
            app,
            ["analyze", "--config", str(config_file), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Total Processed" in result.output
        assert mock_pipe.enrich_track.call_count == 2


def test_cli_tune_status(tmp_path: Path) -> None:
    """Test tune status command with missing and populated model files."""
    missing_model = tmp_path / "missing.json"
    res_empty = runner.invoke(app, ["tune", "status", "--model-path", str(missing_model)])
    assert res_empty.exit_code == 0
    assert "No trained personalized mood model found" in res_empty.output

    # Create dummy model JSON with valid anchor vectors
    dummy_vec = [0.01] * 1280
    payload = {
        "version": "1.0",
        "moods": {
            "Chill Hang": {
                "anchors": [dummy_vec],
                "track_count": 12,
                "coherence": 0.85,
                "threshold": 0.72,
            }
        },
    }
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(payload), encoding="utf-8")
    res_model = runner.invoke(app, ["tune", "status", "--model-path", str(model_file)])
    assert res_model.exit_code == 0
    assert "Chill Hang" in res_model.output
    assert "12" in res_model.output
    assert "0.850" in res_model.output
    assert "0.720" in res_model.output


def test_cli_tune_train_no_playlists(tmp_path: Path) -> None:
    """Test tune train command when Plex has no matching playlists."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
plex:
  url: "http://mockplex:32400"
  token: "test"
  library_name: "Music"
database:
  sqlite_path: "test.db"
processing:
  batch_size: 10
  dry_run: true
""",
        encoding="utf-8",
    )

    with patch("resonate.cli.tune_cmd.PlexSync") as mock_plex_cls:
        mock_plex = MagicMock()
        mock_plex.fetch_mood_anchor_playlists.return_value = {}
        mock_plex_cls.return_value = mock_plex

        result = runner.invoke(
            app,
            ["tune", "train", "--config", str(config_file)],
        )
        assert result.exit_code == 0
        assert "No playlists matching prefix" in result.output


def test_cli_tune_train_success(tmp_path: Path) -> None:
    """Test tune train command successfully calibrates heads from Plex anchor playlists."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
plex:
  url: "http://mockplex:32400"
  token: "test"
  library_name: "Music"
database:
  sqlite_path: "test.db"
processing:
  batch_size: 10
  dry_run: true
""",
        encoding="utf-8",
    )

    audio_file = tmp_path / "song.flac"
    audio_file.write_bytes(b"dummy audio")
    model_file = tmp_path / "calibrated_heads.json"

    dummy_track = TrackItem(
        rating_key="1",
        title="Song A",
        artist="Artist A",
        file_path=str(audio_file),
    )

    with (
        patch("resonate.cli.tune_cmd.PlexSync") as mock_plex_cls,
        patch("resonate.cli.tune_cmd.EssentiaAnalyzer") as mock_essentia_cls,
    ):
        mock_plex = MagicMock()
        mock_plex.fetch_mood_anchor_playlists.return_value = {"Chill Hang": [dummy_track]}
        mock_plex_cls.return_value = mock_plex

        mock_essentia = MagicMock()
        mock_essentia.extract_embeddings.return_value = np.ones(1280, dtype=np.float32)
        mock_essentia_cls.return_value = mock_essentia

        result = runner.invoke(
            app,
            [
                "tune",
                "train",
                "--config",
                str(config_file),
                "--model-path",
                str(model_file),
            ],
        )
        assert result.exit_code == 0
        assert "Chill Hang" in result.output
        assert "Calibrated Personalized Mood Heads" in result.output
        assert model_file.exists()


def test_cli_tune_test_command(tmp_path: Path) -> None:
    """Test tune test command for error path and matched prediction path."""
    # 1. Error path: missing model
    res_no_model = runner.invoke(
        app,
        ["tune", "test", "/fake/path.flac", "--model-path", str(tmp_path / "missing.json")],
    )
    assert res_no_model.exit_code == 0
    assert "No trained personalized mood model found" in res_no_model.output

    # 2. Seed a valid model
    model_file = tmp_path / "test_model.json"
    dummy_vec = [1.0 / (1280**0.5)] * 1280
    model_payload = {
        "version": "1.0",
        "moods": {
            "Chill Hang": {
                "anchors": [dummy_vec],
                "track_count": 5,
                "coherence": 0.88,
                "threshold": 0.70,
            }
        },
    }
    model_file.write_text(json.dumps(model_payload), encoding="utf-8")

    # 3. Error path: audio file not found
    res_no_file = runner.invoke(
        app,
        ["tune", "test", "/nonexistent/path.flac", "--model-path", str(model_file)],
    )
    assert res_no_file.exit_code == 0
    assert "Audio file not found" in res_no_file.output

    # 4. Happy path: audio file exists and matches
    audio_file = tmp_path / "track.flac"
    audio_file.write_bytes(b"dummy")

    with patch("resonate.cli.tune_cmd.EssentiaAnalyzer") as mock_essentia_cls:
        mock_essentia = MagicMock()
        mock_essentia.extract_embeddings.return_value = np.array(dummy_vec, dtype=np.float32)
        mock_essentia_cls.return_value = mock_essentia

        res_match = runner.invoke(
            app,
            ["tune", "test", str(audio_file), "--model-path", str(model_file)],
        )
        assert res_match.exit_code == 0
        assert "Assigned Mood:" in res_match.output
        assert "Matched Personalized Moods" in res_match.output
        assert "★ Assigned" in res_match.output

    # 5. Plex ratingKey lookup
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
plex:
  url: "http://mockplex:32400"
  token: "test"
  library_name: "Music"
""",
        encoding="utf-8",
    )

    with (
        patch("resonate.cli.tune_cmd.PlexSync") as mock_plex_cls,
        patch("resonate.cli.tune_cmd.EssentiaAnalyzer") as mock_essentia_cls,
    ):
        mock_plex = MagicMock()
        mock_plex.fetch_track_by_key.side_effect = lambda rating_key, **kwargs: (
            TrackItem(
                rating_key="16012",
                title="Hanginaround",
                artist="Counting Crows",
                file_path=str(audio_file),
            )
            if str(rating_key) == "16012"
            else None
        )
        mock_plex_cls.return_value = mock_plex

        mock_essentia = MagicMock()
        mock_essentia.extract_embeddings.return_value = np.array(dummy_vec, dtype=np.float32)
        mock_essentia_cls.return_value = mock_essentia

        # 5a. Missing key in Plex
        res_key_missing = runner.invoke(
            app,
            [
                "tune",
                "test",
                "--key",
                "99999",
                "--config",
                str(config_file),
                "--model-path",
                str(model_file),
            ],
        )
        assert res_key_missing.exit_code == 0
        assert "not found in Plex" in res_key_missing.output

        # 5b. Valid key in Plex
        res_key_found = runner.invoke(
            app,
            [
                "tune",
                "test",
                "--key",
                "16012",
                "--config",
                str(config_file),
                "--model-path",
                str(model_file),
            ],
        )
        assert res_key_found.exit_code == 0
        assert "Hanginaround" in res_key_found.output
        assert "Counting Crows" in res_key_found.output
        assert "Matched Personalized Moods" in res_key_found.output
        assert "Chill Hang" in res_key_found.output


def test_cli_tune_test_mood_flag(tmp_path: Path) -> None:
    """Test targeted mood testing (--mood) covering match, rejection, and error paths."""
    dummy_vec = [1.0 / (1280**0.5)] * 1280
    model_file = tmp_path / "test_model.json"
    model_payload = {
        "version": "1.0",
        "moods": {
            "TestMood": {
                "anchors": [dummy_vec],
                "track_count": 5,
                "coherence": 0.88,
                "threshold": 0.70,
            }
        },
    }
    model_file.write_text(json.dumps(model_payload), encoding="utf-8")

    audio_file = tmp_path / "track.flac"
    audio_file.write_bytes(b"dummy")

    with patch("resonate.cli.tune_cmd.EssentiaAnalyzer") as mock_essentia_cls:
        mock_essentia = MagicMock()
        mock_essentia.extract_embeddings.return_value = np.array(dummy_vec, dtype=np.float32)
        mock_essentia_cls.return_value = mock_essentia

        # 1. Unknown mood error
        res_unknown = runner.invoke(
            app,
            [
                "tune",
                "test",
                str(audio_file),
                "--model-path",
                str(model_file),
                "--mood",
                "UnknownMood",
            ],
        )
        assert "not found in trained model" in res_unknown.output

        # 2. Targeted match (case-insensitive)
        res_match = runner.invoke(
            app,
            [
                "tune",
                "test",
                str(audio_file),
                "--model-path",
                str(model_file),
                "--mood",
                "testmood",
            ],
        )
        assert res_match.exit_code == 0
        assert "Target Mood Evaluation: TestMood -> ✓ MATCH" in res_match.output
        assert "★ #1 Assigned Mood" in res_match.output

        # 3. Targeted rejection (high threshold model)
        high_thresh_model = tmp_path / "high_thresh.json"
        high_thresh_model.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "moods": {
                        "TestMood": {
                            "anchors": [dummy_vec],
                            "track_count": 5,
                            "coherence": 0.88,
                            "threshold": 1.05,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        res_reject = runner.invoke(
            app,
            [
                "tune",
                "test",
                str(audio_file),
                "--model-path",
                str(high_thresh_model),
                "--mood",
                "TestMood",
            ],
        )
        assert res_reject.exit_code == 0
        assert "Target Mood Evaluation: TestMood -> ✗ REJECTED" in res_reject.output
        assert "Below 'TestMood' threshold" in res_reject.output


def test_cli_tune_test_batch_playlist(tmp_path: Path) -> None:
    """Test batch mood testing from Plex playlist and negative library sampling."""
    dummy_vec = [1.0 / (1280**0.5)] * 1280
    model_file = tmp_path / "model.json"
    model_file.write_text(
        json.dumps(
            {"version": "1.0", "moods": {"TestMood": {"anchors": [dummy_vec], "threshold": 0.7}}}
        )
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text("plex:\n  url: 'http://localhost'\n  token: 't'\n")

    # 1. No arguments -> usage guidance
    res_no_args = runner.invoke(app, ["tune", "test", "-m", str(model_file)])
    assert "Error: You must provide an audio file" in res_no_args.output

    # 2. Missing playlist handling
    with patch("resonate.cli.tune_cmd.PlexSync") as mock_plex:
        mock_plex.return_value.fetch_mood_playlist_tracks.return_value = (None, [])
        res_missing = runner.invoke(
            app, ["tune", "test", "-m", str(model_file), "-c", str(config_file), "-M", "TestMood"]
        )
        assert "Mood playlist 'TestMood' not found" in res_missing.output

    # 3. Successful batch execution
    cand_file = tmp_path / "c.mp3"
    cand_file.write_bytes(b"x")
    neg_file = tmp_path / "n.mp3"
    neg_file.write_bytes(b"x")
    cand = TrackItem(rating_key="1", title="CandSong", artist="ArtA", file_path=str(cand_file))
    neg = TrackItem(rating_key="2", title="NegSong", artist="ArtB", file_path=str(neg_file))

    with (
        patch("resonate.cli.tune_cmd.PlexSync") as mock_plex,
        patch("resonate.cli.tune_cmd.EssentiaAnalyzer") as mock_es,
    ):
        mock_plex.return_value.fetch_mood_playlist_tracks.return_value = (
            "resonate_testmood",
            [cand],
        )
        mock_plex.return_value.fetch_random_tracks.return_value = [neg]
        mock_es.return_value.extract_embeddings.side_effect = lambda file_path="": (
            np.array(dummy_vec, dtype=np.float32)
            if file_path == str(cand_file)
            else np.array([-1.0 / (1280**0.5)] * 1280, dtype=np.float32)
        )

        res = runner.invoke(
            app, ["tune", "test", "-m", str(model_file), "-c", str(config_file), "-M", "TestMood"]
        )
        assert res.exit_code == 0
        assert "✓ PASS" in res.output
        assert "✓ REJECTED" in res.output
        assert "Candidate Matches" in res.output
        assert "Negative Specificity" in res.output


def test_find_renamed_local_file_resolution(tmp_path: Path) -> None:
    """Verify _find_renamed_local_file finds renamed files in directory."""
    from resonate.cli.analyze import _find_renamed_local_file

    music_dir = tmp_path / "The Bristles" / "Unknown Album"
    music_dir.mkdir(parents=True)
    renamed_file = music_dir / "1 - Fall In.mp3"
    renamed_file.write_bytes(b"dummy")

    old_path = str(music_dir / "The Bristles - 01 - Fall In.mp3")
    found = _find_renamed_local_file(old_path)
    assert found == str(renamed_file)

    # Missing file with no match returns None
    missing_path = str(music_dir / "Unknown Artist - 05 - Nowhere.mp3")
    assert _find_renamed_local_file(missing_path) is None


def test_clean_cmd_triggers_plex_rescan(tmp_path: Path) -> None:
    """Verify clean_cmd triggers Plex library scan when files are modified."""
    from resonate.modules.cleaner import FileCleanResult, TagChange

    test_file = tmp_path / "song.mp3"
    test_file.write_bytes(b"dummy")

    mock_settings = MagicMock()
    mock_settings.plex.url = "http://plex:32400"
    mock_settings.plex.token = "token"
    mock_settings.plex.library_name = "Music"

    with (
        patch("resonate.cli.clean.TagCleaner") as mock_cleaner_cls,
        patch("resonate.modules.plex.PlexSync") as mock_plex_cls,
        patch("resonate.config.load_config", return_value=mock_settings),
    ):
        mock_cleaner = MagicMock()
        mock_cleaner_cls.return_value = mock_cleaner
        mock_cleaner.clean_path.return_value = [
            FileCleanResult(
                file_path=str(test_file),
                changed=True,
                changes=[TagChange(field="title", old_value="Old", new_value="New")],
            )
        ]

        mock_sync = MagicMock()
        mock_sync.connect.return_value = True
        mock_sync.scan_library.return_value = True
        mock_plex_cls.return_value = mock_sync

        result = runner.invoke(app, ["clean", str(test_file)])
        assert result.exit_code == 0
        assert "Plex library rescan triggered successfully" in result.output
        mock_sync.scan_library.assert_called_once()
