import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

from resonate.main import app
from resonate.models import TrackItem

runner = CliRunner()


def test_cli_help() -> None:
    """Test top-level CLI help command lists all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "check" in result.output
    assert "clean" in result.output
    assert "setup" in result.output
    assert "status" in result.output
    assert "tune" in result.output


def test_cli_analyze_help() -> None:
    """Test analyze subcommand help lists options."""
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--write-id3" in result.output
    assert "--write-plex" in result.output
    assert "--limit" in result.output
    assert "-l" in result.output


def test_cli_clean_help() -> None:
    """Test clean subcommand help lists options."""
    result = runner.invoke(app, ["clean", "--help"])
    assert result.exit_code == 0
    assert "--retailer-tags" in result.output
    assert "--uncensor" in result.output
    assert "--rename-files" in result.output


def test_cli_check_help() -> None:
    """Test check subcommand help lists options."""
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    assert "--raw" in result.output
    assert "--tag" in result.output


def test_cli_setup_help() -> None:
    """Test setup subcommand help lists options."""
    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


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


def test_cli_tune_help() -> None:
    """Test tune subcommand help lists train, status, and test."""
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "status" in result.output
    assert "test" in result.output


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
        assert "Chill Hang" in res_match.output
        assert "Nearest 1 anchor tracks that triggered 'Chill Hang'" in res_match.output
        assert "Matched Personalized Moods" in res_match.output
        assert "★ Assigned" in res_match.output

        # 4b. Verbose flag
        res_verbose = runner.invoke(
            app,
            ["tune", "test", str(audio_file), "--model-path", str(model_file), "--verbose"],
        )
        assert res_verbose.exit_code == 0
        assert "Full Mood Evaluation Breakdown:" in res_verbose.output

        # 4c. No mood matches threshold
        high_thresh_model = tmp_path / "high_thresh.json"
        high_thresh_model.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "moods": {
                        "Chill Hang": {
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
        res_no_match = runner.invoke(
            app,
            ["tune", "test", str(audio_file), "--model-path", str(high_thresh_model)],
        )
        assert res_no_match.exit_code == 0
        assert "Assigned Mood: None" in res_no_match.output
        assert "Top Evaluated Moods (Below Threshold)" in res_no_match.output

        # 4d. Targeted mood flag (--mood)
        # Unknown mood error
        res_unknown = runner.invoke(
            app,
            ["tune", "test", str(audio_file), "--model-path", str(model_file), "--mood", "Unknown"],
        )
        assert "Mood 'Unknown' not found in trained model" in res_unknown.output

        # Target match path (case-insensitive)
        res_target_match = runner.invoke(
            app,
            [
                "tune",
                "test",
                str(audio_file),
                "--model-path",
                str(model_file),
                "--mood",
                "chill hang",
            ],
        )
        assert res_target_match.exit_code == 0
        assert "Target Mood Evaluation: Chill Hang -> ✓ MATCH" in res_target_match.output
        assert "★ #1 Assigned Mood" in res_target_match.output
        assert "Nearest 1 anchor tracks for 'Chill Hang'" in res_target_match.output

        # Target rejected path
        res_target_reject = runner.invoke(
            app,
            [
                "tune",
                "test",
                str(audio_file),
                "--model-path",
                str(high_thresh_model),
                "--mood",
                "Chill Hang",
            ],
        )
        assert res_target_reject.exit_code == 0
        assert "Target Mood Evaluation: Chill Hang -> ✗ REJECTED" in res_target_reject.output
        assert "Below 'Chill Hang' threshold" in res_target_reject.output

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
