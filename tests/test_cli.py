"""Unit tests for Resonate CLI commands and Typer entrypoints."""

from pathlib import Path

from typer.testing import CliRunner

from resonate.main import app

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


def test_cli_analyze_help() -> None:
    """Test analyze subcommand help lists options."""
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--write-id3" in result.output
    assert "--write-plex" in result.output
    assert "--workers" in result.output
    assert "-w" in result.output


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


def test_analyze_cmd_workers_execution(tmp_path: Path) -> None:
    """Test analyze execution with multiple workers processes all tracks."""
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
  workers: 2
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

        # Test with --workers 2 (concurrent)
        result_parallel = runner.invoke(
            app,
            ["analyze", "--config", str(config_file), "--workers", "2", "--dry-run"],
        )
        assert result_parallel.exit_code == 0
        assert "Total Processed" in result_parallel.output
        assert mock_pipe.enrich_track.call_count == 2

        mock_pipe.enrich_track.reset_mock()

        # Test with --workers 1 (serial)
        result_serial = runner.invoke(
            app,
            ["analyze", "--config", str(config_file), "--workers", "1", "--dry-run"],
        )
        assert result_serial.exit_code == 0
        assert "Total Processed" in result_serial.output
        assert mock_pipe.enrich_track.call_count == 2

