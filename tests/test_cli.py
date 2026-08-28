"""Unit tests for Resonate CLI commands and Typer entrypoints."""

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
