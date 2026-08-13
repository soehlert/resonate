"""Pytest unit tests for Resonate configuration loading and Pydantic validation."""

from pathlib import Path

from pytest import MonkeyPatch

from resonate.config import (
    BeetsConfig,
    DatabaseConfig,
    EssentiaConfig,
    LastFmConfig,
    MappingConfig,
    PlexConfig,
    ProcessingConfig,
    ResonateSettings,
    load_config,
)

EXPECTED_TARGET_MOODS = [
    "party",
    "chill hang",
    "energetic",
    "groovy",
    "acoustic",
    "electronic",
    "melancholic",
    "upbeat",
    "dark",
    "happy",
    "relaxed",
    "aggressive",
    "romantic",
    "calm",
    "mellow",
    "lively",
    "funky",
    "intense",
    "hypnotic",
    "atmospheric",
    "bittersweet",
    "intimate",
]


def test_default_config_loading() -> None:
    """Test default configuration model initialization and target moods list."""
    settings = ResonateSettings()
    assert isinstance(settings.plex, PlexConfig)
    assert isinstance(settings.lastfm, LastFmConfig)
    assert isinstance(settings.mapping, MappingConfig)
    assert isinstance(settings.processing, ProcessingConfig)
    assert isinstance(settings.essentia, EssentiaConfig)
    assert isinstance(settings.beets, BeetsConfig)
    assert isinstance(settings.database, DatabaseConfig)

    assert settings.mapping.target_moods == EXPECTED_TARGET_MOODS
    assert settings.mapping.threshold == 0.5


def test_load_config_file(tmp_path: Path) -> None:
    """Test loading configuration settings from a custom YAML file."""
    yaml_content = """
plex:
  url: "http://plex.local:32400"
  token: "secrettoken123"
  library_name: "My Music"
mapping:
  threshold: 0.45
  target_moods:
    - chill
    - energetic
processing:
  batch_size: 50
  dry_run: true
"""
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    settings = load_config(str(config_file))
    assert settings.plex.url == "http://plex.local:32400"
    assert settings.plex.token == "secrettoken123"
    assert settings.plex.library_name == "My Music"
    assert settings.mapping.threshold == 0.45
    assert settings.mapping.target_moods == ["chill", "energetic"]
    assert settings.processing.batch_size == 50
    assert settings.processing.dry_run is True


def test_env_var_overrides(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test environment variable overrides for configuration settings."""
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("RESONATE_PLEX_URL", "http://override:32400")
    monkeypatch.setenv("RESONATE_PLEX_TOKEN", "envtoken")
    monkeypatch.setenv("RESONATE_PROCESSING_BATCH_SIZE", "200")
    monkeypatch.setenv("RESONATE_PROCESSING_DRY_RUN", "true")

    settings = load_config(str(config_file))
    assert settings.plex.url == "http://override:32400"
    assert settings.plex.token == "envtoken"
    assert settings.processing.batch_size == 200
    assert settings.processing.dry_run is True
