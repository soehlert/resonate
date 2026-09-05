"""Pytest unit tests for Resonate configuration loading and Pydantic validation."""

from pathlib import Path

from pytest import MonkeyPatch

from resonate.config import (
    BeetsConfig,
    DatabaseConfig,
    EssentiaConfig,
    LastFmConfig,
    LyricsConfig,
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
    "heavy",
    "nostalgic",
    "trippy",
    "soulful",
    "moody",
]


def test_default_config_loading() -> None:
    """Test default configuration model initialization and target moods list."""
    settings = ResonateSettings()
    assert isinstance(settings.plex, PlexConfig)
    assert isinstance(settings.lastfm, LastFmConfig)
    assert isinstance(settings.mapping, MappingConfig)
    assert isinstance(settings.processing, ProcessingConfig)
    assert isinstance(settings.essentia, EssentiaConfig)
    assert settings.processing.workers == 4
    assert settings.processing.batch_size == 100
    assert settings.processing.dry_run is False
    assert isinstance(settings.beets, BeetsConfig)
    assert isinstance(settings.lyrics, LyricsConfig)
    assert isinstance(settings.database, DatabaseConfig)

    assert settings.lyrics.enabled is True
    assert settings.lyrics.weight == 0.15
    assert settings.lyrics.prefer_embedded is True
    assert settings.lyrics.lrclib_url == "https://lrclib.net"

    assert settings.mapping.target_moods == EXPECTED_TARGET_MOODS
    assert settings.mapping.threshold == 0.45
    assert settings.mapping.genre_threshold == 0.45
    assert settings.mapping.subgenre_threshold == 0.65
    assert settings.mapping.mood_threshold == 0.45


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
  workers: 6
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
    assert settings.processing.workers == 6
    assert settings.processing.dry_run is True


def test_env_var_overrides(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test environment variable overrides for configuration settings."""
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("RESONATE_PLEX_URL", "http://override:32400")
    monkeypatch.setenv("RESONATE_PLEX_TOKEN", "envtoken")
    monkeypatch.setenv("RESONATE_PROCESSING_BATCH_SIZE", "200")
    monkeypatch.setenv("RESONATE_PROCESSING_WORKERS", "8")
    monkeypatch.setenv("RESONATE_PROCESSING_DRY_RUN", "true")
    monkeypatch.setenv("RESONATE_PROCESSING_REPROCESS", "true")

    settings = load_config(str(config_file))
    assert settings.plex.url == "http://override:32400"
    assert settings.plex.token == "envtoken"
    assert settings.processing.batch_size == 200
    assert settings.processing.workers == 8
    assert settings.processing.dry_run is True
    assert settings.processing.reprocess is True


def test_load_lyrics_config_file(tmp_path: Path) -> None:
    """Test loading lyrics configuration from YAML file."""
    yaml_content = """
lyrics:
  enabled: false
  weight: 0.25
  prefer_embedded: false
  lrclib_url: "https://custom-lyrics.net"
"""
    config_file = tmp_path / "lyrics_config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    settings = load_config(str(config_file))
    assert settings.lyrics.enabled is False
    assert settings.lyrics.weight == 0.25
    assert settings.lyrics.prefer_embedded is False
    assert settings.lyrics.lrclib_url == "https://custom-lyrics.net"

