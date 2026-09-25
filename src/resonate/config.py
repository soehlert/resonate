"""Configuration loader and Pydantic settings models for Resonate."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def load_data_file(filename: str) -> dict[str, Any]:
    """Load package YAML data file from resonate/data/ directory."""
    data_path = DATA_DIR / filename
    if data_path.is_file():
        try:
            with data_path.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    return content
        except Exception as err:
            logger.warning(f"Failed to load package data file '{filename}': {err}")
    return {}


_load_data_file = load_data_file


class PlexConfig(BaseModel):
    """Plex server connection settings."""

    url: str = ""
    token: str = ""
    library_name: str = "Music"


class LastFmConfig(BaseModel):
    """Last.fm API settings."""

    api_key: str = ""


class MappingConfig(BaseModel):
    """Mood mapping configuration."""

    target_moods: list[str] = Field(
        default_factory=lambda: load_data_file("target_moods.yaml").get("moods", [])
    )
    threshold: float = 0.45
    genre_threshold: float = 0.45
    subgenre_threshold: float = 0.65
    mood_threshold: float = 0.45
    model_name: str = "all-MiniLM-L6-v2"


class ProcessingConfig(BaseModel):
    """Track processing options."""

    batch_size: int = 100
    dry_run: bool = False
    reprocess: bool = False
    path_map_source: str = ""
    path_map_target: str = ""


class EssentiaConfig(BaseModel):
    """Essentia audio analysis configuration."""

    enabled: bool = True
    models_dir: str = "models"
    model_filename: str = "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    threshold: float = 0.1


class DiscogsConfig(BaseModel):
    """Discogs API settings."""

    api_token: str = ""


class MusicBrainzConfig(BaseModel):
    """MusicBrainz API settings."""

    enabled: bool = True


class MutagenConfig(BaseModel):
    """Mutagen metadata tagging configuration."""

    enabled: bool = True


class DatabaseConfig(BaseModel):
    """Database and state storage configuration."""

    sqlite_path: str = "data/state.sqlite"


class LyricsConfig(BaseModel):
    """Lyrics retrieval and sentiment analysis configuration."""

    enabled: bool = True
    weight: float = 0.15
    prefer_embedded: bool = True
    lrclib_url: str = "https://lrclib.net"


class MoodConflictRule(BaseModel):
    """Directional mood conflict: if any trigger mood is present, drop target moods."""

    if_present: list[str] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)


class GenreMoodSeedRule(BaseModel):
    """Mapping of genre/subgenre families to default mood seeds."""

    genres: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)


class MoodRulesConfig(BaseModel):
    """Configurable mood rules, genre exclusions, and mutual mood conflicts."""

    acoustic_threshold: float = 0.10
    acoustic_mood_thresholds: dict[str, float] = Field(default_factory=dict)
    lyrics_threshold: float = 0.20
    lyrics_mood_thresholds: dict[str, float] = Field(default_factory=dict)
    genre_exclusions: dict[str, list[str]] = Field(default_factory=dict)
    mood_conflicts: list[MoodConflictRule] = Field(default_factory=list)
    genre_mood_seeds: list[GenreMoodSeedRule] = Field(default_factory=list)
    acoustic_mood_mappings: dict[str, list[str]] = Field(default_factory=dict)


class ResonateSettings(BaseModel):
    """Root configuration settings for Resonate."""

    plex: PlexConfig = Field(default_factory=PlexConfig)
    lastfm: LastFmConfig = Field(default_factory=LastFmConfig)
    discogs: DiscogsConfig = Field(default_factory=DiscogsConfig)
    musicbrainz: MusicBrainzConfig = Field(default_factory=MusicBrainzConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    essentia: EssentiaConfig = Field(default_factory=EssentiaConfig)
    mutagen: MutagenConfig = Field(default_factory=MutagenConfig)
    lyrics: LyricsConfig = Field(default_factory=LyricsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mood_rules: MoodRulesConfig = Field(default_factory=MoodRulesConfig)


def load_config(config_path: str = "config.yaml") -> ResonateSettings:
    """Load settings from YAML configuration file with data file defaults and env overrides."""
    config_dict: dict[str, Any] = {}
    path = Path(config_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if isinstance(yaml_content, dict):
                config_dict = yaml_content

    # Load defaults from package data files
    default_mood_rules = load_data_file("mood_rules.yaml")

    # Merge default mood rules with user overrides
    if default_mood_rules:
        if "mood_rules" not in config_dict or not isinstance(config_dict["mood_rules"], dict):
            config_dict["mood_rules"] = dict(default_mood_rules)
        else:
            user_rules = config_dict["mood_rules"]
            merged_mood_rules = dict(default_mood_rules)
            for k, v in user_rules.items():
                if isinstance(v, dict) and isinstance(merged_mood_rules.get(k), dict):
                    sub_merged = dict(merged_mood_rules[k])
                    sub_merged.update(v)
                    merged_mood_rules[k] = sub_merged
                else:
                    merged_mood_rules[k] = v
            config_dict["mood_rules"] = merged_mood_rules

    sections = {
        "plex": PlexConfig,
        "lastfm": LastFmConfig,
        "discogs": DiscogsConfig,
        "musicbrainz": MusicBrainzConfig,
        "mapping": MappingConfig,
        "processing": ProcessingConfig,
        "essentia": EssentiaConfig,
        "mutagen": MutagenConfig,
        "lyrics": LyricsConfig,
        "database": DatabaseConfig,
        "mood_rules": MoodRulesConfig,
    }

    for section_name in sections:
        if section_name not in config_dict or not isinstance(config_dict[section_name], dict):
            config_dict[section_name] = {}

    env_mappings = {
        "RESONATE_PLEX_URL": ("plex", "url"),
        "RESONATE_PLEX_TOKEN": ("plex", "token"),
        "RESONATE_PLEX_LIBRARY_NAME": ("plex", "library_name"),
        "RESONATE_LASTFM_API_KEY": ("lastfm", "api_key"),
        "RESONATE_DATABASE_SQLITE_PATH": ("database", "sqlite_path"),
        "RESONATE_PROCESSING_BATCH_SIZE": ("processing", "batch_size"),
        "RESONATE_PROCESSING_DRY_RUN": ("processing", "dry_run"),
        "RESONATE_PROCESSING_REPROCESS": ("processing", "reprocess"),
        "RESONATE_PROCESSING_OVERWRITE": ("processing", "reprocess"),
    }

    for env_var, (sec, key) in env_mappings.items():
        if env_var in os.environ:
            val: Any = os.environ[env_var]
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif val.isdigit():
                val = int(val)
            config_dict[sec][key] = val

    for env_var, val in os.environ.items():
        if env_var.startswith("RESONATE_") and env_var not in env_mappings:
            parts = env_var[len("RESONATE_") :].lower().split("_", 1)
            if len(parts) == 2:
                sec, key = parts
                if sec in sections:
                    parsed_val: Any = val
                    if val.lower() in ("true", "false"):
                        parsed_val = val.lower() == "true"
                    elif val.isdigit():
                        parsed_val = int(val)
                    config_dict[sec][key] = parsed_val

    return ResonateSettings(**config_dict)
