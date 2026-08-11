"""Interactive setup wizard for Resonate metadata engine using typer and rich."""

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt, Prompt

from resonate.config import load_config

DEFAULT_MOODS = [
    "chill",
    "energetic",
    "upbeat",
    "melancholic",
    "dark",
    "aggressive",
    "happy",
    "groovy",
    "romantic",
    "nostalgic",
    "trippy",
    "soulful",
    "moody",
]


def run_wizard(config_path: str = "config.yaml") -> None:
    """Run interactive setup wizard to configure Plex, Last.fm, mapping, and batch processing."""
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Resonate Interactive Setup Wizard[/bold cyan]",
            border_style="cyan",
        )
    )

    path = Path(config_path)
    existing_settings = load_config(config_path) if path.is_file() else None

    default_plex_url = (
        existing_settings.plex.url
        if existing_settings and existing_settings.plex.url
        else "http://localhost:32400"
    )
    default_plex_token = existing_settings.plex.token if existing_settings else ""
    default_library_name = existing_settings.plex.library_name if existing_settings else "Music"
    default_lastfm_key = existing_settings.lastfm.api_key if existing_settings else ""
    default_batch_size = existing_settings.processing.batch_size if existing_settings else 100
    default_threshold = existing_settings.mapping.threshold if existing_settings else 0.45
    default_path_map_source = (
        existing_settings.processing.path_map_source if existing_settings else "/data/music"
    )
    default_path_map_target = (
        existing_settings.processing.path_map_target if existing_settings else "/music"
    )
    default_moods_str = ", ".join(
        existing_settings.mapping.target_moods
        if existing_settings and existing_settings.mapping.target_moods
        else DEFAULT_MOODS
    )

    console.print("[bold yellow]Plex Configuration[/bold yellow]")
    plex_url = Prompt.ask("Plex Server URL", default=default_plex_url)

    default_plex_token_display = (
        f"{default_plex_token[:4]}...{default_plex_token[-4:]}"
        if len(default_plex_token) > 8
        else ("********" if default_plex_token else "")
    )
    plex_token_input = Prompt.ask(
        "Plex Auth Token",
        default=default_plex_token_display,
        password=True,
    )
    plex_token = (
        default_plex_token if plex_token_input == default_plex_token_display else plex_token_input
    )

    library_name = Prompt.ask("Music Library Name", default=default_library_name)

    console.print("\n[bold yellow]Last.fm Configuration[/bold yellow]")

    default_lastfm_key_display = (
        f"{default_lastfm_key[:4]}...{default_lastfm_key[-4:]}"
        if len(default_lastfm_key) > 8
        else ("********" if default_lastfm_key else "")
    )
    lastfm_api_key_input = Prompt.ask(
        "Last.fm API Key",
        default=default_lastfm_key_display,
        password=True,
    )
    lastfm_api_key = (
        default_lastfm_key
        if lastfm_api_key_input == default_lastfm_key_display
        else lastfm_api_key_input
    )

    console.print("\n[bold yellow]Processing Settings[/bold yellow]")
    batch_size = IntPrompt.ask("Batch Size", default=default_batch_size)
    threshold = FloatPrompt.ask("Confidence Threshold", default=default_threshold)
    path_map_source = Prompt.ask(
        "Plex music directory path (source map)",
        default=default_path_map_source,
    )
    path_map_target = Prompt.ask(
        "Resonate container music directory path (target map)",
        default=default_path_map_target,
    )

    console.print("\n[bold yellow]Target Moods[/bold yellow]")
    moods_input = Prompt.ask("Target Moods (comma-separated)", default=default_moods_str)
    target_moods = [m.strip() for m in moods_input.split(",") if m.strip()] or DEFAULT_MOODS

    existing_yaml_data: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            if isinstance(content, dict):
                existing_yaml_data = content

    updated_config: dict[str, Any] = {
        **existing_yaml_data,
        "moods": target_moods,
        "plex": {
            **existing_yaml_data.get("plex", {}),
            "url": plex_url,
            "token": plex_token,
            "library_name": library_name,
        },
        "lastfm": {
            **existing_yaml_data.get("lastfm", {}),
            "api_key": lastfm_api_key,
        },
        "mapping": {
            **existing_yaml_data.get("mapping", {}),
            "threshold": threshold,
            "target_moods": target_moods,
        },
        "processing": {
            **existing_yaml_data.get("processing", {}),
            "batch_size": batch_size,
            "path_map_source": path_map_source,
            "path_map_target": path_map_target,
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(updated_config, f, sort_keys=False)

    console.print(f"\n[bold green]Configuration successfully saved to {config_path}![/bold green]")
