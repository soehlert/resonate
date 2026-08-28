"""CLI command to display database processing statistics."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from resonate.config import load_config
from resonate.utils.state import StateManager

console = Console()


def status_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to config file"),
    ] = "config.yaml",
) -> None:
    """Display SQLite database processing statistics."""
    settings = load_config(config)
    state_mgr = StateManager(settings.database.sqlite_path)
    db_stats = state_mgr.get_stats()

    table = Table(title=f"Resonate DB Status ({settings.database.sqlite_path})")
    table.add_column("Key", style="bold cyan")
    table.add_column("Count", style="bold magenta")

    table.add_row("Total Processed Tracks", str(db_stats.get("total_processed", 0)))
    table.add_row("Mapped Tracks", str(db_stats.get("mapped", 0)))
    table.add_row("Unmapped Tracks", str(db_stats.get("unmapped", 0)))

    console.print(table)
