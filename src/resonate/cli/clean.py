"""CLI command to sanitize and clean ID3 and audio file tags."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from resonate.modules.cleaner import TagCleaner

console = Console()


def clean_cmd(
    target_path: Annotated[
        str,
        typer.Argument(
            help="Path to an audio file or music directory to clean",
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="Recursively scan subdirectories for audio files",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview tag cleanups without writing changes to disk",
        ),
    ] = False,
    retailer_tags: Annotated[
        bool | None,
        typer.Option(
            "--retailer-tags/--no-retailer-tags",
            help="Clean retailer exclusive noise from album tags (e.g. Best Buy Exclusive)",
        ),
    ] = None,
    uncensor: Annotated[
        bool | None,
        typer.Option(
            "--uncensor/--no-uncensor",
            help="Uncensor masked profanities in track titles (e.g. Hatef--k -> Hatefuck)",
        ),
    ] = None,
    track_numbers: Annotated[
        bool | None,
        typer.Option(
            "--track-numbers/--no-track-numbers",
            help="Normalize track number formatting and extract disc numbers",
        ),
    ] = None,
    whitespace: Annotated[
        bool | None,
        typer.Option(
            "--whitespace/--no-whitespace",
            help="Trim double spaces and empty bracket artifacts",
        ),
    ] = None,
    rename_files: Annotated[
        bool,
        typer.Option(
            "--rename-files",
            help="Rename files to '{track} - {title}.ext' (no leading zeros) matching clean tags",
        ),
    ] = False,
) -> None:
    """Sanitize and clean noisy ID3/audio tags directly in file metadata."""
    if not os.path.exists(target_path):
        console.print(f"[red]Error: Target path '{target_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    any_positive = any(x is True for x in (retailer_tags, uncensor, track_numbers, whitespace))

    if any_positive:
        do_retailer = retailer_tags is True
        do_uncensor = uncensor is True
        do_track_numbers = track_numbers is True
        do_whitespace = whitespace is True
    else:
        do_retailer = retailer_tags is not False
        do_uncensor = uncensor is not False
        do_track_numbers = track_numbers is not False
        do_whitespace = whitespace is not False

    cleaner = TagCleaner(
        clean_retailer=do_retailer,
        uncensor=do_uncensor,
        normalize_track_numbers=do_track_numbers,
        clean_whitespace=do_whitespace,
        rename_files=rename_files,
    )

    active_rules: list[str] = []
    if do_retailer:
        active_rules.append("retailer-tags")
    if do_uncensor:
        active_rules.append("uncensor")
    if do_track_numbers:
        active_rules.append("track-numbers")
    if do_whitespace:
        active_rules.append("whitespace")
    if rename_files:
        active_rules.append("rename-files")

    dry_label = " [yellow][DRY-RUN][/yellow]" if dry_run else ""
    console.print(f"[bold blue]Starting Tag Cleanup on:[/bold blue] {target_path}{dry_label}")
    active_rules_str = ", ".join(active_rules)
    console.print(f"[dim]Active rules: {active_rules_str}[/dim]")

    results = cleaner.clean_path(target_path, recursive=recursive, dry_run=dry_run)

    total_files = len(results)
    changed_files = [r for r in results if r.changed]
    error_files = [r for r in results if r.error]

    if changed_files:
        table = Table(title="Tag Cleanup Summary", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Field", style="bold yellow")
        table.add_column("Before", style="dim")
        table.add_column("After", style="bold green")

        for r in changed_files:
            fname = os.path.basename(r.file_path)
            for c in r.changes:
                table.add_row(fname, c.field, c.old_value, c.new_value)

        console.print(table)

    if error_files:
        err_table = Table(title="Tag Cleanup Errors", style="red")
        err_table.add_column("File", style="cyan")
        err_table.add_column("Error", style="bold red")
        for r in error_files:
            err_table.add_row(os.path.basename(r.file_path), str(r.error))
        console.print(err_table)

    dry_note = (
        "\n[yellow][DRY-RUN ACTIVE] No changes were written to disk.[/yellow]" if dry_run else ""
    )
    summary_panel = Panel(
        f"Total Scanned: {total_files} | "
        f"Files Modified: {len(changed_files)} | "
        f"Errors: {len(error_files)}{dry_note}",
        title="[bold green]Cleanup Complete[/bold green]",
    )
    console.print(summary_panel)
