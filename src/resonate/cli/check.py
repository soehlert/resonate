"""CLI command to inspect and check audio file tags."""

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from resonate.modules.cleaner import inspect_path

console = Console()


def check_cmd(
    target_path: Annotated[
        str,
        typer.Argument(
            help="Path to an audio file or music directory to inspect",
        ),
    ],
    tag: Annotated[
        list[str] | None,
        typer.Option(
            "--tag",
            "-t",
            help="Filter by specific tag name(s) (e.g. album, title, genre, mood, bpm, TALB)",
        ),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Display raw low-level ID3 frames/tags (like mid3v2 -l)",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="Recursively scan subdirectories for audio files",
        ),
    ] = True,
) -> None:
    """Inspect and check ID3/metadata tags for an audio file or directory."""
    if not os.path.exists(target_path):
        console.print(f"[red]Error: Target path '{target_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    results = inspect_path(
        target_path,
        raw=raw,
        tag_filters=tag,
        recursive=recursive,
    )

    if not results:
        console.print(f"[yellow]No audio files found in: {target_path}[/yellow]")
        return

    errors = [r for r in results if r.error]
    valid_results = [r for r in results if not r.error]

    if raw:
        for r in valid_results:
            fname = os.path.basename(r.file_path)
            table = Table(title=f"Raw Tags: {fname}", show_lines=True)
            table.add_column("Frame / Key", style="bold cyan")
            table.add_column("Value", style="green")
            for k, v in r.tags.items():
                table.add_row(str(k), str(v))
            console.print(table)
    elif tag:
        tag_str = ", ".join(tag)
        table = Table(title=f"Tag Check ({tag_str})", show_lines=True)
        table.add_column("File", style="cyan")
        all_tag_keys: set[str] = set()
        for r in valid_results:
            all_tag_keys.update(r.tags.keys())
        sorted_keys = sorted(all_tag_keys)
        for k in sorted_keys:
            table.add_column(k.capitalize(), style="bold yellow")

        for r in valid_results:
            fname = os.path.basename(r.file_path)
            row_vals = [fname]
            for k in sorted_keys:
                v = r.tags.get(k, "")
                if isinstance(v, list):
                    row_vals.append(", ".join(str(x) for x in v))
                else:
                    row_vals.append(str(v) if v is not None else "")
            table.add_row(*row_vals)
        console.print(table)
    else:
        table = Table(title=f"Metadata Tags: {target_path}", show_lines=True)
        table.add_column("File", style="cyan")
        table.add_column("Artist", style="bold white")
        table.add_column("Album", style="bold yellow")
        table.add_column("Title", style="bold magenta")
        table.add_column("Track", style="dim")
        table.add_column("Genre", style="green")
        table.add_column("Mood", style="blue")
        table.add_column("BPM", style="dim")

        for r in valid_results:
            fname = os.path.basename(r.file_path)
            art = str(r.tags.get("artist") or "")
            alb = str(r.tags.get("album") or "")
            tit = str(r.tags.get("title") or "")
            trk = str(r.tags.get("tracknumber") or "")
            gen = r.tags.get("genre")
            gen_str = ", ".join(gen) if isinstance(gen, list) else str(gen or "")
            moo = r.tags.get("mood")
            moo_str = ", ".join(moo) if isinstance(moo, list) else str(moo or "")
            bpm = str(r.tags.get("bpm") or "")
            table.add_row(fname, art, alb, tit, trk, gen_str, moo_str, bpm)
        console.print(table)

    if errors:
        err_table = Table(title="Errors", style="red")
        err_table.add_column("File", style="cyan")
        err_table.add_column("Error", style="bold red")
        for r in errors:
            err_table.add_row(os.path.basename(r.file_path), str(r.error))
        console.print(err_table)

    console.print(f"[dim]Total Scanned: {len(results)} audio file(s)[/dim]")
