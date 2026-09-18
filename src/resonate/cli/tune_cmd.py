"""CLI commands to train and test personalized mood models from Plex playlists."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from resonate.config import load_config
from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.personalized_tuning import DEFAULT_MODEL_PATH, PersonalizedMoodTuner
from resonate.modules.plex import PlexSync

console = Console()
tune_app = typer.Typer(
    name="tune",
    help="Train and manage personalized mood models from Plex playlists.",
    add_completion=False,
)


@tune_app.command("train")
def tune_train_cmd(
    prefix: Annotated[
        str,
        typer.Option("--prefix", "-p", help="Playlist prefix to auto-discover in Plex"),
    ] = "resonate_",
    k: Annotated[
        int,
        typer.Option("--k", "-k", help="Number of nearest neighbors for k-NN consensus"),
    ] = 3,
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = "config.yaml",
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Output path for trained mood model JSON"),
    ] = DEFAULT_MODEL_PATH,
) -> None:
    """Auto-discover resonate_* playlists in Plex and calibrate personalized mood centroids."""
    settings = load_config(config)

    console.print(
        Panel.fit(
            f"[bold blue]Personalized Mood Tuning[/bold blue]\n"
            f"Plex Server: {settings.plex.url} | Library: {settings.plex.library_name}\n"
            f"Playlist Prefix: [cyan]'{prefix}'[/cyan] | k-NN: [green]{k}[/green] | "
            f"Output: [yellow]{model_path}[/yellow]",
            border_style="blue",
        )
    )

    plex_sync = PlexSync(
        url=settings.plex.url,
        token=settings.plex.token,
        library_name=settings.plex.library_name,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_scan = progress.add_task("Scanning Plex for anchor playlists...", total=None)
        anchor_playlists = plex_sync.fetch_mood_anchor_playlists(
            prefix=prefix,
            path_map_source=settings.processing.path_map_source,
            path_map_target=settings.processing.path_map_target,
        )
        progress.update(task_scan, completed=True, visible=False)

    if not anchor_playlists:
        console.print(
            f"[bold yellow]No playlists matching prefix '[cyan]{prefix}[/cyan]' "
            "found in Plex.[/bold yellow]\n"
            "Create a playlist in Plex starting with this prefix "
            "(e.g. [cyan]resonate_chill_hang[/cyan]) "
            "and add 10–15 anchor tracks to train your model."
        )
        return

    essentia_analyzer = EssentiaAnalyzer(models_dir="models")
    mood_embeddings: dict[str, list] = {}
    mood_track_names: dict[str, list[str]] = {}

    for mood, tracks in anchor_playlists.items():
        console.print(
            f"\n[bold green]Processing anchor playlist for mood:[/bold green] "
            f"[bold cyan]{mood}[/bold cyan] ({len(tracks)} tracks)"
        )
        embeddings_list = []
        names_list = []
        for t in tracks:
            path = t.file_path or ""
            if not path or not os.path.exists(path):
                console.print(
                    f"  [dim yellow]Notice:[/dim yellow] Track [dim]'{t.title}'[/dim] "
                    f"not found at [dim]'{path or 'unknown'}'[/dim] - skipping."
                )
                continue

            try:
                emb = essentia_analyzer.extract_embeddings(file_path=path)
                if emb is not None:
                    embeddings_list.append(emb)
                    names_list.append(f"'{t.title}' by {t.artist}")
                    console.print(
                        f"  [green]✓[/green] Extracted EffNet embedding: "
                        f"[cyan]{t.title}[/cyan] by [yellow]{t.artist}[/yellow]"
                    )
                else:
                    console.print(
                        f"  [red]✗[/red] Failed to extract embedding: [dim]{t.title}[/dim]"
                    )
            except Exception as err:
                console.print(f"  [red]✗[/red] Error on [dim]{t.title}[/dim]: {err}")

        if embeddings_list:
            mood_embeddings[mood] = embeddings_list
            mood_track_names[mood] = names_list

    if not mood_embeddings:
        console.print(
            "[bold red]No valid audio embeddings extracted from any anchor playlist.[/bold red]"
        )
        return

    # Fit tuner
    tuner = PersonalizedMoodTuner(model_path=model_path)
    tuner.fit(mood_embeddings, target_k=k, track_names=mood_track_names)
    tuner.save_model()

    # Display calibrated summary table
    table = Table(title="Calibrated Personalized Mood Heads")
    table.add_column("Mood", style="bold cyan")
    table.add_column("Anchor Tracks", style="bold magenta", justify="center")
    table.add_column("Coherence", style="bold green", justify="center")
    table.add_column("Threshold", style="bold yellow", justify="center")
    table.add_column("k-NN", style="bold blue", justify="center")

    for mood, meta in tuner.mood_heads.items():
        k_val = meta.get("k", tuner.get_k(mood))
        table.add_row(
            mood,
            str(meta.get("track_count", 0)),
            f"{meta.get('coherence', 0.0):.3f}",
            f"{meta.get('threshold', 0.0):.3f}",
            f"{k_val}-NN",
        )

    console.print("\n")
    console.print(table)
    console.print(
        f"\n[bold green]Successfully trained and saved model to[/bold green] "
        f"[yellow]{model_path}[/yellow]!"
    )


@tune_app.command("status")
def tune_status_cmd(
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Path to trained mood model JSON"),
    ] = DEFAULT_MODEL_PATH,
) -> None:
    """Display status and metrics for the currently trained personalized mood model."""
    tuner = PersonalizedMoodTuner(model_path=model_path)
    if not tuner.load_model():
        console.print(
            f"[yellow]No trained personalized mood model found at '{model_path}'.[/yellow]\n"
            "Run [bold cyan]resonate tune train[/bold cyan] to calibrate moods from Plex."
        )
        return

    table = Table(title=f"Personalized Mood Model Status ({model_path})")
    table.add_column("Mood", style="bold cyan")
    table.add_column("Anchor Tracks", style="bold magenta", justify="center")
    table.add_column("Coherence", style="bold green", justify="center")
    table.add_column("Threshold", style="bold yellow", justify="center")
    table.add_column("k-NN", style="bold blue", justify="center")

    for mood, meta in tuner.mood_heads.items():
        k_val = meta.get("k", tuner.get_k(mood))
        table.add_row(
            mood,
            str(meta.get("track_count", 0)),
            f"{meta.get('coherence', 0.0):.3f}",
            f"{meta.get('threshold', 0.0):.3f}",
            f"{k_val}-NN",
        )

    console.print(table)


@tune_app.command("test")
def tune_test_cmd(
    file_path: Annotated[
        str | None,
        typer.Argument(help="Local audio file path to test"),
    ] = None,
    key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="Plex ratingKey of track to test"),
    ] = None,
    top_anchors: Annotated[
        int,
        typer.Option("--top-anchors", "-t", help="Number of nearest anchor tracks to display"),
    ] = 5,
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to config file"),
    ] = "config.yaml",
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Path to trained mood model JSON"),
    ] = DEFAULT_MODEL_PATH,
) -> None:
    """Score a single audio file or Plex track against trained personalized mood centroids."""
    tuner = PersonalizedMoodTuner(model_path=model_path)
    if not tuner.load_model():
        console.print(f"[red]No trained personalized mood model found at '{model_path}'.[/red]")
        return

    resolved_file = file_path
    track_desc = ""
    if not resolved_file and key:
        settings = load_config(config)
        plex_sync = PlexSync(
            url=settings.plex.url,
            token=settings.plex.token,
            library_name=settings.plex.library_name,
        )
        matched = plex_sync.fetch_track_by_key(
            rating_key=key,
            path_map_source=settings.processing.path_map_source,
            path_map_target=settings.processing.path_map_target,
        )
        if matched and matched.file_path:
            resolved_file = matched.file_path
            track_desc = f"[cyan]'{matched.title}'[/cyan] by [yellow]{matched.artist}[/yellow]"
        elif not matched:
            console.print(f"[red]Track with ratingKey '{key}' not found in Plex.[/red]")
            return

    if not resolved_file or not os.path.exists(resolved_file):
        console.print(f"[red]Audio file not found: '{resolved_file or 'None'}'[/red]")
        return

    essentia_analyzer = EssentiaAnalyzer(models_dir="models")
    emb = essentia_analyzer.extract_embeddings(file_path=resolved_file)
    if emb is None:
        console.print("[red]Failed to extract EffNet embedding from audio file.[/red]")
        return

    scores = tuner.score_all(emb)
    if track_desc:
        console.print(f"\n[bold]Testing track:[/bold] {track_desc} [dim](ratingKey={key})[/dim]")
        console.print(f"[dim]Audio file:[/dim] {resolved_file}")
    else:
        console.print(f"\n[bold]Testing file:[/bold] [cyan]{resolved_file}[/cyan]")

    matches = [s for s in scores if s[3]]
    if matches:
        console.print("\n[bold green]Matched Personalized Moods:[/bold green]")
        for m, score, threshold, _ in matches:
            k_val = tuner.get_k(m)
            console.print(
                f"  [bold green]✓[/bold green] [bold cyan]{m}[/bold cyan]: "
                f"similarity = [green]{score:.3f}[/green] "
                f"[dim](threshold: {threshold:.3f}, {k_val}-NN)[/dim]"
            )
            top_neighbors = tuner.get_top_neighbors(emb, m, n_neighbors=top_anchors)
            for idx, (aname, asim) in enumerate(top_neighbors, start=1):
                star = " ★" if idx <= k_val else ""
                style = "green" if idx <= k_val else "dim"
                console.print(f"      [{style}]{idx}. {aname} ({asim:.3f}){star}[/{style}]")

    non_matches = [s for s in scores if not s[3]]
    if non_matches:
        header = (
            "\n[bold dim]Other Evaluated Moods:[/bold dim]"
            if matches
            else "\n[yellow]No moods matched above threshold:[/yellow]"
        )
        console.print(header)
        for m, score, threshold, _ in non_matches:
            k_val = tuner.get_k(m)
            console.print(
                f"  [dim red]✗[/dim red] [cyan]{m}[/cyan]: "
                f"similarity = [yellow]{score:.3f}[/yellow] "
                f"[dim](threshold: {threshold:.3f}, {k_val}-NN)[/dim]"
            )
            top_neighbors = tuner.get_top_neighbors(emb, m, n_neighbors=top_anchors)
            for idx, (aname, asim) in enumerate(top_neighbors, start=1):
                star = " ★" if idx <= k_val else ""
                style = "yellow" if idx <= k_val else "dim"
                console.print(f"      [{style}]{idx}. {aname} ({asim:.3f}){star}[/{style}]")
