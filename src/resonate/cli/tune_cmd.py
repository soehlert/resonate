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
from resonate.models import TrackItem
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
    """Auto-discover resonate_* playlists in Plex and calibrate personalized mood anchor heads."""
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
    mood: Annotated[
        str | None,
        typer.Option("--mood", "-M", help="Specific canonical mood to test against"),
    ] = None,
    negative_samples: Annotated[
        int,
        typer.Option(
            "--negative-samples",
            "-n",
            help="Number of random tracks to sample from Plex for negative path testing",
        ),
    ] = 10,
    prefix: Annotated[
        str,
        typer.Option("--prefix", "-p", help="Playlist prefix to auto-discover in Plex"),
    ] = "resonate_",
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Display nearest anchors for all evaluated moods"),
    ] = False,
) -> None:
    """Score a single audio file or Plex track against trained personalized mood anchor heads."""
    tuner = PersonalizedMoodTuner(model_path=model_path)
    if not tuner.load_model():
        console.print(f"[red]No trained personalized mood model found at '{model_path}'.[/red]")
        return

    target_mood = None
    if mood:
        mood_map = {m.lower(): m for m in tuner.tuned_moods}
        target_mood = mood_map.get(mood.strip().lower())
        if not target_mood:
            console.print(
                f"[red]Mood '{mood}' not found in trained model. "
                f"Available moods: {', '.join(tuner.tuned_moods)}[/red]"
            )
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

    if not resolved_file and not key:
        if not target_mood:
            console.print(
                "[red]Error: You must provide an audio file, a Plex --key, "
                "or a target --mood to test.[/red]\n"
                "Examples:\n"
                "  • Test entire mood playlist + negative sampling:\n"
                "      [cyan]resonate tune test --mood 'Chill Hang'[/cyan]\n"
                "  • Test single Plex track:\n"
                "      [cyan]resonate tune test --key 16012 --mood 'Chill Hang'[/cyan]\n"
                "  • Test local file:\n"
                "      [cyan]resonate tune test /path/to/song.mp3[/cyan]"
            )
            return

        _run_mood_playlist_and_negative_test(
            target_mood=target_mood,
            tuner=tuner,
            config_path=config,
            prefix=prefix,
            negative_count=negative_samples,
            top_anchors=top_anchors,
            verbose=verbose,
        )
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
    predicted = tuner.predict(emb, top_k=1)
    assigned_mood = predicted[0][0] if predicted else None

    if track_desc:
        console.print(f"\n[bold]Testing track:[/bold] {track_desc} [dim](ratingKey={key})[/dim]")
        console.print(f"[dim]Audio file:[/dim] {resolved_file}")
    else:
        console.print(f"\n[bold]Testing file:[/bold] [cyan]{resolved_file}[/cyan]")

    if target_mood:
        target_tuple = next((s for s in scores if s[0] == target_mood), None)
        if not target_tuple:
            console.print(f"[red]Could not score mood '{target_mood}'.[/red]")
            return
        _, t_score, t_threshold, t_match = target_tuple
        t_margin = t_score - t_threshold
        k_val = tuner.get_k(target_mood)

        if t_match:
            status_text = (
                f"[bold green]✓ MATCH[/bold green] "
                f"[dim](score: {t_score:.3f} | threshold: {t_threshold:.3f} | "
                f"margin: {t_margin:+.3f})[/dim]"
            )
        else:
            status_text = (
                f"[dim red]✗ REJECTED[/dim red] "
                f"[dim](score: {t_score:.3f} | threshold: {t_threshold:.3f} | "
                f"margin: {t_margin:+.3f})[/dim]"
            )
        console.print(
            f"\n[bold]Target Mood Evaluation:[/bold] "
            f"[bold cyan]{target_mood}[/bold cyan] -> {status_text}"
        )

        if t_match:
            if assigned_mood == target_mood:
                console.print(
                    "[bold green]Standing:[/bold green] "
                    "[bold]★ #1 Assigned Mood[/bold] (Wins overall)"
                )
            else:
                winner_tuple = next((s for s in scores if s[0] == assigned_mood), None)
                w_score_str = f"{winner_tuple[1]:.3f}" if winner_tuple else "higher"
                console.print(
                    f"[yellow]Standing:[/yellow] Runner-up (Matches '{target_mood}', "
                    f"but '{assigned_mood}' scored higher: {w_score_str})"
                )
        else:
            console.print(
                f"[dim]Standing: Below '{target_mood}' threshold (margin: {t_margin:+.3f})[/dim]"
            )

        n_anchors = top_anchors if verbose else k_val
        top_neighbors = tuner.get_top_neighbors(emb, target_mood, n_neighbors=n_anchors)
        console.print(
            f"\n[bold]Nearest {len(top_neighbors)} anchor tracks for '{target_mood}':[/bold]"
        )
        for idx, (aname, asim) in enumerate(top_neighbors, start=1):
            star = " ★" if idx <= k_val else ""
            style = "green" if idx <= k_val and t_match else "dim"
            console.print(f"  {idx}. [{style}]{aname} ({asim:.3f}){star}[/{style}]")
        return

    if assigned_mood:
        assigned_tuple = next((s for s in scores if s[0] == assigned_mood), None)
        if assigned_tuple:
            _, a_score, a_threshold, _ = assigned_tuple
            a_margin = a_score - a_threshold
            k_val = tuner.get_k(assigned_mood)
            console.print(
                f"\n[bold green]Assigned Mood:[/bold green] [bold cyan]{assigned_mood}[/bold cyan] "
                f"[dim](score: {a_score:.3f} | threshold: {a_threshold:.3f} | "
                f"margin: {a_margin:+.3f})[/dim]"
            )
            top_neighbors = tuner.get_top_neighbors(emb, assigned_mood, n_neighbors=k_val)
            console.print(
                f"\n[bold]Nearest {k_val} anchor tracks that triggered "
                f"'[cyan]{assigned_mood}[/cyan]':[/bold]"
            )
            for idx, (aname, asim) in enumerate(top_neighbors, start=1):
                console.print(f"  {idx}. [green]{aname}[/green] [dim]({asim:.3f})[/dim]")
    else:
        console.print(
            "\n[yellow]Assigned Mood: None[/yellow] [dim](No moods met their threshold)[/dim]"
        )

    matches = [s for s in scores if s[3]]
    if matches:
        table = Table(
            title="Matched Personalized Moods",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Mood", style="bold cyan")
        table.add_column("Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Margin", justify="right")
        table.add_column("Status")

        display_limit = len(matches) if verbose else min(5, len(matches))
        for m, score, threshold, _ in matches[:display_limit]:
            margin = score - threshold
            if m == assigned_mood:
                status = "[bold green]★ Assigned[/bold green]"
                margin_str = f"[bold green]{margin:+.3f}[/bold green]"
            else:
                status = "[dim]Runner-up[/dim]"
                margin_str = f"[green]{margin:+.3f}[/green]"
            table.add_row(m, f"{score:.3f}", f"{threshold:.3f}", margin_str, status)

        console.print()
        console.print(table)
        if not verbose and len(matches) > display_limit:
            omitted = len(matches) - display_limit
            plural = "s" if omitted > 1 else ""
            console.print(f"[dim]({omitted} other weaker match{plural} omitted)[/dim]")
    else:
        table = Table(
            title="Top Evaluated Moods (Below Threshold)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Mood", style="bold cyan")
        table.add_column("Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Margin", justify="right")
        table.add_column("Status")

        display_limit = len(scores) if verbose else min(5, len(scores))
        for m, score, threshold, _ in scores[:display_limit]:
            margin = score - threshold
            status = "[dim yellow]Miss[/dim yellow]"
            margin_str = f"[dim red]{margin:+.3f}[/dim red]"
            table.add_row(m, f"{score:.3f}", f"{threshold:.3f}", margin_str, status)

        console.print()
        console.print(table)

    if verbose:
        console.print("\n[bold dim]Full Mood Evaluation Breakdown:[/bold dim]")
        for m, score, threshold, is_match in scores:
            k_val = tuner.get_k(m)
            symbol = "[bold green]✓[/bold green]" if is_match else "[dim red]✗[/dim red]"
            console.print(
                f"  {symbol} [cyan]{m}[/cyan]: score = {score:.3f} "
                f"[dim](threshold: {threshold:.3f}, {k_val}-NN)[/dim]"
            )
            top_neighbors = tuner.get_top_neighbors(emb, m, n_neighbors=top_anchors)
            for idx, (aname, asim) in enumerate(top_neighbors, start=1):
                star = " ★" if idx <= k_val else ""
                style = "green" if idx <= k_val and is_match else "dim"
                console.print(f"      [{style}]{idx}. {aname} ({asim:.3f}){star}[/{style}]")


def _evaluate_track_mood(
    track: TrackItem,
    target_mood: str,
    tuner: PersonalizedMoodTuner,
    analyzer: EssentiaAnalyzer,
) -> tuple[float, float, str | None, bool] | None:
    """Extract embedding and evaluate track against target mood and overall winner."""
    if not track.file_path or not os.path.exists(track.file_path):
        return None
    emb = analyzer.extract_embeddings(file_path=track.file_path)
    if emb is None:
        return None
    scores = tuner.score_all(emb)
    predicted = tuner.predict(emb, top_k=1)
    assigned = predicted[0][0] if predicted else None
    target_tuple = next((s for s in scores if s[0] == target_mood), None)
    if not target_tuple:
        return None
    _, score, threshold, is_match = target_tuple
    return score, threshold, assigned, is_match


def _run_mood_playlist_and_negative_test(
    target_mood: str,
    tuner: PersonalizedMoodTuner,
    config_path: str,
    prefix: str,
    negative_count: int,
    top_anchors: int,
    verbose: bool,
) -> None:
    """Execute batch mood playlist evaluation with random negative library sampling."""
    settings = load_config(config_path)
    plex = PlexSync(
        url=settings.plex.url,
        token=settings.plex.token,
        library_name=settings.plex.library_name,
    )

    console.print(
        Panel.fit(
            f"[bold blue]Personalized Mood Evaluation: [cyan]{target_mood}[/cyan][/bold blue]\n"
            f"Playlist Prefix: [cyan]'{prefix}'[/cyan] | "
            f"Negative Sample: [yellow]{negative_count}[/yellow]",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_pl = progress.add_task(f"Fetching '{prefix}{target_mood}' playlist...", total=None)
        playlist_title, candidate_tracks = plex.fetch_mood_playlist_tracks(
            mood=target_mood,
            prefix=prefix,
            path_map_source=settings.processing.path_map_source,
            path_map_target=settings.processing.path_map_target,
        )
        progress.update(task_pl, completed=True, visible=False)

    if not candidate_tracks:
        expected_pl = f"{prefix}{target_mood.lower().replace(' ', '_')}"
        console.print(
            f"[bold red]Mood playlist '[cyan]{target_mood}[/cyan]' not found in Plex.[/bold red]\n"
            f"Expected playlist name like: [yellow]{expected_pl}[/yellow]"
        )
        return

    analyzer = EssentiaAnalyzer(models_dir="models")

    # 1. Candidate Playlist Matches
    console.print(
        f"\n[bold]==================================================[/bold]\n"
        f"[bold cyan] 1. TESTING CANDIDATE PLAYLIST: {playlist_title} "
        f"({len(candidate_tracks)} tracks)[/bold cyan]\n"
        f"[bold]==================================================[/bold]"
    )

    cand_pass, cand_runner_up, cand_miss = 0, 0, 0
    rivals: dict[str, int] = {}
    tested_keys: set[str] = set()

    for track in candidate_tracks:
        if track.rating_key:
            tested_keys.add(str(track.rating_key))
        eval_res = _evaluate_track_mood(track, target_mood, tuner, analyzer)
        if not eval_res:
            continue
        score, threshold, assigned, is_match = eval_res
        hdr = (
            f"[cyan]'{track.title}'[/cyan] by [yellow]{track.artist}[/yellow] "
            f"[dim](ratingKey={track.rating_key})[/dim]"
        )

        if is_match and assigned == target_mood:
            cand_pass += 1
            margin = score - threshold
            console.print(
                f"  [green]✓ PASS[/green] {hdr}\n"
                f"         [bold green]★ #1 Assigned[/bold green] "
                f"(score: {score:.3f} | threshold: {threshold:.3f} | margin: {margin:+.3f})"
            )
        elif is_match:
            cand_runner_up += 1
            rivals[assigned or "None"] = rivals.get(assigned or "None", 0) + 1
            console.print(
                f"  [yellow]⚠️ RIVAL[/yellow] {hdr}\n"
                f"         [yellow]Runner-up[/yellow] for '{target_mood}', but "
                f"[bold magenta]'{assigned}'[/bold magenta] won"
            )
        else:
            cand_miss += 1
            console.print(
                f"  [red]✗ MISS[/red] {hdr}\n"
                f"         [dim red]Below threshold[/dim red] "
                f"(score: {score:.3f} < {threshold:.3f})"
            )

    # 2. Negative Path (Random Library Sampling)
    neg_rejected, neg_false_pos = 0, 0
    if negative_count > 0:
        console.print(
            f"\n[bold]==================================================[/bold]\n"
            f"[bold magenta] 2. TESTING NEGATIVE PATH: RANDOM LIBRARY SAMPLE "
            f"({negative_count} tracks)[/bold magenta]\n"
            f"[bold]==================================================[/bold]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_neg = progress.add_task("Sampling random library tracks...", total=None)
            neg_tracks = plex.fetch_random_tracks(
                count=negative_count,
                exclude_keys=tested_keys,
                path_map_source=settings.processing.path_map_source,
                path_map_target=settings.processing.path_map_target,
            )
            progress.update(task_neg, completed=True, visible=False)

        for track in neg_tracks:
            eval_res = _evaluate_track_mood(track, target_mood, tuner, analyzer)
            if not eval_res:
                continue
            score, threshold, assigned, is_match = eval_res
            hdr = (
                f"[cyan]'{track.title}'[/cyan] by [yellow]{track.artist}[/yellow] "
                f"[dim](ratingKey={track.rating_key})[/dim]"
            )

            if not is_match:
                neg_rejected += 1
                console.print(
                    f"  [green]✓ REJECTED[/green] {hdr}\n"
                    f"             [dim](score: {score:.3f} < {threshold:.3f})[/dim]"
                )
            else:
                neg_false_pos += 1
                console.print(
                    f"  [red]⚠️ FALSE POSITIVE[/red] {hdr}\n"
                    f"             [bold red]Matched '{target_mood}'[/bold red] "
                    f"({score:.3f} >= {threshold:.3f})"
                )

    # 3. Summary Report Card
    tot_c = cand_pass + cand_runner_up + cand_miss
    tot_n = neg_rejected + neg_false_pos

    table = Table(title=f"Personalized Mood Validation Summary: {target_mood}", show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Result", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Details", style="dim")

    if tot_c:
        table.add_row(
            "Candidate Matches",
            f"[green]{cand_pass}/{tot_c}[/green]",
            f"[bold green]{cand_pass / tot_c * 100:.1f}%[/bold green]",
            f"Playlist: {playlist_title}",
        )
        if cand_runner_up:
            r_str = ", ".join(f"{r} ({c})" for r, c in rivals.items())
            table.add_row(
                "Candidate Rivals",
                f"[yellow]{cand_runner_up}/{tot_c}[/yellow]",
                f"[yellow]{cand_runner_up / tot_c * 100:.1f}%[/yellow]",
                f"Beaten by: {r_str}",
            )
        if cand_miss:
            table.add_row(
                "Candidate Misses",
                f"[red]{cand_miss}/{tot_c}[/red]",
                f"[red]{cand_miss / tot_c * 100:.1f}%[/red]",
                "Below threshold",
            )
    if tot_n:
        table.add_row(
            "Negative Specificity",
            f"[green]{neg_rejected}/{tot_n}[/green]",
            f"[bold green]{neg_rejected / tot_n * 100:.1f}%[/bold green]",
            "Random library sample",
        )
        if neg_false_pos:
            table.add_row(
                "False Positives",
                f"[red]{neg_false_pos}/{tot_n}[/red]",
                f"[red]{neg_false_pos / tot_n * 100:.1f}%[/red]",
                f"Erroneously matched '{target_mood}'",
            )

    console.print("")
    console.print(table)
