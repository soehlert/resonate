"""CLI command to enrich Plex music library with genres, subgenres, moods, and BPM."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from resonate.config import load_config
from resonate.engine.mood_rules import DEFAULT_MOOD_TAGS
from resonate.engine.pipeline import EnrichmentPipeline
from resonate.engine.taxonomy import DEFAULT_PRIMARY_GENRES, DEFAULT_SUB_GENRES
from resonate.models import ProcessingResult, TrackItem
from resonate.modules.beets import BeetsTagger
from resonate.modules.bpm import BpmDetector
from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.lyrics import LyricsFetcher
from resonate.modules.mutagen import MutagenTagger
from resonate.modules.plex import PlexSync
from resonate.modules.tag_mapper import TagMapper
from resonate.providers.discogs import DiscogsProvider
from resonate.providers.lastfm import LastFmProvider
from resonate.providers.manager import ProviderManager
from resonate.providers.musicbrainz import MusicBrainzProvider
from resonate.utils.state import StateManager

console = Console()
logger = logging.getLogger(__name__)


def analyze_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to config file"),
    ] = "config.yaml",
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch chunk size for processing and DB commits (default: 100)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview mode — runs calculations without saving to SQLite, files, or Plex",
        ),
    ] = False,
    reprocess: Annotated[
        bool,
        typer.Option(
            "--reprocess",
            help="Re-analyzes tracks even if they were already processed in SQLite",
        ),
    ] = False,
    artist: Annotated[
        str | None,
        typer.Option("--artist", "-a", help="Filter Plex tracks by artist name"),
    ] = None,
    track: Annotated[
        str | None,
        typer.Option("--track", "-t", help="Filter Plex tracks by song/track title"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Limit total number of tracks to process in this run"),
    ] = None,
    random_sample: Annotated[
        bool,
        typer.Option("--random", "-r", help="Randomize order of tracks before applying limit"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed track-by-track pipeline progress"),
    ] = False,
    write_plex: Annotated[
        bool,
        typer.Option("--write-plex", help="Writes the enriched tags directly to Plex Media Server"),
    ] = False,
    write_id3: Annotated[
        bool,
        typer.Option(
            "--write-id3",
            help="Writes and embeds the enriched tags into local audio files on disk",
        ),
    ] = False,
    write_blank_tags: Annotated[
        bool,
        typer.Option(
            "--write-blank-tags",
            help="Only populate empty fields in ID3/Plex, leaving existing tags untouched",
        ),
    ] = False,
    genre: Annotated[
        bool,
        typer.Option("--genre", help="Enable primary genre classification"),
    ] = False,
    subgenre: Annotated[
        bool,
        typer.Option("--subgenre", help="Enable sub-genre/style classification"),
    ] = False,
    mood: Annotated[
        bool,
        typer.Option("--mood", help="Enable mood classification"),
    ] = False,
    bpm: Annotated[
        bool,
        typer.Option("--bpm", help="Enable BPM audio analysis"),
    ] = False,
) -> None:
    """Enrich Plex music library with genres, sub-genres, moods, and BPM analysis.

    Default Behavior:
      By default, running './resonate analyze' executes ALL enrichments:
        - Primary Genre classification
        - Sub-Genre & Style tagging
        - Acoustic Mood synthesis
        - BPM tempo audio analysis

    Selective Enrichments:
      If you only want specific enrichments, pass one or more flags:
        - ./resonate analyze --genre (only primary genre)
        - ./resonate analyze --subgenre (only sub-genres/styles)
        - ./resonate analyze --mood (only acoustic moods)
        - ./resonate analyze --bpm (only BPM detection)
        - ./resonate analyze --genre --mood (only genre and mood)

    Common Examples:
      - Preview run on 10 tracks without modifying files or Plex:
          ./resonate analyze --dry-run -v -l 10
      - Full enrichment writing to local audio files and Plex:
          ./resonate analyze --write-id3 --write-plex
      - Only populate blank/missing tags (leaving existing tags untouched):
          ./resonate analyze --write-id3 --write-plex --write-blank-tags
    """
    settings = load_config(config)

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    if batch_size is not None:
        settings.processing.batch_size = batch_size
    if dry_run:
        settings.processing.dry_run = dry_run

    state_mgr = StateManager(settings.database.sqlite_path)
    plex_sync = PlexSync(
        url=settings.plex.url,
        token=settings.plex.token,
        library_name=settings.plex.library_name,
    )

    console.print(f"[bold blue]Connecting to Plex Server at {settings.plex.url}...[/bold blue]")
    plex_sync.connect()

    run_all = not (genre or subgenre or mood or bpm)
    do_genre = genre or run_all
    do_subgenre = subgenre or run_all
    do_mood = mood or run_all
    do_bpm = bpm or run_all

    features_str = []
    if do_genre:
        features_str.append("Genre")
    if do_subgenre:
        features_str.append("Sub-Genre")
    if do_mood:
        features_str.append("Mood")
    if do_bpm:
        features_str.append("BPM")

    enrich_list_str = ", ".join(features_str)
    console.print(
        Panel.fit(
            f"[bold blue]Starting Metadata Enrichment[/bold blue]\n"
            f"Config: {config} | Batch Size: {settings.processing.batch_size} | "
            f"Dry Run: {settings.processing.dry_run} | Verbose: {verbose}\n"
            f"Write Plex: {write_plex} | Write ID3: {write_id3} | "
            f"Write Blank Tags Only: {write_blank_tags}\n"
            f"Enrichments: {enrich_list_str}",
            border_style="blue",
        )
    )

    req_limit = limit if limit is not None else "all"
    filter_parts = []
    if artist:
        filter_parts.append(f"artist '{artist}'")
    if track:
        filter_parts.append(f"track '{track}'")
    filter_str = f" by {', '.join(filter_parts)}" if filter_parts else ""
    console.print(
        f"Grabbing {req_limit} songs{filter_str} "
        f"from Plex library '{settings.plex.library_name}'..."
    )
    all_tracks = plex_sync.fetch_audio_tracks(limit=None, artist=artist, track_title=track)

    processed_keys = state_mgr.get_processed_keys()
    should_reprocess = reprocess or settings.processing.reprocess
    should_overwrite_tags = not write_blank_tags

    local_tracks: list[TrackItem] = []
    skipped_count = 0
    for t in all_tracks:
        resolved_path = t.file_path or ""
        if (
            resolved_path
            and settings.processing.path_map_source
            and settings.processing.path_map_target
        ):
            if resolved_path.startswith(settings.processing.path_map_source):
                resolved_path = resolved_path.replace(
                    settings.processing.path_map_source,
                    settings.processing.path_map_target,
                    1,
                )
        if resolved_path and os.path.exists(resolved_path):
            local_tracks.append(t)
        else:
            skipped_count += 1
            if verbose:
                console.print(
                    f"  [yellow]Notice:[/yellow] Track '{t.title}' by '{t.artist}' "
                    f"not found locally at '{resolved_path or 'unknown'}' - skipping."
                )

    unprocessed_tracks = [
        t for t in local_tracks if should_reprocess or t.rating_key not in processed_keys
    ]

    if random_sample:
        random.shuffle(unprocessed_tracks)

    if limit is not None:
        unprocessed_tracks = unprocessed_tracks[:limit]

    total_tracks = len(unprocessed_tracks)
    reprocess_str = " (reprocessing)" if should_reprocess else ""
    console.print(
        f"Found [bold cyan]{len(all_tracks)}[/bold cyan] total tracks on Plex, "
        f"[bold yellow]{skipped_count}[/bold yellow] missing on disk, "
        f"[bold yellow]{len(processed_keys)}[/bold yellow] already processed{reprocess_str}. "
        f"Processing [bold green]{total_tracks}[/bold green] tracks."
    )

    if total_tracks == 0:
        console.print("[green]No new tracks to process![/green]")
        return

    # Initialize ProviderManager with pluggable providers
    providers = [
        LastFmProvider(api_key=settings.lastfm.api_key, api_secret=settings.lastfm.api_secret),
        MusicBrainzProvider(),
    ]
    if settings.discogs.api_token:
        providers.append(DiscogsProvider(api_token=settings.discogs.api_token))

    provider_mgr = ProviderManager(providers=providers, state_manager=state_mgr)

    genre_mapper = TagMapper(
        target_moods=DEFAULT_PRIMARY_GENRES,
        model_name=settings.mapping.model_name,
        threshold=settings.mapping.genre_threshold,
    )
    subgenre_mapper = TagMapper(
        target_moods=DEFAULT_SUB_GENRES,
        model_name=settings.mapping.model_name,
        threshold=settings.mapping.subgenre_threshold,
    )
    mood_mapper = TagMapper(
        target_moods=settings.mapping.target_moods or DEFAULT_MOOD_TAGS,
        model_name=settings.mapping.model_name,
        threshold=settings.mapping.mood_threshold,
    )

    essentia_analyzer = EssentiaAnalyzer(
        models_dir=settings.essentia.models_dir,
        model_filename=settings.essentia.model_filename,
    )
    beets_tagger = BeetsTagger(
        binary_path=settings.beets.binary_path,
        enabled=settings.beets.enabled,
    )
    mutagen_tagger = MutagenTagger(enabled=settings.mutagen.enabled)
    bpm_detector = BpmDetector()
    lyrics_fetcher = LyricsFetcher(
        state_manager=state_mgr,
        prefer_embedded=settings.lyrics.prefer_embedded,
        lrclib_url=settings.lyrics.lrclib_url,
    )

    pipeline = EnrichmentPipeline(
        provider_manager=provider_mgr,
        genre_mapper=genre_mapper,
        subgenre_mapper=subgenre_mapper,
        mood_mapper=mood_mapper,
        essentia_analyzer=essentia_analyzer,
        bpm_detector=bpm_detector,
        lyrics_fetcher=lyrics_fetcher,
        mutagen_tagger=mutagen_tagger,
        state_manager=state_mgr,
    )

    processed_count = 0
    genre_matches_count = 0
    subgenre_matches_count = 0
    mood_matches_count = 0
    bpm_detected_count = 0
    mutagen_writes_count = 0
    plex_syncs_count = 0
    skipped_tracks_count = 0

    bsize = settings.processing.batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Processing tracks...", total=total_tracks)

        for i in range(0, total_tracks, bsize):
            batch = unprocessed_tracks[i : i + bsize]
            batch_results: list[ProcessingResult] = []

            for track_item in batch:
                if not verbose:
                    desc_text = f"[cyan]Processing: '{track_item.title}' by {track_item.artist}..."
                    progress.update(task_id, description=desc_text)
                else:
                    console.print(
                        f"\n[bold magenta]Processing track:[/bold magenta] "
                        f"[cyan]'{track_item.title}'[/cyan] by "
                        f"[yellow]{track_item.artist}[/yellow] (ratingKey={track_item.rating_key})"
                    )

                resolved_path = track_item.file_path or ""
                if (
                    resolved_path
                    and settings.processing.path_map_source
                    and settings.processing.path_map_target
                ):
                    if resolved_path.startswith(settings.processing.path_map_source):
                        resolved_path = resolved_path.replace(
                            settings.processing.path_map_source,
                            settings.processing.path_map_target,
                            1,
                        )

                # Execute Enrichment Pipeline
                enrichment = pipeline.enrich_track(
                    track=track_item,
                    resolved_path=resolved_path,
                    do_genre=do_genre,
                    do_subgenre=do_subgenre,
                    do_mood=do_mood,
                    do_bpm=do_bpm,
                    write_tags=write_id3 or settings.mutagen.enabled,
                    overwrite_tags=should_overwrite_tags,
                    dry_run=settings.processing.dry_run,
                    target_moods=settings.mapping.target_moods or DEFAULT_MOOD_TAGS,
                    essentia_threshold=settings.mapping.mood_threshold,
                )

                if enrichment.primary_genre:
                    genre_matches_count += 1
                if enrichment.subgenres:
                    subgenre_matches_count += 1
                if enrichment.moods:
                    mood_matches_count += 1
                if enrichment.bpm:
                    bpm_detected_count += 1
                if enrichment.mutagen_updated:
                    mutagen_writes_count += 1

                # Plex Server Sync
                if write_plex:
                    genre_list = (
                        [enrichment.primary_genre] if enrichment.primary_genre else []
                    ) + enrichment.subgenres
                    success_plex = plex_sync.update_track_metadata(
                        rating_key=track_item.rating_key,
                        genres=genre_list if (do_genre or do_subgenre) else None,
                        moods=enrichment.moods if do_mood else None,
                        bpm=enrichment.bpm if do_bpm else None,
                        overwrite_tags=should_overwrite_tags,
                        dry_run=settings.processing.dry_run,
                    )
                    if success_plex:
                        plex_syncs_count += 1

                # Beets fallback tagger (if enabled)
                if do_mood and enrichment.moods and settings.beets.enabled and resolved_path:
                    beets_tagger.update_file_mood(
                        resolved_path, enrichment.moods[0], dry_run=settings.processing.dry_run
                    )

                # Live Transformation Table during verbose or dry-run
                if verbose or settings.processing.dry_run:
                    existing_genres = (
                        [g.tag for g in getattr(track_item, "genres", [])]
                        if hasattr(track_item, "genres")
                        else []
                    )
                    existing_moods = (
                        [m.tag for m in getattr(track_item, "moods", [])]
                        if hasattr(track_item, "moods")
                        else []
                    )
                    existing_bpm = getattr(track_item, "bpm", 0) or 0

                    table_title = (
                        f"Live Transformation: '{track_item.title}' by {track_item.artist}"
                    )
                    table = Table(
                        title=table_title,
                        show_header=True,
                        header_style="bold magenta",
                    )
                    table.add_column("Metadata Field", style="bold cyan")
                    table.add_column("Before (Plex/File)", style="yellow")
                    table.add_column("After (Enriched for TuneBox)", style="green")

                    before_p = existing_genres[0] if existing_genres else "(None)"
                    after_p = enrichment.primary_genre or "(Unchanged)"
                    table.add_row("Primary Genre", before_p, after_p)

                    before_sub = (
                        ", ".join(existing_genres[1:]) if len(existing_genres) > 1 else "(None)"
                    )
                    after_sub = (
                        ", ".join(enrichment.subgenres) if enrichment.subgenres else "(None)"
                    )
                    table.add_row("Sub-Genres / Styles", before_sub, after_sub)

                    before_m = ", ".join(existing_moods) if existing_moods else "(None)"
                    after_m = ", ".join(enrichment.moods) if enrichment.moods else "(None)"
                    table.add_row("Moods", before_m, after_m)

                    before_b = f"{existing_bpm} BPM" if existing_bpm else "0 / (None)"
                    after_b = f"{enrichment.bpm} BPM" if enrichment.bpm else "(Not detected)"
                    table.add_row("BPM", before_b, after_b)

                    console.print(table)
                    if settings.processing.dry_run:
                        console.print(
                            "    [bold yellow][DRY-RUN ACTIVE] "
                            "No changes saved to audio files or Plex database.[/bold yellow]\n"
                        )

                any_enriched = (
                    enrichment.primary_genre
                    or enrichment.subgenres
                    or enrichment.moods
                    or enrichment.bpm
                )
                if not any_enriched:
                    skipped_tracks_count += 1

                processed_count += 1
                progress.advance(task_id)

                primary_mood = enrichment.moods[0] if enrichment.moods else None
                batch_results.append(
                    ProcessingResult(
                        rating_key=track_item.rating_key,
                        title=track_item.title,
                        artist=track_item.artist,
                        mapped_mood=primary_mood,
                        confidence=1.0 if primary_mood else 0.0,
                        source="hybrid",
                        timestamp=time.time(),
                    )
                )

            if not settings.processing.dry_run:
                state_mgr.save_results_batch(batch_results)

    table = Table(title="Analysis Summary Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Processed", str(processed_count))
    if do_genre:
        table.add_row("Genres Mapped", str(genre_matches_count))
    if do_subgenre:
        table.add_row("Sub-Genres Mapped", str(subgenre_matches_count))
    if do_mood:
        table.add_row("Moods Mapped", str(mood_matches_count))
    if do_bpm:
        table.add_row("BPMs Estimated", str(bpm_detected_count))
    if write_id3:
        table.add_row("Files Tagged (Mutagen)", str(mutagen_writes_count))
    if write_plex:
        table.add_row("Plex Sync Updates", str(plex_syncs_count))
    table.add_row("Skipped / Unmapped", str(skipped_tracks_count))

    console.print(table)
