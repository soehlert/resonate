"""Main entrypoint orchestrator for Resonate CLI app."""

import html
import logging
import os
import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from resonate.config import load_config
from resonate.models import ProcessingResult
from resonate.modules.beets import BeetsTagger
from resonate.modules.bpm import BpmDetector
from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.external_metadata import DiscogsFetcher, MusicBrainzFetcher
from resonate.modules.lastfm import LastFmFetcher
from resonate.modules.mutagen import MutagenTagger
from resonate.modules.plex import PlexSync
from resonate.modules.tag_mapper import (
    DEFAULT_MOOD_TAGS,
    DEFAULT_PRIMARY_GENRES,
    DEFAULT_SUB_GENRES,
    TagMapper,
    get_genre_seeded_moods,
    resolve_mood_conflicts,
)
from resonate.utils.state import StateManager
from resonate.wizard import run_wizard

app = typer.Typer(
    name="resonate",
    help="Resonate: Music metadata enrichment engine for genres, moods, and BPM.",
    add_completion=False,
)
console = Console()

GENRE_KEYWORDS = {
    "rock",
    "punk",
    "metal",
    "hardcore",
    "pop",
    "jazz",
    "blues",
    "folk",
    "country",
    "classical",
    "hiphop",
    "hip hop",
    "rap",
    "electronic",
    "techno",
    "house",
    "indie",
    "alternative",
    "reggae",
    "ska",
    "grunge",
    "synthpop",
    "instrumental",
}

RECOGNIZED_MOOD_KEYWORDS = {
    "party",
    "dance",
    "club",
    "lively",
    "fun",
    "celebration",
    "festive",
    "hangout",
    "chill",
    "mellow",
    "feel-good",
    "friendly",
    "upbeat",
    "relaxed",
    "calm",
    "energetic",
    "intense",
    "driving",
    "powerful",
    "aggressive",
    "hardcore",
    "thrash",
    "nyhc",
    "metalcore",
    "grindcore",
    "rowdy",
    "groovy",
    "funky",
    "rhythmic",
    "soulful",
    "boogie",
    "smooth",
    "acoustic",
    "unplugged",
    "intimate",
    "organic",
    "warm",
    "romantic",
    "electronic",
    "synth",
    "hypnotic",
    "futuristic",
    "atmospheric",
    "melancholic",
    "sad",
    "bittersweet",
    "somber",
    "brooding",
    "gloomy",
    "emotional",
    "happy",
    "dark",
    "heavy",
    "space",
    "summer",
    "ballad",
    "dream",
    "inspiring",
    "motivational",
    "cool",
    "hype",
    "gritty",
    "laid-back",
    "conscious",
    "street",
    "vibes",
    "flow",
    "surf",
}


@app.command(name="setup")
@app.command(name="wizard")
def wizard_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to config file to save wizard settings"),
    ] = "config.yaml",
) -> None:
    """Run interactive setup wizard to configure Resonate."""
    run_wizard(config_path=config)


def is_valid_mood_tag(tag: str, artist: str, album: str | None) -> bool:
    tag_lower = tag.lower().strip()

    # 1. Skip if it is a genre keyword
    if any(g in tag_lower for g in GENRE_KEYWORDS):
        return False

    # 2. Skip if it contains the artist name or any part of it (if longer than 3 chars)
    artist_lower = artist.lower().strip()
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return False
    artist_words = [w.strip() for w in artist_lower.split() if len(w.strip()) > 3]
    if any(w in tag_lower for w in artist_words):
        return False

    # 3. Skip if it contains the album name
    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return False
        album_words = [w.strip() for w in album_lower.split() if len(w.strip()) > 3]
        if any(w in tag_lower for w in album_words):
            return False

    # 4. Skip if it contains any digits (like 90s, 1999)
    if any(c.isdigit() for c in tag_lower):
        return False

    # 5. Whitelist Filter: Must match or contain a recognized mood keyword
    words = tag_lower.replace("-", " ").split()
    if not (
        any(w in RECOGNIZED_MOOD_KEYWORDS for w in words)
        or any(k in tag_lower for k in RECOGNIZED_MOOD_KEYWORDS)
    ):
        return False

    # 6. Skip common non-mood/boilerplate descriptors
    boilerplate = {
        "chicago",
        "american",
        "us",
        "uk",
        "british",
        "english",
        "australian",
        "canadian",
        "german",
        "french",
        "japanese",
        "seen live",
        "live",
        "favorites",
        "favourite",
        "favorite",
        "love",
        "heard on",
        "pandora",
        "spotify",
        "playlist",
        "track",
        "song",
        "album",
        "artist",
        "music",
        "singer",
        "songwriter",
        "band",
        "great",
        "nice",
        "awesome",
        "good",
        "cool",
        "mp3",
        "tag",
        "recommend",
        "soundtrack",
        "ost",
        "theme",
        "version",
        "remix",
        "cover",
    }
    if any(b in tag_lower for b in boilerplate):
        return False

    return True


def is_valid_subgenre_tag(tag: str, artist: str, album: str | None) -> bool:
    """Filter out non-genre tags, playlists, TV shows, and decades from subgenre candidates."""
    tag_lower = tag.lower().strip()

    # 1. Skip if it contains digits (like 2010s, 70s, 2017 albums, s36, 1982)
    if any(c.isdigit() for c in tag_lower):
        return False

    # 2. Skip if it contains the artist name or any part of it
    artist_lower = artist.lower().strip()
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return False
    artist_words = [w.strip() for w in artist_lower.split() if len(w.strip()) > 3]
    if any(w in tag_lower for w in artist_words):
        return False

    # 3. Skip if it contains the album name
    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return False
        album_words = [w.strip() for w in album_lower.split() if len(w.strip()) > 3]
        if any(w in tag_lower for w in album_words):
            return False

    # Whitelist recognized compound subgenres before boilerplate check
    # so 'singer-songwriter' is never dropped by single words 'singer' or 'songwriter'
    compound_subgenre_whitelist = {
        "singer-songwriter",
        "singer songwriter",
        "indie rock",
        "indie pop",
        "indie folk",
        "folk rock",
        "pop rock",
        "punk rock",
        "hard rock",
        "soft rock",
        "acoustic rock",
        "country rock",
        "southern rock",
        "roots rock",
        "garage rock",
        "psychedelic rock",
        "progressive rock",
        "prog rock",
        "heavy metal",
        "progressive metal",
        "thrash metal",
        "death metal",
        "black metal",
        "doom metal",
        "power metal",
        "hardcore punk",
        "skate punk",
        "pop-punk",
        "post-hardcore",
        "post-punk",
        "post-rock",
        "synth-pop",
        "synthpop",
        "synthwave",
        "dance-pop",
        "trip-hop",
        "hip-hop",
        "hip hop",
        "lo-fi",
        "contemporary r&b",
        "alt-country",
        "blues rock",
        "electric blues",
        "chicago blues",
        "delta blues",
        "chamber music",
    }
    if tag_lower in compound_subgenre_whitelist:
        return True

    # 4. Skip common non-genre/boilerplate/playlist/TV descriptors
    boilerplate = {
        "fav",
        "favorites",
        "favourite",
        "favorite",
        "personal favourites",
        "seen live",
        "live",
        "heard on",
        "pandora",
        "spotify",
        "playlist",
        "track",
        "song",
        "album",
        "albums",
        "artist",
        "music",
        "singer",
        "songwriter",
        "band",
        "great",
        "nice",
        "awesome",
        "good",
        "mp3",
        "tag",
        "recommend",
        "soundtrack",
        "ost",
        "theme",
        "version",
        "remix",
        "cover",
        "gtst",
        "ludo sanders",
        "series",
        "tv",
        "label",
        "catfish",
        "radio",
        "bagel",
        "nachspiel",
        "gr last",
    }
    if any(b in tag_lower for b in boilerplate):
        return False

    return True


@app.command(name="analyze")
@app.command(name="enrich")
def analyze_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to config file"),
    ] = "config.yaml",
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", "-b", help="Batch size for processing"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Perform analysis without committing metadata changes"),
    ] = False,
    artist: Annotated[
        str | None,
        typer.Option("--artist", "-a", help="Filter Plex tracks by artist name"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Limit number of tracks to process"),
    ] = None,
    random_sample: Annotated[
        bool,
        typer.Option("--random", "-r", help="Randomize order of tracks before applying limit"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed track-by-track pipeline progress"),
    ] = False,
    sync_plex: Annotated[
        bool,
        typer.Option("--sync-plex", help="Push enriched tags directly to Plex Media Server"),
    ] = False,
    write_id3: Annotated[
        bool,
        typer.Option("--write-id3", help="Embed tags in FLAC/MP3 files using mutagen"),
    ] = False,
    overwrite_tags: Annotated[
        bool,
        typer.Option("--overwrite-tags", help="Overwrite existing metadata tags on disk/Plex"),
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
    """Enrich Plex music library with genres, sub-genres, moods, and BPM analysis."""
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

    # Determine which features to run (defaults to all if none specified)
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

    console.print(
        Panel.fit(
            f"[bold blue]Starting Metadata Enrichment[/bold blue]\n"
            f"Config: {config} | Batch Size: {settings.processing.batch_size} | "
            f"Dry Run: {settings.processing.dry_run} | Verbose: {verbose}\n"
            f"Sync Plex: {sync_plex} | Write ID3: {write_id3} | Overwrite Tags: {overwrite_tags}\n"
            f"Enrichments: {', '.join(features_str)}",
            border_style="blue",
        )
    )

    req_limit = limit if limit is not None else "all"
    artist_str = f" by artist '{artist}'" if artist else ""
    console.print(
        f"Grabbing {req_limit} songs{artist_str} "
        f"from Plex library '{settings.plex.library_name}'..."
    )
    all_tracks = plex_sync.fetch_audio_tracks(limit=None, artist=artist)

    processed_keys = state_mgr.get_processed_keys()
    should_overwrite = settings.processing.overwrite

    # Filter tracks to only those that exist locally on the host/container filesystem
    local_tracks = []
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
        t for t in local_tracks if should_overwrite or t.rating_key not in processed_keys
    ]

    if random_sample:
        import random

        random.shuffle(unprocessed_tracks)

    if limit is not None:
        unprocessed_tracks = unprocessed_tracks[:limit]

    total_tracks = len(unprocessed_tracks)
    overwrite_str = " (overwriting)" if should_overwrite else ""
    console.print(
        f"Found [bold cyan]{len(all_tracks)}[/bold cyan] total tracks on Plex, "
        f"[bold yellow]{skipped_count}[/bold yellow] not found locally, "
        f"[bold yellow]{len(processed_keys)}[/bold yellow] already processed{overwrite_str}. "
        f"Processing [bold green]{total_tracks}[/bold green] tracks."
    )

    if total_tracks == 0:
        console.print("[green]No new tracks to process![/green]")
        return

    # Initialize fetchers/analyzers/taggers
    lastfm_fetcher = LastFmFetcher(
        api_key=settings.lastfm.api_key,
        api_secret=settings.lastfm.api_secret,
    )
    mb_fetcher = MusicBrainzFetcher()
    discogs_fetcher = DiscogsFetcher(api_token=settings.discogs.api_token)

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

    # Track stats
    processed_count = 0
    genre_matches_count = 0
    subgenre_matches_count = 0
    mood_matches_count = 0
    bpm_detected_count = 0
    mutagen_writes_count = 0
    plex_syncs_count = 0
    skipped_tracks_count = 0

    bsize = settings.processing.batch_size

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

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

            for track in batch:
                if not verbose:
                    progress.update(
                        task_id,
                        description=f"[cyan]Processing: '{track.title}' by {track.artist}...",
                    )
                else:
                    console.print(
                        f"\n[bold magenta]Processing track:[/bold magenta] "
                        f"[cyan]'{track.title}'[/cyan] by [yellow]{track.artist}[/yellow] "
                        f"(ratingKey={track.rating_key})"
                    )

                # Resolve local path
                resolved_path = track.file_path or ""
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

                # 1. Fetch tags
                raw_tags = []
                if do_genre or do_subgenre or do_mood:
                    if verbose:
                        console.print(
                            "  [blue]Phase 1 (Metadata):[/blue] "
                            "Fetching tags from Last.fm & MusicBrainz..."
                        )

                    track_tags = lastfm_fetcher.get_track_tags(track.artist, track.title)
                    raw_tags.extend(track_tags)
                    track_specific_tags = list(track_tags)

                    if track.album:
                        album_tags = lastfm_fetcher.get_album_tags(track.artist, track.album)
                        raw_tags.extend(album_tags)
                    artist_tags = lastfm_fetcher.get_artist_tags(track.artist)
                    raw_tags.extend(artist_tags)

                    mb_tags = mb_fetcher.get_recording_tags(track.artist, track.title)
                    raw_tags.extend(mb_tags)
                    track_specific_tags.extend(mb_tags)

                    if settings.discogs.api_token:
                        discogs_tags = discogs_fetcher.get_release_genres(track.artist, track.title)
                        raw_tags.extend(discogs_tags)

                    raw_tags = [
                        html.unescape(t).strip()
                        for t in raw_tags
                        if isinstance(t, str) and t.strip()
                    ]
                    track_specific_tags = [
                        html.unescape(t).strip()
                        for t in track_specific_tags
                        if isinstance(t, str) and t.strip()
                    ]
                    seen = set()
                    raw_tags = [
                        t for t in raw_tags if not (t.lower() in seen or seen.add(t.lower()))
                    ]
                    track_seen = set()
                    track_specific_tags = [
                        t
                        for t in track_specific_tags
                        if not (t.lower() in track_seen or track_seen.add(t.lower()))
                    ]

                    if verbose:
                        console.print(
                            f"    Raw consolidated tags: {raw_tags if raw_tags else 'None'}"
                        )

                # 2. Map Genres and Moods
                mapped_genre = None
                mapped_subgenres = []
                mapped_moods = []

                if do_genre:
                    if raw_tags:
                        # Filter raw_tags so only tags with valid genre keywords are evaluated
                        genre_filtered_tags = [
                            t
                            for t in raw_tags
                            if any(g in t.lower().strip() for g in GENRE_KEYWORDS)
                        ]
                        tags_to_match = genre_filtered_tags if genre_filtered_tags else raw_tags
                        genre_matches = genre_mapper.match_genre_consensus(tags_to_match)
                        if genre_matches:
                            from collections import Counter

                            core_keywords = {
                                "rock",
                                "pop",
                                "hip-hop",
                                "hip hop",
                                "rap",
                                "gangsta rap",
                                "reggae",
                                "jazz",
                                "blues",
                                "metal",
                                "classical",
                                "electronic",
                                "country",
                                "folk",
                                "punk",
                                "soul",
                                "r&b",
                            }
                            genre_counts = Counter()
                            for g_name, raw_t, _score, raw_pos in genre_matches:
                                raw_lower = raw_t.lower().strip()
                                weight = 3 if any(ck in raw_lower for ck in core_keywords) else 1
                                # Bonus weight if raw tag is among top 3 consensus tags
                                if raw_pos < 3:
                                    weight += 5
                                genre_counts[g_name] += weight

                            mapped_genre = genre_counts.most_common(1)[0][0]
                            genre_matches_count += 1
                            if verbose:
                                console.print(
                                    f"    [green]Mapped Primary Genre:[/green] '{mapped_genre}'"
                                )

                    # Fallback to audio genre model if no primary genre mapped
                    has_audio = resolved_path and os.path.exists(resolved_path)
                    if not mapped_genre and settings.essentia.enabled and has_audio:
                        e_genre, e_subgenres = essentia_analyzer.analyze_genre_waveform(
                            resolved_path,
                            genre_mapper=genre_mapper,
                            subgenre_mapper=subgenre_mapper,
                        )
                        if e_genre:
                            mapped_genre = e_genre
                            genre_matches_count += 1
                            if verbose:
                                console.print(f"    [green]Audio Genre:[/green] '{mapped_genre}'")
                        if e_subgenres and not mapped_subgenres:
                            mapped_subgenres = e_subgenres
                            subgenre_matches_count += 1
                            if verbose:
                                console.print(
                                    f"    [green]Audio Sub-Genres:[/green] {mapped_subgenres}"
                                )

                if do_subgenre and raw_tags and not mapped_subgenres:
                    generic_primary = {
                        "rock",
                        "pop",
                        "metal",
                        "jazz",
                        "blues",
                        "country",
                        "folk",
                        "rap",
                        "hip hop",
                        "hiphop",
                        "electronic",
                        "dance",
                        "punk",
                    }
                    tags_for_subgenre = track_specific_tags if track_specific_tags else raw_tags
                    filtered_subgenre_tags = [
                        t
                        for t in tags_for_subgenre
                        if is_valid_subgenre_tag(t, track.artist, track.album)
                        and t.lower().strip() not in generic_primary
                    ]
                    subgenre_matches = subgenre_mapper.match_multiple_tags(
                        filtered_subgenre_tags if filtered_subgenre_tags else tags_for_subgenre,
                        max_matches=3,
                    )
                    mapped_subgenres = [s[0] for s in subgenre_matches]
                    # Fall back to raw_tags if track_specific_tags returned no sub-genres
                    if not mapped_subgenres and track_specific_tags and raw_tags:
                        filtered_fallback = [
                            t
                            for t in raw_tags
                            if is_valid_subgenre_tag(t, track.artist, track.album)
                            and t.lower().strip() not in generic_primary
                        ]
                        subgenre_matches = subgenre_mapper.match_multiple_tags(
                            filtered_fallback if filtered_fallback else raw_tags, max_matches=3
                        )
                        mapped_subgenres = [s[0] for s in subgenre_matches]
                    if mapped_subgenres:
                        subgenre_matches_count += 1
                        if verbose:
                            console.print(
                                f"    [green]Mapped Sub-Genres/Styles:[/green] {mapped_subgenres}"
                            )

                    # Elevate Primary Genre from generic Rock/Pop to Punk or Metal
                    # if subgenres indicate punk or metal
                    if mapped_genre in {"Rock", "Pop"} and mapped_subgenres:
                        if any(
                            s.lower()
                            in {
                                "punk rock",
                                "hardcore punk",
                                "post-hardcore",
                                "skate punk",
                                "pop-punk",
                            }
                            for s in mapped_subgenres
                        ):
                            mapped_genre = "Punk"
                        elif any(
                            s.lower()
                            in {
                                "heavy metal",
                                "thrash metal",
                                "death metal",
                                "black metal",
                                "doom metal",
                                "power metal",
                                "progressive metal",
                            }
                            for s in mapped_subgenres
                        ):
                            mapped_genre = "Metal"

                    # Reconcile Punk or Metal if acoustic/folk indicators are present
                    if mapped_genre in {"Punk", "Metal"}:
                        acoustic_chill_subgenres = {
                            "acoustic rock",
                            "soft rock",
                            "folk rock",
                            "indie folk",
                            "singer-songwriter",
                            "americana",
                            "lo-fi",
                            "chamber music",
                            "bluegrass",
                        }
                        has_acoustic_subgenre = any(
                            s.lower() in acoustic_chill_subgenres for s in mapped_subgenres
                        )
                        acoustic_keywords = {
                            "acoustic",
                            "mellow",
                            "chill",
                            "folk",
                            "unplugged",
                            "ambient",
                            "laid-back",
                        }
                        raw_acoustic_count = sum(
                            1 for t in raw_tags if any(ak in t.lower() for ak in acoustic_keywords)
                        )
                        has_punk_metal_subgenre = any(
                            s.lower()
                            in {
                                "punk rock",
                                "hardcore punk",
                                "post-hardcore",
                                "skate punk",
                                "pop-punk",
                                "heavy metal",
                                "thrash metal",
                                "death metal",
                                "black metal",
                                "doom metal",
                                "power metal",
                                "progressive metal",
                            }
                            for s in mapped_subgenres
                        )
                        if has_acoustic_subgenre or (
                            raw_acoustic_count >= 2 and not has_punk_metal_subgenre
                        ):
                            non_punk_metal = [
                                g
                                for g, _ in genre_counts.most_common()
                                if g not in {"Punk", "Metal"}
                            ]
                            if non_punk_metal:
                                mapped_genre = non_punk_metal[0]
                            elif any(
                                s.lower() in {"acoustic rock", "soft rock"}
                                for s in mapped_subgenres
                            ):
                                mapped_genre = "Rock"
                            elif any(
                                s.lower()
                                in {
                                    "folk rock",
                                    "indie folk",
                                    "singer-songwriter",
                                    "bluegrass",
                                    "americana",
                                }
                                for s in mapped_subgenres
                            ):
                                mapped_genre = "Folk"
                            elif any(s.lower() == "lo-fi" for s in mapped_subgenres):
                                mapped_genre = "Indie"
                            else:
                                mapped_genre = "Rock"
                            if verbose:
                                console.print(
                                    f"    [yellow]Reconciled Genre:[/yellow] '{mapped_genre}'"
                                )

                # 3. Detect BPM
                detected_bpm = None
                if do_bpm:
                    if resolved_path and os.path.exists(resolved_path):
                        if verbose:
                            console.print(
                                "  [blue]Phase 2 (BPM Detection):[/blue] Estimating audio tempo..."
                            )
                        detected_bpm = bpm_detector.detect_bpm(resolved_path)
                        if detected_bpm:
                            bpm_detected_count += 1
                            if verbose:
                                console.print(f"    [green]Detected BPM:[/green] {detected_bpm}")
                        elif verbose:
                            console.print("    [yellow]BPM Detection Failed[/yellow]")
                    elif verbose:
                        console.print(
                            f"  [yellow]Phase 2 (BPM Detection):[/yellow] "
                            f"Skipped - Audio file not found at '{resolved_path}'"
                        )

                if do_mood:
                    filtered_mood_tags = [
                        t
                        for t in track_specific_tags
                        if is_valid_mood_tag(t, track.artist, track.album)
                    ]
                    text_mood_matches = mood_mapper.match_multiple_tags(filtered_mood_tags)
                    text_mapped_moods = [m[0] for m in text_mood_matches]

                    candidate_seeds = list(
                        set(
                            text_mapped_moods
                            + (
                                get_genre_seeded_moods(mapped_subgenres)
                                if mapped_subgenres
                                else (
                                    get_genre_seeded_moods([mapped_genre]) if mapped_genre else []
                                )
                            )
                        )
                    )

                    e_mapped_moods = []
                    if settings.essentia.enabled:
                        if resolved_path and os.path.exists(resolved_path):
                            if verbose:
                                console.print(
                                    "  [blue]Phase 2.5 (Essentia Waveform Analysis):[/blue] "
                                    "Analyzing local waveform..."
                                )
                            e_moods, e_score, e_top = essentia_analyzer.analyze_waveform(
                                resolved_path,
                                settings.mapping.target_moods,
                                tag_mapper=mood_mapper,
                                bpm=detected_bpm,
                                candidate_seeds=candidate_seeds,
                            )
                            if verbose and e_top:
                                console.print("    [blue]Essentia Model Top Predictions:[/blue]")
                                for idx, (lbl, val) in enumerate(e_top, 1):
                                    console.print(f"      {idx}. '{lbl}': {val:.4f}")

                            if e_moods and e_score >= settings.essentia.threshold:
                                e_mapped_moods = e_moods
                                if verbose:
                                    console.print(
                                        f"    [green]Essentia Waveform Matches:[/green] {e_moods} "
                                        f"(score: {e_score:.4f} >= threshold: "
                                        f"{settings.essentia.threshold})"
                                    )
                            elif verbose:
                                console.print(
                                    "    [yellow]No Essentia Match:[/yellow] "
                                    "Waveform classification score below threshold"
                                )
                        elif verbose:
                            console.print(
                                f"  [yellow]Phase 2.5 (Essentia Waveform Analysis):[/yellow] "
                                f"Skipped - Audio file not found at '{resolved_path}'"
                            )

                    # Combine text tags, genre-seeded moods, and Essentia waveform predictions
                    combined_moods = list(text_mapped_moods)

                    # Seed natural acoustic moods from mapped sub-genres/styles
                    if mapped_subgenres:
                        seeded_moods = get_genre_seeded_moods(mapped_subgenres)
                        e_pred_dict = {p[0].lower(): float(p[1]) for p in e_top} if e_top else {}
                        is_raw_energetic = e_pred_dict.get("energetic", 0.0) >= 0.15
                        is_raw_heavy = e_pred_dict.get("heavy", 0.0) >= 0.08
                        is_raw_aggressive = e_pred_dict.get("aggressive", 0.0) >= 0.05
                        is_raw_dark = e_pred_dict.get("dark", 0.0) >= 0.10
                        melancholic_cluster_score = sum(
                            e_pred_dict.get(k, 0.0)
                            for k in ["sad", "ballad", "emotional", "melancholic"]
                        )
                        is_raw_melancholic_heavy = (
                            melancholic_cluster_score >= 0.25
                            or e_pred_dict.get("sad", 0.0) >= 0.15
                            or e_pred_dict.get("ballad", 0.0) >= 0.15
                            or e_pred_dict.get("emotional", 0.0) >= 0.15
                        )
                        is_high_tempo = detected_bpm is not None and detected_bpm >= 125

                        is_rowdy_or_heavy = (
                            is_raw_energetic
                            or is_raw_heavy
                            or is_raw_aggressive
                            or is_raw_dark
                            or is_raw_melancholic_heavy
                            or any(
                                em.lower()
                                in {
                                    "energetic",
                                    "heavy",
                                    "aggressive",
                                    "intense",
                                    "dark",
                                    "rowdy",
                                }
                                for em in e_mapped_moods
                            )
                            or any(
                                sg.lower()
                                in {
                                    "hard rock",
                                    "heavy metal",
                                    "thrash metal",
                                    "hardcore punk",
                                    "post-hardcore",
                                    "grunge",
                                    "punk rock",
                                    "skate punk",
                                    "progressive metal",
                                }
                                for sg in mapped_subgenres
                            )
                        )

                        is_low_tempo = detected_bpm is not None and detected_bpm < 100
                        is_slow_and_not_heavy = is_low_tempo and not (
                            is_raw_heavy or is_raw_aggressive
                        )

                        for sm in seeded_moods:
                            sm_l = sm.lower()
                            if sm_l == "chill hang":
                                # Millennial Indie, Americana, and Mellow Alt-Rock:
                                # Valid as Chill Hang unless rowdy, heavy, dark, or aggressive
                                if not (is_high_tempo or is_rowdy_or_heavy):
                                    if sm not in combined_moods:
                                        combined_moods.append(sm)
                            elif sm_l in {"rowdy", "aggressive", "heavy"}:
                                # Suppress rowdy/aggressive seeds on slow non-heavy tracks
                                if not is_slow_and_not_heavy:
                                    if (
                                        not e_mapped_moods
                                        or sm in text_mapped_moods
                                        or sm in e_mapped_moods
                                    ):
                                        if sm not in combined_moods:
                                            combined_moods.append(sm)
                            elif not e_mapped_moods:
                                if sm not in combined_moods:
                                    combined_moods.append(sm)
                            elif sm in text_mapped_moods or sm in e_mapped_moods:
                                if sm not in combined_moods:
                                    combined_moods.append(sm)

                    for em in e_mapped_moods:
                        if em not in combined_moods:
                            combined_moods.append(em)

                    # Apply BPM-Grounded Mood Validation across ALL combined moods
                    if detected_bpm is not None:
                        if detected_bpm >= 130:
                            # 130+ BPM is Energetic; strip Lively
                            combined_moods = [m for m in combined_moods if m.lower() != "lively"]
                        elif 110 <= detected_bpm < 130:
                            # 110-130 BPM is Lively; convert Energetic to Lively
                            combined_moods = [
                                "Lively" if m.lower() == "energetic" else m for m in combined_moods
                            ]
                            seen_m: set[str] = set()
                            combined_moods = [
                                m
                                for m in combined_moods
                                if not (m.lower() in seen_m or seen_m.add(m.lower()))
                            ]
                        else:
                            # Below 110 BPM is neither Energetic nor Lively
                            combined_moods = [
                                m
                                for m in combined_moods
                                if m.lower() not in {"energetic", "lively"}
                            ]

                    # Resolve mutually exclusive mood conflicts (e.g. Dark/Heavy vs Upbeat/Romantic)
                    combined_moods = resolve_mood_conflicts(combined_moods)

                    # Prioritize specific emotional/acoustic moods first
                    specific_moods = [
                        m for m in combined_moods if m.lower() not in {"energetic", "lively"}
                    ]
                    generic_moods = [
                        m for m in combined_moods if m.lower() in {"energetic", "lively"}
                    ]
                    mapped_moods = specific_moods + generic_moods
                    if mapped_moods:
                        mood_matches_count += 1
                        if verbose:
                            console.print(f"    [green]Mapped Moods:[/green] {mapped_moods}")

                # 4. Tag Writing (Mutagen)
                if write_id3 and resolved_path and os.path.exists(resolved_path):
                    if verbose:
                        console.print(
                            "  [blue]Phase 3 (Tag Writing):[/blue] Embedding tags in local file..."
                        )

                    genre_list = ([mapped_genre] if mapped_genre else []) + mapped_subgenres
                    success_mutagen = mutagen_tagger.update_file_tags(
                        file_path=resolved_path,
                        genres=genre_list if (do_genre or do_subgenre) else None,
                        moods=mapped_moods if do_mood else None,
                        bpm=detected_bpm if do_bpm else None,
                        overwrite_tags=overwrite_tags,
                        dry_run=settings.processing.dry_run,
                    )
                    if success_mutagen:
                        mutagen_writes_count += 1
                        if verbose:
                            console.print("    Local file metadata update: Success")

                # 5. Sync to Plex
                if sync_plex:
                    if verbose:
                        console.print(
                            "  [blue]Phase 3.5 (Plex Sync):[/blue] "
                            "Updating Plex Server track database..."
                        )

                    genre_list = ([mapped_genre] if mapped_genre else []) + mapped_subgenres
                    success_plex = plex_sync.update_track_metadata(
                        rating_key=track.rating_key,
                        genres=genre_list if (do_genre or do_subgenre) else None,
                        moods=mapped_moods if do_mood else None,
                        bpm=detected_bpm if do_bpm else None,
                        overwrite_tags=overwrite_tags,
                        dry_run=settings.processing.dry_run,
                    )
                    if success_plex:
                        plex_syncs_count += 1
                        if verbose:
                            console.print("    Plex server track metadata update: Success")

                # Beets fallback tagger for backwards compatibility (moods only)
                if do_mood and mapped_moods and settings.beets.enabled:
                    beets_tagger.update_file_mood(
                        resolved_path, mapped_moods[0], dry_run=settings.processing.dry_run
                    )

                # Render Live Transformation Report Table during verbose or dry-run
                if verbose or settings.processing.dry_run:
                    existing_genres = (
                        [g.tag for g in getattr(track, "genres", [])]
                        if hasattr(track, "genres")
                        else []
                    )
                    existing_moods = (
                        [m.tag for m in getattr(track, "moods", [])]
                        if hasattr(track, "moods")
                        else []
                    )
                    existing_bpm = getattr(track, "bpm", 0) or 0

                    table = Table(
                        title=f"Live Metadata Transformation: '{track.title}' by {track.artist}",
                        show_header=True,
                        header_style="bold magenta",
                    )
                    table.add_column("Metadata Field", style="bold cyan")
                    table.add_column("Before (Plex/File)", style="yellow")
                    table.add_column("After (Enriched for TuneBox)", style="green")

                    before_p_genre = existing_genres[0] if existing_genres else "(None)"
                    after_p_genre = mapped_genre if mapped_genre else "(Unchanged)"
                    table.add_row("Primary Genre", before_p_genre, after_p_genre)

                    before_sub = (
                        ", ".join(existing_genres[1:]) if len(existing_genres) > 1 else "(None)"
                    )
                    after_sub = ", ".join(mapped_subgenres) if mapped_subgenres else "(None)"
                    table.add_row("Sub-Genres / Styles", before_sub, after_sub)

                    before_m = ", ".join(existing_moods) if existing_moods else "(None)"
                    after_m = ", ".join(mapped_moods) if mapped_moods else "(None)"
                    table.add_row("Moods", before_m, after_m)

                    before_b = f"{existing_bpm} BPM" if existing_bpm else "0 / (None)"
                    after_b = f"{detected_bpm} BPM" if detected_bpm else "(Not detected)"
                    table.add_row("BPM", before_b, after_b)

                    console.print(table)
                    if settings.processing.dry_run:
                        console.print(
                            "    [bold yellow][DRY-RUN ACTIVE] "
                            "No changes saved to audio files or Plex database.[/bold yellow]\n"
                        )

                # Keep track of skipped
                any_enriched = mapped_genre or mapped_subgenres or mapped_moods or detected_bpm
                if not any_enriched:
                    skipped_tracks_count += 1

                processed_count += 1
                progress.advance(task_id)

                # Record results
                primary_mood = mapped_moods[0] if mapped_moods else None
                batch_results.append(
                    ProcessingResult(
                        rating_key=track.rating_key,
                        title=track.title,
                        artist=track.artist,
                        mapped_mood=primary_mood,
                        confidence=1.0 if primary_mood else 0.0,
                        source="hybrid",
                        timestamp=time.time(),
                    )
                )

            # Phase 4: Bulk save batch
            if not settings.processing.dry_run:
                state_mgr.save_results_batch(batch_results)

    # Print summary Table
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
    if sync_plex:
        table.add_row("Plex Sync Updates", str(plex_syncs_count))
    table.add_row("Skipped / Unmapped", str(skipped_tracks_count))

    console.print(table)


@app.command(name="status")
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


if __name__ == "__main__":
    app()
