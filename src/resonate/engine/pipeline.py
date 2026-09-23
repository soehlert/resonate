"""Track enrichment pipeline orchestrator connecting providers, ML, taxonomy, and taggers."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from typing import TYPE_CHECKING

from resonate.engine.mood_rules import (
    GENRE_KEYWORDS,
    get_genre_seeded_moods,
    is_valid_mood_tag,
    synthesize_track_moods,
)
from resonate.engine.taxonomy import (
    deduplicate_subgenres,
    is_valid_subgenre_tag,
    promote_genre_by_subgenres,
    sanitize_subgenres_for_genre,
)
from resonate.models import (
    LyricsAnalysisResult,
    TrackEnrichmentResult,
    TrackItem,
)
from resonate.utils.audio import is_valid_audio_header

if TYPE_CHECKING:
    from resonate.modules.bpm import BpmDetector
    from resonate.modules.essentia import EssentiaAnalyzer
    from resonate.modules.lyrics import LyricsFetcher
    from resonate.modules.mutagen import MutagenTagger
    from resonate.modules.personalized_tuning import PersonalizedMoodTuner
    from resonate.modules.tag_mapper import TagMapper
    from resonate.providers.manager import ProviderManager
    from resonate.utils.state import StateManager

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """End-to-end music track enrichment pipeline coordinating all stages."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        genre_mapper: TagMapper,
        subgenre_mapper: TagMapper,
        mood_mapper: TagMapper,
        essentia_analyzer: EssentiaAnalyzer | None = None,
        bpm_detector: BpmDetector | None = None,
        lyrics_fetcher: LyricsFetcher | None = None,
        mutagen_tagger: MutagenTagger | None = None,
        state_manager: StateManager | None = None,
        personalized_tuner: PersonalizedMoodTuner | None = None,
    ) -> None:
        """Initialize EnrichmentPipeline with required mapper and provider components."""
        self.provider_manager = provider_manager
        self.genre_mapper = genre_mapper
        self.subgenre_mapper = subgenre_mapper
        self.mood_mapper = mood_mapper
        self.essentia_analyzer = essentia_analyzer
        self.bpm_detector = bpm_detector
        self.lyrics_fetcher = lyrics_fetcher
        self.mutagen_tagger = mutagen_tagger
        self.state_manager = state_manager
        if personalized_tuner is not None:
            self.personalized_tuner = personalized_tuner
        else:
            try:
                from resonate.modules.personalized_tuning import PersonalizedMoodTuner

                tuner = PersonalizedMoodTuner()
                self.personalized_tuner = tuner if tuner.load_model() else None
            except Exception:
                self.personalized_tuner = None

    def enrich_track(
        self,
        track: TrackItem,
        resolved_path: str | None = None,
        do_genre: bool = True,
        do_subgenre: bool = True,
        do_mood: bool = True,
        do_bpm: bool = True,
        write_tags: bool = False,
        overwrite_tags: bool = False,
        dry_run: bool = False,
        target_moods: list[str] | None = None,
        essentia_threshold: float = 0.35,
    ) -> TrackEnrichmentResult:
        """Enrich a single music track through all processing phases and return typed result."""
        phase_timings: dict[str, float] = {}
        t_start = time.perf_counter()

        # 1. External Metadata Discovery (Concurrent with SQLite Caching)
        t0 = time.perf_counter()
        raw_tags, track_specific, has_verified, resolved_art = (
            self.provider_manager.get_tags_for_track(
                artist=track.artist,
                title=track.title,
                album=track.album,
                album_artist=getattr(track, "album_artist", None),
            )
        )
        phase_timings["metadata"] = time.perf_counter() - t0

        mapped_genre: str | None = None
        mapped_subgenres: list[str] = []
        mapped_moods: list[str] = []
        detected_bpm: int | None = None
        lyrics_res: LyricsAnalysisResult | None = None

        has_audio = bool(resolved_path and os.path.exists(resolved_path))

        # 2. Genre & Subgenre Mapping
        t_genre = time.perf_counter()
        if do_genre and raw_tags:
            # Check track-specific tags first before falling back to album/artist tags
            track_genre_filtered = [
                t for t in track_specific if any(g in t.lower().strip() for g in GENRE_KEYWORDS)
            ]
            genre_tags_to_match = (
                track_genre_filtered
                if track_genre_filtered
                else [t for t in raw_tags if any(g in t.lower().strip() for g in GENRE_KEYWORDS)]
                or raw_tags
            )
            genre_matches = self.genre_mapper.match_genre_consensus(genre_tags_to_match)
            if genre_matches:
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
                genre_counts: Counter[str] = Counter()
                for g_name, raw_t, _score, raw_pos in genre_matches:
                    raw_lower = raw_t.lower().strip()
                    weight = 3 if any(ck in raw_lower for ck in core_keywords) else 1
                    if raw_pos < 3:
                        weight += 5
                    genre_counts[g_name] += weight

                mapped_genre = genre_counts.most_common(1)[0][0]

        # Shared Audio Buffers (single-pass 90s decode at 44.1kHz, resampled to 16kHz)
        audio_44k = None
        audio_16k = None
        audio_loaded = False

        def get_audio_buffers():
            nonlocal audio_44k, audio_16k, audio_loaded
            if not audio_loaded:
                audio_loaded = True
                if has_audio and resolved_path and is_valid_audio_header(resolved_path):
                    t_audio = time.perf_counter()
                    try:
                        import essentia.standard as es

                        audio_44k = es.EasyLoader(
                            filename=resolved_path, sampleRate=44100, startTime=0, endTime=90
                        )()
                        audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(
                            audio_44k
                        )
                    except Exception as err:
                        logger.debug(
                            f"Failed single-pass audio decode for '{resolved_path}': {err}"
                        )
                    phase_timings["audio_decode"] = time.perf_counter() - t_audio
            return audio_44k, audio_16k

        # Audio Waveform Genre Fallback (if no tag match or solely unverified artist tags)
        if (
            (not mapped_genre or not has_verified)
            and self.essentia_analyzer
            and has_audio
            and resolved_path
        ):
            _, buf_16k = get_audio_buffers()
            e_genre, e_subgenres = self.essentia_analyzer.analyze_genre_waveform(
                resolved_path,
                genre_mapper=self.genre_mapper,
                subgenre_mapper=self.subgenre_mapper,
                audio=buf_16k,
            )
            if e_genre:
                mapped_genre = e_genre
            if e_subgenres and not mapped_subgenres:
                mapped_subgenres = e_subgenres

        # Subgenre Classification (Track-level tags strictly prioritized over album tags)
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
            # 1. Try track-specific subgenre tags first
            track_sg_tags = [
                t
                for t in track_specific
                if is_valid_subgenre_tag(t, resolved_art, track.album)
                and t.lower().strip() not in generic_primary
            ]
            if track_sg_tags:
                sg_matches = self.subgenre_mapper.match_subgenre_consensus(
                    track_sg_tags,
                    max_matches=3,
                )
                mapped_subgenres = [s[0] for s in sg_matches]

            # 2. Fallback to album/raw tags only if track has no specific subgenre tags
            if not mapped_subgenres:
                filtered_sg_tags = [
                    t
                    for t in raw_tags
                    if is_valid_subgenre_tag(t, resolved_art, track.album)
                    and t.lower().strip() not in generic_primary
                ]
                sg_matches = self.subgenre_mapper.match_subgenre_consensus(
                    filtered_sg_tags if filtered_sg_tags else raw_tags,
                    max_matches=3,
                )
                mapped_subgenres = [s[0] for s in sg_matches]

        # Taxonomy Hierarchy Promotion (e.g. Rock -> Punk/Metal)
        if mapped_genre in {"Rock", "Pop"} and mapped_subgenres:
            promoted, _decision = promote_genre_by_subgenres(mapped_genre, mapped_subgenres)
            if promoted:
                mapped_genre = promoted

        # Cross-Family Sanitization and Deduplication
        if mapped_subgenres:
            mapped_subgenres = sanitize_subgenres_for_genre(
                mapped_genre, mapped_subgenres, raw_tags
            )
            mapped_subgenres = deduplicate_subgenres(mapped_genre, mapped_subgenres)
        if do_genre or do_subgenre:
            phase_timings["genre_tax"] = time.perf_counter() - t_genre

        # 3. Essentia Waveform Analysis & Acoustic Mood Prediction
        t_mood = time.perf_counter()
        e_mapped_moods: list[str] = []
        e_top: list[tuple[str, float]] = []
        text_mapped_moods: list[str] = []

        if do_mood:
            filtered_mood_tags = [
                t for t in track_specific if is_valid_mood_tag(t, resolved_art, track.album)
            ]
            text_mood_matches = self.mood_mapper.match_multiple_tags(filtered_mood_tags)
            text_mapped_moods = [m[0] for m in text_mood_matches]

        # Candidate seeds for Essentia only include track-specific moods (not album seeds)
        candidate_seeds = list(set(text_mapped_moods))

        effnet_embeddings = None
        if self.essentia_analyzer and has_audio and resolved_path:
            target_list = target_moods or self.mood_mapper.target_moods
            _, buf_16k = get_audio_buffers()
            effnet_embeddings = self.essentia_analyzer.extract_embeddings(
                audio=buf_16k, file_path=resolved_path
            )
            if effnet_embeddings is not None:
                e_moods, e_score, e_top = self.essentia_analyzer.predict_moods(
                    embeddings=effnet_embeddings,
                    target_moods=target_list,
                    tag_mapper=self.mood_mapper,
                    bpm=None,
                    candidate_seeds=candidate_seeds,
                )
                if e_moods and e_score >= essentia_threshold:
                    e_mapped_moods = e_moods

        pers_moods: list[tuple[str, float]] = []
        if (
            self.personalized_tuner
            and self.personalized_tuner.is_trained
            and effnet_embeddings is not None
        ):
            pers_moods = self.personalized_tuner.predict(effnet_embeddings, top_k=1)

        if do_mood:
            phase_timings["mood_ml"] = time.perf_counter() - t_mood

        # 4. Detect BPM
        t_bpm = time.perf_counter()
        if do_bpm and self.bpm_detector and has_audio and resolved_path:
            buf_44k, _ = get_audio_buffers()
            detected_bpm = self.bpm_detector.detect_bpm(
                resolved_path,
                genre_hint=mapped_genre,
                subgenres=mapped_subgenres,
                raw_tags=raw_tags,
                audio_predictions=e_top,
                audio=buf_44k,
            )
        if do_bpm:
            phase_timings["bpm"] = time.perf_counter() - t_bpm

        # 5. Lyrics Retrieval & Sentiment/Mood Analysis
        t_lyrics = time.perf_counter()
        if self.lyrics_fetcher:
            lyrics_text, lyrics_src = self.lyrics_fetcher.get_lyrics(
                artist=resolved_art,
                title=track.title,
                album=track.album,
                file_path=resolved_path,
            )
            if lyrics_text:
                lyrics_res = self.lyrics_fetcher.analyze_lyrics(
                    lyrics_text=lyrics_text,
                    source=lyrics_src,
                    tag_mapper=self.mood_mapper,
                )
        phase_timings["lyrics"] = time.perf_counter() - t_lyrics

        # 6. Synthesize Final Moods
        if do_mood:
            seeded = (
                get_genre_seeded_moods(mapped_subgenres)
                if mapped_subgenres
                else []
            )
            mapped_moods = synthesize_track_moods(
                text_moods=text_mapped_moods,
                seeded_moods=seeded,
                essentia_moods=e_mapped_moods,
                essentia_top=e_top,
                detected_bpm=detected_bpm,
                lyrics_analysis=lyrics_res,
                primary_genre=mapped_genre,
                subgenres=mapped_subgenres,
                raw_tags=raw_tags,
                personalized_moods=pers_moods,
            )

        # 7. Write Embedded Mutagen Audio Tags
        t_mutagen = time.perf_counter()
        mutagen_updated = False
        if (
            write_tags
            and self.mutagen_tagger
            and self.mutagen_tagger.enabled
            and has_audio
            and resolved_path
        ):
            genres_to_write = ([mapped_genre] if mapped_genre else []) + mapped_subgenres
            mutagen_updated = self.mutagen_tagger.update_file_tags(
                file_path=resolved_path,
                genres=genres_to_write,
                moods=mapped_moods,
                bpm=detected_bpm,
                overwrite_tags=overwrite_tags,
                dry_run=dry_run,
            )
        if write_tags:
            phase_timings["mutagen"] = time.perf_counter() - t_mutagen

        total_duration_ms = (time.perf_counter() - t_start) * 1000

        return TrackEnrichmentResult(
            rating_key=track.rating_key,
            title=track.title,
            artist=resolved_art,
            album=track.album,
            resolved_path=resolved_path,
            primary_genre=mapped_genre,
            subgenres=mapped_subgenres,
            moods=mapped_moods,
            bpm=detected_bpm,
            lyrics_valence=lyrics_res.valence_score if lyrics_res else None,
            raw_tags=raw_tags,
            track_specific_tags=track_specific,
            essentia_predictions=e_top,
            has_verified_tags=has_verified,
            mutagen_updated=mutagen_updated,
            plex_updated=False,
            skipped=False,
            duration_ms=total_duration_ms,
            phase_timings=phase_timings,
        )
