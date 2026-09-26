"""Track enrichment pipeline orchestrator connecting providers, ML, taxonomy, and taggers."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

from resonate.config import MoodRulesConfig
from resonate.engine.mood_rules import (
    GENRE_KEYWORDS,
    get_genre_seeded_moods,
    is_valid_mood_tag,
    synthesize_track_moods,
)
from resonate.engine.taxonomy import (
    DEFAULT_PRIMARY_GENRES,
    _get_family_for_tag,
    deduplicate_subgenres,
    filter_subgenres_by_family,
    is_valid_subgenre_tag,
    promote_genre_by_subgenres,
)
from resonate.engine.tracer import DecisionTracer
from resonate.models import (
    LyricsAnalysisResult,
    TrackEnrichmentResult,
    TrackItem,
)

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
        mood_rules: MoodRulesConfig | None = None,
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
        if mood_rules is not None:
            self.mood_rules = mood_rules
        else:
            try:
                from resonate.config import load_config

                self.mood_rules = load_config().mood_rules
            except Exception:
                self.mood_rules = MoodRulesConfig()
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
        tracer = DecisionTracer()
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
                if resolved_path:
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

        # Subgenre Classification (Track-level tags strictly prioritized)
        sg_matches: list[tuple[str, str, float]] = []
        track_sg_tags: list[str] = []
        if do_subgenre and raw_tags:
            generic_primary = {g.lower() for g in DEFAULT_PRIMARY_GENRES}
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
                    max_matches=10,
                )
                mapped_subgenres = [s[0] for s in sg_matches]

        # Audio Waveform Genre & Subgenre Fallback (runs if genre or subgenres are blank/unverified)
        needs_genre_fallback = (not mapped_genre or not has_verified) and do_genre
        needs_subgenre_fallback = not mapped_subgenres and do_subgenre
        if (
            (needs_genre_fallback or needs_subgenre_fallback)
            and self.essentia_analyzer
            and resolved_path
        ):
            _, buf_16k = get_audio_buffers()
            genre_res = self.essentia_analyzer.analyze_genre_waveform(
                resolved_path,
                genre_mapper=self.genre_mapper,
                subgenre_mapper=self.subgenre_mapper,
                audio=buf_16k,
            )
            if isinstance(genre_res, tuple) and len(genre_res) == 2:
                essentia_genre, essentia_subgenres = genre_res
                if needs_genre_fallback and essentia_genre:
                    mapped_genre = essentia_genre
                if needs_subgenre_fallback and essentia_subgenres:
                    mapped_subgenres = essentia_subgenres

        # Taxonomy Hierarchy Promotion (e.g. Rock -> Punk/Metal)
        if mapped_genre in {"Rock", "Pop"} and mapped_subgenres:
            subgenre_scores = (
                {s[0]: s[2] for s in sg_matches}
                if sg_matches
                else {s: 1.0 for s in mapped_subgenres}
            )
            promoted, _decision = promote_genre_by_subgenres(
                mapped_genre, subgenre_scores, raw_tags=raw_tags
            )
            if promoted:
                mapped_genre = promoted

        # Primary Genre Family Filtering and Deduplication
        if mapped_subgenres:
            mapped_subgenres = filter_subgenres_by_family(mapped_genre, mapped_subgenres)
            mapped_subgenres = deduplicate_subgenres(mapped_genre, mapped_subgenres)[:3]

        # Artist tags fallback: If family filtering eliminated all subgenres
        # (or track/audio left none), inspect artist tags last.
        if not mapped_subgenres and raw_tags and do_subgenre:
            generic_primary = {g.lower() for g in DEFAULT_PRIMARY_GENRES}
            filtered_sg_tags = [
                t
                for t in raw_tags
                if is_valid_subgenre_tag(t, resolved_art, track.album)
                and t.lower().strip() not in generic_primary
            ]
            artist_sg_matches = self.subgenre_mapper.match_subgenre_consensus(
                filtered_sg_tags if filtered_sg_tags else raw_tags,
                max_matches=10,
            )
            artist_subgenres = [s[0] for s in artist_sg_matches]
            if artist_subgenres:
                family_counts: Counter[str] = Counter()
                for sg in artist_subgenres:
                    fam = _get_family_for_tag(sg)
                    if fam:
                        family_counts[fam] += 1
                if family_counts:
                    top_family = family_counts.most_common(1)[0][0]
                    # If primary genre was unverified, artist subgenres push up to primary genre
                    if not has_verified and mapped_genre != top_family:
                        old_genre = mapped_genre
                        mapped_genre = top_family
                        tracer.record(
                            f"Artist subgenre fallback pushed primary genre from '{old_genre}' "
                            f"to '{mapped_genre}'"
                        )

                if mapped_genre in {"Rock", "Pop", "Reggae"}:
                    subgenre_scores = {s[0]: s[2] for s in artist_sg_matches}
                    promoted, _decision = promote_genre_by_subgenres(
                        mapped_genre, subgenre_scores, raw_tags=raw_tags
                    )
                    if promoted:
                        old_genre = mapped_genre
                        mapped_genre = promoted
                        tracer.record(
                            f"Taxonomy promotion pushed primary genre from '{old_genre}' "
                            f"to '{mapped_genre}'"
                        )

                rescued_subgenres = filter_subgenres_by_family(mapped_genre, artist_subgenres)
                if rescued_subgenres:
                    mapped_subgenres = deduplicate_subgenres(mapped_genre, rescued_subgenres)[:3]
                    tracer.record(
                        f"Artist subgenre fallback applied (strict family adherence to "
                        f"'{mapped_genre}'): {mapped_subgenres}"
                    )
        if do_genre or do_subgenre:
            phase_timings["genre_tax"] = time.perf_counter() - t_genre
            tracer.record(
                f"Taxonomy Consensus: Primary='{mapped_genre}', Subgenres={mapped_subgenres}"
            )

        # 3. Essentia Waveform Analysis & Acoustic Mood Prediction
        t_mood = time.perf_counter()
        essentia_mapped_moods: list[str] = []
        essentia_top_preds: list[tuple[str, float]] = []
        text_mapped_moods: list[str] = []

        raw_mood_seeds: list[str] = []
        if do_mood:
            filtered_mood_tags = [
                t for t in track_specific if is_valid_mood_tag(t, resolved_art, track.album)
            ]
            text_mood_matches = self.mood_mapper.match_multiple_tags(filtered_mood_tags)
            text_mapped_moods = [m[0] for m in text_mood_matches]

            if not text_mapped_moods and raw_tags:
                raw_mood_filtered = [
                    t for t in raw_tags if is_valid_mood_tag(t, resolved_art, track.album)
                ]
                if raw_mood_filtered:
                    raw_matches = self.mood_mapper.match_multiple_tags(raw_mood_filtered)
                    raw_mood_seeds = [m[0] for m in raw_matches]

        # Candidate seeds for Essentia only include track-specific moods (not album seeds)
        candidate_seeds = list(set(text_mapped_moods))

        effnet_embeddings = None
        if self.essentia_analyzer and resolved_path:
            target_list = target_moods or self.mood_mapper.target_moods
            _, buf_16k = get_audio_buffers()
            effnet_embeddings = self.essentia_analyzer.extract_embeddings(
                audio=buf_16k, file_path=resolved_path
            )
            if effnet_embeddings is not None:
                predicted_moods, pred_score, top_predictions = self.essentia_analyzer.predict_moods(
                    embeddings=effnet_embeddings,
                    target_moods=target_list,
                    tag_mapper=self.mood_mapper,
                    bpm=None,
                    candidate_seeds=candidate_seeds,
                    mood_thresholds=self.mood_rules.acoustic_mood_thresholds,
                )
                if predicted_moods and pred_score >= essentia_threshold:
                    essentia_mapped_moods = predicted_moods
                essentia_top_preds = top_predictions

        pers_moods: list[tuple[str, float]] = []
        if (
            self.personalized_tuner
            and self.personalized_tuner.is_trained
            and effnet_embeddings is not None
        ):
            pers_moods = self.personalized_tuner.predict(
                effnet_embeddings,
                top_k=1,
                default_threshold=self.mood_rules.anchor_threshold,
            )

        if do_mood:
            phase_timings["mood_ml"] = time.perf_counter() - t_mood

        # 4. Detect BPM
        t_bpm = time.perf_counter()
        if do_bpm and self.bpm_detector and resolved_path:
            buf_44k, _ = get_audio_buffers()
            detected_bpm = self.bpm_detector.detect_bpm(
                resolved_path,
                genre_hint=mapped_genre,
                subgenres=mapped_subgenres,
                raw_tags=raw_tags,
                audio_predictions=essentia_top_preds,
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
                get_genre_seeded_moods(
                    mapped_subgenres,
                    genre_mood_seeds=self.mood_rules.genre_mood_seeds,
                )
                if mapped_subgenres
                else []
            )
            mapped_moods = synthesize_track_moods(
                text_moods=text_mapped_moods,
                seeded_moods=seeded,
                essentia_moods=essentia_mapped_moods,
                essentia_top=essentia_top_preds,
                detected_bpm=detected_bpm,
                lyrics_analysis=lyrics_res,
                primary_genre=mapped_genre,
                subgenres=mapped_subgenres,
                raw_tags=raw_tags,
                raw_mood_seeds=raw_mood_seeds,
                personalized_moods=pers_moods,
                genre_exclusions=self.mood_rules.genre_exclusions,
                mood_conflicts=self.mood_rules.mood_conflicts,
                lyrics_threshold=self.mood_rules.lyrics_threshold,
                lyrics_mood_thresholds=self.mood_rules.lyrics_mood_thresholds,
                acoustic_threshold=self.mood_rules.acoustic_threshold,
                acoustic_mood_thresholds=self.mood_rules.acoustic_mood_thresholds,
                acoustic_mood_mappings=self.mood_rules.acoustic_mood_mappings,
                anchor_reinforcement_threshold=self.mood_rules.anchor_reinforcement_threshold,
                tracer=tracer,
            )

        # 7. Write Embedded Mutagen Audio Tags
        t_mutagen = time.perf_counter()
        mutagen_updated = False
        if write_tags and self.mutagen_tagger and self.mutagen_tagger.enabled and resolved_path:
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
            essentia_predictions=essentia_top_preds,
            has_verified_tags=has_verified,
            mutagen_updated=mutagen_updated,
            plex_updated=False,
            skipped=False,
            duration_ms=total_duration_ms,
            phase_timings=phase_timings,
            decision_trace=tracer.events,
        )
