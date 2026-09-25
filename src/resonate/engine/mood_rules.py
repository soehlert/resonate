"""Acoustic mood heuristics, genre seeding, BPM gating, and conflict resolution."""

from __future__ import annotations

import logging

from resonate.config import (
    GenreMoodSeedRule,
    MoodConflictRule,
    MoodRulesConfig,
    load_config,
    load_data_file,
)
from resonate.engine.subgenre_registry import NATIONALITY_STRINGS
from resonate.engine.tag_filter import (
    GENRE_KEYWORDS,
    RECOGNIZED_MOOD_KEYWORDS,
    is_artist_or_album_match,
    is_boilerplate_tag,
)
from resonate.engine.tracer import DecisionTracer
from resonate.models import LyricsAnalysisResult, TraceAction

logger = logging.getLogger(__name__)


def _get_default_rules() -> MoodRulesConfig:
    try:
        return load_config().mood_rules
    except Exception:
        return MoodRulesConfig()


DEFAULT_GENRE_EXCLUSIONS: dict[str, list[str]] = _get_default_rules().genre_exclusions
DEFAULT_MOOD_CONFLICTS: list[MoodConflictRule] = _get_default_rules().mood_conflicts
DEFAULT_GENRE_MOOD_SEEDS: list[GenreMoodSeedRule] = _get_default_rules().genre_mood_seeds
DEFAULT_ACOUSTIC_THRESHOLD: float = _get_default_rules().acoustic_threshold
DEFAULT_ACOUSTIC_MOOD_THRESHOLDS: dict[str, float] = _get_default_rules().acoustic_mood_thresholds
DEFAULT_ANCHOR_REINFORCEMENT_THRESHOLD: float = _get_default_rules().anchor_reinforcement_threshold
DEFAULT_ACOUSTIC_MOOD_MAPPINGS: dict[str, list[str]] = (
    _get_default_rules().acoustic_mood_mappings
    or load_data_file("mood_rules.yaml").get("acoustic_mood_mappings", {})
)


def _get_default_acoustic_mood_mappings() -> dict[str, list[str]]:
    defaults = dict(DEFAULT_ACOUSTIC_MOOD_MAPPINGS)
    try:
        configured = _get_default_rules().acoustic_mood_mappings
        if configured:
            defaults.update(configured)
    except Exception:
        pass
    return defaults


DEFAULT_TARGET_MOODS: list[str] = load_data_file("target_moods.yaml").get("moods", [])

DEFAULT_MOOD_TAGS: list[str] = [m.title() for m in DEFAULT_TARGET_MOODS]

ESSENTIA_MOOD_MAP: dict[str, str] = load_data_file("essentia_moods.yaml").get(
    "essentia_to_mood", {}
)


def is_valid_mood_tag(tag: str, artist: str, album: str | None = None) -> bool:
    """Filter out non-mood tags, genres, playlists, and artists from mood candidates."""
    tag_lower = tag.lower().strip()

    if any(g in tag_lower for g in GENRE_KEYWORDS):
        return False

    if is_artist_or_album_match(tag_lower, artist, album):
        return False

    if any(c.isdigit() for c in tag_lower):
        return False

    words = tag_lower.replace("-", " ").split()
    if not (
        any(w in RECOGNIZED_MOOD_KEYWORDS for w in words)
        or any(k in tag_lower for k in RECOGNIZED_MOOD_KEYWORDS)
    ):
        return False

    if any(n in tag_lower for n in NATIONALITY_STRINGS):
        return False

    if is_boilerplate_tag(tag_lower):
        return False

    return True


def get_genre_seeded_moods(
    subgenres: list[str],
    genre_mood_seeds: list[GenreMoodSeedRule] | dict[str, list[str]] | None = None,
) -> list[str]:
    """Get natural acoustic mood seeds based on mapped sub-genres/styles."""
    if not subgenres:
        return []
    rules = genre_mood_seeds if genre_mood_seeds is not None else DEFAULT_GENRE_MOOD_SEEDS
    if not rules:
        return []

    subgenres_lower = {sg.lower() for sg in subgenres}
    seeded: list[str] = []

    if isinstance(rules, list):
        for rule in rules:
            rule_genres = (
                {g.lower() for g in rule.genres}
                if hasattr(rule, "genres")
                else {g.lower() for g in rule.get("genres", [])}
            )
            rule_moods = rule.moods if hasattr(rule, "moods") else rule.get("moods", [])
            if any(sg in rule_genres for sg in subgenres_lower):
                for mood in rule_moods:
                    if mood not in seeded:
                        seeded.append(mood)
    elif isinstance(rules, dict):
        for g, moods in rules.items():
            if g.lower() in subgenres_lower:
                for mood in moods:
                    if mood not in seeded:
                        seeded.append(mood)

    return seeded


def is_mood_excluded_by_genre(
    mood: str,
    subgenres: list[str],
    primary_genre: str | None,
    raw_tags: list[str],
    genre_exclusions: dict[str, list[str]] | None = None,
) -> bool:
    """Check if a mood is excluded for the track's genre unless explicitly tagged in raw_tags."""
    exclusions = genre_exclusions if genre_exclusions is not None else DEFAULT_GENRE_EXCLUSIONS
    if not exclusions:
        return False

    mood_lower = mood.lower()
    excluded_genres: list[str] = []
    for excluded_mood, genres in exclusions.items():
        if excluded_mood.lower() == mood_lower:
            excluded_genres.extend([g.lower() for g in genres])

    if not excluded_genres:
        return False

    all_genres = [sg.lower() for sg in subgenres]
    if primary_genre:
        all_genres.append(primary_genre.lower())

    is_excluded_genre = any(eg == g or eg in g for eg in excluded_genres for g in all_genres)
    if not is_excluded_genre:
        return False

    # Allow the mood if raw_tags has an explicit tag matching the mood
    has_explicit_raw_tag = any(mood_lower in tag.lower() for tag in raw_tags)
    return not has_explicit_raw_tag


def resolve_mood_conflicts(
    moods: list[str],
    mood_conflicts: list[MoodConflictRule] | None = None,
    mood_scores: dict[str, float] | None = None,
    tracer: DecisionTracer | None = None,
    decision_trace: list[str] | None = None,
) -> list[str]:
    """Resolve mutually exclusive mood conflicts based on priority rules and evidence scores."""
    if not moods:
        return []
    if tracer is None:
        tracer = DecisionTracer(messages=decision_trace, enabled=decision_trace is not None)

    active_rules = mood_conflicts if mood_conflicts is not None else DEFAULT_MOOD_CONFLICTS

    for rule in active_rules:
        if isinstance(rule, dict):
            if_present = rule.get("if_present", [])
            drop = rule.get("drop", [])
        else:
            if_present = rule.if_present
            drop = rule.drop

        trigger_set = {t.lower() for t in if_present}
        drop_set = {d.lower() for d in drop}

        triggers_found = [m for m in moods if m.lower() in trigger_set]
        targets_found = [m for m in moods if m.lower() in drop_set]

        if triggers_found and targets_found:
            for target in targets_found:
                if target in moods:
                    tracer.drop(target, f"conflict rule triggered by {triggers_found}")
                    moods = [m for m in moods if m != target]

    return moods


def apply_bpm_mood_rules(moods: list[str], detected_bpm: int | None) -> list[str]:
    """Pass through candidate moods without applying hard BPM tempo vetoes."""
    return moods


def synthesize_track_moods(
    text_moods: list[str],
    seeded_moods: list[str],
    essentia_moods: list[str],
    essentia_top: list[tuple[str, float]],
    detected_bpm: int | None,
    lyrics_analysis: LyricsAnalysisResult | None,
    primary_genre: str | None,
    subgenres: list[str],
    raw_tags: list[str],
    raw_mood_seeds: list[str] | None = None,
    max_moods: int = 3,
    personalized_moods: list[tuple[str, float]] | None = None,
    genre_exclusions: dict[str, list[str]] | None = None,
    mood_conflicts: list[MoodConflictRule] | None = None,
    lyrics_threshold: float = 0.20,
    lyrics_mood_thresholds: dict[str, float] | None = None,
    acoustic_threshold: float = 0.10,
    acoustic_mood_thresholds: dict[str, float] | None = None,
    acoustic_mood_mappings: dict[str, list[str]] | None = None,
    anchor_reinforcement_threshold: float | None = None,
    tracer: DecisionTracer | None = None,
    decision_trace: list[str] | None = None,
) -> list[str]:
    """Synthesis engine combining text, personalized anchors, audio waveform, and BPM gating."""
    if tracer is None:
        tracer = DecisionTracer(messages=decision_trace, enabled=decision_trace is not None)

    candidate_scores: dict[str, float] = {}
    combined: list[str] = list(text_moods)
    for m in text_moods:
        candidate_scores[m] = 0.85
    if text_moods:
        tracer.record(f"Provider/Text tags mapped candidate moods: {text_moods}")

    # Personalized Anchor Moods (User-calibrated anchors take top priority, at most 1 mood)
    if personalized_moods:
        raw_mappings = (
            acoustic_mood_mappings
            if acoustic_mood_mappings is not None
            else _get_default_acoustic_mood_mappings()
        )
        norm_mappings = {k.lower(): {t.lower() for t in v} for k, v in raw_mappings.items()}

        active_reinforcement_threshold = (
            anchor_reinforcement_threshold
            if anchor_reinforcement_threshold is not None
            else DEFAULT_ANCHOR_REINFORCEMENT_THRESHOLD
        )

        for personalized_mood, _personalized_score in personalized_moods:
            if is_mood_excluded_by_genre(
                personalized_mood, subgenres, primary_genre, raw_tags, genre_exclusions
            ):
                tracer.reject(
                    personalized_mood,
                    f"anchor score {_personalized_score:.2f} excluded by genre rules",
                )
                continue

            # Require acoustic reinforcement or provider text tag agreement
            has_acoustic_backing = False
            top_acoustic_score = 0.0
            acoustic_backing_tags: list[str] = []
            if essentia_top:
                target_lower = personalized_mood.lower()
                allowed_tags = norm_mappings.get(target_lower, set())
                for e_tag, e_score in essentia_top:
                    e_tag_lower = e_tag.lower()
                    mapped_e_mood = ESSENTIA_MOOD_MAP.get(e_tag_lower, "").lower()
                    if (
                        e_tag_lower in allowed_tags
                        or mapped_e_mood in allowed_tags
                        or e_tag_lower == target_lower
                        or mapped_e_mood == target_lower
                    ):
                        top_acoustic_score += e_score
                        acoustic_backing_tags.append(f"{e_tag}={e_score:.2f}")
                top_acoustic_score = min(1.0, round(top_acoustic_score, 4))
                if top_acoustic_score >= active_reinforcement_threshold:
                    has_acoustic_backing = True

            has_tag_backing = any(m.lower() == personalized_mood.lower() for m in text_moods)

            if not (has_acoustic_backing or has_tag_backing):
                score_fmt = (
                    f"{top_acoustic_score:.3f} < {active_reinforcement_threshold:.3f}"
                    if f"{top_acoustic_score:.2f}" == f"{active_reinforcement_threshold:.2f}"
                    else f"{top_acoustic_score:.2f} < {active_reinforcement_threshold:.2f}"
                )
                skip_reason = (
                    f"anchor score {_personalized_score:.2f} lacks acoustic reinforcement "
                    f"({score_fmt}) and text tag agreement"
                )
                tracer.skip("Personalized anchor", personalized_mood, skip_reason)
                continue

            if personalized_mood not in combined:
                combined.append(personalized_mood)
                candidate_scores[personalized_mood] = float(_personalized_score)
                backing_details: list[str] = []
                if has_acoustic_backing:
                    backing_details.append(
                        f"acoustic backing: {', '.join(acoustic_backing_tags)} "
                        f"({top_acoustic_score:.2f} >= {active_reinforcement_threshold:.2f})"
                    )
                if has_tag_backing:
                    backing_details.append("text tag agreement")
                backing_msg = f" [{'; '.join(backing_details)}]" if backing_details else ""
                accept_msg = (
                    f"Personalized anchor accepted: '{personalized_mood}' "
                    f"(score={_personalized_score:.2f}){backing_msg}"
                )
                tracer.record(accept_msg, action=TraceAction.ACCEPT)
            # Enforce at most 1 personalized anchor mood
            break

    for essentia_mood in essentia_moods:
        if len(combined) >= max_moods:
            break
        if essentia_mood not in combined:
            combined.append(essentia_mood)
            candidate_scores[essentia_mood] = 0.50
            tracer.accept("Essentia classifier", essentia_mood)

    # Populate from Essentia top acoustic predictions without force-padding to max_moods
    if len(combined) < max_moods and essentia_top:
        active_acoustic_threshold = (
            acoustic_threshold if acoustic_threshold is not None else DEFAULT_ACOUSTIC_THRESHOLD
        )
        active_acoustic_mood_thresholds = (
            acoustic_mood_thresholds
            if acoustic_mood_thresholds is not None
            else DEFAULT_ACOUSTIC_MOOD_THRESHOLDS
        )
        for tag, score in essentia_top:
            tag_lower = tag.lower()
            target_mood = ESSENTIA_MOOD_MAP.get(tag_lower)
            if not target_mood and any(d.lower() == tag_lower for d in DEFAULT_TARGET_MOODS):
                target_mood = next(
                    d.title() for d in DEFAULT_TARGET_MOODS if d.lower() == tag_lower
                )
            if not target_mood:
                continue

            required_threshold = active_acoustic_mood_thresholds.get(
                target_mood, active_acoustic_threshold
            )
            if score < required_threshold:
                tracer.skip(
                    "Essentia acoustic",
                    tag,
                    f"score {score:.2f} < {required_threshold:.2f} threshold",
                )
                continue

            if target_mood not in combined:
                combined.append(target_mood)
                candidate_scores[target_mood] = max(
                    candidate_scores.get(target_mood, 0.0), float(score)
                )
                tracer.accept("Essentia acoustic", f"{tag} -> {target_mood}", score)
            else:
                reinforce_msg = (
                    f"Essentia acoustic '{tag}' (score={score:.2f}) "
                    f"reinforces existing '{target_mood}'"
                )
                tracer.record(reinforce_msg, action=TraceAction.ACCEPT)
            if len(combined) >= max_moods:
                break

    # Lyrics Analysis
    if lyrics_analysis and lyrics_analysis.lyrics_text:
        val_str = f"{lyrics_analysis.valence_score:.2f}"
        tracer.record(f"Lyrics retrieved ({lyrics_analysis.source}): valence={val_str}")
        mood_thresholds = lyrics_mood_thresholds if lyrics_mood_thresholds is not None else {}
        for lyrics_mood, lyrics_score in lyrics_analysis.mood_scores.items():
            required_threshold = mood_thresholds.get(lyrics_mood, lyrics_threshold)
            if lyrics_score < required_threshold:
                tracer.skip(
                    "Lyrics mood",
                    lyrics_mood,
                    f"score {lyrics_score:.2f} < {required_threshold:.2f}",
                )
                continue
            if lyrics_mood not in combined:
                combined.append(lyrics_mood)
                candidate_scores[lyrics_mood] = float(lyrics_score)
                tracer.accept("Lyrics mood", lyrics_mood, lyrics_score)
    else:
        tracer.record("Lyrics not found (checked embedded tags, sidecar, and LRCLIB)")

    # Filter out moods excluded by genre rules unless explicitly tagged
    kept_moods: list[str] = []
    for m in combined:
        if is_mood_excluded_by_genre(m, subgenres, primary_genre, raw_tags, genre_exclusions):
            tracer.drop(m, f"genre exclusion rule for {primary_genre or ''}/{subgenres}")
        else:
            kept_moods.append(m)
    combined = kept_moods

    # Fallback: If no moods found from text, audio, or lyrics,
    # fallback to recognized provider mood tags
    if not combined and raw_mood_seeds:
        for raw_mood in raw_mood_seeds:
            if len(combined) >= max_moods:
                break
            if raw_mood not in combined and not is_mood_excluded_by_genre(
                raw_mood, subgenres, primary_genre, raw_tags, genre_exclusions
            ):
                combined.append(raw_mood)
                candidate_scores[raw_mood] = 0.40
                tracer.record(
                    f"Provider tag fallback applied (from raw tags): '{raw_mood}'",
                    action=TraceAction.ACCEPT,
                )
    elif raw_mood_seeds:
        tracer.record(
            "Provider tag fallback skipped: higher-priority candidate moods already present "
            f"({combined})"
        )

    # Fallback: If still no moods found, seed from subgenre taxonomy
    if not combined and seeded_moods:
        for seeded_mood in seeded_moods:
            if len(combined) >= max_moods:
                break
            if seeded_mood not in combined and not is_mood_excluded_by_genre(
                seeded_mood, subgenres, primary_genre, raw_tags, genre_exclusions
            ):
                combined.append(seeded_mood)
                candidate_scores[seeded_mood] = 0.35
        if combined:
            tracer.record(
                f"Genre-seeded fallback applied (no previous moods): {combined}",
                action=TraceAction.ACCEPT,
            )
    elif seeded_moods:
        tracer.record(
            "Genre-seeded fallback skipped: higher-priority candidate moods already present "
            f"({combined})"
        )

    # BPM Tempo Gating
    combined = apply_bpm_mood_rules(combined, detected_bpm)

    # Mutual Exclusion Conflict Resolution
    combined = resolve_mood_conflicts(
        combined,
        mood_conflicts=mood_conflicts,
        mood_scores=candidate_scores,
        tracer=tracer,
        decision_trace=decision_trace,
    )

    sorted_final = combined[:max_moods]
    final_result = [m.title() for m in sorted_final if m and m.strip().lower() != "none"]
    tracer.record(f"Final Resolved Moods: {final_result}", action=TraceAction.ACCEPT)
    return final_result
