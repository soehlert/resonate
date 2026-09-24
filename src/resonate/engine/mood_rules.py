"""Acoustic mood heuristics, genre seeding, BPM gating, and conflict resolution."""

from __future__ import annotations

import logging

from resonate.config import MoodConflictRule, MoodRulesConfig
from resonate.models import LyricsAnalysisResult

logger = logging.getLogger(__name__)

DEFAULT_GENRE_EXCLUSIONS: dict[str, list[str]] = MoodRulesConfig().genre_exclusions
DEFAULT_MOOD_CONFLICTS: list[MoodConflictRule] = MoodRulesConfig().conflicts

GENRE_KEYWORDS: set[str] = {
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

RECOGNIZED_MOOD_KEYWORDS: set[str] = {
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

DEFAULT_TARGET_MOODS: list[str] = [
    "party",
    "chill hang",
    "energetic",
    "groovy",
    "acoustic",
    "electronic",
    "melancholic",
    "upbeat",
    "dark",
    "happy",
    "relaxed",
    "aggressive",
    "romantic",
    "calm",
    "mellow",
    "lively",
    "funky",
    "intense",
    "hypnotic",
    "atmospheric",
    "bittersweet",
    "intimate",
]

DEFAULT_MOOD_TAGS: list[str] = [
    "Party",
    "Chill Hang",
    "Energetic",
    "Groovy",
    "Acoustic",
    "Electronic",
    "Melancholic",
    "Lively",
    "Relaxed",
    "Romantic",
    "Calm",
    "Upbeat",
    "Dark",
    "Happy",
    "Fun",
    "Celebration",
    "Festive",
    "Mellow",
    "Feel-Good",
    "Friendly",
    "Intense",
    "Driving",
    "Powerful",
    "Aggressive",
    "Rowdy",
    "Funky",
    "Rhythmic",
    "Soulful",
    "Smooth",
    "Unplugged",
    "Intimate",
    "Organic",
    "Warm",
    "Hypnotic",
    "Futuristic",
    "Atmospheric",
    "Bittersweet",
]

ESSENTIA_MOOD_MAP: dict[str, str] = {
    "sexy": "Romantic",
    "love": "Romantic",
    "romantic": "Romantic",
    "sad": "Melancholic",
    "ballad": "Melancholic",
    "emotional": "Melancholic",
    "melancholic": "Melancholic",
    "relaxing": "Relaxed",
    "relaxed": "Relaxed",
    "meditative": "Calm",
    "calm": "Calm",
    "soft": "Mellow",
    "mellow": "Mellow",
    "heavy": "Heavy",
    "party": "Party",
    "fun": "Party",
    "dark": "Dark",
    "drama": "Atmospheric",
    "dramatic": "Atmospheric",
    "epic": "Atmospheric",
    "dream": "Atmospheric",
    "space": "Atmospheric",
    "atmospheric": "Atmospheric",
    "happy": "Happy",
    "positive": "Happy",
    "groovy": "Groovy",
    "energetic": "Energetic",
    "upbeat": "Upbeat",
    "uplifting": "Upbeat",
    "inspiring": "Upbeat",
    "motivational": "Upbeat",
    "hopeful": "Upbeat",
    "action": "Intense",
    "intense": "Intense",
    "powerful": "Intense",
    "chill": "Chill Hang",
    "chillout": "Chill Hang",
}

GENRE_MOOD_SEEDS: dict[str, list[str]] = {
    "Punk Rock": ["Rowdy", "Aggressive"],
    "Skate Punk": ["Rowdy", "Aggressive"],
    "Pop-Punk": ["Rowdy", "Upbeat"],
    "Hardcore": ["Aggressive", "Heavy"],
    "Hard Rock": ["Heavy"],
    "Heavy Metal": ["Heavy", "Aggressive", "Intense"],
    "Grunge": ["Heavy", "Dark"],
    "Industrial": ["Dark", "Intense", "Energetic"],
    "Dance-Pop": ["Party", "Upbeat"],
    "Disco": ["Party", "Groovy"],
    "Funk": ["Groovy", "Funky"],
    "Funk Rock": ["Groovy", "Funky"],
    "Ska": ["Upbeat"],
    "Ska Punk": ["Rowdy", "Upbeat"],
    "Reggae": ["Chill Hang", "Groovy", "Soulful"],
    "Roots Reggae": ["Chill Hang", "Groovy", "Soulful"],
    "Dub": ["Trippy", "Atmospheric", "Groovy"],
    "Reggae Rock": ["Chill Hang", "Upbeat"],
    "Classical": ["Atmospheric", "Emotional"],
    "Baroque": ["Atmospheric", "Calm"],
    "Chamber Music": ["Calm", "Mellow", "Atmospheric", "Intimate"],
    "Symphonic": ["Atmospheric", "Emotional", "Intense"],
    "Symphony": ["Atmospheric", "Emotional", "Intense"],
    "Opera": ["Atmospheric", "Intense", "Romantic", "Emotional"],
    "Soft Rock": ["Mellow", "Relaxed"],
    "Acoustic Rock": ["Acoustic", "Mellow"],
    "Singer-Songwriter": ["Acoustic", "Melancholic"],
    "Post-Punk": ["Dark", "Intense", "Energetic"],
    "Gothic": ["Dark", "Atmospheric"],
    "Darkwave": ["Dark", "Electronic", "Atmospheric"],
    "Slowcore": ["Melancholic", "Mellow", "Atmospheric"],
    "Sadcore": ["Melancholic", "Mellow", "Atmospheric"],
    "Emo": ["Melancholic"],
    "Motown": ["Soulful", "Groovy"],
    "Neo-Soul": ["Soulful", "Groovy"],
    "Psychedelic Rock": ["Trippy", "Atmospheric"],
    "Shoegaze": ["Atmospheric", "Trippy", "Intense"],
    "Post-Rock": ["Atmospheric", "Mellow"],
    "Dream Pop": ["Atmospheric", "Mellow", "Romantic"],
    "Indie Rock": ["Chill Hang"],
    "Indie Pop": ["Chill Hang"],
    "Indie Folk": ["Chill Hang", "Acoustic"],
    "Lo-Fi": ["Chill Hang", "Mellow"],
    "Americana": ["Chill Hang", "Acoustic"],
    "Alternative Rock": ["Chill Hang"],
    "Country": ["Acoustic", "Soulful"],
    "Country Rock": ["Acoustic", "Upbeat", "Chill Hang"],
    "Alt-Country": ["Acoustic", "Melancholic", "Chill Hang"],
    "Outlaw Country": ["Rowdy", "Acoustic"],
    "Bluegrass": ["Acoustic", "Lively", "Upbeat"],
    "Folk": ["Acoustic", "Melancholic"],
    "Folk Rock": ["Acoustic", "Chill Hang"],
    "Hip-Hop": ["Groovy", "Soulful"],
    "Rap": ["Rowdy", "Energetic"],
    "East Coast Hip Hop": ["Groovy", "Soulful"],
    "West Coast Hip Hop": ["Groovy", "Mellow"],
    "G-Funk": ["Groovy", "Mellow"],
    "Boom Bap": ["Groovy", "Soulful"],
    "Trap": ["Intense", "Dark", "Rowdy"],
    "Southern Rap": ["Intense", "Dark", "Rowdy"],
    "Gangsta Rap": ["Intense", "Dark", "Aggressive", "Rowdy"],
    "Hardcore Hip Hop": ["Intense", "Dark", "Aggressive", "Rowdy"],
    "Conscious Hip Hop": ["Soulful", "Mellow"],
    "Alternative Hip Hop": ["Groovy", "Soulful"],
    "Cloud Rap": ["Melancholic", "Mellow", "Atmospheric"],
    "Emo Rap": ["Melancholic", "Mellow", "Atmospheric"],
    "R&B": ["Soulful", "Groovy"],
    "Contemporary R&B": ["Soulful", "Groovy"],
    "Rockabilly": ["Upbeat", "Lively"],
    "Rock and Roll": ["Upbeat", "Rowdy", "Lively"],
    "Progressive Metal": ["Heavy", "Intense", "Atmospheric"],
    "Alternative Metal": ["Heavy", "Intense"],
    "Funk Metal": ["Funky", "Heavy"],
    "Nu-Metal": ["Heavy", "Intense"],
    "Industrial Metal": ["Heavy", "Intense", "Dark"],
    "Sludge Metal": ["Heavy", "Dark"],
    "Thrash Metal": ["Heavy", "Aggressive", "Intense"],
    "Death Metal": ["Heavy", "Aggressive", "Intense"],
    "Black Metal": ["Heavy", "Dark", "Intense"],
    "Doom Metal": ["Heavy", "Dark"],
    "Blues": ["Soulful", "Melancholic"],
    "Delta Blues": ["Melancholic", "Soulful", "Acoustic"],
    "Chicago Blues": ["Soulful", "Groovy", "Melancholic"],
    "Electric Blues": ["Soulful", "Groovy", "Melancholic"],
    "Blues Rock": ["Soulful", "Groovy"],
    "Jazz": ["Soulful", "Mellow"],
    "Big Band": ["Upbeat", "Lively"],
    "Swing": ["Upbeat", "Lively"],
    "Cool Jazz": ["Mellow", "Relaxed", "Atmospheric"],
    "Modal Jazz": ["Mellow", "Atmospheric"],
    "Bebop": ["Energetic", "Intense"],
    "Hard Bop": ["Soulful", "Energetic"],
    "Soul Jazz": ["Soulful", "Groovy"],
    "Vocal Jazz": ["Soulful", "Romantic"],
    "Bossa Nova": ["Chill Hang", "Relaxed"],
    "Latin Jazz": ["Lively", "Groovy"],
    "Smooth Jazz": ["Mellow", "Relaxed"],
    "Jazz Fusion": ["Intense", "Energetic"],
    "Free Jazz": ["Intense", "Experimental"],
    "Dixieland": ["Upbeat", "Lively"],
    "Gypsy Jazz": ["Upbeat", "Lively"],
}

MUTUALLY_EXCLUSIVE_MOODS: list[set[str]] = [
    {"Upbeat", "Dark"},
    {"Upbeat", "Melancholic"},
    {"Upbeat", "Heavy"},
    {"Romantic", "Heavy"},
    {"Romantic", "Aggressive"},
    {"Romantic", "Dark"},
    {"Acoustic", "Heavy"},
    {"Acoustic", "Aggressive"},
    {"Mellow", "Heavy"},
    {"Mellow", "Aggressive"},
    {"Calm", "Energetic"},
    {"Calm", "Rowdy"},
    {"Calm", "Intense"},
    {"Calm", "Heavy"},
    {"Calm", "Aggressive"},
]


def is_valid_mood_tag(tag: str, artist: str, album: str | None = None) -> bool:
    """Filter out non-mood tags, genres, playlists, and artists from mood candidates."""
    tag_lower = tag.lower().strip()

    if any(g in tag_lower for g in GENRE_KEYWORDS):
        return False

    artist_lower = artist.lower().strip()
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return False
    artist_words = [w.strip() for w in artist_lower.split() if len(w.strip()) > 3]
    if any(w in tag_lower for w in artist_words):
        return False

    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return False
        album_words = [w.strip() for w in album_lower.split() if len(w.strip()) > 3]
        if any(w in tag_lower for w in album_words):
            return False

    if any(c.isdigit() for c in tag_lower):
        return False

    words = tag_lower.replace("-", " ").split()
    if not (
        any(w in RECOGNIZED_MOOD_KEYWORDS for w in words)
        or any(k in tag_lower for k in RECOGNIZED_MOOD_KEYWORDS)
    ):
        return False

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

    genre_boilerplate = {
        "hard rock",
        "hardnheavy",
        "hard n heavy",
        "punk rock",
        "heavy metal",
        "alternative rock",
        "grunge",
        "alt rock",
        "indie rock",
        "pop rock",
        "metalcore",
        "death metal",
        "black metal",
        "thrash metal",
        "nu metal",
    }
    if any(gb in tag_lower for gb in genre_boilerplate):
        return False

    return True


def get_genre_seeded_moods(subgenres: list[str]) -> list[str]:
    """Get natural acoustic mood seeds based on mapped sub-genres/styles."""
    seeded: list[str] = []
    for sg in subgenres:
        if sg in GENRE_MOOD_SEEDS:
            for mood in GENRE_MOOD_SEEDS[sg]:
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
    conflicts: list[MoodConflictRule] | None = None,
) -> list[str]:
    """Resolve mutually exclusive mood conflicts based on directional priority rules."""
    if not moods:
        return []

    rules = conflicts if conflicts is not None else DEFAULT_MOOD_CONFLICTS
    for rule in rules:
        if isinstance(rule, dict):
            if_present = rule.get("if_present", [])
            drop = rule.get("drop", [])
        else:
            if_present = rule.if_present
            drop = rule.drop

        trigger_set = {t.lower() for t in if_present}
        drop_set = {d.lower() for d in drop}

        if any(m.lower() in trigger_set for m in moods):
            moods = [m for m in moods if m.lower() not in drop_set]

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
    max_moods: int = 3,
    personalized_moods: list[tuple[str, float]] | None = None,
    genre_exclusions: dict[str, list[str]] | None = None,
    mood_conflicts: list[MoodConflictRule] | None = None,
) -> list[str]:
    """Synthesis engine combining text, personalized anchors, audio waveform, and BPM gating."""
    combined: list[str] = list(text_moods)

    essentia_scores = {p[0].lower(): float(p[1]) for p in essentia_top} if essentia_top else {}
    is_raw_heavy = essentia_scores.get("heavy", 0.0) >= 0.08
    love_happy_sum = essentia_scores.get("love", 0.0) + essentia_scores.get("happy", 0.0)

    is_rowdy_or_heavy = (
        (primary_genre in {"Metal", "Punk"} if primary_genre else False)
        or is_raw_heavy
        or any(
            essentia_mood.lower() in {"heavy", "aggressive", "intense", "rowdy"}
            for essentia_mood in essentia_moods
        )
    )

    # Personalized Anchor Moods (User-calibrated anchors take top priority, at most 1 mood)
    if personalized_moods:
        for personalized_mood, _personalized_score in personalized_moods:
            personalized_mood_lower = personalized_mood.lower()
            if (
                personalized_mood_lower in {"chill hang", "calm", "mellow", "relaxed"}
                and is_rowdy_or_heavy
            ):
                continue
            if is_mood_excluded_by_genre(
                personalized_mood, subgenres, primary_genre, raw_tags, genre_exclusions
            ):
                continue
            if personalized_mood not in combined:
                combined.append(personalized_mood)
            # Enforce at most 1 personalized anchor mood
            break

    for seeded_mood in seeded_moods:
        seeded_mood_lower = seeded_mood.lower()
        if seeded_mood_lower == "chill hang":
            if not is_rowdy_or_heavy and seeded_mood not in combined:
                combined.append(seeded_mood)
        elif seeded_mood not in combined:
            combined.append(seeded_mood)

    for essentia_mood in essentia_moods:
        if len(combined) >= max_moods:
            break
        if essentia_mood not in combined:
            combined.append(essentia_mood)

    # Populate from Essentia top acoustic predictions without force-padding to max_moods
    if len(combined) < max_moods and essentia_top:
        for tag, score in essentia_top:
            if score < 0.10:
                continue
            tag_lower = tag.lower()
            if tag_lower in {"energetic", "lively"} and score < 0.25:
                continue
            if tag_lower in {"love", "sexy"} and not (score >= 0.25 or love_happy_sum > 0.25):
                continue
            target_mood = ESSENTIA_MOOD_MAP.get(tag_lower)
            if not target_mood and any(d.lower() == tag_lower for d in DEFAULT_TARGET_MOODS):
                target_mood = next(
                    d.title() for d in DEFAULT_TARGET_MOODS if d.lower() == tag_lower
                )
            if target_mood and target_mood not in combined:
                combined.append(target_mood)
            if len(combined) >= max_moods:
                break

    # Lyrics Analysis
    if lyrics_analysis and lyrics_analysis.lyrics_text:
        for lyrics_mood, lyrics_score in lyrics_analysis.mood_scores.items():
            if lyrics_mood in {"Dark", "Melancholic"}:
                if lyrics_score >= 0.35 and lyrics_analysis.valence_score < -0.15:
                    if lyrics_mood not in combined:
                        combined.append(lyrics_mood)
            elif lyrics_mood in {"Romantic", "Happy"}:
                if lyrics_score >= 0.35 and lyrics_analysis.valence_score > 0.15:
                    if lyrics_mood not in combined:
                        combined.append(lyrics_mood)
            elif lyrics_score >= 0.40 and lyrics_mood not in combined:
                combined.append(lyrics_mood)

    # BPM Tempo Gating
    combined = apply_bpm_mood_rules(combined, detected_bpm)

    # Filter out moods excluded by genre rules unless explicitly tagged
    combined = [
        m
        for m in combined
        if not is_mood_excluded_by_genre(m, subgenres, primary_genre, raw_tags, genre_exclusions)
    ]

    # Mutual Exclusion Conflict Resolution
    combined = resolve_mood_conflicts(combined, conflicts=mood_conflicts)

    # Prioritize specific emotional/acoustic moods first
    specific_moods = [m for m in combined if m.lower() not in {"energetic", "lively"}]
    tempo_moods = [m for m in combined if m.lower() in {"energetic", "lively"}]
    sorted_final = (specific_moods + tempo_moods)[:max_moods]

    return [m for m in sorted_final if m and m.strip().lower() != "none"]
