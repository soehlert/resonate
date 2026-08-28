"""Acoustic mood heuristics, genre seeding, BPM gating, and conflict resolution."""

from __future__ import annotations

import logging

from resonate.models import LyricsAnalysisResult

logger = logging.getLogger(__name__)

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
    "Sad",
    "Bittersweet",
    "Somber",
    "Brooding",
    "Gloomy",
    "Emotional",
]

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
    "Rockabilly": ["Upbeat", "Chill Hang"],
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


def resolve_mood_conflicts(moods: list[str]) -> list[str]:
    """Resolve mutually exclusive mood conflicts (e.g. Dark vs Upbeat, Heavy vs Chill Hang)."""
    if not moods:
        return []

    mood_lower_set = {m.lower() for m in moods}

    # If Acoustic or Mellow is present, drop Heavy, Aggressive, and Rowdy
    if any(m in mood_lower_set for m in {"acoustic", "mellow", "meditative", "calm"}):
        moods = [m for m in moods if m.lower() not in {"heavy", "aggressive", "rowdy"}]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Rowdy, Dark, Melancholic, or Ballad is present, drop Chill Hang
    if any(
        m in mood_lower_set
        for m in {"heavy", "aggressive", "rowdy", "dark", "melancholic", "ballad"}
    ):
        moods = [m for m in moods if m.lower() != "chill hang"]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Dark, or Melancholic is present, drop Happy and Upbeat
    if any(m in mood_lower_set for m in {"heavy", "aggressive", "dark", "melancholic"}):
        moods = [m for m in moods if m.lower() not in {"happy", "upbeat"}]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy or Aggressive is present, drop Groovy
    if any(m in mood_lower_set for m in {"heavy", "aggressive"}):
        moods = [m for m in moods if m.lower() != "groovy"]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Dark, Rowdy, or Hardcore is present, drop Romantic
    if any(m in mood_lower_set for m in {"heavy", "aggressive", "dark", "rowdy", "hardcore"}):
        moods = [m for m in moods if m.lower() != "romantic"]
        mood_lower_set = {m.lower() for m in moods}

    # Energetic and Lively mutual exclusion (keep Energetic, drop Lively)
    if "energetic" in mood_lower_set and "lively" in mood_lower_set:
        moods = [m for m in moods if m.lower() != "lively"]

    return moods


def apply_bpm_mood_rules(moods: list[str], detected_bpm: int | None) -> list[str]:
    """Apply BPM tempo gating across candidate moods."""
    if detected_bpm is None or not moods:
        return moods

    if detected_bpm >= 130:
        # 130+ BPM is Energetic; strip Lively
        return [m for m in moods if m.lower() != "lively"]
    elif 110 <= detected_bpm < 130:
        # 110-130 BPM is Lively; convert Energetic to Lively
        res = ["Lively" if m.lower() == "energetic" else m for m in moods]
        seen: set[str] = set()
        return [m for m in res if not (m.lower() in seen or seen.add(m.lower()))]
    else:
        # Below 110 BPM is neither Energetic nor Lively
        return [m for m in moods if m.lower() not in {"energetic", "lively"}]


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
) -> list[str]:
    """Pure synthesis engine combining text, audio waveform, lyrics valence, and BPM gating."""
    combined: list[str] = list(text_moods)

    e_pred_dict = {p[0].lower(): float(p[1]) for p in essentia_top} if essentia_top else {}
    is_raw_energetic = e_pred_dict.get("energetic", 0.0) >= 0.15
    is_raw_heavy = e_pred_dict.get("heavy", 0.0) >= 0.08
    is_raw_aggressive = e_pred_dict.get("aggressive", 0.0) >= 0.05
    is_raw_dark = e_pred_dict.get("dark", 0.0) >= 0.08
    has_grunge = any("grunge" in sg.lower() for sg in subgenres)
    is_grunge_heavy_or_energetic = has_grunge and (
        is_raw_energetic or is_raw_heavy or is_raw_aggressive
    )

    is_rowdy_or_heavy = (
        (primary_genre in {"Metal", "Punk"} if primary_genre else False)
        or is_grunge_heavy_or_energetic
        or is_raw_heavy
        or is_raw_aggressive
        or is_raw_dark
        or any(
            em.lower() in {"heavy", "aggressive", "intense", "dark", "rowdy"}
            for em in essentia_moods
        )
    )

    is_low_tempo = detected_bpm is not None and detected_bpm < 100
    is_slow_and_not_heavy = is_low_tempo and not (is_raw_heavy or is_raw_aggressive)

    for sm in seeded_moods:
        sm_l = sm.lower()
        if sm_l == "chill hang":
            if not is_rowdy_or_heavy and sm not in combined:
                combined.append(sm)
        elif sm_l in {"rowdy", "aggressive", "heavy"}:
            if not is_slow_and_not_heavy:
                if sm in text_moods or sm in essentia_moods:
                    if sm not in combined:
                        combined.append(sm)
        elif sm in text_moods or sm in essentia_moods:
            if sm not in combined:
                combined.append(sm)

    for em in essentia_moods:
        if em not in combined:
            combined.append(em)

    # Populate from Essentia top acoustic predictions if combined is not full
    if len(combined) < max_moods and essentia_top:
        for tag, _score in essentia_top:
            tag_lower = tag.lower()
            if tag_lower in {"heavy", "aggressive", "rowdy", "dark"} and not (
                is_raw_heavy or is_raw_aggressive or is_raw_dark
            ):
                continue
            matching_default = next(
                (d for d in DEFAULT_MOOD_TAGS if d.lower() == tag_lower), None
            )
            if matching_default and matching_default not in combined:
                combined.append(matching_default)
            if len(combined) >= max_moods:
                break

    # Lyrics Analysis
    if lyrics_analysis and lyrics_analysis.lyrics_text:
        is_high_tempo_upbeat = (
            detected_bpm is not None and detected_bpm >= 120
        ) and not (is_raw_heavy or is_raw_dark or is_raw_aggressive)

        if (
            lyrics_analysis.valence_score < -0.30
            or lyrics_analysis.mood_scores.get("Dark", 0.0) >= 0.35
        ):
            if not is_high_tempo_upbeat or lyrics_analysis.valence_score < -0.50:
                combined = [
                    m for m in combined
                    if m.lower() not in {"happy", "upbeat", "chill hang"}
                ]

        for lm_tag, lm_score in lyrics_analysis.mood_scores.items():
            if lm_tag in {"Dark", "Melancholic"}:
                if (
                    is_high_tempo_upbeat
                    and lm_tag == "Dark"
                    and lyrics_analysis.valence_score > -0.50
                ):
                    continue
                if lm_score >= 0.35 and lyrics_analysis.valence_score < -0.15:
                    if lm_tag not in combined:
                        combined.append(lm_tag)
            elif lm_tag in {"Romantic", "Happy"}:
                if lm_score >= 0.35 and lyrics_analysis.valence_score > 0.15:
                    if lm_tag not in combined:
                        combined.append(lm_tag)
            elif lm_score >= 0.40 and lm_tag not in combined:
                combined.append(lm_tag)

    # BPM Tempo Gating
    combined = apply_bpm_mood_rules(combined, detected_bpm)

    # Mutual Exclusion Conflict Resolution
    combined = resolve_mood_conflicts(combined)

    # Prioritize specific emotional/acoustic moods first
    specific_moods = [m for m in combined if m.lower() not in {"energetic", "lively"}]
    tempo_moods = [m for m in combined if m.lower() in {"energetic", "lively"}]
    sorted_final = (specific_moods + tempo_moods)[:max_moods]

    return sorted_final
