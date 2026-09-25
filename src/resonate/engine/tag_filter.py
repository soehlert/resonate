"""Tag validation utilities and boilerplate/metadata noise filters."""

from __future__ import annotations

from resonate.config import load_data_file

_filter_data = load_data_file("tag_filters.yaml")

BOILERPLATE_TAGS: set[str] = set(_filter_data.get("boilerplate_tags", []))
GENRE_KEYWORDS: set[str] = set(_filter_data.get("genre_keywords", []))
GENRE_STOP_WORDS: set[str] = set(_filter_data.get("genre_stop_words", []))
RECOGNIZED_MOOD_KEYWORDS: set[str] = set(_filter_data.get("recognized_mood_keywords", []))


def is_artist_or_album_match(tag_lower: str, artist: str, album: str | None = None) -> bool:
    """Return True if tag contains or matches the artist or album name."""
    artist_lower = artist.lower().strip()
    if not artist_lower:
        return False
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return True

    artist_words = [
        w.strip(" \t\n\r:;,.!?()[]{}\"'")
        for w in artist_lower.split()
        if len(w.strip(" \t\n\r:;,.!?()[]{}\"'")) > 3
        and w.strip(" \t\n\r:;,.!?()[]{}\"'") not in GENRE_STOP_WORDS
    ]
    if any(w in tag_lower for w in artist_words):
        return True

    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return True
        album_words = [
            w.strip(" \t\n\r:;,.!?()[]{}\"'")
            for w in album_lower.split()
            if len(w.strip(" \t\n\r:;,.!?()[]{}\"'")) > 3
            and w.strip(" \t\n\r:;,.!?()[]{}\"'") not in GENRE_STOP_WORDS
        ]
        if any(w in tag_lower for w in album_words):
            return True

    return False


def is_boilerplate_tag(tag_lower: str) -> bool:
    """Check if tag contains common non-genre/non-mood boilerplate strings."""
    return any(b in tag_lower for b in BOILERPLATE_TAGS)


__all__ = [
    "BOILERPLATE_TAGS",
    "GENRE_KEYWORDS",
    "GENRE_STOP_WORDS",
    "RECOGNIZED_MOOD_KEYWORDS",
    "is_artist_or_album_match",
    "is_boilerplate_tag",
]
