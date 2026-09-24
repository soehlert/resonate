"""Genre taxonomy hierarchy, family promotion rules, and subgenre filtering engine."""

from __future__ import annotations

import logging
import re
from collections import Counter

from resonate.engine.subgenre_registry import (
    COMPOUND_SUBGENRE_WHITELIST,
    DEFAULT_PRIMARY_GENRES,
    DEFAULT_SUB_GENRES,
    GENERIC_MODIFIERS,
    GENRE_STOP_WORDS,
    MUTUALLY_EXCLUSIVE_STYLES,
    NATIONALITY_STRINGS,
    PRIMARY_GENRE_STEMS,
    SUB_GENRE_STEMS,
    SUBGENRE_REGISTRY,
    SUBGENRE_TO_FAMILY,
    SubgenreSpec,
)
from resonate.models import TaxonomyDecision

logger = logging.getLogger(__name__)


def is_valid_subgenre_tag(tag: str, artist: str, album: str | None = None) -> bool:
    """Filter out non-genre tags, playlists, TV shows, and decades from subgenre candidates."""
    tag_lower = tag.lower().strip()

    # 1. Skip if contains digits (decades like 80s, 2010s, 1994, s36)
    if any(c.isdigit() for c in tag_lower):
        return False

    # Known canonical compound subgenres (e.g. "rock and roll", "post-punk") pass immediately
    if tag_lower in COMPOUND_SUBGENRE_WHITELIST:
        return True

    # Reject non-whitelisted conjunction mashup tags like "rock and punk", "alternative and rock"
    if re.search(r"\b(and|&)\b", tag_lower):
        return False

    # 2. Skip if it contains the artist name or any significant word of it
    artist_lower = artist.lower().strip()
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return False
    artist_words = [
        w.strip(" \t\n\r:;,.!?()[]{}\"'")
        for w in artist_lower.split()
        if len(w.strip(" \t\n\r:;,.!?()[]{}\"'")) > 3
        and w.strip(" \t\n\r:;,.!?()[]{}\"'") not in GENRE_STOP_WORDS
    ]
    if any(w in tag_lower for w in artist_words):
        return False

    # 3. Skip if it contains the album name or any significant word of it
    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return False
        album_words = [
            w.strip(" \t\n\r:;,.!?()[]{}\"'")
            for w in album_lower.split()
            if len(w.strip(" \t\n\r:;,.!?()[]{}\"'")) > 3
            and w.strip(" \t\n\r:;,.!?()[]{}\"'") not in GENRE_STOP_WORDS
        ]
        if any(w in tag_lower for w in album_words):
            return False

    # 4. Skip common non-genre/boilerplate/playlist descriptors
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
    }
    if any(b in tag_lower for b in boilerplate):
        return False

    return True


def promote_genre_by_subgenres(
    mapped_genre: str | None, mapped_subgenres: list[str]
) -> tuple[str | None, TaxonomyDecision | None]:
    """Elevate generic Rock or Pop if child subgenres strictly outnumber parent."""
    if not mapped_genre or mapped_genre not in {"Rock", "Pop", "Reggae"} or not mapped_subgenres:
        return mapped_genre, None

    subgenre_family_counts: Counter[str] = Counter()
    for sg in mapped_subgenres:
        fam = SUBGENRE_TO_FAMILY.get(sg.lower())
        if fam == "HardRock":
            fam = "Rock"
        if fam and fam in DEFAULT_PRIMARY_GENRES:
            subgenre_family_counts[fam] += 1

    parent_family = mapped_genre
    parent_count = subgenre_family_counts.get(parent_family, 0)

    top_candidates = [
        (fam, cnt) for fam, cnt in subgenre_family_counts.most_common() if fam != parent_family
    ]
    if top_candidates:
        top_child_family, top_child_count = top_candidates[0]
        if top_child_count > parent_count:
            decision = TaxonomyDecision(
                original_genre=mapped_genre,
                promoted_genre=top_child_family,
                reason=(
                    f"Child family '{top_child_family}' subgenres ({top_child_count}) "
                    f"strictly outnumber parent '{parent_family}' subgenres ({parent_count})"
                ),
                contributing_subgenres=[
                    sg
                    for sg in mapped_subgenres
                    if (
                        SUBGENRE_TO_FAMILY.get(sg.lower()) == top_child_family
                        or (
                            top_child_family == "Rock"
                            and SUBGENRE_TO_FAMILY.get(sg.lower()) == "HardRock"
                        )
                    )
                ],
                confidence=1.0,
            )
            return top_child_family, decision

    return mapped_genre, None


def sanitize_subgenres_for_genre(
    mapped_genre: str | None, mapped_subgenres: list[str], raw_tags: list[str]
) -> list[str]:
    """Apply cross-family sanity guards to strip incompatible subgenres."""
    if not mapped_genre or not mapped_subgenres:
        return mapped_subgenres

    raw_clean_set = {r.lower().strip() for r in raw_tags}

    # 1. Punk / Metal / Rock: strip Hip-Hop subgenres unless explicit hip-hop tags are present
    if mapped_genre in {"Punk", "Metal", "Rock"}:
        if not any(r in {"hip-hop", "hip hop", "rap", "hiphop"} for r in raw_clean_set):
            mapped_subgenres = [
                s for s in mapped_subgenres if SUBGENRE_TO_FAMILY.get(s.lower()) != "Hip-Hop"
            ]

    # 2. Hip-Hop / Rap: strip Metal / Punk subgenres unless explicit metal/punk tags are present
    elif mapped_genre in {"Hip-Hop", "Rap"}:
        if not any(r in {"metal", "heavy metal", "punk", "punk rock"} for r in raw_clean_set):
            mapped_subgenres = [
                s
                for s in mapped_subgenres
                if SUBGENRE_TO_FAMILY.get(s.lower()) not in {"Metal", "Punk"}
            ]

    # 3. Classical: strip incompatible modern rock/pop/metal subgenres
    elif mapped_genre == "Classical":
        incompatible_classical_families = {"Rock", "Metal", "Punk", "Hip-Hop", "Funk", "Country"}
        mapped_subgenres = [
            s
            for s in mapped_subgenres
            if SUBGENRE_TO_FAMILY.get(s.lower()) not in incompatible_classical_families
        ]

    return mapped_subgenres


def deduplicate_subgenres(primary_genre: str | None, subgenres: list[str]) -> list[str]:
    """Deduplicate subgenres and remove primary genre exact matches and mutual style conflicts."""
    if not subgenres:
        return []

    seen: set[str] = set()
    cleaned: list[str] = []
    primary_lower = primary_genre.lower().strip() if primary_genre else ""

    for s in subgenres:
        s_clean = s.strip()
        s_lower = s_clean.lower()
        if not s_clean:
            continue
        if s_lower == primary_lower:
            continue
        if s_lower not in seen:
            seen.add(s_lower)
            cleaned.append(s_clean)

    # Filter mutually exclusive styles
    final: list[str] = []
    for item in cleaned:
        conflict = False
        for group in MUTUALLY_EXCLUSIVE_STYLES:
            if item in group and any(e in group for e in final):
                conflict = True
                break
        if not conflict:
            final.append(item)

    return final


__all__ = [
    "COMPOUND_SUBGENRE_WHITELIST",
    "DEFAULT_PRIMARY_GENRES",
    "DEFAULT_SUB_GENRES",
    "GENERIC_MODIFIERS",
    "GENRE_STOP_WORDS",
    "MUTUALLY_EXCLUSIVE_STYLES",
    "NATIONALITY_STRINGS",
    "PRIMARY_GENRE_STEMS",
    "SUB_GENRE_STEMS",
    "SUBGENRE_REGISTRY",
    "SUBGENRE_TO_FAMILY",
    "SubgenreSpec",
    "deduplicate_subgenres",
    "is_valid_subgenre_tag",
    "promote_genre_by_subgenres",
    "sanitize_subgenres_for_genre",
]
