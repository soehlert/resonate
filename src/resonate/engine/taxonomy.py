"""Genre taxonomy hierarchy, family promotion rules, and subgenre filtering engine."""

from __future__ import annotations

import logging
import re
from collections import Counter

from resonate.engine.subgenre_registry import (
    COMPOUND_SUBGENRE_WHITELIST,
    DEFAULT_PRIMARY_GENRES,
    DEFAULT_SUB_GENRES,
    FAMILY_TO_PRIMARY,
    GENERIC_MODIFIERS,
    MUTUALLY_EXCLUSIVE_STYLES,
    NATIONALITY_STRINGS,
    PRIMARY_GENRE_STEMS,
    SUB_GENRE_STEMS,
    SUBGENRE_REGISTRY,
    SUBGENRE_TO_FAMILY,
    SubgenreSpec,
)
from resonate.engine.tag_filter import is_artist_or_album_match, is_boilerplate_tag
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

    # 2. Skip if it matches the artist or album name
    if is_artist_or_album_match(tag_lower, artist, album):
        return False

    # 3. Skip common non-genre/boilerplate/playlist descriptors
    if is_boilerplate_tag(tag_lower):
        return False

    return True



def _get_family_for_tag(tag: str) -> str | None:
    """Resolve raw tag or subgenre string to its canonical primary genre family."""
    t_clean = tag.lower().strip()
    for g in DEFAULT_PRIMARY_GENRES:
        if t_clean == g.lower():
            return g
    for g, stems in PRIMARY_GENRE_STEMS.items():
        if t_clean in stems:
            return g
    fam = SUBGENRE_TO_FAMILY.get(t_clean)
    if not fam:
        return None
    fam = FAMILY_TO_PRIMARY.get(fam, fam)
    if fam in DEFAULT_PRIMARY_GENRES:
        return fam
    return None


def promote_genre_by_subgenres(
    mapped_genre: str | None,
    subgenre_scores: dict[str, float],
    raw_tags: list[str] | None = None,
) -> tuple[str | None, TaxonomyDecision | None]:
    """Elevate generic Rock, Pop, or Reggae if child subgenres strictly outscore parent."""
    if not mapped_genre or mapped_genre not in {"Rock", "Pop", "Reggae"} or not subgenre_scores:
        return mapped_genre, None

    subgenre_family_scores: Counter[str] = Counter()
    for sg, score in subgenre_scores.items():
        fam = SUBGENRE_TO_FAMILY.get(sg.lower())
        if not fam:
            continue
        fam = FAMILY_TO_PRIMARY.get(fam, fam)
        if fam in DEFAULT_PRIMARY_GENRES:
            subgenre_family_scores[fam] += score

    parent_family = mapped_genre
    parent_score = subgenre_family_scores.get(parent_family, 0.0)

    # Factor in explicit raw tags not already accounted for in subgenre scores
    if raw_tags:
        subgenre_tag_set = {sg.lower() for sg in subgenre_scores}
        for t in raw_tags:
            t_clean = t.lower().strip()
            if t_clean in subgenre_tag_set:
                continue
            tag_fam = _get_family_for_tag(t_clean)
            if tag_fam == parent_family:
                parent_score += 1.0
            elif tag_fam and tag_fam in DEFAULT_PRIMARY_GENRES:
                subgenre_family_scores[tag_fam] += 1.0

    top_candidates = [
        (fam, score) for fam, score in subgenre_family_scores.most_common() if fam != parent_family
    ]
    if top_candidates:
        top_child_family, top_child_score = top_candidates[0]
        if top_child_score > parent_score:
            total_score = top_child_score + parent_score
            conf = round(top_child_score / total_score, 2) if total_score > 0 else 1.0
            decision = TaxonomyDecision(
                original_genre=mapped_genre,
                promoted_genre=top_child_family,
                reason=(
                    f"Child family '{top_child_family}' score ({top_child_score:.2f}) "
                    f"strictly outnumbers parent '{parent_family}' score ({parent_score:.2f})"
                ),
                contributing_subgenres=[
                    sg
                    for sg in subgenre_scores
                    if SUBGENRE_TO_FAMILY.get(sg.lower()) == top_child_family
                ],
                confidence=conf,
            )
            return top_child_family, decision

    return mapped_genre, None


def filter_subgenres_by_family(
    primary_genre: str | None, subgenres: list[str]
) -> list[str]:
    """Filter subgenres to those matching the primary genre's family."""
    if not primary_genre or not subgenres:
        return subgenres

    return [s for s in subgenres if _get_family_for_tag(s) == primary_genre]


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
    "MUTUALLY_EXCLUSIVE_STYLES",

    "NATIONALITY_STRINGS",
    "PRIMARY_GENRE_STEMS",
    "SUB_GENRE_STEMS",
    "SUBGENRE_REGISTRY",
    "SUBGENRE_TO_FAMILY",
    "SubgenreSpec",
    "deduplicate_subgenres",
    "filter_subgenres_by_family",
    "is_valid_subgenre_tag",
    "promote_genre_by_subgenres",
]
