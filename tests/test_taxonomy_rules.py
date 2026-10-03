"""Unit tests for taxonomy rules, primary genre stem matching, and subgenre consensus."""

import pytest

from resonate.engine.taxonomy import (
    DEFAULT_SUB_GENRES,
    is_valid_subgenre_tag,
    promote_genre_by_subgenres,
)
from resonate.modules.tag_mapper import TagMapper


@pytest.fixture(scope="module")
def subgenre_mapper() -> TagMapper:
    """Shared TagMapper for subgenre matching."""
    return TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)


# --- 1. Subgenre Disambiguation & Compound Rules ---


@pytest.mark.parametrize(
    ("raw_tags", "expected_in", "expected_not_in"),
    [
        (["pop rock"], ["Pop Rock"], ["Post-Rock"]),
        (["american"], [], ["Americana"]),
    ],
)
def test_subgenre_disambiguation(
    subgenre_mapper: TagMapper,
    raw_tags: list[str],
    expected_in: list[str],
    expected_not_in: list[str],
) -> None:
    """Verify subgenre matching precision and prevention of false positive mappings."""
    results = subgenre_mapper.match_multiple_tags(raw_tags)
    matched = [r[0] for r in results]
    for expected in expected_in:
        assert expected in matched, f"Expected '{expected}' in {matched} for tags {raw_tags}"
    for not_expected in expected_not_in:
        assert not_expected not in matched, f"Unexpected '{not_expected}' in {matched}"


@pytest.mark.parametrize(
    ("tags", "expected_in", "expected_not_in"),
    [
        (["hardcore"], [], ["Hardcore Punk"]),
        (["hardcore", "punk"], ["Hardcore Punk"], ["Hardcore Hip Hop"]),
    ],
)
def test_contextual_modifier_disambiguation(
    subgenre_mapper: TagMapper,
    tags: list[str],
    expected_in: list[str],
    expected_not_in: list[str],
) -> None:
    """Verify contextual modifier combinations require appropriate genre context."""
    results = subgenre_mapper.match_multiple_tags(tags)
    matched = [r[0] for r in results]
    for expected in expected_in:
        assert expected in matched
    for not_expected in expected_not_in:
        assert not_expected not in matched


def test_tail_tag_cannot_introduce_unrelated_subgenre(
    subgenre_mapper: TagMapper,
) -> None:
    """Verify noise tags at index > 5 do not override dominant candidate subgenres."""
    raw_tags = [
        "alternative rock",
        "rock",
        "hard rock",
        "alternative",
        "grunge",
        "post-grunge",
        "2005",
        "00s",
        "american",
        "alternative metal",
        "alternative and punk",
    ]
    results = subgenre_mapper.match_multiple_tags(raw_tags)
    matched = [r[0] for r in results]
    assert "Punk Rock" not in matched


# --- 2. Tag Validation & Stop-Word Filtering ---


@pytest.mark.parametrize(
    ("tag", "artist", "album", "expected_valid"),
    [
        ("blues rock", "Test Artist", "Test Album", True),
        ("seen live", "Test Artist", "Test Album", False),
    ],
)
def test_is_valid_subgenre_tag(
    tag: str,
    artist: str,
    album: str,
    expected_valid: bool,
) -> None:
    """Verify tag noise, broadcaster/boilerplate, and generic decade filters."""
    assert is_valid_subgenre_tag(tag, artist, album) is expected_valid


# --- 3. Primary Genre Promotion Rules ---


@pytest.mark.parametrize(
    ("parent_genre", "subgenre_scores", "expected_promoted"),
    [
        ("Rock", {"Punk Rock": 1.0}, "Punk"),
        ("Pop", {"Pop-Punk": 1.0}, "Punk"),
    ],
)
def test_promote_genre_by_subgenres(
    parent_genre: str,
    subgenre_scores: dict[str, float],
    expected_promoted: str,
) -> None:
    """Verify specific child subgenres elevate generic parent genres to Punk."""
    promoted, _decision = promote_genre_by_subgenres(parent_genre, subgenre_scores)
    assert promoted == expected_promoted


def test_promote_genre_by_subgenres_scoring() -> None:
    """Verify child family strictly outscoring parent promotes umbrella genre."""
    promoted, decision = promote_genre_by_subgenres(
        "Rock",
        {"Folk Punk": 1.0, "Indie Rock": 0.96},
    )
    assert promoted == "Punk"
    assert decision is not None
    assert decision.promoted_genre == "Punk"

    not_promoted, decision = promote_genre_by_subgenres(
        "Rock",
        {"Punk Rock": 1.0, "Classic Rock": 1.5},
    )
    assert not_promoted == "Rock"
    assert decision is None


def test_filter_subgenres_by_family_cross_family_support() -> None:
    """Verify cross-family subgenres survive under all their declared parent families."""
    from resonate.engine.taxonomy import filter_subgenres_by_family

    subgenres = ["Blues Rock", "Garage Rock", "Hard Rock"]

    blues_survivors = filter_subgenres_by_family("Blues", subgenres)
    rock_survivors = filter_subgenres_by_family("Rock", subgenres)
    hiphop_survivors = filter_subgenres_by_family("Hip-Hop", subgenres)

    assert blues_survivors == ["Blues Rock", "Garage Rock"]
    assert rock_survivors == ["Blues Rock", "Garage Rock", "Hard Rock"]
    assert hiphop_survivors == []


def test_promote_genre_non_promotable_parent_retained() -> None:
    """Verify non-promotable parent genre is never demoted."""
    promoted, decision = promote_genre_by_subgenres(
        "Metal",
        {"Heavy Metal": 1.0, "Thrash Metal": 1.0},
    )
    assert promoted == "Metal"
    assert decision is None


def test_all_primary_genres_have_mood_seeds() -> None:
    """Verify all 17 default primary genres have non-empty mood seeds for fallback."""
    from resonate.engine.mood_rules import get_genre_seeded_moods
    from resonate.engine.taxonomy import DEFAULT_PRIMARY_GENRES

    for primary in DEFAULT_PRIMARY_GENRES:
        seeds = get_genre_seeded_moods(subgenres=[], primary_genre=primary)
        assert len(seeds) > 0, f"Primary genre '{primary}' must have at least one mood seed"

