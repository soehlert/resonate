"""Unit tests for taxonomy rules, primary genre stem matching, and subgenre consensus."""

import pytest

from resonate.engine.taxonomy import (
    DEFAULT_PRIMARY_GENRES,
    DEFAULT_SUB_GENRES,
    is_valid_subgenre_tag,
    promote_genre_by_subgenres,
)
from resonate.modules.tag_mapper import TagMapper


@pytest.fixture(scope="module")
def primary_genre_mapper() -> TagMapper:
    """Shared TagMapper for primary genre matching."""
    return TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)


@pytest.fixture(scope="module")
def subgenre_mapper() -> TagMapper:
    """Shared TagMapper for subgenre matching."""
    return TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)


# --- 1. Primary Genre Consensus & Stem Matching ---


@pytest.mark.parametrize(
    ("tag", "expected_genre"),
    [
        ("alternative rock", "Rock"),
        ("rock and roll", "Rock"),
        ("punk rock", "Punk"),
        ("heavy metal", "Metal"),
        ("skate punk", "Punk"),
        ("gangsta rap", "Hip-Hop"),
        ("bebop jazz", "Jazz"),
        ("chicago blues", "Blues"),
        ("alt-country", "Country"),
        ("indie folk", "Folk"),
        ("ambient electronic", "Electronic"),
        ("neo-soul", "Soul"),
        ("roots reggae", "Reggae"),
        ("baroque classical", "Classical"),
        ("indie pop", "Pop"),
        ("glam rock", "Rock"),
        ("glam", "Rock"),
        ("classic country", "Country"),
        ("gospel", "Soul"),
    ],
)
def test_primary_genre_mapping(
    primary_genre_mapper: TagMapper,
    tag: str,
    expected_genre: str,
) -> None:
    """Verify multi-word consensus and single stem tags map to expected primary genres."""
    results = primary_genre_mapper.match_multiple_tags([tag])
    assert results, f"Failed to match raw tag '{tag}'"
    matched_genres = [r[0] for r in results]
    assert expected_genre in matched_genres


# --- 2. Subgenre Disambiguation & Compound Rules ---


@pytest.mark.parametrize(
    ("raw_tags", "expected_in", "expected_not_in"),
    [
        # Pop rock must never map to Post-Rock
        (["pop rock"], ["Pop Rock"], ["Post-Rock"]),
        # Nationality tag must never map to Americana
        (["american"], [], ["Americana"]),
        # Rock and Roll vs Rockabilly mutual disambiguation
        (["rock and roll"], ["Rock and Roll"], ["Rockabilly"]),
        (["rock n roll"], ["Rock and Roll"], ["Rockabilly"]),
        (["rockabilly"], ["Rockabilly"], ["Rock and Roll"]),
        (["rock", "punk"], ["Punk Rock"], ["Rockabilly"]),
        (["rock"], [], ["Rockabilly"]),
        # Indie alone does not map to Indie Folk
        (["indie"], [], ["Indie Folk"]),
        # Garage rock + indie matches Garage Rock and Indie Rock, not Indie Folk
        (["garage rock", "indie rock", "indie"], ["Garage Rock", "Indie Rock"], ["Indie Folk"]),
        # Prog rock stem matching
        (["progressive rock", "prog"], ["Prog Rock"], []),
        # Industrial matches Industrial subgenre, NOT Industrial Metal
        (["industrial"], ["Industrial"], ["Industrial Metal"]),
        # Orchestra and Chamber music must not match Big Band
        (["orchestra", "symphonic"], ["Symphonic"], ["Big Band"]),
        (["string quartet", "chamber music"], ["Chamber Music"], ["Big Band"]),
        # Missing subgenre stems
        (["thrash metal"], ["Thrash Metal"], []),
        (["hardcore punk"], ["Hardcore Punk"], []),
        (["instrumental rock"], ["Instrumental Rock"], []),
        (["instrumental"], ["Instrumental"], []),
        (["oldies"], ["Oldies"], []),
        (["blue-eyed soul", "neo soul"], ["Blue-Eyed Soul", "Neo-Soul"], []),
        (["glam rock"], ["Glam Rock"], []),
        (["glam"], ["Glam Rock"], []),
        (["rock", "glam"], ["Glam Rock"], []),
        (["classic country"], ["Classic Country"], []),
        (["gospel"], ["Gospel"], []),
        (["alternative and punk"], ["Alternative Punk"], []),
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
        # Standalone modifier ignored
        (["hardcore"], [], ["Hardcore Hip Hop", "Hardcore Punk"]),
        # Combined with genre keywords
        (["hardcore", "punk"], ["Hardcore Punk"], ["Hardcore Hip Hop"]),
        (["hardcore", "hip hop"], ["Hardcore Hip Hop"], ["Hardcore Punk"]),
        # Generic modifiers require full phrases
        (["southern"], [], ["Southern Rock"]),
        (["roots"], [], ["Roots Rock"]),
        (["progressive"], [], ["Progressive Metal"]),
        (["southern rock"], ["Southern Rock"], []),
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


# --- 3. Tag Validation & Stop-Word Filtering ---


@pytest.mark.parametrize(
    ("tag", "artist", "album", "expected_valid"),
    [
        ("singer-songwriter", "Artist Name", "Album", True),
        ("blues rock", "Artist Name", "Greatest Hits: 30 Years of Rock", True),
        ("hard rock", "Artist Name", "Greatest Hits: 30 Years of Rock", True),
        ("rock & roll", "Artist Name", "Greatest Hits: 30 Years of Rock", True),
        ("disco", "ABBA", "ABBA - Disco", True),
        ("americana", "Hurray for the Riff Raff", "Americana Sessions", True),
        ("album rock", "Artist Name", "Album", False),
        ("seen live", "Artist Name", "Album", False),
        ("favourites", "Artist Name", "Album", False),
        ("90s", "Artist Name", "Album", False),
        ("2006", "Artist Name", "Album", False),
        ("alternative and punk", "Artist Name", "Album", True),
        ("classic country", "Artist Name", "Album", True),
        ("gospel", "Artist Name", "Album", True),
        ("rock and punk", "Artist Name", "Album", False),
        ("folk and punk", "Artist Name", "Album", False),
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


# --- 4. Primary Genre Promotion Rules ---


@pytest.mark.parametrize(
    ("parent_genre", "subgenre_scores", "expected_promoted"),
    [
        ("Pop", {"Pop-Punk": 1.0}, "Punk"),
        ("Rock", {"Pop-Punk": 1.0}, "Punk"),
        ("Reggae", {"Ska Punk": 1.0}, "Punk"),
        ("Rock", {"Ska Punk": 1.0}, "Punk"),
        ("Rock", {"Hardcore Punk": 1.0, "Punk Rock": 1.0}, "Punk"),
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


def test_promote_genre_by_subgenres_symmetric_raw_tags() -> None:
    """Verify raw tags matching child family give credit to child family."""
    promoted, decision = promote_genre_by_subgenres(
        "Rock",
        {"Punk Rock": 1.0},
        raw_tags=["rock", "punk"],
    )
    assert promoted == "Punk"
    assert decision is not None
    assert decision.promoted_genre == "Punk"

    not_promoted, decision = promote_genre_by_subgenres(
        "Rock",
        {"Punk Rock": 1.0},
        raw_tags=["rock", "classic rock"],
    )
    assert not_promoted == "Rock"
    assert decision is None


def test_promote_genre_pop_to_soul_with_blue_eyed_soul() -> None:
    """Verify Pop is promoted to Soul when Neo-Soul and Blue-Eyed Soul outscore Pop."""
    promoted, decision = promote_genre_by_subgenres(
        "Pop",
        {"Neo-Soul": 0.96, "Blue-Eyed Soul": 0.92},
        raw_tags=["pop", "Neo Soul", "Blue-Eyed Soul"],
    )
    assert promoted == "Soul"
    assert decision is not None
    assert decision.promoted_genre == "Soul"
    assert "Neo-Soul" in decision.contributing_subgenres
    assert "Blue-Eyed Soul" in decision.contributing_subgenres
