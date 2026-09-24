"""Unit tests for taxonomy rules, primary genre stem matching, and subgenre consensus."""

import html

import pytest

from resonate.engine.mood_rules import synthesize_track_moods
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
        ("acoustic rock", "Rock"),
        ("rock and roll", "Rock"),
        ("rock n roll", "Rock"),
        ("rockabilly", "Rock"),
        ("soft rock", "Rock"),
        ("hard rock", "Rock"),
        ("indie rock", "Rock"),
        ("garage rock", "Rock"),
        ("classic rock", "Rock"),
        ("punk rock", "Punk"),
        ("indie pop", "Pop"),
        ("heavy metal", "Metal"),
        ("skate punk", "Punk"),
        ("gangsta rap", "Hip-Hop"),
        ("bebop jazz", "Jazz"),
        ("chicago blues", "Blues"),
        ("alt-country", "Country"),
        ("indie folk", "Folk"),
        ("ambient electronic", "Electronic"),
        ("techno", "Electronic"),
        ("neo-soul", "Soul"),
        ("roots reggae", "Reggae"),
        ("baroque classical", "Classical"),
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


def test_subgenre_consensus_voting_outvotes_isolated_minority(
    subgenre_mapper: TagMapper,
) -> None:
    """Verify subgenre consensus voting prioritizes cluster consensus over isolated tags."""
    raw_tags = ["hard rock", "alternative", "alt-country", "alternative rock", "country-rock"]
    matches = subgenre_mapper.match_subgenre_consensus(raw_tags, max_matches=3)
    matched = [m[0] for m in matches]

    assert "Alternative Rock" in matched
    assert "Alt-Country" in matched
    assert "Hard Rock" not in matched


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
        ("album rock", "Artist Name", "Album", False),
        ("seen live", "Artist Name", "Album", False),
        ("favourites", "Artist Name", "Album", False),
        ("90s", "Artist Name", "Album", False),
        ("2006", "Artist Name", "Album", False),
        ("alternative and punk", "Artist Name", "Album", False),
        ("rock and punk", "Artist Name", "Album", False),
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
    ("parent_genre", "subgenres", "expected_promoted"),
    [
        ("Pop", ["Pop-Punk"], "Punk"),
        ("Rock", ["Pop-Punk"], "Punk"),
        ("Reggae", ["Ska Punk"], "Punk"),
        ("Rock", ["Ska Punk"], "Punk"),
        ("Rock", ["Hardcore Punk", "Punk Rock"], "Punk"),
    ],
)
def test_promote_genre_by_subgenres(
    parent_genre: str,
    subgenres: list[str],
    expected_promoted: str,
) -> None:
    """Verify specific child subgenres elevate generic parent genres to Punk."""
    promoted, _decision = promote_genre_by_subgenres(parent_genre, subgenres)
    assert promoted == expected_promoted


# --- 5. Tag Preprocessing & Audio Confidence Floored Fallbacks ---


def test_html_unescaping_tags() -> None:
    """Verify HTML entities like '&amp;' in raw provider tags are properly unescaped."""
    raw_tag = "Rock &amp; Roll"
    assert html.unescape(raw_tag) == "Rock & Roll"


def test_essentia_sub_10_percent_predictions_ignored() -> None:
    """Verify Essentia fallback predictions below 0.10 confidence floor are ignored."""
    moods = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[
            ("love", 0.16),
            ("ballad", 0.15),
            ("melodic", 0.11),
            ("meditative", 0.06),
            ("energetic", 0.05),
        ],
        detected_bpm=141,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Classic Rock", "Pop Rock", "Psychedelic Rock"],
        raw_tags=["classic rock", "pop rock", "rock"],
    )
    assert "Calm" not in moods
    assert "Energetic" not in moods
    assert moods == ["Melancholic"]


def test_rap_metal_subgenre_mapping() -> None:
    """Verify rap metal and rapcore map to Rap Metal under Metal family, not Rap or Hip-Hop."""
    from resonate.engine.taxonomy import DEFAULT_SUB_GENRES, SUBGENRE_TO_FAMILY
    from resonate.modules.tag_mapper import TagMapper

    sm = TagMapper(target_moods=DEFAULT_SUB_GENRES)

    rap_metal_matches = sm.match_multiple_tags(["rap metal"], max_matches=2)
    assert rap_metal_matches
    assert rap_metal_matches[0][0] == "Rap Metal"
    assert not any(m[0] in {"Rap", "Hip-Hop"} for m in rap_metal_matches)

    rapcore_matches = sm.match_multiple_tags(["rapcore"], max_matches=2)
    assert rapcore_matches
    assert rapcore_matches[0][0] == "Rap Metal"
    assert not any(m[0] in {"Rap", "Hip-Hop"} for m in rapcore_matches)

    assert SUBGENRE_TO_FAMILY.get("rap metal") == "Metal"
    assert SUBGENRE_TO_FAMILY.get("rapcore") == "Metal"


def test_fusion_rap_metal_consensus_avoids_hiphop() -> None:
    """Verify fusion tracks with rap metal tags do not get classified as Hip-Hop or Rap."""
    from collections import Counter

    from resonate.engine.taxonomy import (
        DEFAULT_PRIMARY_GENRES,
        DEFAULT_SUB_GENRES,
        deduplicate_subgenres,
        is_valid_subgenre_tag,
        promote_genre_by_subgenres,
        sanitize_subgenres_for_genre,
    )
    from resonate.modules.tag_mapper import TagMapper

    raw_tags = [
        "rock",
        "alternative rock",
        "rap metal",
        "alternative metal",
        "rapcore",
        "hard rock",
    ]

    gm = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES)
    primary_matches = gm.match_genre_consensus(raw_tags)
    assert not any(m[0] == "Hip-Hop" for m in primary_matches)

    genre_counts: Counter[str] = Counter()
    for g_name, raw_t, _score, raw_pos in primary_matches:
        raw_lower = raw_t.lower().strip()
        weight = 3 if raw_lower in {"rock", "metal"} else 1
        if raw_pos < 3:
            weight += 5
        genre_counts[g_name] += weight

    mapped_genre = genre_counts.most_common(1)[0][0]
    assert mapped_genre in {"Rock", "Metal"}

    sm = TagMapper(target_moods=DEFAULT_SUB_GENRES)
    filtered_sg_tags = [
        t
        for t in raw_tags
        if is_valid_subgenre_tag(t, "Generic Artist", "Generic Album")
        and t.lower().strip()
        not in {
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
    ]
    sg_matches = sm.match_subgenre_consensus(filtered_sg_tags, max_matches=3)
    mapped_subgenres = [s[0] for s in sg_matches]

    if mapped_genre in {"Rock", "Pop"} and mapped_subgenres:
        promoted, _decision = promote_genre_by_subgenres(mapped_genre, mapped_subgenres)
        if promoted:
            mapped_genre = promoted

    mapped_subgenres = sanitize_subgenres_for_genre(mapped_genre, mapped_subgenres, raw_tags)
    mapped_subgenres = deduplicate_subgenres(mapped_genre, mapped_subgenres)

    assert mapped_genre in {"Rock", "Metal"}
    assert "Rap Metal" in mapped_subgenres
    assert "Rap" not in mapped_subgenres
    assert "Hip-Hop" not in mapped_subgenres


def test_pure_hip_hop_tags_retain_rap_classification() -> None:
    """Verify authentic hip-hop tags resolve to Hip-Hop and Rap, never Metal or Rap Metal."""
    from collections import Counter

    from resonate.engine.taxonomy import (
        DEFAULT_PRIMARY_GENRES,
        DEFAULT_SUB_GENRES,
        deduplicate_subgenres,
        is_valid_subgenre_tag,
        sanitize_subgenres_for_genre,
    )
    from resonate.modules.tag_mapper import TagMapper

    raw_tags = ["hip hop", "rap", "boom bap", "east coast hip hop"]

    gm = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES)
    primary_matches = gm.match_genre_consensus(raw_tags)
    assert any(m[0] == "Hip-Hop" for m in primary_matches)
    assert not any(m[0] == "Metal" for m in primary_matches)

    genre_counts: Counter[str] = Counter()
    for g_name, raw_t, _score, raw_pos in primary_matches:
        raw_lower = raw_t.lower().strip()
        weight = 3 if raw_lower in {"hip hop", "rap", "hip-hop"} else 1
        if raw_pos < 3:
            weight += 5
        genre_counts[g_name] += weight

    mapped_genre = genre_counts.most_common(1)[0][0]
    assert mapped_genre == "Hip-Hop"

    sm = TagMapper(target_moods=DEFAULT_SUB_GENRES)
    filtered_sg_tags = [
        t
        for t in raw_tags
        if is_valid_subgenre_tag(t, "Generic Rapper", "Generic Album")
        and t.lower().strip()
        not in {
            "rock",
            "pop",
            "metal",
            "jazz",
            "blues",
            "country",
            "folk",
            "electronic",
            "dance",
            "punk",
        }
    ]
    sg_matches = sm.match_subgenre_consensus(filtered_sg_tags, max_matches=3)
    mapped_subgenres = [s[0] for s in sg_matches]

    mapped_subgenres = sanitize_subgenres_for_genre(mapped_genre, mapped_subgenres, raw_tags)
    mapped_subgenres = deduplicate_subgenres(mapped_genre, mapped_subgenres)

    assert mapped_genre == "Hip-Hop"
    expected_rap_subgenres = {"Rap", "Boom Bap", "East Coast Hip Hop"}
    assert any(s in expected_rap_subgenres for s in mapped_subgenres)
    assert "Rap Metal" not in mapped_subgenres
    assert "Alternative Metal" not in mapped_subgenres
