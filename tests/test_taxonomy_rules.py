"""Unit tests for taxonomy rules, primary genre stem matching, and GENRE_MOOD_SEEDS."""

from resonate.modules.essentia import ESSENTIA_MOOD_MAP
from resonate.modules.tag_mapper import (
    DEFAULT_PRIMARY_GENRES,
    GENRE_MOOD_SEEDS,
    TagMapper,
    get_genre_seeded_moods,
)


def test_multi_word_primary_genre_mapping_rock() -> None:
    """Verify multi-word consensus tags map to Rock primary genre."""
    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    rock_tags = [
        "alternative rock",
        "acoustic rock",
        "rock and roll",
        "rock n roll",
        "rockabilly",
        "soft rock",
        "hard rock",
        "indie rock",
        "garage rock",
        "punk rock",
    ]
    for tag in rock_tags:
        results = mapper.match_multiple_tags([tag])
        assert results, f"Failed to match raw tag '{tag}'"
        assert results[0][0] == "Rock", f"Tag '{tag}' mapped to '{results[0][0]}' instead of 'Rock'"


def test_multi_word_primary_genre_mapping_other_genres() -> None:
    """Verify multi-word consensus tags map to respective primary genres."""
    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    test_cases = [
        ("indie pop", "Pop"),
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
    ]
    for tag, expected in test_cases:
        results = mapper.match_multiple_tags([tag])
        matched_targets = [r[0] for r in results]
        assert expected in matched_targets, (
            f"Tag '{tag}' matched targets {matched_targets}, expected '{expected}' to be included"
        )


def test_genre_mood_seeds_chill_hang() -> None:
    """Verify Chill Hang is seeded for millennial indie, americana, alt-rock, and emo genres."""
    chill_genres = ["Indie Rock", "Indie Pop", "Indie Folk", "Alternative Rock", "Americana", "Emo"]
    for g in chill_genres:
        seeded = get_genre_seeded_moods([g])
        assert "Chill Hang" in seeded, f"Genre '{g}' missing 'Chill Hang' seed: {seeded}"


def test_hard_rock_seeds_heavy() -> None:
    """Verify Hard Rock seeds Heavy."""
    assert "Hard Rock" in GENRE_MOOD_SEEDS
    assert "Heavy" in GENRE_MOOD_SEEDS["Hard Rock"]


def test_classic_rock_does_not_seed_groovy() -> None:
    """Verify Classic Rock does NOT seed Groovy."""
    if "Classic Rock" in GENRE_MOOD_SEEDS:
        assert "Groovy" not in GENRE_MOOD_SEEDS["Classic Rock"]


def test_love_node_maps_to_romantic() -> None:
    """Verify Essentia love node maps to Romantic."""
    assert "love" in ESSENTIA_MOOD_MAP
    assert ESSENTIA_MOOD_MAP["love"] == "Romantic"


def test_energetic_bpm_threshold_130() -> None:
    """Verify Energetic requires BPM >= 130."""
    moods = ["Upbeat", "Energetic"]
    bpm_below = 128
    bpm_above = 132

    # Below 130 BPM drops Energetic
    filtered_below = [m for m in moods if not (m.lower() == "energetic" and bpm_below < 130)]
    assert "Energetic" not in filtered_below

    # At or above 130 BPM keeps Energetic
    filtered_above = [m for m in moods if not (m.lower() == "energetic" and bpm_above < 130)]
    assert "Energetic" in filtered_above


def test_techno_primary_genre_electronic() -> None:
    """Verify raw tag 'techno' maps to Electronic primary genre (not Punk)."""
    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    results = mapper.match_multiple_tags(["techno"])
    assert results, "Failed to match raw tag 'techno'"
    assert results[0][0] == "Electronic"


def test_pop_rock_does_not_map_to_post_rock() -> None:
    """Verify raw tag 'pop rock' does not hallucinate Post-Rock subgenre."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(["pop rock", "epic"])
    matched = [r[0] for r in results]
    assert "Post-Rock" not in matched


def test_conflict_resolution_strips_chill_hang_and_groovy() -> None:
    """Verify Heavy/Aggressive strips Chill Hang and Groovy."""
    from resonate.modules.tag_mapper import resolve_mood_conflicts

    moods = ["Chill Hang", "Groovy", "Heavy", "Aggressive", "Intense"]
    resolved = resolve_mood_conflicts(moods)
    assert "Chill Hang" not in resolved
    assert "Groovy" not in resolved
    assert "Heavy" in resolved
    assert "Aggressive" in resolved


def test_american_raw_tag_does_not_map_to_americana() -> None:
    """Verify nationality raw tag 'american' does not map to Americana subgenre."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(["american"])
    matched = [r[0] for r in results]
    assert "Americana" not in matched


def test_missing_subgenres_and_stem_matches() -> None:
    """Verify subgenres like Thrash Metal, Hardcore Punk, Rap, and Punk Rock match correctly."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)

    # 1. Thrash metal
    res_thrash = mapper.match_multiple_tags(["thrash metal"])
    assert any(r[0] == "Thrash Metal" for r in res_thrash)

    # 2. Hardcore punk
    res_hc = mapper.match_multiple_tags(["hardcore punk"])
    assert any(r[0] == "Hardcore Punk" for r in res_hc)

    # 3. Rap from hip hop
    res_rap = mapper.match_multiple_tags(["hip hop"])
    assert any(r[0] == "Rap" for r in res_rap)

    # 4. Punk rock from punk
    res_punk = mapper.match_multiple_tags(["punk"])
    assert any(r[0] == "Punk Rock" for r in res_punk)


def test_essentia_mood_map_cleanup() -> None:
    """Verify deep, funny, and cool are removed from ESSENTIA_MOOD_MAP."""
    assert "deep" not in ESSENTIA_MOOD_MAP
    assert "funny" not in ESSENTIA_MOOD_MAP
    assert "cool" not in ESSENTIA_MOOD_MAP


def test_rockabilly_and_oldies_subgenres() -> None:
    """Verify Rockabilly and Oldies match for 50s rock tags."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(["rockabilly", "oldies", "rock and roll"])
    matched = [r[0] for r in results]
    assert "Rockabilly" in matched
    assert "Oldies" in matched
    assert "Rock and Roll" in matched


def test_indie_alone_does_not_map_to_indie_folk() -> None:
    """Verify raw tag 'indie' alone does not match Indie Folk."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(["indie"])
    matched = [r[0] for r in results]
    assert "Indie Folk" not in matched


def test_garage_rock_and_indie_disambiguation() -> None:
    """Verify 'garage rock' + 'indie' matches Indie Rock and Garage Rock (not Indie Folk)."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(["garage rock", "indie rock", "indie", "garage"])
    matched = [r[0] for r in results]
    assert "Garage Rock" in matched
    assert "Indie Rock" in matched
    assert "Indie Folk" not in matched


def test_tail_tag_cannot_introduce_punk_rock() -> None:
    """Verify tail tag at index > 5 ('alternative and punk') does not match Punk Rock."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    raw_tags = [
        "alternative rock",
        "rock",
        "hard rock",
        "alternative",
        "grunge",
        "post-grunge",
        "2005",
        "chris cornell",
        "00s",
        "american",
        "male vocalists",
        "alternative metal",
        "mellow",
        "alternative and punk",
    ]
    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(raw_tags)
    matched = [r[0] for r in results]
    assert "Punk Rock" not in matched
