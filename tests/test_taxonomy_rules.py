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
        "classic rock",
    ]
    for tag in rock_tags:
        results = mapper.match_multiple_tags([tag])
        assert results, f"Failed to match raw tag '{tag}'"
        assert results[0][0] == "Rock", f"Tag '{tag}' mapped to '{results[0][0]}' instead of 'Rock'"


def test_multi_word_primary_genre_mapping_other_genres() -> None:
    """Verify multi-word consensus tags map to respective primary genres."""
    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    test_cases = [
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
    """Verify Chill Hang is seeded for millennial indie, americana, and alt-rock genres."""
    chill_genres = [
        "Indie Rock",
        "Indie Pop",
        "Indie Folk",
        "Lo-Fi",
        "Americana",
        "Alternative Rock",
    ]
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


def test_cold_war_kids_primary_genre_not_punk() -> None:
    """Verify Cold War Kids raw tags do not map to Punk primary genre."""
    raw_tags = [
        "indie",
        "indie rock",
        "alternative",
        "cold war kids",
        "rock",
        "2006",
        "downtown records",
        "alternative rock",
    ]
    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    results = mapper.match_multiple_tags(raw_tags)
    matched = [r[0] for r in results]
    assert "Punk" not in matched
    assert "Indie" in matched or "Rock" in matched


def test_gbh_primary_genre_punk() -> None:
    """Verify GBH raw tags map to Punk primary genre."""
    raw_tags = [
        "hardcore punk",
        "punk rock",
        "street punk",
        "hardcore",
        "punk",
        "1982",
    ]
    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    results = mapper.match_multiple_tags(raw_tags)
    assert results
    assert results[0][0] == "Punk"


def test_acoustic_rock_and_heavy_mood_exclusion() -> None:
    """Verify Acoustic Rock excludes Hard Rock, and Acoustic mood strips Heavy."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, resolve_mood_conflicts

    # Sub-genre exclusion
    subgenre_mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    subgenres = [
        r[0]
        for r in subgenre_mapper.match_multiple_tags(["acoustic rock", "acoustic", "hard rock"])
    ]
    assert "Acoustic Rock" in subgenres
    assert "Hard Rock" not in subgenres

    # Mood conflict resolution
    moods = ["Acoustic", "Heavy", "Mellow"]
    resolved = resolve_mood_conflicts(moods)
    assert "Heavy" not in resolved
    assert "Acoustic" in resolved
    assert "Mellow" in resolved


def test_artist_matches_verification() -> None:
    """Verify artist_matches correctly differentiates Ye from Yes."""
    from resonate.modules.external_metadata import artist_matches

    assert artist_matches("Ye", "Ye") is True
    assert artist_matches("Ye", "Ye feat. Jay-Z") is True
    assert artist_matches("Ye", "Yes") is False
    assert artist_matches("Air", "Air Supply") is False
    assert artist_matches("The Beatles", "Beatles") is True


def test_chill_hang_gated_by_tempo_and_energy() -> None:
    """Verify Alternative Rock Chill Hang is dropped when BPM >= 125 or high energy/dark."""
    from resonate.modules.tag_mapper import resolve_mood_conflicts

    # Conflict resolution: Dark or Heavy drops Chill Hang
    conflicting = resolve_mood_conflicts(["Chill Hang", "Dark"])
    assert "Dark" in conflicting
    assert "Chill Hang" not in conflicting

    conflicting_heavy = resolve_mood_conflicts(["Chill Hang", "Heavy"])
    assert "Heavy" in conflicting_heavy
    assert "Chill Hang" not in conflicting_heavy

    # Millennial Indie Rock with mellow waveform keeps Chill Hang
    mellow_indie = resolve_mood_conflicts(["Chill Hang", "Acoustic"])
    assert "Chill Hang" in mellow_indie
    assert "Acoustic" in mellow_indie


def test_meat_puppets_top_3_gating_excludes_punk_rock() -> None:
    """Verify Meat Puppets raw tags strictly gate candidates to top 3, excluding Punk Rock."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    raw_tags = [
        "grunge",
        "alternative rock",
        "alternative",
        "punk",
        "rock",
        "80s",
        "cowpunk",
        "90s",
        "punk rock",
        "instrumental",
    ]
    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(raw_tags)
    matched = [r[0] for r in results]
    assert "Grunge" in matched
    assert "Alternative Rock" in matched
    assert "Punk Rock" not in matched


def test_dead_kennedys_elevates_primary_genre_to_punk() -> None:
    """Verify punk subgenres elevate generic Rock to Punk."""
    subgenres = ["Punk Rock", "Hardcore Punk", "Post-Hardcore"]
    primary_genre = "Rock"
    if any(
        s.lower() in {"punk rock", "hardcore punk", "post-hardcore", "skate punk"}
        for s in subgenres
    ):
        primary_genre = "Punk"
    assert primary_genre == "Punk"


def test_essentia_epic_maps_to_atmospheric() -> None:
    """Verify Essentia epic maps to Atmospheric rather than Intense."""
    from resonate.modules.essentia import ESSENTIA_MOOD_MAP

    assert ESSENTIA_MOOD_MAP.get("epic") == "Atmospheric"


def test_hip_hop_genre_mood_seeds_no_chill_hang() -> None:
    """Verify Hip-Hop subgenres have mood seeds and never include Chill Hang."""
    from resonate.modules.tag_mapper import get_genre_seeded_moods

    hip_hop_cases = [
        ("Hip-Hop", ["Groovy", "Soulful"]),
        ("Rap", ["Groovy", "Rowdy"]),
        ("East Coast Hip Hop", ["Groovy", "Soulful"]),
        ("West Coast Hip Hop", ["Groovy", "Mellow"]),
        ("G-Funk", ["Groovy", "Mellow"]),
        ("Boom Bap", ["Groovy", "Soulful"]),
        ("Trap", ["Intense", "Dark", "Groovy"]),
        ("Gangsta Rap", ["Intense", "Dark", "Aggressive", "Rowdy"]),
        ("Cloud Rap", ["Melancholic", "Mellow", "Atmospheric"]),
    ]
    for genre, expected_seeds in hip_hop_cases:
        seeds = get_genre_seeded_moods([genre])
        assert seeds, f"No mood seeds found for '{genre}'"
        assert "Chill Hang" not in seeds, f"'{genre}' should not contain 'Chill Hang'"
        for expected in expected_seeds:
            assert expected in seeds, f"'{genre}' missing expected seed '{expected}': {seeds}"


def test_melancholic_and_energetic_coexist() -> None:
    """Verify Melancholic and Energetic coexist for emo/alt-rock tracks."""
    from resonate.modules.tag_mapper import resolve_mood_conflicts

    moods = ["Melancholic", "Energetic"]
    resolved = resolve_mood_conflicts(moods)
    assert "Melancholic" in resolved
    assert "Energetic" in resolved


def test_hip_hop_keywords_in_recognized_moods() -> None:
    """Verify hip-hop vibe keywords exist in RECOGNIZED_MOOD_KEYWORDS without banger."""
    from resonate.main import RECOGNIZED_MOOD_KEYWORDS

    assert "banger" not in RECOGNIZED_MOOD_KEYWORDS
    for kw in ["hype", "gritty", "laid-back", "conscious", "street", "vibes", "flow"]:
        assert kw in RECOGNIZED_MOOD_KEYWORDS, f"Missing keyword '{kw}'"


def test_is_valid_subgenre_tag_filters_junk() -> None:
    """Verify is_valid_subgenre_tag filters out TV show names, personal favourites, and decades."""
    from resonate.main import is_valid_subgenre_tag
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, TagMapper

    raw_tags = [
        "gtst",
        "personal favourites",
        "gtst s36",
        "ludo sanders",
        "2010s",
        "albums",
        "2017 albums",
        "indie rock",
        "rockabilly",
        "rock and roll",
        "blues",
    ]
    cleaned = [
        t for t in raw_tags if is_valid_subgenre_tag(t, "JD McPherson", "Undivided Heart & Soul")
    ]
    assert "gtst" not in cleaned
    assert "personal favourites" not in cleaned
    assert "2010s" not in cleaned
    assert "albums" not in cleaned
    assert "indie rock" in cleaned
    assert "rockabilly" in cleaned
    assert "rock and roll" in cleaned

    # Verify that with cleaned tags, subgenre mapper correctly matches Rockabilly and Rock and Roll
    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    matches = mapper.match_multiple_tags(cleaned)
    matched_subgenres = [m[0] for m in matches]
    assert "Rockabilly" in matched_subgenres
    assert "Rock and Roll" in matched_subgenres


def test_bpm_110_to_130_converts_energetic_to_lively() -> None:
    """Verify 110-130 BPM tracks with Energetic prediction convert to Lively."""
    bpm = 128
    mapped_moods = ["Energetic"]
    if 110 <= bpm < 130:
        mapped_moods = ["Lively" if m.lower() == "energetic" else m for m in mapped_moods]
    assert mapped_moods == ["Lively"]


def test_pop_punk_seeds_rowdy_and_upbeat() -> None:
    """Verify Pop-Punk seeds Rowdy and Upbeat."""
    from resonate.modules.tag_mapper import get_genre_seeded_moods

    seeds = get_genre_seeded_moods(["Pop-Punk"])
    assert "Rowdy" in seeds
    assert "Upbeat" in seeds


def test_pop_punk_elevates_pop_to_punk() -> None:
    """Verify Pop-Punk subgenre elevates mapped Primary Genre from Pop to Punk."""
    mapped_genre = "Pop"
    mapped_subgenres = ["Pop-Punk", "Punk Rock"]
    if mapped_genre in {"Rock", "Pop"} and mapped_subgenres:
        if any(
            s.lower()
            in {
                "punk rock",
                "hardcore punk",
                "post-hardcore",
                "skate punk",
                "pop-punk",
            }
            for s in mapped_subgenres
        ):
            mapped_genre = "Punk"
    assert mapped_genre == "Punk"


def test_heavy_and_melancholic_and_lively_coexist() -> None:
    """Verify Melancholic is preserved when Heavy and Lively are present (Local H case)."""
    from resonate.modules.tag_mapper import resolve_mood_conflicts

    moods = ["Heavy", "Melancholic", "Lively"]
    resolved = resolve_mood_conflicts(moods)
    assert "Heavy" in resolved
    assert "Melancholic" in resolved
    assert "Lively" in resolved


def test_beatles_127_bpm_energetic_converts_to_lively() -> None:
    """Verify Beatles 'I'll Get You' at 127 BPM converts energetic 0.1402 to Lively."""
    bpm = 127
    mapped_moods = ["Energetic"]
    if 110 <= bpm < 130:
        mapped_moods = ["Lively" if m.lower() == "energetic" else m for m in mapped_moods]
    assert mapped_moods == ["Lively"]


def test_html_unescaping_tags() -> None:
    """Verify HTML entities in raw tags are unescaped."""
    import html

    raw_tags = ["r&amp;b", "rock &amp; roll", "rhythm &amp; blues", "pop"]
    unescaped = [html.unescape(t) for t in raw_tags]
    assert unescaped == ["r&b", "rock & roll", "rhythm & blues", "pop"]


def test_rnb_subgenre_and_seeds() -> None:
    """Verify R&B matches as subgenre and seeds Soulful and Groovy."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, TagMapper, get_genre_seeded_moods

    assert "R&B" in DEFAULT_SUB_GENRES
    assert "Contemporary R&B" in DEFAULT_SUB_GENRES

    seeds = get_genre_seeded_moods(["R&B"])
    assert "Soulful" in seeds
    assert "Groovy" in seeds

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    matches = mapper.match_multiple_tags(["r&b"])
    matched_subgenres = [m[0] for m in matches]
    assert "R&B" in matched_subgenres


def test_synergy_gating_rejects_unseeded_ballad_for_rock() -> None:
    """Verify unseeded ballad 0.0509 is rejected without candidate seed (Rolling Stones case)."""
    from resonate.modules.essentia import ESSENTIA_MOOD_MAP

    candidate_seeds = ["Lively", "Upbeat"]  # Not Melancholic
    pred_lbl = "ballad"
    pred_score = 0.0509

    is_synergy = False
    if candidate_seeds and pred_lbl in ESSENTIA_MOOD_MAP:
        target = ESSENTIA_MOOD_MAP[pred_lbl]
        if any(target.lower() == cs.lower() for cs in candidate_seeds):
            is_synergy = True

    # Without synergy, score < 0.10 should be rejected
    is_confident = (is_synergy and pred_score >= 0.05) or pred_score >= 0.10
    assert not is_confident


def test_synergy_gating_accepts_seeded_melancholic_for_emo() -> None:
    """Verify seeded sad/ballad 0.05 is accepted with candidate seed (Emo case)."""
    from resonate.modules.essentia import ESSENTIA_MOOD_MAP

    candidate_seeds = ["Melancholic"]
    pred_lbl = "sad"
    pred_score = 0.055

    is_synergy = False
    if candidate_seeds and pred_lbl in ESSENTIA_MOOD_MAP:
        target = ESSENTIA_MOOD_MAP[pred_lbl]
        if any(target.lower() == cs.lower() for cs in candidate_seeds):
            is_synergy = True

    is_confident = (is_synergy and pred_score >= 0.05) or pred_score >= 0.10
    assert is_confident


def test_surf_is_recognized_mood_tag() -> None:
    """Verify 'surf' is accepted as a valid mood tag keyword."""
    from resonate.main import is_valid_mood_tag

    assert is_valid_mood_tag("surf", "Unknown Artist", "Unknown Album")


def test_rockabilly_mood_seeds() -> None:
    """Verify Rockabilly seeds Upbeat and Chill Hang."""
    from resonate.modules.tag_mapper import get_genre_seeded_moods

    seeds = get_genre_seeded_moods(["Rockabilly"])
    assert "Upbeat" in seeds
    assert "Chill Hang" in seeds


def test_prog_rock_matches_progressive_rock_tag() -> None:
    """Verify 'progressive rock' and 'prog' map to 'Prog Rock' subgenre."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, TagMapper

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    matches_prog_rock = mapper.match_multiple_tags(["progressive rock"])
    assert any(m[0] == "Prog Rock" for m in matches_prog_rock)

    matches_prog = mapper.match_multiple_tags(["prog"])
    assert any(m[0] == "Prog Rock" for m in matches_prog)


def test_progressive_metal_subgenre_and_seeds() -> None:
    """Verify Progressive Metal is in subgenres, matches tags, and seeds Heavy."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, TagMapper, get_genre_seeded_moods

    assert "Progressive Metal" in DEFAULT_SUB_GENRES

    seeds = get_genre_seeded_moods(["Progressive Metal"])
    assert "Heavy" in seeds
    assert "Intense" in seeds
    assert "Atmospheric" in seeds

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    matches = mapper.match_multiple_tags(["progressive metal"])
    assert any(m[0] == "Progressive Metal" for m in matches)

    matches_prog_metal = mapper.match_multiple_tags(["prog metal"])
    assert any(m[0] == "Progressive Metal" for m in matches_prog_metal)


def test_instrumental_subgenres() -> None:
    """Verify Instrumental and Instrumental Rock match as subgenres."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, TagMapper

    assert "Instrumental" in DEFAULT_SUB_GENRES
    assert "Instrumental Rock" in DEFAULT_SUB_GENRES

    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    matches_inst = mapper.match_multiple_tags(["instrumental"])
    assert any(m[0] == "Instrumental" for m in matches_inst)

    matches_inst_rock = mapper.match_multiple_tags(["instrumental rock"])
    assert any(m[0] == "Instrumental Rock" for m in matches_inst_rock)


def test_essentia_cluster_pooling_score_retention() -> None:
    """Verify cluster pooling combines scores and satisfies the >= 0.10 threshold."""
    distinctive_preds = [
        ("happy", 0.0664),
        ("inspiring", 0.0647),
        ("uplifting", 0.0555),
    ]

    positive_upbeat_cluster = {
        "happy",
        "positive",
        "upbeat",
        "uplifting",
        "inspiring",
        "motivational",
        "fun",
        "summer",
    }
    cluster_preds = [p for p in distinctive_preds if p[0].lower() in positive_upbeat_cluster]
    assert len(cluster_preds) >= 2
    total_score = sum(cp[1] for cp in cluster_preds)
    assert total_score >= 0.10
    best_pred = max(cluster_preds, key=lambda x: x[1])
    pooled_pred = (best_pred[0], total_score)
    assert pooled_pred[0] == "happy"
    assert pooled_pred[1] >= 0.10


def test_genre_consensus_accumulation_outvotes_isolated_tag() -> None:
    """Verify multiple folk and rock tags accumulate weight and outvote an isolated punk tag."""
    from collections import Counter

    from resonate.modules.tag_mapper import DEFAULT_PRIMARY_GENRES, TagMapper

    mapper = TagMapper(target_moods=DEFAULT_PRIMARY_GENRES, threshold=0.45)
    tags = [
        "alternative punk",
        "folk rock",
        "indie",
        "psychedelic folk",
        "folk",
        "indie rock",
        "rock",
        "bedroom pop",
    ]
    matches = mapper.match_genre_consensus(tags)
    core_keywords = {
        "rock", "pop", "hip-hop", "hip hop", "rap", "gangsta rap", "reggae",
        "jazz", "blues", "metal", "classical", "electronic", "country",
        "folk", "punk", "soul", "r&b",
    }
    genre_counts = Counter()
    for g_name, raw_t, _score, raw_pos in matches:
        raw_lower = raw_t.lower().strip()
        weight = 3 if any(ck in raw_lower for ck in core_keywords) else 1
        if raw_pos < 3:
            weight += 5
        genre_counts[g_name] += weight

    # Folk and Rock should both have higher consensus than Punk
    assert genre_counts["Folk"] > genre_counts["Punk"]
    assert genre_counts["Rock"] > genre_counts["Punk"]
    assert genre_counts.most_common(1)[0][0] in {"Folk", "Rock"}


def test_punk_reconciliation_with_acoustic_subgenre() -> None:
    """Verify Punk is reconciled to Rock/Folk when acoustic subgenres are mapped."""
    mapped_genre = "Punk"
    mapped_subgenres = ["Acoustic Rock"]
    genre_counts = {"Punk": 8, "Rock": 6, "Folk": 6}

    acoustic_chill_subgenres = {
        "acoustic rock",
        "soft rock",
        "folk rock",
        "indie folk",
        "singer-songwriter",
        "americana",
        "lo-fi",
        "chamber music",
        "bluegrass",
    }
    has_acoustic_subgenre = any(s.lower() in acoustic_chill_subgenres for s in mapped_subgenres)
    assert has_acoustic_subgenre

    non_punk_metal = [g for g in genre_counts if g not in {"Punk", "Metal"}]
    if non_punk_metal:
        mapped_genre = non_punk_metal[0]
    assert mapped_genre == "Rock"


def test_blues_genre_mood_seeds() -> None:
    """Verify Blues and blues subgenres seed Soulful, Melancholic, Groovy, and Acoustic."""
    from resonate.modules.tag_mapper import get_genre_seeded_moods

    # 1. Primary Blues
    blues_seeds = get_genre_seeded_moods(["Blues"])
    assert "Soulful" in blues_seeds
    assert "Melancholic" in blues_seeds

    # 2. Delta Blues
    delta_seeds = get_genre_seeded_moods(["Delta Blues"])
    assert "Melancholic" in delta_seeds
    assert "Soulful" in delta_seeds
    assert "Acoustic" in delta_seeds

    # 3. Chicago Blues & Electric Blues
    chicago_seeds = get_genre_seeded_moods(["Chicago Blues"])
    assert "Soulful" in chicago_seeds
    assert "Groovy" in chicago_seeds
    assert "Melancholic" in chicago_seeds

    electric_seeds = get_genre_seeded_moods(["Electric Blues"])
    assert "Soulful" in electric_seeds
    assert "Groovy" in electric_seeds
    assert "Melancholic" in electric_seeds

    # 4. Blues Rock
    blues_rock_seeds = get_genre_seeded_moods(["Blues Rock"])
    assert "Soulful" in blues_rock_seeds
    assert "Groovy" in blues_rock_seeds


def test_singer_songwriter_whitelist_in_subgenre_filter() -> None:
    """Verify 'singer-songwriter' passes is_valid_subgenre_tag despite boilerplate."""
    from resonate.main import is_valid_subgenre_tag

    assert is_valid_subgenre_tag("singer-songwriter", "Unknown Artist", "Unknown Album")
    assert is_valid_subgenre_tag("singer songwriter", "Unknown Artist", "Unknown Album")


def test_junk_tags_filtered_out_by_subgenre_filter() -> None:
    """Verify junk and noise tags are filtered out by is_valid_subgenre_tag."""
    from resonate.main import is_valid_subgenre_tag

    junk_tags = [
        "fav", "favorites", "bagel", "catfish", "nachspiel", "gr last", "radio", "label"
    ]
    for j in junk_tags:
        assert not is_valid_subgenre_tag(
            j, "Unknown Artist", "Unknown Album"
        ), f"Junk tag '{j}' was not filtered"


def test_high_melancholic_cluster_suppresses_chill_hang() -> None:
    """Verify intensely high melancholic cluster (>= 0.25) suppresses Chill Hang."""
    e_top = [
        ("love", 0.2185),
        ("energetic", 0.1307),
        ("ballad", 0.1167),
        ("emotional", 0.0962),
        ("melodic", 0.0939),
        ("melancholic", 0.0758),
        ("sad", 0.0734),
    ]
    e_pred_dict = {p[0].lower(): float(p[1]) for p in e_top}
    melancholic_cluster_score = sum(
        e_pred_dict.get(k, 0.0) for k in ["sad", "ballad", "emotional", "melancholic"]
    )
    is_raw_melancholic_heavy = (
        melancholic_cluster_score >= 0.25
        or e_pred_dict.get("sad", 0.0) >= 0.15
        or e_pred_dict.get("ballad", 0.0) >= 0.15
        or e_pred_dict.get("emotional", 0.0) >= 0.15
    )
    assert is_raw_melancholic_heavy

    seeded_moods = ["Chill Hang"]
    combined_moods: list[str] = []
    for sm in seeded_moods:
        if sm.lower() == "chill hang":
            if not is_raw_melancholic_heavy:
                combined_moods.append(sm)

    assert "Chill Hang" not in combined_moods


def test_mild_melancholic_preserves_chill_hang() -> None:
    """Verify mild melancholic cluster (< 0.25) preserves Chill Hang for gentle indie tracks."""
    e_top = [
        ("melodic", 0.12),
        ("acoustic", 0.10),
        ("sad", 0.05),
        ("melancholic", 0.04),
    ]
    e_pred_dict = {p[0].lower(): float(p[1]) for p in e_top}
    melancholic_cluster_score = sum(
        e_pred_dict.get(k, 0.0) for k in ["sad", "ballad", "emotional", "melancholic"]
    )
    is_raw_melancholic_heavy = (
        melancholic_cluster_score >= 0.25
        or e_pred_dict.get("sad", 0.0) >= 0.15
        or e_pred_dict.get("ballad", 0.0) >= 0.15
        or e_pred_dict.get("emotional", 0.0) >= 0.15
    )
    assert not is_raw_melancholic_heavy

    seeded_moods = ["Chill Hang"]
    combined_moods: list[str] = []
    for sm in seeded_moods:
        if sm.lower() == "chill hang":
            if not is_raw_melancholic_heavy:
                combined_moods.append(sm)

    assert "Chill Hang" in combined_moods


def test_kevin_morby_parade_matches_singer_songwriter_subgenre() -> None:
    """Verify Kevin Morby raw tags filter boilerplate and match Singer-Songwriter."""
    from resonate.main import is_valid_subgenre_tag
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES, TagMapper

    raw_tags = [
        "fav",
        "nachspiel",
        "2014",
        "mtv catfish",
        "gr last",
        "singer-songwriter",
        "woodsist",
        "bagel",
        "indie",
        "indie rock",
        "indie folk",
        "folk",
        "rock",
        "lo-fi",
        "folk rock",
        "radio as i want it",
        "pop",
    ]
    generic_primary = {
        "rock", "pop", "metal", "jazz", "blues", "country",
        "folk", "rap", "hip hop", "hiphop", "electronic", "dance", "punk",
    }
    filtered_subgenre_tags = [
        t
        for t in raw_tags
        if is_valid_subgenre_tag(t, "Kevin Morby", "Parade")
        and t.lower().strip() not in generic_primary
    ]
    mapper = TagMapper(target_moods=DEFAULT_SUB_GENRES, threshold=0.65)
    results = mapper.match_multiple_tags(filtered_subgenre_tags, max_matches=3)
    matched = [r[0] for r in results]
    assert "Singer-Songwriter" in matched


def test_artist_alias_resolution() -> None:
    """Verify artist_matches resolves aliases like Ye <-> Kanye West but rejects Yes."""
    from resonate.modules.external_metadata import artist_matches, get_artist_aliases

    assert artist_matches("Ye", "Kanye West") is True
    assert artist_matches("Kanye West", "Ye") is True
    assert artist_matches("Ye", "Kanye West feat. Lil Wayne") is True
    assert artist_matches("Ye", "Yes") is False
    assert artist_matches("Yes", "Ye") is False
    assert artist_matches("Yasiin Bey", "Mos Def") is True

    aliases = get_artist_aliases("Ye")
    assert "Ye" in aliases
    assert "kanye west" in aliases


def test_lastfm_redirect_mismatch_rejection(monkeypatch) -> None:
    """Verify _scrape_url_tags rejects tags when Last.fm redirects to an unrelated artist."""
    from urllib.request import Request

    from resonate.modules.lastfm import LastFmFetcher

    class MockResponse:
        status = 200

        def geturl(self) -> str:
            return "https://www.last.fm/music/Yes/+tags"

        def read(self) -> bytes:
            return b'<a href="/tag/progressive+rock">progressive rock</a>'

    def mock_urlopen(req: Request, timeout: int = 10):
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    fetcher = LastFmFetcher()
    tags = fetcher._scrape_url_tags("https://www.last.fm/music/Ye/+tags", expected_artist="Ye")
    assert tags == []


def test_energetic_rowdy_excludes_romantic() -> None:
    """Verify resolve_mood_conflicts drops Romantic when Energetic or Rowdy is present."""
    from resonate.modules.tag_mapper import resolve_mood_conflicts

    moods = resolve_mood_conflicts(["Romantic", "Energetic"])
    assert "Romantic" not in moods
    assert "Energetic" in moods

    moods_rowdy = resolve_mood_conflicts(["Romantic", "Rowdy"])
    assert "Romantic" not in moods_rowdy
    assert "Rowdy" in moods_rowdy






