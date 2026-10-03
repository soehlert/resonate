"""Pytest unit tests for TagMapper tag mapping logic and threshold verification."""

from unittest.mock import MagicMock

from resonate.modules.tag_mapper import TagMapper


def test_empty_raw_tags_return_none() -> None:
    """Verify empty raw tags list returns (None, None, None, 0.0)."""
    mapper = TagMapper(threshold=0.45)
    mood, _, _, score = mapper.map_tags([])
    assert mood is None
    assert score == 0.0


def test_matching_tag_above_threshold() -> None:
    """Verify matching tag with score >= threshold returns mapped mood."""
    mapper = TagMapper(target_moods=["chill", "energetic"], threshold=0.45)
    mood, _, _, score = mapper.map_tags(["chillout", "ambient"])
    assert mood == "chill"
    assert score >= 0.45


def test_low_score_below_threshold() -> None:
    """Verify low similarity score below threshold returns None mood and score."""
    mapper = TagMapper(threshold=0.95)
    mood, _, _, score = mapper.map_tags(["randomtagxyz"])
    assert mood is None
    assert score < 0.95


def test_mocked_sentence_transformer_embedding() -> None:
    """Verify TagMapper logic with mocked SentenceTransformer embeddings model."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, *args, **kwargs: [
        [1.0, 0.0] if "chill" in t else [0.0, 1.0] for t in texts
    ]

    mapper = TagMapper(
        target_moods=["chill", "energetic"],
        threshold=0.45,
        model=mock_model,
    )
    mood, _, _, score = mapper.map_tags(["chill"])
    assert mood == "chill"
    assert score >= 0.45


def test_subgenre_consensus_bypasses_sentence_transformer() -> None:
    """Verify subgenre matching does not invoke SentenceTransformer model.encode."""
    from resonate.modules.tag_mapper import DEFAULT_SUB_GENRES

    mock_model = MagicMock()
    mapper = TagMapper(
        target_moods=DEFAULT_SUB_GENRES,
        threshold=0.50,
        model=mock_model,
    )
    # Reset call count from any initialization
    mock_model.encode.reset_mock()

    matches = mapper.match_subgenre_consensus(["hardcore", "punk rock"])
    # Should find subgenres via dictionary/stems
    assert any(m[0] == "Punk Rock" for m in matches)
    # SentenceTransformer encode must NEVER be called during subgenre matching
    assert mock_model.encode.call_count == 0


def test_primary_genre_match_multiple_bypasses_sentence_transformer() -> None:
    """Verify primary genre matching in match_multiple_tags does not call model.encode."""
    from resonate.modules.tag_mapper import DEFAULT_PRIMARY_GENRES

    mock_model = MagicMock()
    mapper = TagMapper(
        target_moods=DEFAULT_PRIMARY_GENRES,
        threshold=0.50,
        model=mock_model,
    )
    mock_model.encode.reset_mock()

    matches = mapper.match_multiple_tags(["punk rock", "hardcore"])
    assert any(m[0] == "Punk" for m in matches)
    assert mock_model.encode.call_count == 0


def test_match_multiple_tags_rank_decay_toggle() -> None:
    """Verify apply_rank_decay=False bypasses top-5 candidate gating and rank factor penalty."""
    mapper = TagMapper(target_moods=["chill"], threshold=0.50)
    tags = ["tag0", "tag1", "tag2", "tag3", "tag4", "chill"]

    decayed = mapper.match_multiple_tags(tags, apply_rank_decay=True)
    no_decay = mapper.match_multiple_tags(tags, apply_rank_decay=False)

    assert len(decayed) == 1
    assert len(no_decay) == 1
    # Under rank decay, exact match was gated at index 5 and fell through to decayed similarity
    assert decayed[0][2] < 1.0
    # Without rank decay, exact match is immediately accepted with full score 1.0
    assert no_decay[0][2] == 1.0


def test_forward_vector_mapping_prevents_opposite_mood_fanout() -> None:
    """Verify single tag maps strictly to its best match and does not pull conflicting opposites."""
    from resonate.engine.mood_rules import DEFAULT_MOOD_TAGS

    mapper = TagMapper(target_moods=DEFAULT_MOOD_TAGS, threshold=0.50)
    # Cheerful must map strictly to Upbeat, never fan out into Melancholic or Relaxed
    cheerful_matches = mapper.match_multiple_tags(["cheerful"])
    matched_moods = [m[0] for m in cheerful_matches]
    assert matched_moods == ["Upbeat"]
    assert "Melancholic" not in matched_moods
    assert "Relaxed" not in matched_moods

    # Lively must map strictly to Lively, never fan out into Relaxed or Calm
    lively_matches = mapper.match_multiple_tags(["lively"])
    lively_moods = [m[0] for m in lively_matches]
    assert lively_moods == ["Lively"]
    assert "Relaxed" not in lively_moods
    assert "Calm" not in lively_moods


def test_all_278_allmusic_tags_recognized_and_mapped() -> None:
    """Verify all 278 AllMusic mood tags are recognized and map to canonical target moods."""
    from resonate.config import load_data_file
    from resonate.engine.mood_rules import DEFAULT_TARGET_MOODS, MOOD_ALIAS_MAP, is_valid_mood_tag

    target_data = load_data_file("target_moods.yaml")
    aliases_by_canonical = target_data.get("aliases", {})
    all_allmusic_tags: list[str] = []
    for alias_list in aliases_by_canonical.values():
        all_allmusic_tags.extend(alias_list)

    assert len(all_allmusic_tags) == 278
    assert len(set(all_allmusic_tags)) == 278

    canonical_set = {m.lower() for m in DEFAULT_TARGET_MOODS}
    for tag in all_allmusic_tags:
        assert is_valid_mood_tag(tag, "Artist", "Album"), f"Tag '{tag}' should be valid mood"
        mapped_canonical = MOOD_ALIAS_MAP.get(tag.lower().strip())
        assert mapped_canonical is not None, f"Tag '{tag}' should be in MOOD_ALIAS_MAP"
        assert mapped_canonical.lower() in canonical_set


def test_multi_tag_allmusic_alias_resolution_and_conflict_pruning() -> None:
    """Verify multiple raw AllMusic tags map to canonical moods and conflict rules prune opposing tags."""
    from resonate.engine.mood_rules import synthesize_track_moods

    raw_mood_tags = [
        "Swaggering",
        "Confrontational",
        "Celebratory",
        "Rebellious",
        "Reckless",
        "Freewheeling",
        "Raucous",
        "Fiery",
        "Outrageous",
        "Brash",
    ]
    mapper = TagMapper()
    matches = mapper.match_multiple_tags(raw_mood_tags, apply_rank_decay=False)
    mapped_target_moods = {m[0] for m in matches}

    assert "Rowdy" in mapped_target_moods
    assert "Aggressive" in mapped_target_moods
    assert "Party" in mapped_target_moods
    assert "Chill Hang" in mapped_target_moods

    result = synthesize_track_moods(
        text_moods=matches,
        seeded_moods=["Soulful", "Groovy"],
        essentia_moods=[],
        essentia_top=[],
        detected_bpm=99,
        lyrics_analysis=None,
        primary_genre="Hip-Hop",
        subgenres=["East Coast Hip Hop", "Rap"],
        raw_tags=["hip hop", "rap"],
        max_moods=5,
    )
    # Conflict rule: [Heavy, Aggressive, Rowdy, Ballad] -> drop: [Chill Hang]
    assert "Chill Hang" not in result
    assert "Soulful" not in result
    assert "Rowdy" in result
    assert "Aggressive" in result
    assert "Party" in result





