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

