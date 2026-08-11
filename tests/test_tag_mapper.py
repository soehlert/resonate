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
    mapper = TagMapper(threshold=0.45)
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
        [1.0, 0.0] if t == "chill" else [0.0, 1.0] for t in texts
    ]

    mapper = TagMapper(
        target_moods=["chill", "energetic"],
        threshold=0.45,
        model=mock_model,
    )
    mood, _, _, score = mapper.map_tags(["chill"])
    assert mood == "chill"
    assert score >= 0.45
