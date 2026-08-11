"""Unit tests for TagMapper mapping logic with mocked SentenceTransformers embeddings."""

from unittest.mock import MagicMock, patch

import numpy as np

from resonate.modules.tag_mapper import (
    TagMapper,
)


@patch("sentence_transformers.SentenceTransformer")
def test_tag_mapper_single_mapping(mock_transformer_cls):
    """Verify that TagMapper maps a raw tag to the closest target tag."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model

    # Let's mock encode to return pre-defined embeddings.
    # The order of targets:
    # DEFAULT_PRIMARY_GENRES = ["Rock", "Pop", ...]
    # We can design the returned vectors to have high cosine similarity for the target we want.
    # Rock is [1.0, 0.0]
    # Pop is [0.0, 1.0]
    # Raw tag "grunge" is encoded as [0.98, 0.2] (close to Rock)

    def mock_encode(texts, **kwargs):
        embeddings = []
        for text in texts:
            if text == "Rock":
                embeddings.append([1.0, 0.0])
            elif text == "Pop":
                embeddings.append([0.0, 1.0])
            elif text == "grunge":
                embeddings.append([0.98, 0.2])  # Cosine similarity to Rock is high
            else:
                embeddings.append([0.1, 0.1])
        return np.array(embeddings)

    mock_model.encode.side_effect = mock_encode

    # Initialize TagMapper with Rock and Pop
    mapper = TagMapper(target_moods=["Rock", "Pop"], threshold=0.5)

    # Verify mapping
    mapped_mood, best_mood, best_raw_tag, score = mapper.match_tags(["grunge"])

    assert mapped_mood == "Rock"
    assert best_mood == "Rock"
    assert best_raw_tag == "grunge"
    assert score > 0.9


@patch("sentence_transformers.SentenceTransformer")
def test_tag_mapper_multiple_mappings(mock_transformer_cls):
    """Verify that TagMapper's match_multiple_tags maps multiple raw tags above threshold."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model

    def mock_encode(texts, **kwargs):
        embeddings = []
        for text in texts:
            if text == "Indie Rock":
                embeddings.append([1.0, 0.0])
            elif text == "Synthpop":
                embeddings.append([0.0, 1.0])
            elif text == "indie":
                embeddings.append([0.95, 0.1])  # Close to Indie Rock
            elif text == "synth":
                embeddings.append([0.1, 0.95])  # Close to Synthpop
            else:
                embeddings.append([0.1, 0.1])
        return np.array(embeddings)

    mock_model.encode.side_effect = mock_encode

    mapper = TagMapper(target_moods=["Indie Rock", "Synthpop"], threshold=0.5)

    results = mapper.match_multiple_tags(["indie", "synth"])

    assert len(results) == 2
    target_tags = [r[0] for r in results]
    assert "Indie Rock" in target_tags
    assert "Synthpop" in target_tags
