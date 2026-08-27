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
            if "Rock" in text:
                embeddings.append([1.0, 0.0])
            elif "Pop" in text:
                embeddings.append([0.0, 1.0])
            elif "grunge" in text:
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


@patch("sentence_transformers.SentenceTransformer")
def test_homonym_band_disambiguation_logic(mock_transformer_cls):
    """Verify that verified album/track tags map correctly without pooled artist tags."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model

    def mock_encode(texts, **kwargs):
        embeddings = []
        for text in texts:
            if "Electronic" in text or "ambient" in text:
                embeddings.append([1.0, 0.0])
            elif "Metal" in text or "doom metal" in text:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([0.1, 0.1])
        return np.array(embeddings)

    mock_model.encode.side_effect = mock_encode

    mapper = TagMapper(target_moods=["Electronic", "Metal"], threshold=0.5)

    # Simulated verified release tags for Sleepwalkers (electronic) vs pooled artist tags
    verified_tags = ["ambient", "downtempo"]
    pooled_artist_tags = ["death doom metal", "doom metal", "hardcore"]

    # 1. Matching verified tags directly yields Electronic
    mapped_genre, _, _, score = mapper.match_tags(verified_tags)
    assert mapped_genre == "Electronic"
    assert score > 0.8

    # 2. In contrast, pooled tags would have yielded Metal
    pooled_genre, _, _, pooled_score = mapper.match_tags(pooled_artist_tags)
    assert pooled_genre == "Metal"
    assert pooled_score > 0.8

