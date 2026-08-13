"""Tag mapper module using SentenceTransformers for mood embedding matching."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MOODS = [
    "chill",
    "energetic",
    "melancholic",
    "upbeat",
    "dark",
    "happy",
    "relaxed",
    "aggressive",
]


class TagMapper:
    """Map raw music tags to target moods via vector embeddings."""

    def __init__(
        self,
        target_moods: list[str] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        model: Any | None = None,
        threshold: float = 0.45,
    ) -> None:
        """Initialize TagMapper with target moods and SentenceTransformer model."""
        self.target_moods = target_moods if target_moods is not None else DEFAULT_TARGET_MOODS
        self.model_name = model_name
        self._model = model
        self.threshold = threshold
        self.target_embeddings: Any = None

        if self._model is not None and self.target_moods:
            self._init_embeddings()

    def _get_model(self) -> Any:
        """Lazy load or return existing SentenceTransformer model."""
        if self._model is None:
            try:
                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()

                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as err:
                logger.warning(
                    f"Failed to load SentenceTransformer model '{self.model_name}': {err}"
                )
                self._model = None

        if self._model is not None and self.target_embeddings is None and self.target_moods:
            self._init_embeddings()
        return self._model

    def _encode(self, model: Any, texts: list[str]) -> Any:
        """Encode texts using model, handling convert_to_tensor parameter gracefully."""
        try:
            return model.encode(texts, convert_to_tensor=False)
        except TypeError:
            return model.encode(texts)

    def _init_embeddings(self) -> None:
        """Pre-compute embeddings for target moods."""
        if self._model is not None and self.target_moods:
            try:
                self.target_embeddings = self._encode(self._model, self.target_moods)
            except Exception as err:
                logger.warning(f"Failed to pre-compute embeddings for target moods: {err}")
                self.target_embeddings = None

    def match_tags(
        self, raw_tags: list[str], threshold: float | None = None
    ) -> tuple[str | None, str | None, str | None, float]:
        """Match raw tags against target moods using cosine similarity."""
        cutoff = threshold if threshold is not None else self.threshold
        if not raw_tags or not self.target_moods:
            return (None, None, None, 0.0)

        model = self._get_model()
        if model is None or self.target_embeddings is None:
            return (None, None, None, 0.0)

        try:
            raw_embeddings = self._encode(model, raw_tags)
        except Exception as err:
            logger.warning(f"Failed to encode raw tags: {err}")
            return (None, None, None, 0.0)

        raw_arr = np.asarray(raw_embeddings, dtype=np.float32)
        target_arr = np.asarray(self.target_embeddings, dtype=np.float32)

        raw_norms = np.linalg.norm(raw_arr, axis=1, keepdims=True)
        raw_norms = np.maximum(raw_norms, 1e-9)
        target_norms = np.linalg.norm(target_arr, axis=1, keepdims=True)
        target_norms = np.maximum(target_norms, 1e-9)

        raw_norm = raw_arr / raw_norms
        target_norm = target_arr / target_norms

        sim_matrix = np.dot(raw_norm, target_norm.T)

        max_flat_idx = int(np.argmax(sim_matrix))
        best_tag_idx, best_mood_idx = np.unravel_index(max_flat_idx, sim_matrix.shape)
        max_score = float(sim_matrix[best_tag_idx, best_mood_idx])
        best_mood = self.target_moods[best_mood_idx]
        best_raw_tag = raw_tags[best_tag_idx]

        if max_score >= cutoff:
            return (best_mood, best_mood, best_raw_tag, max_score)
        return (None, best_mood, best_raw_tag, max_score)

    def map_tags(
        self, raw_tags: list[str], threshold: float | None = None
    ) -> tuple[str | None, str | None, str | None, float]:
        """Alias for match_tags to map raw tags to target moods."""
        return self.match_tags(raw_tags, threshold=threshold)

    def match_multiple_tags(
        self, raw_tags: list[str], threshold: float | None = None
    ) -> list[tuple[str, str, float]]:
        """Match raw tags against all target tags and return all that exceed the threshold."""
        cutoff = threshold if threshold is not None else self.threshold
        if not raw_tags or not self.target_moods:
            return []

        model = self._get_model()
        if model is None or self.target_embeddings is None:
            return []

        try:
            raw_embeddings = self._encode(model, raw_tags)
        except Exception as err:
            logger.warning(f"Failed to encode raw tags: {err}")
            return []

        raw_arr = np.asarray(raw_embeddings, dtype=np.float32)
        target_arr = np.asarray(self.target_embeddings, dtype=np.float32)

        raw_norms = np.linalg.norm(raw_arr, axis=1, keepdims=True)
        raw_norms = np.maximum(raw_norms, 1e-9)
        target_norms = np.linalg.norm(target_arr, axis=1, keepdims=True)
        target_norms = np.maximum(target_norms, 1e-9)

        raw_norm = raw_arr / raw_norms
        target_norm = target_arr / target_norms

        sim_matrix = np.dot(raw_norm, target_norm.T)

        matched_results = []
        for col_idx, target_tag in enumerate(self.target_moods):
            row_idx = int(np.argmax(sim_matrix[:, col_idx]))
            score = float(sim_matrix[row_idx, col_idx])
            if score >= cutoff:
                matched_results.append((target_tag, raw_tags[row_idx], score))

        matched_results.sort(key=lambda x: x[2], reverse=True)
        return matched_results


DEFAULT_PRIMARY_GENRES = [
    "Rock",
    "Pop",
    "Indie",
    "Hip-Hop",
    "Electronic",
    "Jazz",
    "Blues",
    "Country",
    "Folk",
    "R&B",
    "Metal",
    "Punk",
    "Reggae",
    "Latin",
]

DEFAULT_SUB_GENRES = [
    "Americana",
    "Alternative Rock",
    "Hard Rock",
    "Grunge",
    "Indie Rock",
    "Synthpop",
    "Downtempo",
    "Lo-Fi",
    "Motown",
    "Shoegaze",
    "Garage Rock",
    "Post-Rock",
    "Classic Rock",
    "Acoustic Rock",
    "House",
    "Techno",
]

DEFAULT_MOOD_TAGS = [
    "Party",
    "Chill Hang",
    "Energetic",
    "Groovy",
    "Acoustic",
    "Electronic",
    "Melancholic",
    "Lively",
    "Relaxed",
]
