"""Personalized mood classification engine based on anchor track EffNet embeddings."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join("models", "personalized_mood_heads.json")


class PersonalizedMoodTuner:
    """Manages anchor-based personalized mood centroids and inference scoring."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """Initialize tuner with target model JSON file path."""
        self.model_path = model_path
        self.mood_heads: dict[str, dict[str, Any]] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._thresholds: dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        """Return True if at least one mood centroid is calibrated and loaded."""
        return len(self._centroids) > 0

    @property
    def tuned_moods(self) -> list[str]:
        """Return list of canonical mood names currently tuned."""
        return sorted(self._centroids.keys())

    def fit(self, mood_embeddings: dict[str, list[np.ndarray]]) -> None:
        """Compute unit centroids and adaptive thresholds from anchor embeddings.

        Args:
            mood_embeddings: Mapping from canonical mood name to a list of
                frame-level (num_frames, 1280) or pre-pooled (1280,) numpy arrays.
        """
        self.mood_heads.clear()
        self._centroids.clear()
        self._thresholds.clear()

        raw_centroids: dict[str, np.ndarray] = {}
        intra_similarities: dict[str, list[float]] = {}

        for mood, vectors in mood_embeddings.items():
            if not vectors:
                continue

            # Pool frame-level embeddings to 1D vectors if needed
            pooled_list: list[np.ndarray] = []
            for vec in vectors:
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=0)
                if arr.size == 0:
                    continue
                # Normalize individual track vector
                norm = np.linalg.norm(arr)
                if norm > 1e-9:
                    pooled_list.append(arr / norm)

            if not pooled_list:
                continue

            # Compute mean centroid
            stacked = np.stack(pooled_list, axis=0)
            mean_vec = np.mean(stacked, axis=0)
            centroid_norm = np.linalg.norm(mean_vec)
            if centroid_norm < 1e-9:
                continue
            unit_centroid = mean_vec / centroid_norm

            raw_centroids[mood] = unit_centroid

            # Measure intra-cluster cosine similarities: s_i = v_i . c_m
            sims = [float(np.dot(v, unit_centroid)) for v in pooled_list]
            intra_similarities[mood] = sims

        # Second pass: calculate calibrated thresholds
        for mood, unit_centroid in raw_centroids.items():
            sims = intra_similarities[mood]
            mean_sim = float(np.mean(sims)) if sims else 0.80
            std_sim = float(np.std(sims)) if len(sims) > 1 else 0.04

            # Baseline threshold: captures core anchor cluster without outlier drag
            if len(sims) >= 3:
                base_threshold = mean_sim - 1.25 * std_sim
            else:
                base_threshold = mean_sim - 0.06

            calibrated_threshold = max(0.75, min(0.88, base_threshold))

            # Cross-mood contrastive margin:
            # Ensure threshold is safely above the similarity to any other centroid
            for other_mood, other_centroid in raw_centroids.items():
                if other_mood == mood:
                    continue
                inter_sim = float(np.dot(unit_centroid, other_centroid))
                if inter_sim >= calibrated_threshold:
                    # Nudge threshold to be slightly above the mid-point between clusters
                    calibrated_threshold = max(
                        calibrated_threshold, min(0.92, (inter_sim + mean_sim) / 2.0)
                    )

            self._centroids[mood] = unit_centroid
            self._thresholds[mood] = float(round(calibrated_threshold, 3))
            self.mood_heads[mood] = {
                "centroid": unit_centroid.tolist(),
                "threshold": float(round(calibrated_threshold, 3)),
                "track_count": len(sims),
                "coherence": float(round(mean_sim, 3)),
            }

    def predict(self, track_embedding: np.ndarray, top_k: int = 2) -> list[tuple[str, float]]:
        """Score a track's EffNet embedding against calibrated mood centroids.

        Args:
            track_embedding: Array of shape (1280,) or (num_frames, 1280).
            top_k: Maximum number of matching moods to return.

        Returns:
            List of (canonical_mood, score) tuples exceeding their calibrated threshold.
        """
        if not self.is_trained or track_embedding is None:
            return []

        arr = np.asarray(track_embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)

        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            return []
        unit_track = arr / norm

        matches: list[tuple[str, float]] = []
        for mood, centroid in self._centroids.items():
            sim = float(np.dot(unit_track, centroid))
            threshold = self._thresholds.get(mood, 0.75)
            if sim >= threshold:
                matches.append((mood, round(sim, 3)))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def score_all(self, track_embedding: np.ndarray) -> list[tuple[str, float, float, bool]]:
        """Score a track's EffNet embedding against all calibrated heads with diagnostics.

        Returns:
            List of (canonical_mood, similarity, threshold, is_match) tuples
            sorted by score descending.
        """
        if not self.is_trained or track_embedding is None:
            return []

        arr = np.asarray(track_embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)

        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            return []
        unit_track = arr / norm

        results: list[tuple[str, float, float, bool]] = []
        for mood, centroid in self._centroids.items():
            sim = float(np.dot(unit_track, centroid))
            threshold = self._thresholds.get(mood, 0.75)
            is_match = sim >= threshold
            results.append((mood, round(sim, 3), round(threshold, 3), is_match))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save_model(self, path: str | None = None) -> None:
        """Save tuned centroids and thresholds to human-readable JSON."""
        target_path = path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        payload = {
            "version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "mood_count": len(self.mood_heads),
            "moods": self.mood_heads,
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved {len(self.mood_heads)} personalized mood heads to '{target_path}'.")

    def load_model(self, path: str | None = None) -> bool:
        """Load tuned centroids and thresholds from JSON file."""
        target_path = path or self.model_path
        if not os.path.exists(target_path) or os.path.getsize(target_path) < 10:
            return False

        try:
            with open(target_path, encoding="utf-8") as f:
                data = json.load(f)

            moods_data = data.get("moods", {})
            self.mood_heads = moods_data
            self._centroids.clear()
            self._thresholds.clear()

            for mood, meta in moods_data.items():
                centroid_list = meta.get("centroid", [])
                threshold = meta.get("threshold", 0.70)
                if centroid_list:
                    vec = np.asarray(centroid_list, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        self._centroids[mood] = vec / norm
                        self._thresholds[mood] = float(threshold)

            return self.is_trained
        except Exception as err:
            logger.warning(f"Failed to load personalized mood model from '{target_path}': {err}")
            return False
