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
    """Manages anchor-based personalized mood scoring using adaptive k-NN."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """Initialize tuner with target model JSON file path."""
        self.model_path = model_path
        self.mood_heads: dict[str, dict[str, Any]] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._anchors: dict[str, np.ndarray] = {}
        self._anchor_names: dict[str, list[str]] = {}
        self._thresholds: dict[str, float] = {}
        self._k_values: dict[str, int] = {}

    @property
    def is_trained(self) -> bool:
        """Return True if at least one mood centroid is calibrated and loaded."""
        return len(self._centroids) > 0

    @property
    def tuned_moods(self) -> list[str]:
        """Return list of canonical mood names currently tuned."""
        return sorted(self._centroids.keys())

    def get_k(self, mood: str) -> int:
        """Return the k-nearest-neighbors value for a given mood."""
        return self._k_values.get(mood, 1)

    @staticmethod
    def _compute_k(n_tracks: int, target_k: int = 3) -> int:
        """Determine adaptive k based on anchor track count (default: 3)."""
        return min(target_k, max(1, n_tracks))

    def fit(
        self,
        mood_embeddings: dict[str, list[np.ndarray]],
        target_k: int = 3,
        track_names: dict[str, list[str]] | None = None,
    ) -> None:
        """Compute unit centroids, store anchor matrices, and calibrate adaptive k-NN thresholds.

        Args:
            mood_embeddings: Mapping from canonical mood name to a list of
                frame-level (num_frames, 1280) or pre-pooled (1280,) numpy arrays.
            target_k: Consensus neighborhood size (default: 3).
            track_names: Optional mapping from mood name to anchor track display titles.
        """
        self.mood_heads.clear()
        self._centroids.clear()
        self._anchors.clear()
        self._anchor_names.clear()
        self._thresholds.clear()
        self._k_values.clear()

        raw_centroids: dict[str, np.ndarray] = {}
        raw_anchors: dict[str, np.ndarray] = {}
        raw_k: dict[str, int] = {}
        raw_names: dict[str, list[str]] = {}
        loo_metrics: dict[str, tuple[float, float]] = {}

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

            n_tracks = len(pooled_list)
            anchors_arr = np.stack(pooled_list, axis=0)
            raw_anchors[mood] = anchors_arr

            # Mean centroid
            mean_vec = np.mean(anchors_arr, axis=0)
            centroid_norm = np.linalg.norm(mean_vec)
            if centroid_norm < 1e-9:
                continue
            unit_centroid = mean_vec / centroid_norm
            raw_centroids[mood] = unit_centroid

            # Adaptive k
            k = self._compute_k(n_tracks, target_k=target_k)
            raw_k[mood] = k
            if track_names and mood in track_names:
                raw_names[mood] = track_names[mood][:n_tracks]
            else:
                raw_names[mood] = [f"Anchor {i+1}" for i in range(n_tracks)]

            # Leave-One-Out (LOO) cross-validation for coherence and threshold calibration
            if n_tracks == 1:
                loo_scores = [1.0]
            else:
                k_loo = min(k, n_tracks - 1)
                loo_scores = []
                # Pairwise cosine similarity matrix: (N, N)
                sim_matrix = np.dot(anchors_arr, anchors_arr.T)
                for i in range(n_tracks):
                    other_sims = np.delete(sim_matrix[i], i)
                    top_k_sims = np.sort(other_sims)[-k_loo:]
                    loo_scores.append(float(np.mean(top_k_sims)))

            mean_sim = float(np.mean(loo_scores))
            std_sim = float(np.std(loo_scores)) if len(loo_scores) > 1 else 0.04
            loo_metrics[mood] = (mean_sim, std_sim)

        # Second pass: calculate calibrated thresholds with contrastive margins
        for mood, unit_centroid in raw_centroids.items():
            mean_sim, std_sim = loo_metrics[mood]
            anchors_arr = raw_anchors[mood]
            k = raw_k[mood]
            n_tracks = len(anchors_arr)

            # Baseline threshold: captures core anchor clusters
            if n_tracks >= 3:
                base_threshold = mean_sim - 1.25 * std_sim
            else:
                base_threshold = mean_sim - 0.06

            calibrated_threshold = max(0.75, min(0.88, base_threshold))

            # Cross-mood contrastive margin:
            # Ensure threshold is safely above the similarity to any other mood centroid
            for other_mood, other_centroid in raw_centroids.items():
                if other_mood == mood:
                    continue
                inter_sim = float(np.dot(unit_centroid, other_centroid))
                if inter_sim >= calibrated_threshold:
                    calibrated_threshold = max(
                        calibrated_threshold, min(0.92, (inter_sim + mean_sim) / 2.0)
                    )

            self._centroids[mood] = unit_centroid
            self._anchors[mood] = anchors_arr
            self._anchor_names[mood] = raw_names.get(mood, [])
            self._thresholds[mood] = float(round(calibrated_threshold, 3))
            self._k_values[mood] = k

            self.mood_heads[mood] = {
                "centroid": unit_centroid.tolist(),
                "anchors": anchors_arr.tolist(),
                "anchor_names": raw_names.get(mood, []),
                "threshold": float(round(calibrated_threshold, 3)),
                "k": k,
                "track_count": n_tracks,
                "coherence": float(round(mean_sim, 3)),
            }

    def _score_track_for_mood(self, unit_track: np.ndarray, mood: str) -> float:
        """Compute k-NN similarity score of a unit track vector against a mood's anchors."""
        anchors = self._anchors.get(mood)
        if anchors is not None and len(anchors) > 0:
            sims = np.dot(anchors, unit_track)
            k = min(self._k_values.get(mood, 1), len(sims))
            top_k_sims = np.sort(sims)[-k:]
            return float(np.mean(top_k_sims))

        # Fallback to single centroid if anchors are not available (e.g. legacy model)
        centroid = self._centroids.get(mood)
        if centroid is not None:
            return float(np.dot(unit_track, centroid))
        return 0.0

    def predict(self, track_embedding: np.ndarray, top_k: int = 2) -> list[tuple[str, float]]:
        """Score a track's EffNet embedding against calibrated mood heads.

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
        for mood in self._centroids:
            score = self._score_track_for_mood(unit_track, mood)
            threshold = self._thresholds.get(mood, 0.75)
            if score >= threshold:
                matches.append((mood, round(score, 3)))

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
        for mood in self._centroids:
            score = self._score_track_for_mood(unit_track, mood)
            threshold = self._thresholds.get(mood, 0.75)
            is_match = score >= threshold
            results.append((mood, round(score, 3), round(threshold, 3), is_match))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_top_neighbors(
        self,
        track_embedding: np.ndarray,
        mood: str,
        n_neighbors: int = 3,
    ) -> list[tuple[str, float]]:
        """Return the top-n nearest anchor track labels and cosine similarities for a mood."""
        if track_embedding is None or mood not in self._anchors:
            return []

        arr = np.asarray(track_embedding, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            return []
        unit_track = arr / norm

        anchors = self._anchors[mood]
        sims = np.dot(anchors, unit_track)
        names = self._anchor_names.get(mood, [])

        top_indices = np.argsort(sims)[::-1][:n_neighbors]
        results: list[tuple[str, float]] = []
        for idx in top_indices:
            label = names[idx] if idx < len(names) else f"Anchor {idx + 1}"
            results.append((label, float(round(float(sims[idx]), 3))))
        return results

    def save_model(self, path: str | None = None) -> None:
        """Save tuned centroids, anchors, and thresholds to human-readable JSON."""
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
        """Load tuned centroids, anchors, and thresholds from JSON file."""
        target_path = path or self.model_path
        if not os.path.exists(target_path) or os.path.getsize(target_path) < 10:
            return False

        try:
            with open(target_path, encoding="utf-8") as f:
                data = json.load(f)

            moods_data = data.get("moods", {})
            self.mood_heads = moods_data
            self._centroids.clear()
            self._anchors.clear()
            self._anchor_names.clear()
            self._thresholds.clear()
            self._k_values.clear()

            for mood, meta in moods_data.items():
                threshold = meta.get("threshold", 0.75)
                self._thresholds[mood] = float(threshold)

                # Centroid
                centroid_list = meta.get("centroid", [])
                if centroid_list:
                    vec = np.asarray(centroid_list, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        self._centroids[mood] = vec / norm

                # Anchors
                anchors_list = meta.get("anchors", [])
                if anchors_list:
                    arr = np.asarray(anchors_list, dtype=np.float32)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-9, 1.0, norms)
                    self._anchors[mood] = arr / norms
                    k_val = meta.get("k", self._compute_k(len(arr)))
                    self._k_values[mood] = int(k_val)
                    self._anchor_names[mood] = meta.get("anchor_names", [])
                elif mood in self._centroids:
                    # Legacy fallback: use centroid as 1-NN anchor
                    self._anchors[mood] = self._centroids[mood].reshape(1, -1)
                    self._k_values[mood] = 1
                    self._anchor_names[mood] = [f"{mood} Centroid"]

            return self.is_trained
        except Exception as err:
            logger.warning(f"Failed to load personalized mood model from '{target_path}': {err}")
            return False
