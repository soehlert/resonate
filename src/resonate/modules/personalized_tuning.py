"""Personalized mood classification engine based on anchor track EffNet embeddings."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np

from resonate.config import load_data_file

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join("models", "personalized_mood_heads.json")
DEFAULT_ANCHOR_THRESHOLD = float(
    load_data_file("mood_rules.yaml").get("anchor_threshold", 0.65)
)


class PersonalizedMoodTuner:
    """Manages anchor-based personalized mood scoring using adaptive k-NN."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        default_threshold: float = DEFAULT_ANCHOR_THRESHOLD,
    ) -> None:
        """Initialize tuner with target model JSON file path and default threshold."""
        self.model_path = model_path
        self.default_threshold = default_threshold
        self.mood_heads: dict[str, dict[str, Any]] = {}
        self._anchors: dict[str, np.ndarray] = {}
        self._anchor_names: dict[str, list[str]] = {}
        self._thresholds: dict[str, float] = {}
        self._k_values: dict[str, int] = {}

    @property
    def is_trained(self) -> bool:
        """Return True if at least one mood head is calibrated and loaded."""
        return len(self._anchors) > 0

    @property
    def tuned_moods(self) -> list[str]:
        """Return list of canonical mood names currently tuned."""
        return sorted(self._anchors.keys())

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
        """Store anchor matrices and calibrate adaptive k-NN thresholds.

        Args:
            mood_embeddings: Mapping from canonical mood name to a list of
                frame-level (num_frames, 1280) or pre-pooled (1280,) numpy arrays.
            target_k: Consensus neighborhood size (default: 3).
            track_names: Optional mapping from mood name to anchor track display titles.
        """
        self.mood_heads.clear()
        self._anchors.clear()
        self._anchor_names.clear()
        self._thresholds.clear()
        self._k_values.clear()

        raw_anchors: dict[str, np.ndarray] = {}
        raw_k: dict[str, int] = {}
        raw_names: dict[str, list[str]] = {}
        anchor_consensus_metrics: dict[str, tuple[float, float]] = {}

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

            # Adaptive k
            k = self._compute_k(n_tracks, target_k=target_k)
            raw_k[mood] = k
            if track_names and mood in track_names:
                raw_names[mood] = track_names[mood][:n_tracks]
            else:
                raw_names[mood] = [f"Anchor {i+1}" for i in range(n_tracks)]

            # Leave-one-out peer consensus evaluation for coherence and threshold calibration
            if n_tracks == 1:
                anchor_peer_similarities = [1.0]
            else:
                peer_k = min(k, n_tracks - 1)
                anchor_peer_similarities = []
                # Pairwise cosine similarity matrix: (N, N)
                sim_matrix = np.dot(anchors_arr, anchors_arr.T)
                for i in range(n_tracks):
                    other_sims = np.delete(sim_matrix[i], i)
                    top_k_sims = np.sort(other_sims)[-peer_k:]
                    anchor_peer_similarities.append(float(np.mean(top_k_sims)))

            mean_similarity = float(np.mean(anchor_peer_similarities))
            std_similarity = (
                float(np.std(anchor_peer_similarities))
                if len(anchor_peer_similarities) > 1
                else 0.04
            )
            anchor_consensus_metrics[mood] = (mean_similarity, std_similarity)

        # Calibrate thresholds purely from anchor consensus distributions
        for mood, anchors_arr in raw_anchors.items():
            mean_similarity, std_similarity = anchor_consensus_metrics[mood]
            k = raw_k[mood]
            n_tracks = len(anchors_arr)

            # Set threshold slightly below the mean peer similarity (~0.75 std dev)
            # to accommodate ~80% of anchor tracks without being overly permissive
            if n_tracks >= 3:
                base_threshold = mean_similarity - 0.75 * std_similarity
            else:
                base_threshold = mean_similarity - 0.05

            # Bounded threshold:
            # - Floor (0.58): prevents matching ambient background noise on diffuse playlists
            # - Ceiling (0.84): prevents impossible matching bars on tight playlists
            calibrated_threshold = max(0.58, min(0.84, base_threshold))

            self._anchors[mood] = anchors_arr
            self._anchor_names[mood] = raw_names.get(mood, [])
            self._thresholds[mood] = float(round(calibrated_threshold, 3))
            self._k_values[mood] = k

            self.mood_heads[mood] = {
                "anchors": anchors_arr.tolist(),
                "anchor_names": raw_names.get(mood, []),
                "threshold": float(round(calibrated_threshold, 3)),
                "k": k,
                "track_count": n_tracks,
                "coherence": float(round(mean_similarity, 3)),
            }

    def _score_track_for_mood(self, unit_track: np.ndarray, mood: str) -> float:
        """Compute k-NN similarity score of a unit track vector against a mood's anchors."""
        anchors = self._anchors.get(mood)
        if anchors is not None and len(anchors) > 0:
            sims = np.dot(anchors, unit_track)
            k = min(self._k_values.get(mood, 1), len(sims))
            top_k_sims = np.sort(sims)[-k:]
            return float(np.mean(top_k_sims))
        return 0.0

    def predict(
        self,
        track_embedding: np.ndarray,
        top_k: int = 1,
        default_threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Score a track's EffNet embedding against calibrated mood heads.

        Uses competitive Winner-Take-All matching: only candidate moods with the
        highest consensus score are eligible, preventing lower-ranked moods with
        lower thresholds from matching as backdoors.

        Args:
            track_embedding: Array of shape (1280,) or (num_frames, 1280).
            top_k: Maximum number of matching moods to return (default: 1).
            default_threshold: Optional fallback threshold override.

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

        all_scores: list[tuple[str, float]] = []
        for mood in self._anchors:
            score = self._score_track_for_mood(unit_track, mood)
            all_scores.append((mood, score))

        if not all_scores:
            return []

        # Sort all moods by similarity descending
        all_scores.sort(key=lambda x: x[1], reverse=True)

        active_default = (
            default_threshold if default_threshold is not None else self.default_threshold
        )
        matches: list[tuple[str, float]] = []
        for mood, score in all_scores:
            threshold = self._thresholds.get(mood, active_default)
            if score >= threshold:
                matches.append((mood, round(score, 3)))
                if len(matches) >= top_k:
                    break
            else:
                # If the highest-scoring mood fails its threshold, lower-ranked
                # moods with lower thresholds must not backdoor match!
                break

        return matches

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
        for mood in self._anchors:
            score = self._score_track_for_mood(unit_track, mood)
            threshold = self._thresholds.get(mood, self.default_threshold)
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
        """Save tuned anchors and thresholds to human-readable JSON."""
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
        """Load tuned anchors and thresholds from JSON file."""
        target_path = path or self.model_path
        if not os.path.exists(target_path) or os.path.getsize(target_path) < 10:
            return False

        try:
            with open(target_path, encoding="utf-8") as f:
                data = json.load(f)

            moods_data = data.get("moods", {})
            self.mood_heads = moods_data
            self._anchors.clear()
            self._anchor_names.clear()
            self._thresholds.clear()
            self._k_values.clear()

            for mood, meta in moods_data.items():
                threshold = meta.get("threshold", 0.65)
                self._thresholds[mood] = float(threshold)

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

            return self.is_trained
        except Exception as err:
            logger.warning(f"Failed to load personalized mood model from '{target_path}': {err}")
            return False
