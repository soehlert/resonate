import logging
import os
from collections import Counter
from typing import Any

import numpy as np

from resonate.engine.mood_rules import (
    DEFAULT_MOOD_TAGS,
    DEFAULT_TARGET_MOODS,
)
from resonate.engine.taxonomy import (
    DEFAULT_PRIMARY_GENRES,
    DEFAULT_SUB_GENRES,
    GENERIC_MODIFIERS,
    NATIONALITY_STRINGS,
    PRIMARY_GENRE_STEMS,
    SUB_GENRE_STEMS,
    SUBGENRE_REGISTRY,
)

# Squelch Hugging Face Hub token warnings and progress bars
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


SUBGENRE_DESCRIPTIONS: dict[str, str] = {
    spec.name: spec.description for spec in SUBGENRE_REGISTRY if spec.description
}

MOOD_DESCRIPTIONS: dict[str, str] = {
    # Moods / Vibes
    "Party": "Party music, energetic celebration fun club dance party",
    "Chill Hang": (
        "Chill hang music, millennial indie rock, indie pop, indie folk, "
        "laid-back americana, relaxed mellow easygoing listening, lo-fi chill"
    ),
    "Energetic": "Energetic music, driving high-intensity powerful energetic energy",
    "Groovy": "Groovy music, rhythmic funk bass dance groove",
    "Acoustic": "Acoustic music, unplugged acoustic guitar organic sound",
    "Electronic": "Electronic music, synthesizer electronic beat synth sound",
    "Melancholic": "Melancholic music, sad bittersweet somber melancholic ballad",
    "Lively": "Lively music, bright upbeat active lively animated pop",
    "Relaxed": "Relaxed music, calm peaceful gentle relaxed quiet sound",
    "Romantic": "Romantic music, intimate passionate love romantic ballad",
    "Calm": "Calm music, peaceful quiet meditative calm sound",
    "Upbeat": "Upbeat music, happy cheerful feel-good upbeat pop",
    "Dark": "Dark music, brooding minor key heavy dark atmospheric",
    "Happy": "Happy music, joyful bright happy feel-good song",
    "Mellow": "Mellow music, soft gentle relaxed mellow acoustic",
    "Heavy": "Heavy music, aggressive heavy guitar distortion loud rock",
    "Aggressive": "Aggressive music, intense rowdy loud aggressive metal punk",
    "Soulful": "Soulful music, smooth vocal R&B soulful emotional blues",
    "Trippy": "Trippy music, hypnotic psychedelic spacey trippy sound",
}

CONTEXTUAL_DESCRIPTIONS: dict[str, str] = {**SUBGENRE_DESCRIPTIONS, **MOOD_DESCRIPTIONS}
NORM_CONTEXTUAL_DESCRIPTIONS: dict[str, str] = {
    k.lower(): v for k, v in CONTEXTUAL_DESCRIPTIONS.items()
}


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

    def warmup(self) -> None:
        """Pre-load model and pre-compute target embeddings."""
        self._get_model()

    def _get_model(self) -> Any:
        """Lazy load or return existing SentenceTransformer model."""
        if self._model is None:
            try:
                import os

                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                from sentence_transformers import SentenceTransformer

                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
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
        """Pre-compute embeddings for target moods using contextual descriptions."""
        if self._model is not None and self.target_moods:
            try:
                descriptions = [
                    NORM_CONTEXTUAL_DESCRIPTIONS.get(tm.lower(), f"{tm} music")
                    for tm in self.target_moods
                ]
                self.target_embeddings = self._encode(self._model, descriptions)
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

    def _score_candidate_tag(
        self,
        target_tag: str,
        raw: str,
        raw_tags: list[str] | None = None,
    ) -> float | None:
        """Evaluate match between a raw tag and a target taxonomy tag, returning base score."""
        raw_clean = raw.lower().strip()
        target_clean = target_tag.lower().strip()
        raw_norm = raw_clean.replace("-", " ").replace("/", " ")
        target_norm = target_clean.replace("-", " ").replace("/", " ")

        # 1. Exact string match
        if raw_clean == target_clean or (
            len(raw_clean) > 3
            and (raw_clean == target_clean.replace("-", " ") or raw_norm == target_norm)
        ):
            return 1.0

        raw_words = set(raw_norm.split())
        target_words = set(target_norm.split())
        is_compound = len(target_words) > 1

        # 2. Contextual disambiguation for Indie and Hardcore
        if raw_tags:
            if target_tag == "Indie Rock" and "indie" in raw_words:
                if any(w in r.lower() for r in raw_tags[:3] for w in ["rock", "garage"]):
                    return 0.95
            elif target_tag == "Indie Pop" and "indie" in raw_words:
                if any(w in r.lower() for r in raw_tags[:3] for w in ["pop", "dance"]):
                    return 0.95
            elif target_tag == "Indie Folk" and "indie" in raw_words:
                if any(w in r.lower() for r in raw_tags[:3] for w in ["folk", "acoustic"]):
                    return 0.95
            elif target_tag == "Hardcore Punk" and "hardcore" in raw_words:
                if any(w in r.lower() for r in raw_tags for w in ["punk", "punk rock", "nyhc"]):
                    return 0.95
            elif target_tag == "Hardcore Hip Hop" and "hardcore" in raw_words:
                if any(
                    w in r.lower()
                    for r in raw_tags
                    for w in ["hip hop", "hip-hop", "rap", "hiphop"]
                ):
                    return 0.95

        # 3. Data-driven taxonomy stem matching
        if target_tag in PRIMARY_GENRE_STEMS:
            for stem in PRIMARY_GENRE_STEMS[target_tag]:
                stem_match = (
                    stem in raw_words
                    if " " not in stem and "-" not in stem
                    else (
                        stem in raw_clean or stem.replace("-", " ") in raw_clean.replace("-", " ")
                    )
                )
                if stem_match:
                    if target_tag == "Rock" and any(p in raw_clean for p in ["punk", "metal"]):
                        continue
                    if target_tag == "Hip-Hop" and any(
                        m in raw_clean for m in ["metal", "rapcore", "rock"]
                    ):
                        continue
                    return 0.95

        if target_tag in SUB_GENRE_STEMS:
            for stem in SUB_GENRE_STEMS[target_tag]:
                stem_clean = stem.replace("-", " ").replace("/", " ")
                if raw_clean == stem or raw_norm == stem_clean:
                    return 0.95

        # 4. Word-stem substring inclusion (compound targets only, non-generic modifiers)
        if is_compound and raw_norm not in GENERIC_MODIFIERS and raw_clean not in GENERIC_MODIFIERS:
            if len(raw_clean) >= 3 and (
                raw_clean in target_clean
                or raw_norm in target_norm
                or target_norm in raw_norm
                or (raw_words and raw_words.issubset(target_words))
            ):
                if (
                    target_tag in {"Americana", "Country", "Folk"}
                    and raw_clean in NATIONALITY_STRINGS
                ):
                    return None
                return 0.95

        return None

    def match_multiple_tags(
        self, raw_tags: list[str], threshold: float | None = None, max_matches: int = 3
    ) -> list[tuple[str, str, float]]:
        """Match raw tags against target tags using configured threshold and capping top results."""
        cutoff = threshold if threshold is not None else self.threshold
        if not raw_tags or not self.target_moods:
            return []

        is_genre_or_subgenre = (
            self.target_moods == DEFAULT_SUB_GENRES or self.target_moods == DEFAULT_PRIMARY_GENRES
        )

        sim_matrix = None
        if not is_genre_or_subgenre:
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
        # Track candidates discovered from top consensus tags (raw_tags[:5])
        top_consensus_candidates: set[str] = set()

        for col_idx, target_tag in enumerate(self.target_moods):
            best_raw: str | None = None
            best_score: float = 0.0
            best_raw_idx: int = -1

            for raw_idx, raw in enumerate(raw_tags):
                # Top-5 candidate gating: only raw_tags[:5] can introduce new candidates
                if raw_idx >= 5 and target_tag not in top_consensus_candidates:
                    continue

                base_score = self._score_candidate_tag(target_tag, raw, raw_tags)
                if base_score is not None:
                    rank_factor = max(0.50, 1.0 - (raw_idx * 0.04))
                    score = base_score * rank_factor
                    if score > best_score:
                        best_score = score
                        best_raw = raw
                        best_raw_idx = raw_idx

            if best_raw is not None and best_score > 0:
                matched_results.append((target_tag, best_raw, best_score))
                if best_raw_idx < 3:
                    top_consensus_candidates.add(target_tag)
                continue

            # Disable fuzzy vector similarity for sub-genres and primary genres
            if is_genre_or_subgenre or sim_matrix is None:
                continue

            # Instrumentation tags (acoustic, electronic) require explicit keyword hits
            if target_tag.lower() in ["acoustic", "electronic"]:
                continue
            row_idx = int(np.argmax(sim_matrix[:, col_idx]))
            matched_raw = raw_tags[row_idx].lower().strip()
            # Skip fuzzy matching if the matched raw tag is a generic primary genre name
            generic_primary_words = {g.lower() for g in DEFAULT_PRIMARY_GENRES}
            if matched_raw in generic_primary_words:
                continue
            score = float(sim_matrix[row_idx, col_idx])
            # Rank-weighted scoring based on Last.fm community consensus order
            rank_factor = max(0.40, 1.0 - (row_idx * 0.05))
            effective_score = score * rank_factor
            if effective_score >= cutoff * 0.80:
                matched_results.append((target_tag, raw_tags[row_idx], effective_score))

        # Deduplicate and keep highest effective_score for each target_tag
        unique_matches: dict[str, tuple[str, float]] = {}
        for tgt, raw, score in matched_results:
            if tgt not in unique_matches or score > unique_matches[tgt][1]:
                unique_matches[tgt] = (raw, score)

        sorted_results = sorted(
            [(k, v[0], v[1]) for k, v in unique_matches.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        final_results = sorted_results

        # For subgenres, gate matches strictly to candidates introduced in the top 3 tags
        if self.target_moods == DEFAULT_SUB_GENRES and top_consensus_candidates:
            final_results = [item for item in final_results if item[0] in top_consensus_candidates]

        return final_results[:max_matches]

    def match_genre_consensus(self, raw_tags: list[str]) -> list[tuple[str, str, float, int]]:
        """Match all raw tags against primary genres without single-match deduplication.

        Returns a list of (target_genre, raw_tag, score, raw_idx) for every matching raw tag.
        """
        if not raw_tags or not self.target_moods:
            return []

        matched_results: list[tuple[str, str, float, int]] = []
        for raw_idx, raw in enumerate(raw_tags):
            rank_factor = max(0.50, 1.0 - (raw_idx * 0.04))
            for target_tag in self.target_moods:
                base_score = self._score_candidate_tag(target_tag, raw, raw_tags)
                if base_score is not None:
                    score = base_score * rank_factor
                    matched_results.append((target_tag, raw, score, raw_idx))

        return matched_results

    def match_subgenre_consensus(
        self, raw_tags: list[str], max_matches: int = 3
    ) -> list[tuple[str, str, float]]:
        """Match subgenres using direct additive tag consensus."""
        if not raw_tags or not self.target_moods:
            return []

        raw_matches: list[tuple[str, str, float, int]] = []
        for tag_index, raw_tag in enumerate(raw_tags):
            single_matches = self.match_multiple_tags([raw_tag], max_matches=2)
            if single_matches and single_matches[0][2] >= 1.0:
                single_matches = [single_matches[0]]
            rank_factor = max(0.50, 1.0 - (tag_index * 0.04))
            for target_subgenre, matched_raw, match_score in single_matches:
                raw_matches.append(
                    (target_subgenre, matched_raw, match_score * rank_factor, tag_index)
                )

        if not raw_matches:
            return []

        # 1. Accumulate individual subgenre scores directly
        subgenre_scores: Counter[str] = Counter()
        subgenre_raw_map: dict[str, str] = {}
        for target_subgenre, matched_raw, score, _idx in raw_matches:
            subgenre_scores[target_subgenre] += score
            if target_subgenre not in subgenre_raw_map:
                subgenre_raw_map[target_subgenre] = matched_raw

        sorted_subgenres = sorted(
            subgenre_scores.items(), key=lambda item: item[1], reverse=True
        )

        # 2. Retain top subgenres meeting minimum consensus support (>= 15% of top score)
        top_score = sorted_subgenres[0][1]
        final = [
            (target_subgenre, subgenre_raw_map.get(target_subgenre, ""), score)
            for target_subgenre, score in sorted_subgenres
            if score >= top_score * 0.15
        ]
        return final[:max_matches]


__all__ = [
    "CONTEXTUAL_DESCRIPTIONS",
    "DEFAULT_MOOD_TAGS",
    "DEFAULT_PRIMARY_GENRES",
    "DEFAULT_SUB_GENRES",
    "DEFAULT_TARGET_MOODS",
    "GENERIC_MODIFIERS",
    "NATIONALITY_STRINGS",
    "PRIMARY_GENRE_STEMS",
    "SUB_GENRE_STEMS",
    "TagMapper",
]

