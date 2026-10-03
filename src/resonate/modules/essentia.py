import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _load_essentia_mood_map() -> dict[str, str]:
    """Load Essentia class to mood mapping from package data."""
    try:
        from resonate.config import load_data_file

        return load_data_file("essentia_moods.yaml").get("essentia_to_mood", {})
    except Exception as err:
        logger.warning(f"Failed to load essentia_moods.yaml: {err}")
        return {}


ESSENTIA_MOOD_MAP: dict[str, str] = _load_essentia_mood_map()


class EssentiaAnalyzer:
    """Analyze audio files for mood classification using Essentia."""

    def __init__(self, models_dir: str = "models", model_filename: str = "model.pb") -> None:
        """Initialize EssentiaAnalyzer with models directory and model filename."""
        self.models_dir = models_dir
        self.model_filename = model_filename
        self.model_path = os.path.join(models_dir, model_filename)
        self._predictors: dict[str, Any] = {}
        self._meta_cache: dict[str, tuple[list[str], str, str]] = {}

    def _get_model_meta(self, model_path: str) -> tuple[list[str], str, str]:
        """Get cached class labels and input/output tensor names for a model."""
        json_path = os.path.splitext(model_path)[0] + ".json"
        if json_path in self._meta_cache:
            return self._meta_cache[json_path]

        model_classes: list[str] = []
        input_name = "serving_default_model_Placeholder"
        output_name = "PartitionedCall:0"
        if os.path.exists(json_path) and os.path.getsize(json_path) > 100:
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                    model_classes = meta.get("classes", [])
                    schema = meta.get("schema", {})
                    inputs = schema.get("inputs", [])
                    outputs = schema.get("outputs", [])
                    if inputs and "name" in inputs[0]:
                        input_name = inputs[0]["name"]
                    if outputs:
                        pred_outputs = [
                            o for o in outputs if o.get("output_purpose") == "predictions"
                        ]
                        if pred_outputs and "name" in pred_outputs[0]:
                            output_name = pred_outputs[0]["name"]
                        elif "name" in outputs[0]:
                            output_name = outputs[0]["name"]
            except Exception as json_err:
                logger.warning(f"Failed to read model metadata JSON: {json_err}")

        res = (model_classes, input_name, output_name)
        self._meta_cache[json_path] = res
        return res

    def load_audio(self, file_path: str, audio: Any | None = None) -> Any | None:
        """Load audio file into 16kHz mono float32 numpy array for model inference."""
        if audio is not None:
            try:
                import numpy as np

                return np.asarray(audio, dtype=np.float32)
            except Exception as err:
                logger.warning(f"Failed to convert provided audio buffer: {err}")
                return None

        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found: {file_path}")
            return None

        try:
            from resonate.utils.audio import calculate_audio_window, decode_audio

            start_sec, end_sec = calculate_audio_window(file_path, target_duration=90.0)
            _, audio_16k = decode_audio(file_path, start_sec=start_sec, end_sec=end_sec)
            return audio_16k
        except Exception as err:
            logger.warning(f"Failed to load audio from '{file_path}': {err}")
            return None

    def extract_embeddings(
        self, audio: Any | None = None, file_path: str | None = None
    ) -> Any | None:
        """Extract Discogs-EffNet embeddings from audio array or file."""
        if audio is None:
            if not file_path:
                return None
            audio = self.load_audio(file_path)
            if audio is None:
                return None

        embedding_model_filename = "discogs-effnet-bs64-1.pb"
        embedding_model_path = os.path.join(self.models_dir, embedding_model_filename)
        if (
            not os.path.exists(embedding_model_path)
            or os.path.getsize(embedding_model_path) < 10000
        ):
            logger.warning(
                f"Essentia embedding model not found/invalid: {embedding_model_path}. "
                "Please ensure it is downloaded to the models directory."
            )
            return None

        try:
            import essentia.standard as es

            emb_key = f"effnet:{embedding_model_path}:PartitionedCall:1"
            if emb_key not in self._predictors:
                self._predictors[emb_key] = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=embedding_model_path, output="PartitionedCall:1"
                )
            return self._predictors[emb_key](audio)
        except Exception as err:
            logger.warning(f"Failed to extract EffNet embeddings: {err}")
            return None

    def predict_moods(
        self,
        embeddings: Any,
        target_moods: list[str],
        tag_mapper: Any = None,
        bpm: int | None = None,
        candidate_seeds: list[str] | None = None,
        mood_thresholds: dict[str, float] | None = None,
    ) -> tuple[list[str], float, list[tuple[str, float]]]:
        """Predict moods from pre-extracted Discogs-EffNet embeddings."""
        if embeddings is None or len(embeddings) == 0:
            return ([], 0.0, [])

        model_path = self.model_path
        if not os.path.exists(model_path) or (
            os.path.exists(model_path) and os.path.getsize(model_path) < 10000
        ):
            jamendo_path = os.path.join(
                self.models_dir, "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
            )
            discogs_path = os.path.join(self.models_dir, "genre_discogs400-discogs-effnet-1.pb")
            if os.path.exists(jamendo_path) and os.path.getsize(jamendo_path) > 10000:
                model_path = jamendo_path
            elif os.path.exists(discogs_path) and os.path.getsize(discogs_path) > 10000:
                model_path = discogs_path
            else:
                logger.warning(f"No valid Essentia graph model found in '{self.models_dir}'")
                return ([], 0.0, [])

        try:
            import essentia.standard as es
            import numpy as np

            model_classes, input_name, output_name = self._get_model_meta(model_path)

            head_key = f"head:{model_path}:{input_name}:{output_name}"
            if head_key not in self._predictors:
                self._predictors[head_key] = es.TensorflowPredict2D(
                    graphFilename=model_path, input=input_name, output=output_name
                )
            predictions = self._predictors[head_key](embeddings)

            if predictions is None or len(predictions) == 0:
                return ([], 0.0, [])

            # Average predictions across all frames/patches
            if hasattr(predictions, "ndim") and predictions.ndim > 1:
                scores = np.mean(predictions, axis=0)
            else:
                scores = predictions

            max_score: float = 0.0
            top_predictions = []

            if model_classes and len(model_classes) == len(scores):
                # Evaluate top 10 candidates so distinctive mood classes at ranks 4-10 are preserved
                top_indices = np.argsort(scores)[::-1][:10]
                top_predictions = [(model_classes[idx], float(scores[idx])) for idx in top_indices]

                # We have the model's output classes. Find the highest scoring class
                best_class_idx = int(np.argmax(scores))
                max_score = float(scores[best_class_idx])

                # Filter out generic tempo and utility predictions
                generic_labels = {
                    "background",
                    "film",
                    "soundtrack",
                    "documentary",
                    "adventure",
                    "commercial",
                    "advertising",
                    "corporate",
                    "presentation",
                    "game",
                    "trailer",
                    "melodic",
                }

                distinctive_preds = [
                    p for p in top_predictions if p[0].lower() not in generic_labels
                ]
                active_thresholds = mood_thresholds
                if active_thresholds is None:
                    try:
                        from resonate.config import load_config

                        active_thresholds = load_config().mood_rules.acoustic_mood_thresholds
                    except Exception:
                        active_thresholds = {}

                confident_preds = []
                for p in distinctive_preds:
                    class_name = p[0].lower()
                    score = p[1]
                    target = ESSENTIA_MOOD_MAP.get(class_name)
                    if not target and target_moods:
                        for tm in target_moods:
                            if tm.lower() == class_name:
                                target = tm
                                break
                    if not target:
                        target = class_name.title()

                    # Synergy match with track-specific candidate seeds at >= 0.05
                    is_synergy = False
                    if candidate_seeds:
                        if any(target.lower() == cs.lower() for cs in candidate_seeds):
                            is_synergy = True

                    required_threshold = active_thresholds.get(target, 0.10)
                    if is_synergy and score >= 0.05:
                        confident_preds.append(p)
                    elif score >= required_threshold:
                        confident_preds.append(p)

                # Adaptive fallback: if no confident predictions,
                # lower to 0.08 for distinctive classes respecting configured thresholds
                if not confident_preds:
                    for p in distinctive_preds:
                        class_name = p[0].lower()
                        score = p[1]
                        target = ESSENTIA_MOOD_MAP.get(class_name)
                        if not target and target_moods:
                            for tm in target_moods:
                                if tm.lower() == class_name:
                                    target = tm
                                    break
                        if not target:
                            target = class_name.title()
                        fallback_threshold = active_thresholds.get(target, 0.08)
                        if score >= fallback_threshold:
                            confident_preds.append(p)

                # Map predicted top classes to target moods using ESSENTIA_MOOD_MAP + tag_mapper
                mapped_moods = []
                for p in confident_preds:
                    class_name = p[0].lower()
                    if class_name in ESSENTIA_MOOD_MAP:
                        target = ESSENTIA_MOOD_MAP[class_name]
                        if target not in mapped_moods:
                            mapped_moods.append(target)
                    elif tag_mapper is not None:
                        matches = tag_mapper.match_multiple_tags([p[0]], threshold=0.45)
                        for m in matches:
                            if m[0] not in mapped_moods:
                                mapped_moods.append(m[0])

                matched_score = 0.0
                for p in confident_preds:
                    class_name = p[0].lower()
                    target_m = ESSENTIA_MOOD_MAP.get(class_name)
                    if target_m and target_m in mapped_moods:
                        matched_score = max(matched_score, float(p[1]))
                if matched_score == 0.0 and mapped_moods:
                    matched_score = float(max_score)

                if mapped_moods:
                    return (mapped_moods, matched_score, top_predictions)
                # Fallback to direct substring matching if no tag_mapper is active
                matched_direct = []
                for mood in target_moods:
                    if any(mood.lower() in p[0].lower() for p in top_predictions):
                        matched_direct.append(mood)
                return (matched_direct, matched_score, top_predictions)
            else:
                top_indices = np.argsort(scores)[::-1][:3]
                top_predictions = [
                    (
                        target_moods[idx] if idx < len(target_moods) else f"class_{idx}",
                        float(scores[idx]),
                    )
                    for idx in top_indices
                ]

                # Default behavior: assume model outputs match target_moods order
                best_mood = None
                for i, score in enumerate(scores):
                    val = float(score)
                    if val > max_score and i < len(target_moods):
                        max_score = val
                        best_mood = target_moods[i]

                if best_mood is not None:
                    return ([best_mood], max_score, top_predictions)
                return ([], max_score, top_predictions)

        except Exception as err:
            logger.warning(f"Error during Essentia mood prediction: {err}")
            return ([], 0.0, [])

    def analyze_genre_waveform(
        self,
        file_path: str,
        genre_mapper: Any = None,
        subgenre_mapper: Any = None,
        audio: Any = None,
        metadata_tags: list[str] | None = None,
        metadata_primary_genre: str | None = None,
        tracer: Any = None,
    ) -> tuple[str | None, list[str]]:
        """Predict Primary Genre and Sub-Genres with top-3 candidate corroboration."""
        embeddings = self.extract_embeddings(audio=audio, file_path=file_path)
        if embeddings is None:
            return (None, [])

        genre_model_path = os.path.join(self.models_dir, "genre_discogs400-discogs-effnet-1.pb")
        if not os.path.exists(genre_model_path):
            return (None, [])

        try:
            import essentia.standard as es

            predictors = self._predictors
            genre_key = (
                f"head:{genre_model_path}:serving_default_model_Placeholder:PartitionedCall:0"
            )
            if genre_key not in predictors:
                predictors[genre_key] = es.TensorflowPredict2D(
                    graphFilename=genre_model_path,
                    input="serving_default_model_Placeholder",
                    output="PartitionedCall:0",
                )
            predictions = predictors[genre_key](embeddings)
            scores = predictions.mean(axis=0)

            labels, _, _ = self._get_model_meta(genre_model_path)

            if not labels or len(labels) != len(scores):
                return (None, [])

            top_indices = scores.argsort()[::-1][:10]
            top_preds = [
                (labels[idx], float(scores[idx])) for idx in top_indices if scores[idx] >= 0.05
            ]

            if not top_preds:
                return (None, [])

            from resonate.engine.taxonomy import (
                DEFAULT_PRIMARY_GENRES,
                _get_families_for_tag,
                _get_family_for_tag,
                deduplicate_subgenres,
                filter_subgenres_by_family,
            )

            # Build candidates list for top acoustic predictions
            candidates: list[tuple[str | None, list[str], float]] = []
            all_styles: list[str] = []
            for label, score in top_preds:
                parts = label.split("---")
                g_part = parts[0].strip()
                s_part = parts[1].strip() if len(parts) > 1 else g_part
                all_styles.append(s_part)
                if len(parts) > 1:
                    all_styles.append(f"{g_part} {s_part}")

                cand_subs: list[str] = []
                if subgenre_mapper is not None:
                    style_queries = [s_part]
                    if len(parts) > 1:
                        style_queries.append(f"{g_part} {s_part}")
                    s_matches = subgenre_mapper.match_multiple_tags(style_queries)
                    cand_subs = [m[0] for m in s_matches]

                cand_primary: str | None = None
                if cand_subs:
                    top_sub = cand_subs[0]
                    top_sub_fams = _get_families_for_tag(top_sub)
                    if genre_mapper is not None and g_part:
                        g_matches = genre_mapper.match_multiple_tags(
                            [g_part], apply_rank_decay=True
                        )
                        for m in g_matches:
                            if m[0] in top_sub_fams:
                                cand_primary = m[0]
                                break
                    if cand_primary is None:
                        st_fam = _get_family_for_tag(top_sub)
                        if st_fam and st_fam in DEFAULT_PRIMARY_GENRES:
                            cand_primary = st_fam

                if cand_primary is None and genre_mapper is not None and g_part:
                    g_matches = genre_mapper.match_multiple_tags([g_part], apply_rank_decay=True)
                    if g_matches:
                        cand_primary = g_matches[0][0]

                candidates.append((cand_primary, cand_subs, score))

            # Resolve target metadata families for corroboration
            meta_families: set[str] = set()
            if metadata_primary_genre:
                meta_families.add(metadata_primary_genre.lower().strip())
            if metadata_tags:
                for tag in metadata_tags:
                    tag_clean = tag.lower().strip()
                    tag_fams = _get_families_for_tag(tag_clean)
                    for tf in tag_fams:
                        meta_families.add(tf.lower().strip())
                    if genre_mapper is not None:
                        gm = genre_mapper.match_multiple_tags([tag_clean], apply_rank_decay=False)
                        for m in gm:
                            meta_families.add(m[0].lower().strip())
                    if tag_clean in {g.lower() for g in DEFAULT_PRIMARY_GENRES}:
                        meta_families.add(tag_clean)

            # Resolve canonical metadata subgenres for candidate corroboration
            meta_subgenres: set[str] = set()
            if subgenre_mapper is not None and metadata_tags:
                meta_s_matches = subgenre_mapper.match_multiple_tags(
                    metadata_tags, apply_rank_decay=False
                )
                meta_subgenres = {m[0] for m in meta_s_matches}

            # 1. Prefer candidate whose subgenre matches metadata subgenres
            # (direct subgenre corroboration)
            subgenre_corroborated_idx: int | None = None
            if meta_subgenres:
                meta_subs_lower = {s.lower() for s in meta_subgenres}
                for idx, (cand_prim, cand_subs, _cand_sc) in enumerate(candidates[:5]):
                    cand_fam = cand_prim.lower().strip() if cand_prim else ""
                    if cand_fam and meta_families and cand_fam not in meta_families:
                        continue
                    if any(s.lower() in meta_subs_lower for s in cand_subs):
                        subgenre_corroborated_idx = idx
                        break

            chosen_idx = 0
            corroborated = False
            subgenre_corroborated = False

            if subgenre_corroborated_idx is not None:
                chosen_idx = subgenre_corroborated_idx
                corroborated = True
                subgenre_corroborated = True
            elif meta_families:
                for idx, (cand_prim, cand_subs, _cand_sc) in enumerate(candidates[:3]):
                    cand_fam = cand_prim.lower().strip() if cand_prim else ""
                    cand_sub_fams = {
                        f.lower().strip() for s in cand_subs for f in _get_families_for_tag(s)
                    }
                    if (cand_fam and cand_fam in meta_families) or (cand_sub_fams & meta_families):
                        chosen_idx = idx
                        corroborated = True
                        break

            chosen_primary, chosen_subgenres, chosen_score = candidates[chosen_idx]

            if chosen_primary:
                # If a specific candidate was chosen, prioritize its subgenres
                pooled: list[str] = list(chosen_subgenres)
                for cand_p, cand_s, cand_sc in candidates:
                    if cand_p == chosen_primary:
                        is_cand_corroborated = bool(
                            meta_subgenres and any(s.lower() in meta_subs_lower for s in cand_s)
                        ) if meta_subgenres else False
                        if cand_sc >= 0.20 or is_cand_corroborated:
                            for s in cand_s:
                                if s not in pooled:
                                    pooled.append(s)
                surviving = filter_subgenres_by_family(chosen_primary, pooled)
                mapped_subgenres = deduplicate_subgenres(chosen_primary, surviving)
                if not mapped_subgenres and chosen_subgenres:
                    mapped_subgenres = deduplicate_subgenres(chosen_primary, chosen_subgenres)
            else:
                mapped_subgenres = chosen_subgenres

            # Minimum confidence floor for acoustic subgenres when uncorroborated
            # If acoustic prediction has no subgenre corroboration and score < 0.20,
            # discard weak acoustic subgenres so pipeline fallback can rescue with curated metadata.
            if mapped_subgenres and not subgenre_corroborated and chosen_score < 0.20:
                mapped_subgenres = []

            # Trace recording
            if tracer is not None and hasattr(tracer, "record"):
                top_prim, _, top_sc = candidates[0]
                if corroborated and chosen_idx > 0:
                    tracer.record(
                        f"Essentia waveform candidate #{chosen_idx + 1} '{chosen_primary}' "
                        f"(score={chosen_score:.2f}) corroborated by metadata "
                        f"(overriding #{1} '{top_prim}', score={top_sc:.2f})"
                    )
                elif corroborated:
                    tracer.record(
                        f"Essentia waveform genre analysis: Primary='{chosen_primary}', "
                        f"Subgenres={mapped_subgenres} (corroborated #1 acoustic prediction, "
                        f"score={chosen_score:.2f})"
                    )
                elif meta_families:
                    tracer.record(
                        f"Essentia waveform genre analysis: Primary='{chosen_primary}', "
                        f"Subgenres={mapped_subgenres} (audio ground truth override, "
                        f"top-3 acoustic candidates uncorroborated by metadata)"
                    )
                else:
                    tracer.record(
                        f"Essentia waveform genre analysis: Primary='{chosen_primary}', "
                        f"Subgenres={mapped_subgenres}"
                    )

            return (chosen_primary, mapped_subgenres)

        except Exception as err:
            logger.warning(f"Error during Essentia genre waveform analysis: {err}")
            return (None, [])
