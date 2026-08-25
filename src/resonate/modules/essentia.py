import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


ESSENTIA_MOOD_MAP: dict[str, str] = {
    "sexy": "Romantic",
    "love": "Romantic",
    "sad": "Melancholic",
    "ballad": "Melancholic",
    "emotional": "Melancholic",
    "relaxing": "Relaxed",
    "meditative": "Calm",
    "soft": "Mellow",
    "heavy": "Heavy",
    "party": "Party",
    "fun": "Party",
    "dark": "Dark",
    "dramatic": "Dark",
    "happy": "Happy",
    "positive": "Happy",
    "groovy": "Groovy",
    "energetic": "Energetic",
    "upbeat": "Upbeat",
    "inspiring": "Upbeat",
    "motivational": "Upbeat",
    "hopeful": "Upbeat",
    "epic": "Atmospheric",
    "dream": "Atmospheric",
}


class EssentiaAnalyzer:
    """Analyze audio files for mood classification using Essentia."""

    def __init__(self, models_dir: str = "models", model_filename: str = "model.pb") -> None:
        """Initialize EssentiaAnalyzer with models directory and model filename."""
        self.models_dir = models_dir
        self.model_filename = model_filename
        self.model_path = os.path.join(models_dir, model_filename)

    def analyze_waveform(
        self,
        file_path: str,
        target_moods: list[str],
        tag_mapper: Any = None,
        bpm: int | None = None,
    ) -> tuple[list[str], float, list[tuple[str, float]]]:
        """Analyze audio file waveform and predict best matching target moods."""
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found: {file_path}")
            return ([], 0.0, [])

        model_path = self.model_path
        # Auto-fallback if model_path is missing or a tiny 404 file (< 10 KB)
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
        except ImportError:
            logger.warning("Essentia or numpy library is not installed.")
            return ([], 0.0, [])

        try:
            audio = es.MonoLoader(filename=file_path, sampleRate=16000)()

            # Check if there is an associated metadata JSON file containing class labels
            json_path = os.path.splitext(model_path)[0] + ".json"
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

            # If this is a Discogs EffNet based classification head model,
            # we need to extract the EffNet embeddings first, then pass to model.
            model_filename = os.path.basename(model_path)
            if "effnet" in model_filename.lower():
                embedding_model_filename = "discogs-effnet-bs64-1.pb"
                embedding_model_path = os.path.join(self.models_dir, embedding_model_filename)
                invalid_emb = (
                    not os.path.exists(embedding_model_path)
                    or os.path.getsize(embedding_model_path) < 10000
                )
                if invalid_emb:
                    logger.warning(
                        f"Essentia embedding model not found/invalid: {embedding_model_path}. "
                        "Please ensure it is downloaded to the models directory."
                    )
                    return ([], 0.0, [])

                # Step 1: Extract embeddings
                embedding_extractor = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=embedding_model_path, output="PartitionedCall:1"
                )
                embeddings = embedding_extractor(audio)

                # Step 2: Run classification head on embeddings
                model = es.TensorflowPredict2D(
                    graphFilename=model_path, input=input_name, output=output_name
                )
                predictions = model(embeddings)
            else:
                # Standalone model classification
                model = es.TensorflowPredict2D(graphFilename=model_path)
                predictions = model(audio)

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
                    "fast",
                    "lively",
                    "slow",
                    "background",
                    "commercial",
                    "advertising",
                    "corporate",
                    "film",
                    "movie",
                    "documentary",
                    "game",
                    "trailer",
                }
                melodic_score = next(
                    (float(p[1]) for p in top_predictions if p[0].lower() == "melodic"), 0.0
                )
                has_melodic_boost = melodic_score >= 0.10

                distinctive_preds = [
                    p
                    for p in top_predictions
                    if p[0].lower() not in generic_labels and p[0].lower() != "melodic"
                ]
                confident_preds = []
                for p in distinctive_preds:
                    lbl = p[0].lower()
                    score = p[1]
                    # Synergy match with target moods / candidate seeds at >= 0.05
                    is_synergy = False
                    if lbl in ESSENTIA_MOOD_MAP:
                        target = ESSENTIA_MOOD_MAP[lbl]
                        if any(target.lower() == tm.lower() for tm in target_moods):
                            is_synergy = True

                    # Strong synergy match or melodic booster for emotional classes at >= 0.05
                    if (
                        is_synergy or (has_melodic_boost and lbl in ESSENTIA_MOOD_MAP)
                    ) and score >= 0.05:
                        confident_preds.append(p)
                    elif lbl == "love":
                        if score >= 0.16:
                            confident_preds.append(p)
                    elif score >= 0.10:
                        confident_preds.append(p)

                # Adaptive fallback: if no confident predictions,
                # lower to 0.08 for distinctive classes
                if not confident_preds:
                    for p in distinctive_preds:
                        score = p[1]
                        if score >= 0.08:
                            confident_preds.append(p)

                # Map predicted top classes to target moods using ESSENTIA_MOOD_MAP + tag_mapper
                mapped_moods = []
                for p in confident_preds:
                    lbl_lower = p[0].lower()
                    if lbl_lower in ESSENTIA_MOOD_MAP:
                        target = ESSENTIA_MOOD_MAP[lbl_lower]
                        if target not in mapped_moods:
                            mapped_moods.append(target)
                    elif tag_mapper is not None:
                        matches = tag_mapper.match_multiple_tags([p[0]], threshold=0.45)
                        for m in matches:
                            if m[0] not in mapped_moods:
                                mapped_moods.append(m[0])

                # BPM-Grounded Mood Validation (case-insensitive):
                if bpm is not None:
                    if any(m.lower() == "energetic" for m in mapped_moods) and bpm < 130:
                        mapped_moods = [m for m in mapped_moods if m.lower() != "energetic"]
                    if any(m.lower() == "lively" for m in mapped_moods) and bpm < 115:
                        mapped_moods = [m for m in mapped_moods if m.lower() != "lively"]
                    if (
                        any(m.lower() in {"relaxed", "calm", "mellow"} for m in mapped_moods)
                        and bpm > 120
                    ):
                        mapped_moods = [
                            m
                            for m in mapped_moods
                            if m.lower() not in {"relaxed", "calm", "mellow"}
                        ]

                # Standalone Energetic / Lively rule:
                # Energetic is valid standalone if BPM >= 130; Lively if 110 <= BPM < 130.
                # Otherwise they serve as secondary support tags alongside a specific mood.
                specific_support_moods = {
                    "aggressive",
                    "heavy",
                    "party",
                    "upbeat",
                    "groovy",
                    "melancholic",
                    "mellow",
                    "romantic",
                    "dark",
                    "calm",
                    "happy",
                }
                has_specific_support = any(
                    m.lower() in specific_support_moods for m in mapped_moods
                )
                if (
                    bpm is not None
                    and bpm >= 130
                    and any(m.lower() == "energetic" for m in mapped_moods)
                ):
                    has_specific_support = True
                if (
                    bpm is not None
                    and 110 <= bpm < 130
                    and any(m.lower() == "lively" for m in mapped_moods)
                ):
                    has_specific_support = True

                if not has_specific_support:
                    mapped_moods = [
                        m for m in mapped_moods if m.lower() not in {"energetic", "lively"}
                    ]

                matched_score = 0.0
                for p in confident_preds:
                    lbl_lower = p[0].lower()
                    if (
                        lbl_lower in ESSENTIA_MOOD_MAP
                        and ESSENTIA_MOOD_MAP[lbl_lower] in mapped_moods
                    ):
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
                for i, score in enumerate(scores):
                    val = float(score)
                    if val > max_score and i < len(target_moods):
                        max_score = val
                        best_mood = target_moods[i]

                if best_mood is not None:
                    return (best_mood, max_score, top_predictions)
                return (None, max_score, top_predictions)

        except Exception as err:
            logger.warning(f"Error during Essentia waveform analysis: {err}")
            return ([], 0.0, [])

    def analyze_genre_waveform(
        self,
        file_path: str,
        genre_mapper: Any = None,
        subgenre_mapper: Any = None,
    ) -> tuple[str | None, list[str]]:
        """Predict Primary Genre and Sub-Genres using 400 Discogs model."""
        if not os.path.exists(file_path):
            return (None, [])

        model_path = os.path.join(self.models_dir, "discogs-effnet-bs64-1.pb")
        labels_path = os.path.join(self.models_dir, "genre_discogs400-discogs-effnet-1.json")

        if not os.path.exists(model_path):
            return (None, [])

        try:
            import essentia.standard as es

            loader = es.MonoLoader(filename=file_path, sampleRate=16000)
            audio = loader()
            model = es.TensorflowPredictEffnetDiscogs(
                graphFilename=model_path, output="PartitionedCall:0"
            )
            predictions = model(audio)
            scores = predictions.mean(axis=0)

            labels = []
            if os.path.exists(labels_path):
                with open(labels_path) as f:
                    data = json.load(f)
                    labels = data.get("classes", [])

            if not labels or len(labels) != len(scores):
                return (None, [])

            top_indices = scores.argsort()[::-1][:10]
            top_preds = [
                (labels[idx], float(scores[idx])) for idx in top_indices if scores[idx] >= 0.10
            ]

            if not top_preds:
                return (None, [])

            raw_genres = []
            raw_styles = []
            for label, _score in top_preds:
                parts = label.split("---")
                g = parts[0].strip()
                s = parts[1].strip() if len(parts) > 1 else g
                if "Folk" in g:
                    g = "Folk"
                elif "Hip Hop" in g:
                    g = "Hip-Hop"
                raw_genres.append(g)
                raw_styles.append(s)

            mapped_primary = None
            if genre_mapper is not None and raw_genres:
                g_matches = genre_mapper.match_multiple_tags(raw_genres)
                if g_matches:
                    from collections import Counter

                    mapped_primary = Counter([m[0] for m in g_matches]).most_common(1)[0][0]

            mapped_subgenres = []
            if subgenre_mapper is not None and raw_styles:
                s_matches = subgenre_mapper.match_multiple_tags(raw_styles)
                mapped_subgenres = [m[0] for m in s_matches]

            # Elevate Punk/Metal over generic Rock if subgenres indicate punk or metal
            if mapped_subgenres:
                if any("punk" in s.lower() for s in mapped_subgenres) or any(
                    "punk" in s.lower() for s in raw_styles
                ):
                    mapped_primary = "Punk"
                elif any("metal" in s.lower() for s in mapped_subgenres) or any(
                    "metal" in s.lower() for s in raw_styles
                ):
                    mapped_primary = "Metal"

            return (mapped_primary, mapped_subgenres)

        except Exception as err:
            logger.warning(f"Error during Essentia genre waveform analysis: {err}")
            return (None, [])
