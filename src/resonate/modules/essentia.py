"""Audio waveform analyzer module using Essentia TensorFlow models."""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


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

        if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
            logger.warning(f"Essentia model file not found or invalid: {model_path}")
            return ([], 0.0, [])

        try:
            import json

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
                distinctive_preds = [
                    p for p in top_predictions if p[0].lower() not in generic_labels
                ]
                confident_preds = [
                    p for p in distinctive_preds if p[1] >= max_score * 0.50 and p[1] >= 0.12
                ]

                # Map predicted top classes to target moods
                if tag_mapper is not None:
                    top_labels = [p[0] for p in confident_preds]
                    mood_matches = tag_mapper.match_multiple_tags(
                        top_labels, threshold=0.45, max_matches=3
                    )
                    mapped_moods = [m[0] for m in mood_matches]
                    # BPM-Grounded Mood Validation (case-insensitive):
                    if bpm is not None:
                        if any(m.lower() == "energetic" for m in mapped_moods) and bpm < 125:
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

                    # Prioritize specific moods over generic Energetic
                    if len(mapped_moods) > 1 and any(
                        m.lower() == "energetic" for m in mapped_moods
                    ):
                        mapped_moods = [m for m in mapped_moods if m.lower() != "energetic"]
                    if mapped_moods:
                        return (mapped_moods, max_score, top_predictions)
                # Fallback to direct substring matching if no tag_mapper is active
                matched_direct = []
                for mood in target_moods:
                    if any(mood.lower() in p[0].lower() for p in top_predictions):
                        matched_direct.append(mood)
                return (matched_direct, max_score, top_predictions)
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
            return (None, 0.0, [])
