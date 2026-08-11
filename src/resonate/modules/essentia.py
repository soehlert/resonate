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
        self, file_path: str, target_moods: list[str], tag_mapper: Any = None
    ) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Analyze audio file waveform and predict best matching target mood."""
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found: {file_path}")
            return (None, 0.0, [])

        if not os.path.exists(self.model_path):
            logger.warning(f"Essentia model file not found: {self.model_path}")
            return (None, 0.0, [])

        try:
            import json

            import essentia.standard as es
            import numpy as np
        except ImportError:
            logger.warning("Essentia or numpy library is not installed.")
            return (None, 0.0, [])

        try:
            audio = es.MonoLoader(filename=file_path, sampleRate=16000)()

            # Check if there is an associated metadata JSON file containing class labels
            # and inputs/outputs configurations
            json_path = os.path.splitext(self.model_path)[0] + ".json"
            model_classes: list[str] = []
            input_name = "serving_default_model_Placeholder"
            output_name = "PartitionedCall:0"
            if os.path.exists(json_path):
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
            if "effnet" in self.model_filename.lower():
                embedding_model_filename = "discogs-effnet-bs64-1.pb"
                embedding_model_path = os.path.join(self.models_dir, embedding_model_filename)
                if not os.path.exists(embedding_model_path):
                    logger.warning(
                        f"Essentia embedding model file not found: {embedding_model_path}. "
                        f"Please ensure it is downloaded to the models directory."
                    )
                    return (None, 0.0, [])

                # Step 1: Extract embeddings
                embedding_extractor = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=embedding_model_path, output="PartitionedCall:1"
                )
                embeddings = embedding_extractor(audio)

                # Step 2: Run classification head on embeddings
                model = es.TensorflowPredict2D(
                    graphFilename=self.model_path, input=input_name, output=output_name
                )
                predictions = model(embeddings)
            else:
                # Standalone model classification
                model = es.TensorflowPredict2D(graphFilename=self.model_path)
                predictions = model(audio)

            if predictions is None or len(predictions) == 0:
                return (None, 0.0, [])

            # Average predictions across all frames/patches
            if hasattr(predictions, "ndim") and predictions.ndim > 1:
                scores = np.mean(predictions, axis=0)
            else:
                scores = predictions

            best_mood: str | None = None
            max_score: float = 0.0
            top_predictions = []

            if model_classes and len(model_classes) == len(scores):
                top_indices = np.argsort(scores)[::-1][:3]
                top_predictions = [(model_classes[idx], float(scores[idx])) for idx in top_indices]

                # We have the model's output classes. Find the highest scoring class
                best_class_idx = int(np.argmax(scores))
                max_score = float(scores[best_class_idx])
                best_class = model_classes[best_class_idx]

                # Map the predicted class (e.g. "sad") to target moods using TagMapper
                if tag_mapper is not None:
                    mapped_mood, best_mood, best_raw_tag, confidence = tag_mapper.map_tags(
                        [best_class]
                    )
                    if mapped_mood is not None:
                        return (mapped_mood, max_score, top_predictions)
                # Fallback to direct substring matching if no tag_mapper is active
                for mood in target_moods:
                    if mood.lower() in best_class.lower():
                        return (mood, max_score, top_predictions)
                return (best_class, max_score, top_predictions)
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
