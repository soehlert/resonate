"""BPM detector module using librosa for audio analysis."""

import logging
import os

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class BpmDetector:
    """Detect BPM (tempo) of audio files using librosa."""

    def __init__(self) -> None:
        """Initialize BpmDetector."""
        pass

    def detect_bpm(
        self,
        file_path: str,
        genre_hint: str | None = None,
        subgenres: list[str] | None = None,
        raw_tags: list[str] | None = None,
        audio_predictions: list[tuple[str, float]] | None = None,
    ) -> int | None:
        """Estimate BPM from audio file with universal tempo octave disambiguation."""
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found for BPM detection: {file_path}")
            return None

        high_tempo_genres = {
            "punk",
            "punk rock",
            "hardcore punk",
            "pop-punk",
            "skate punk",
            "thrash metal",
            "speed metal",
            "grindcore",
            "drum and bass",
            "dnb",
            "jungle",
            "breakcore",
            "bebop",
            "hard bop",
            "rockabilly",
            "surf rock",
            "surf",
            "bluegrass",
        }
        is_high_tempo_genre = False
        if genre_hint and genre_hint.lower() in high_tempo_genres:
            is_high_tempo_genre = True
        if subgenres and any(sg.lower() in high_tempo_genres for sg in subgenres):
            is_high_tempo_genre = True
        if raw_tags and any(
            any(ht in t.lower() for ht in high_tempo_genres) for t in raw_tags
        ):
            is_high_tempo_genre = True

        is_high_energy = False
        is_calm_relaxing = False
        if audio_predictions:
            for pred_lbl, pred_score in audio_predictions:
                lbl_l = pred_lbl.lower()
                high_energy_labels = {"energetic", "action", "upbeat", "sport", "party", "lively"}
                if lbl_l in high_energy_labels and pred_score >= 0.08:
                    is_high_energy = True
                elif lbl_l in {"relaxing", "calm", "meditative", "soft"} and pred_score >= 0.15:
                    is_calm_relaxing = True

        # Primary: Use Essentia RhythmExtractor2013 for state-of-the-art MIR tempo detection
        try:
            import essentia.standard as es

            audio = es.MonoLoader(filename=file_path, sampleRate=44100)()
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, confidence, estimates, _ = rhythm_extractor(audio)
            if bpm and bpm > 0:
                final_bpm = float(bpm)
                double_bpm = final_bpm * 2
                if double_bpm <= 300 and not is_calm_relaxing:
                    has_double_estimate = False
                    if estimates is not None and hasattr(estimates, "__iter__"):
                        for est in estimates:
                            if isinstance(est, (int, float)) and abs(est - double_bpm) <= 8.0:
                                has_double_estimate = True
                                break

                    # Disambiguate if verified by double harmonic peak or high-tempo genre
                    if has_double_estimate and (is_high_tempo_genre or is_high_energy):
                        final_bpm = double_bpm
                    elif (
                        is_high_tempo_genre
                        and 70 <= final_bpm <= 115
                        and 140 <= double_bpm <= 260
                    ):
                        final_bpm = double_bpm

                return int(round(final_bpm))
        except Exception as es_err:
            logger.debug(
                f"Essentia RhythmExtractor unavailable/failed, falling back to librosa: {es_err}"
            )

        # Fallback: Librosa beat tracking
        try:
            # Load the first 60 seconds at 22050 Hz (default for librosa analysis)
            y, sr = librosa.load(file_path, sr=22050, duration=60)

            # Estimate tempo (BPM)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Handle librosa return formats (which can vary slightly by version)
            if isinstance(tempo, np.ndarray):
                if tempo.size > 0:
                    tempo_val = float(tempo[0])
                else:
                    tempo_val = 0.0
            else:
                tempo_val = float(tempo)

            if tempo_val > 0:
                final_bpm = float(tempo_val)
                double_bpm = final_bpm * 2
                if (
                    is_high_tempo_genre
                    and not is_calm_relaxing
                    and 70 <= final_bpm <= 115
                    and 140 <= double_bpm <= 260
                ):
                    final_bpm = double_bpm
                return int(round(final_bpm))
            return None
        except Exception as err:
            logger.warning(f"Failed to estimate BPM for '{file_path}': {err}")
            return None
