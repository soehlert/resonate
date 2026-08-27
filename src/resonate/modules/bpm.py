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
    ) -> int | None:
        """Estimate BPM from audio file with tempo octave disambiguation."""
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
        }
        is_high_tempo_genre = False
        if genre_hint and genre_hint.lower() in high_tempo_genres:
            is_high_tempo_genre = True
        if subgenres and any(sg.lower() in high_tempo_genres for sg in subgenres):
            is_high_tempo_genre = True

        # Primary: Use Essentia RhythmExtractor2013 for state-of-the-art MIR tempo detection
        try:
            import essentia.standard as es

            audio = es.MonoLoader(filename=file_path, sampleRate=44100)()
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, confidence, estimates, _ = rhythm_extractor(audio)
            if bpm and bpm > 0:
                final_bpm = float(bpm)
                if is_high_tempo_genre and 70 <= final_bpm <= 105:
                    double_bpm = final_bpm * 2
                    has_double_estimate = False
                    if estimates is not None and hasattr(estimates, "__iter__"):
                        for est in estimates:
                            if isinstance(est, (int, float)) and abs(est - double_bpm) <= 6.0:
                                has_double_estimate = True
                                break
                    if has_double_estimate or 140 <= double_bpm <= 215:
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
                if is_high_tempo_genre and 70 <= final_bpm <= 105:
                    double_bpm = final_bpm * 2
                    if 140 <= double_bpm <= 215:
                        final_bpm = double_bpm
                return int(round(final_bpm))
            return None
        except Exception as err:
            logger.warning(f"Failed to estimate BPM for '{file_path}': {err}")
            return None
