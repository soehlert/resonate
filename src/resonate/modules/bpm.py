"""BPM detector module using Essentia RhythmExtractor2013 and Librosa for audio analysis."""

import logging
import os

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Fast breakbeat electronic genres where musical definition is ~160-180 BPM
DOUBLE_TIME_GENRES: set[str] = {
    "drum and bass",
    "dnb",
    "jungle",
    "breakcore",
}


class BpmDetector:
    """Detect BPM (tempo) of audio files using Essentia RhythmExtractor2013 and librosa."""

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
        """Estimate BPM directly from audio file using Essentia MIR rhythm analysis."""
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found for BPM detection: {file_path}")
            return None

        # Check for confirmed Drum & Bass / Jungle half-time breakbeat convention
        is_dnb = False
        if genre_hint and genre_hint.strip().lower() in DOUBLE_TIME_GENRES:
            is_dnb = True
        if subgenres and any(sg.strip().lower() in DOUBLE_TIME_GENRES for sg in subgenres):
            is_dnb = True
        if raw_tags and any(t.strip().lower() in DOUBLE_TIME_GENRES for t in raw_tags):
            is_dnb = True

        # Primary: Use Essentia RhythmExtractor2013 for state-of-the-art MIR tempo detection
        try:
            import essentia.standard as es

            audio = es.MonoLoader(filename=file_path, sampleRate=44100)()
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, _, _, _ = rhythm_extractor(audio)
            if bpm and bpm > 0:
                final_bpm = float(bpm)
                # DnB / Jungle produced at 160-180 BPM where beat trackers detect 80-90 BPM downbeat
                if is_dnb and final_bpm < 100.0:
                    final_bpm *= 2
                return int(round(final_bpm))
        except Exception as es_err:
            logger.debug(
                f"Essentia RhythmExtractor unavailable/failed, falling back to librosa: {es_err}"
            )

        # Fallback: Librosa beat tracking
        try:
            y, sr = librosa.load(file_path, sr=22050, duration=60)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            if isinstance(tempo, np.ndarray):
                tempo_val = float(tempo[0]) if tempo.size > 0 else 0.0
            else:
                tempo_val = float(tempo)

            if tempo_val > 0:
                final_bpm = float(tempo_val)
                if is_dnb and final_bpm < 100.0:
                    final_bpm *= 2
                return int(round(final_bpm))
            return None
        except Exception as err:
            logger.warning(f"Failed to estimate BPM for '{file_path}': {err}")
            return None

