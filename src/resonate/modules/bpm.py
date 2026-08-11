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

    def detect_bpm(self, file_path: str) -> int | None:
        """Estimate BPM from the first 60 seconds of an audio file."""
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found for BPM detection: {file_path}")
            return None

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

            bpm = int(round(tempo_val))
            if bpm > 0:
                return bpm
            return None
        except Exception as err:
            logger.warning(f"Failed to estimate BPM for '{file_path}': {err}")
            return None
