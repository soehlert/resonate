"""BPM detector module using Essentia RhythmExtractor2013 and Librosa for audio analysis."""

import logging
import os
from typing import Any

import librosa
import numpy as np

from resonate.models import BpmCandidate
from resonate.utils.audio import calculate_audio_window

logger = logging.getLogger(__name__)

DOUBLE_TIME_GENRES: set[str] = {
    "drum and bass",
    "dnb",
    "jungle",
    "breakcore",
}


def _extract_librosa_candidates(
    y: np.ndarray, sr: float
) -> tuple[int | None, list[BpmCandidate]]:
    """Extract candidate BPMs and normalized strengths using Librosa tempogram."""
    hop_length = 512
    try:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, aggregate=np.median
        )
    except Exception as exc:
        logger.debug(f"Onset envelope extraction failed: {exc}")
        return None, []

    if not onset_env.any():
        tempo_global, _ = librosa.beat.beat_track(y=y, sr=sr)
        if tempo_global is not None:
            g_bpm = int(round(float(np.atleast_1d(tempo_global)[0])))
            return g_bpm, [BpmCandidate(bpm=g_bpm, strength=1.0)] if g_bpm > 0 else []
        return None, []

    win_length = librosa.time_to_frames(8.0, sr=sr, hop_length=hop_length).item()
    tg = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=win_length
    )
    bpms = librosa.tempo_frequencies(win_length, hop_length=hop_length, sr=sr)
    valid_mask = (bpms >= 35.0) & (bpms <= 260.0)
    sub_bpms = bpms[valid_mask]
    sub_tg = np.mean(tg, axis=-1)[valid_mask]

    max_val = np.max(sub_tg)
    if max_val <= 0:
        return None, []

    norm_tg = sub_tg / max_val
    raw_cands: list[tuple[int, float]] = []
    for i in range(1, len(norm_tg) - 1):
        if norm_tg[i] > norm_tg[i - 1] and norm_tg[i] > norm_tg[i + 1] and norm_tg[i] >= 0.15:
            raw_cands.append((int(round(sub_bpms[i])), float(norm_tg[i])))

    tempo_global, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if tempo_global is not None:
        g_bpm = int(round(float(np.atleast_1d(tempo_global)[0])))
        if g_bpm > 0 and not any(abs(g_bpm - c[0]) <= 3 for c in raw_cands):
            raw_cands.append((g_bpm, 0.90))

    deduped: list[BpmCandidate] = []
    for b_val, s_val in sorted(raw_cands, key=lambda x: x[1], reverse=True):
        if not any(abs(b_val - c.bpm) <= 3 for c in deduped):
            deduped.append(BpmCandidate(bpm=b_val, strength=round(s_val, 2)))

    top = deduped[:5]
    chosen = top[0].bpm if top else None
    return chosen, top


class BpmDetector:
    """Detect BPM (tempo) of audio files using Essentia RhythmExtractor2013 and librosa."""

    def __init__(self) -> None:
        """Initialize BpmDetector."""
        self._rhythm_extractor: Any = None

    def detect_bpm(
        self,
        file_path: str,
        genre_hint: str | None = None,
        subgenres: list[str] | None = None,
        raw_tags: list[str] | None = None,
        audio_predictions: list[tuple[str, float]] | None = None,
        audio: Any = None,
        tracer: Any = None,
    ) -> tuple[int | None, list[BpmCandidate]]:
        """Estimate BPM directly from audio file using Essentia or Librosa MIR rhythm analysis."""
        if not os.path.exists(file_path) and audio is None:
            logger.warning(f"Audio file not found for BPM detection: {file_path}")
            return None, []

        is_dnb = False
        if genre_hint and genre_hint.strip().lower() in DOUBLE_TIME_GENRES:
            is_dnb = True
        if subgenres and any(sg.strip().lower() in DOUBLE_TIME_GENRES for sg in subgenres):
            is_dnb = True
        if raw_tags and any(t.strip().lower() in DOUBLE_TIME_GENRES for t in raw_tags):
            is_dnb = True

        candidates: list[BpmCandidate] = []
        final_bpm: int | None = None

        # 1. Primary: Essentia RhythmExtractor2013
        try:
            import essentia.standard as es

            if audio is not None:
                audio_bpm = np.asarray(audio, dtype=np.float32)
            else:
                start_sec, end_sec = calculate_audio_window(file_path, target_duration=90.0)
                try:
                    audio_bpm = es.EasyLoader(
                        filename=file_path,
                        sampleRate=44100,
                        startTime=start_sec,
                        endTime=end_sec,
                    )()
                except Exception:
                    audio_bpm = es.MonoLoader(filename=file_path, sampleRate=44100)()

            if self._rhythm_extractor is None:
                self._rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, _, estimates, _ = self._rhythm_extractor(audio_bpm)
            if bpm and bpm > 0:
                final_bpm = int(round(float(bpm)))
                candidates.append(BpmCandidate(bpm=final_bpm, strength=1.0))
                if estimates is not None:
                    for est in estimates:
                        est_bpm = int(round(float(est)))
                        if est_bpm > 0 and not any(abs(est_bpm - c.bpm) <= 3 for c in candidates):
                            candidates.append(BpmCandidate(bpm=est_bpm, strength=0.85))
        except Exception as es_err:
            logger.debug(f"Essentia unavailable/failed, falling back to librosa: {es_err}")

        # 2. Fallback: Librosa beat tracking and tempogram candidate analysis
        if final_bpm is None:
            try:
                if audio is not None:
                    y, sr = np.asarray(audio, dtype=np.float32), 44100
                else:
                    start_sec, _ = calculate_audio_window(file_path, target_duration=90.0)
                    y, sr = librosa.load(file_path, sr=22050, offset=start_sec, duration=60)
                final_bpm, candidates = _extract_librosa_candidates(y=y, sr=sr)
            except Exception as err:
                logger.warning(f"Failed to estimate BPM for '{file_path}': {err}")
                return None, []

        # 3. Post-process & DecisionTracer logging
        if is_dnb and final_bpm is not None and final_bpm < 100:
            final_bpm *= 2

        if tracer is not None:
            if candidates:
                cand_str = ", ".join(
                    f"{c.bpm} BPM (strength: {c.strength:.2f})" for c in candidates
                )
                tracer.record(f"BPM candidates evaluated: {cand_str}")
            if final_bpm is not None:
                tracer.accept("BPM Selector", f"{final_bpm} BPM")

        return final_bpm, candidates
