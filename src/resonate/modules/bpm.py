"""BPM detector module using Essentia RhythmExtractor2013 and Librosa for audio analysis."""

import logging
import os
from typing import Any

import librosa
import numpy as np

from resonate.config import BpmConfig
from resonate.models import BpmCandidate
from resonate.utils.audio import calculate_audio_window

logger = logging.getLogger(__name__)


def _extract_librosa_candidates(
    y: np.ndarray, sr: float, max_bpm: float = 200.0
) -> tuple[int | None, list[BpmCandidate]]:
    """Extract candidate BPMs and normalized strengths using Librosa tempogram."""
    hop_length = 512
    try:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, aggregate=np.mean
        )
    except Exception as exc:
        logger.debug(f"Onset envelope extraction failed: {exc}")
        return None, []

    if not onset_env.any():
        tempo_global, _ = librosa.beat.beat_track(y=y, sr=sr)
        if tempo_global is not None:
            g_bpm = int(round(float(np.atleast_1d(tempo_global)[0])))
            if 0 < g_bpm <= max_bpm:
                return g_bpm, [BpmCandidate(bpm=g_bpm, strength=1.0)]
        return None, []

    win_length = librosa.time_to_frames(8.0, sr=sr, hop_length=hop_length).item()
    tg = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=win_length
    )
    bpms = librosa.tempo_frequencies(win_length, hop_length=hop_length, sr=sr)
    valid_mask = (bpms >= 35.0) & (bpms <= max_bpm)
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
        if 0 < g_bpm <= max_bpm and not any(abs(g_bpm - c[0]) <= 3 for c in raw_cands):
            raw_cands.append((g_bpm, 0.90))

    # Temporal segment consensus: evaluate 3 segments across the tempogram time frames
    # to identify the dominant sustained tempo and reject transient intro/outro artifacts.
    n_frames = tg.shape[-1]
    if n_frames >= 30:
        n_segments = 3
        seg_len = n_frames // n_segments
        seg_peaks: list[int] = []
        for s in range(n_segments):
            start_f = s * seg_len
            end_f = (s + 1) * seg_len if s < n_segments - 1 else n_frames
            seg_slice = tg[:, start_f:end_f]
            if seg_slice.shape[-1] > 0:
                seg_profile = np.mean(seg_slice, axis=-1)[valid_mask]
                if np.max(seg_profile) > 0:
                    seg_peaks.append(int(round(sub_bpms[int(np.argmax(seg_profile))])))

        # If at least 2 segments directly agree (within +-3 BPM), prioritize that majority tempo
        direct_majority_bpm: int | None = None
        for p in seg_peaks:
            if sum(1 for other in seg_peaks if abs(p - other) <= 3) >= 2:
                direct_majority_bpm = p
                break

        if direct_majority_bpm is not None:
            if not any(abs(direct_majority_bpm - c[0]) <= 3 for c in raw_cands):
                raw_cands.append((direct_majority_bpm, 1.0))
            else:
                raw_cands = [
                    (c[0], 1.0 if abs(direct_majority_bpm - c[0]) <= 3 else c[1])
                    for c in raw_cands
                ]

    deduped: list[BpmCandidate] = []
    for b_val, s_val in sorted(raw_cands, key=lambda x: x[1], reverse=True):
        if not any(abs(b_val - c.bpm) <= 3 for c in deduped):
            deduped.append(BpmCandidate(bpm=b_val, strength=round(s_val, 2)))

    top = deduped[:5]
    chosen = top[0].bpm if top else None
    return chosen, top


class BpmDetector:
    """Detect BPM (tempo) of audio files using Essentia RhythmExtractor2013 and librosa."""

    def __init__(self, config: BpmConfig | None = None) -> None:
        """Initialize BpmDetector with optional BpmConfig."""
        self.config = config or BpmConfig()
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
        """Estimate BPM and candidate tempos of the overall beat using MIR rhythm analysis."""
        if not os.path.exists(file_path) and audio is None:
            logger.warning(f"Audio file not found for BPM detection: {file_path}")
            return None, []

        # 1. Obtain audio waveform array for MIR rhythm and candidate analysis
        y: np.ndarray | None = None
        sr: float = 44100.0

        if audio is not None:
            y = np.asarray(audio, dtype=np.float32)
            sr = 44100.0
        else:
            start_sec, end_sec = calculate_audio_window(file_path, target_duration=90.0)
            try:
                import essentia.standard as es

                try:
                    y = es.EasyLoader(
                        filename=file_path,
                        sampleRate=44100,
                        startTime=start_sec,
                        endTime=end_sec,
                    )()
                except Exception:
                    y = es.MonoLoader(filename=file_path, sampleRate=44100)()
                sr = 44100.0
            except Exception:
                try:
                    y, sr = librosa.load(file_path, sr=22050, offset=start_sec, duration=60)
                except Exception as err:
                    logger.warning(f"Failed to load audio for BPM detection '{file_path}': {err}")
                    return None, []

        if y is None or len(y) == 0:
            return None, []

        # 2. Extract candidate periodicities of the overall beat via tempogram analysis
        librosa_bpm, candidates = _extract_librosa_candidates(
            y=y, sr=sr, max_bpm=float(self.config.max_promoted_bpm)
        )

        # 3. Base candidate selection from Librosa tempogram analysis
        final_bpm: int | None = librosa_bpm

        # 4. Extract Essentia's second opinion if available
        essentia_bpm: int | None = None
        try:
            import essentia.standard as es

            if self._rhythm_extractor is None:
                self._rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, _, _, _ = self._rhythm_extractor(y)
            if bpm and bpm > 0:
                essentia_bpm = int(round(float(bpm)))
        except Exception as es_err:
            logger.debug(f"Essentia unavailable/failed: {es_err}")

        # Fallback to Essentia if Librosa produced no candidate
        if final_bpm is None or final_bpm <= 0:
            final_bpm = essentia_bpm

        # Ensure the accepted BPM is represented in the candidates list
        if final_bpm is not None and candidates:
            match_idx = next(
                (i for i, c in enumerate(candidates) if abs(c.bpm - final_bpm) <= 2),
                None,
            )
            if match_idx is not None:
                candidates[match_idx].bpm = final_bpm
            else:
                candidates.insert(0, BpmCandidate(bpm=final_bpm, strength=1.0))
        elif final_bpm is not None and not candidates:
            candidates = [BpmCandidate(bpm=final_bpm, strength=1.0)]

        # 5. Harmonic octave consensus and resolution:
        # Check if an octave candidate (~2x or ~0.5x tempo) aligns with Essentia consensus,
        # or has strong standalone Librosa acoustic correlation for double-time.
        # Reject rogue polyrhythms from Essentia (e.g. 1.5x like 111 vs 167).
        if (
            self.config.octave_resolution
            and final_bpm is not None
            and final_bpm > 0
            and candidates
        ):
            base_cand = next((c for c in candidates if c.bpm == final_bpm), None)
            base_strength = base_cand.strength if base_cand else 1.0

            # Log rejection if Essentia returned an unaligned polyrhythm against the base candidate
            if (
                tracer is not None
                and essentia_bpm is not None
                and final_bpm > 0
                and abs(essentia_bpm - final_bpm) > 3
            ):
                ratio = essentia_bpm / final_bpm
                is_aligned_octave = (
                    self.config.octave_min_ratio <= ratio <= self.config.octave_max_ratio
                    or self.config.octave_min_ratio <= (1.0 / ratio) <= self.config.octave_max_ratio
                )
                if not is_aligned_octave:
                    tracer.record(
                        f"Essentia candidate {essentia_bpm} BPM rejected: unaligned polyrhythm "
                        f"against fundamental {final_bpm} BPM"
                    )

            # Find matching octave candidate
            octave_cand: BpmCandidate | None = None
            resolution_type: str = ""
            resolution_reason: str = ""

            for c in candidates:
                is_double = (
                    self.config.octave_min_ratio
                    <= (c.bpm / final_bpm)
                    <= self.config.octave_max_ratio
                )
                is_half = (
                    self.config.octave_min_ratio
                    <= (final_bpm / c.bpm)
                    <= self.config.octave_max_ratio
                )

                if is_double and c.bpm <= self.config.max_promoted_bpm:
                    if essentia_bpm is not None and abs(c.bpm - essentia_bpm) <= 3:
                        octave_cand = c
                        resolution_type = "promoted"
                        resolution_reason = f"corroborated by Essentia {essentia_bpm} BPM"
                        break
                    if (
                        c.strength >= self.config.octave_min_strength
                        and c.strength >= (base_strength - self.config.octave_max_delta)
                    ):
                        octave_cand = c
                        resolution_type = "promoted"
                        resolution_reason = f"strength: {c.strength:.2f}"
                        break
                elif is_half and final_bpm > self.config.max_promoted_bpm:
                    if essentia_bpm is not None and abs(c.bpm - essentia_bpm) <= 3:
                        octave_cand = c
                        resolution_type = "resolved"
                        resolution_reason = f"corroborated by Essentia {essentia_bpm} BPM"
                        break

            if octave_cand is not None:
                if tracer is not None:
                    tracer.record(
                        f"Harmonic octave resolution: {resolution_type} {final_bpm} BPM to "
                        f"{'double-time' if resolution_type == 'promoted' else 'half-time'} "
                        f"{octave_cand.bpm} BPM ({resolution_reason})"
                    )
                final_bpm = octave_cand.bpm

        # 6. Log evaluated overall beat candidates and accepted BPM to DecisionTracer
        if tracer is not None:
            if candidates:
                cand_str = ", ".join(
                    f"{c.bpm} BPM (strength: {c.strength:.2f})" for c in candidates
                )
                tracer.record(f"BPM candidates evaluated: {cand_str}")
            if final_bpm is not None:
                tracer.accept("BPM Selector", f"{final_bpm} BPM")

        return final_bpm, candidates

