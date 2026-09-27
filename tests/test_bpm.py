"""Unit tests for BpmDetector using mocked and synthetic librosa calls."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from resonate.config import BpmConfig
from resonate.engine.tracer import DecisionTracer
from resonate.models import BpmCandidate, TraceAction
from resonate.modules.bpm import BpmDetector


def _mock_essentia(bpm: float) -> dict[str, Any]:
    """Helper to mock Essentia RhythmExtractor2013 returning specified BPM."""
    mock_extractor = MagicMock(return_value=(bpm, None, None, None, None))
    mock_es = MagicMock()
    mock_es.RhythmExtractor2013.return_value = mock_extractor
    mock_pkg = MagicMock()
    mock_pkg.standard = mock_es
    return {"essentia": mock_pkg, "essentia.standard": mock_es}


def test_bpm_file_not_found():
    """Verify that BpmDetector returns None, [] if the target file does not exist."""
    detector = BpmDetector()
    bpm, candidates = detector.detect_bpm("/non/existent/file.mp3")
    assert bpm is None
    assert candidates == []


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_happy_path(mock_beat_track, mock_load, mock_exists):
    """Verify that BpmDetector correctly estimates BPM under happy path."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (120.4, None)

    detector = BpmDetector()
    # Test scalar tempo
    bpm, candidates = detector.detect_bpm("/fake/file.mp3")
    assert bpm == 120
    assert len(candidates) == 1
    assert candidates[0].bpm == 120

    # Test array tempo
    mock_beat_track.return_value = (np.array([135.6]), None)
    bpm, candidates = detector.detect_bpm("/fake/file.mp3")
    assert bpm == 136
    assert len(candidates) == 1
    assert candidates[0].bpm == 136


@patch("os.path.exists")
@patch("librosa.load")
def test_bpm_exception_handling(mock_load, mock_exists):
    """Verify that BpmDetector catches exceptions from librosa and returns None, []."""
    mock_exists.return_value = True
    mock_load.side_effect = RuntimeError("Failed to decode audio file")

    detector = BpmDetector()
    bpm, candidates = detector.detect_bpm("/fake/file.mp3")
    assert bpm is None
    assert candidates == []


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_no_genre_octave_distortion(mock_beat_track, mock_load, mock_exists):
    """Verify BPM detection does not apply artificial genre octave doubling or rules."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (87.0, None)

    detector = BpmDetector()
    bpm_dnb, candidates = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Electronic",
        subgenres=["Drum and Bass"],
        raw_tags=["dnb", "drum and bass", "jungle"],
    )
    assert bpm_dnb == 87
    assert len(candidates) >= 1


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_rock_and_punk_retain_measured_tempo(mock_beat_track, mock_load, mock_exists):
    """Verify Rock, Punk, Surf, and Ballads retain their exact measured tempo."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    detector = BpmDetector()

    # 1. Beach Boys (144 BPM)
    mock_beat_track.return_value = (143.6, None)
    bpm_beach, _ = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Rock and Roll", "Surf Rock"],
        raw_tags=["surf rock", "rock and roll"],
    )
    assert bpm_beach == 144

    # 2. Bad Religion Punk (108 BPM)
    mock_beat_track.return_value = (108.0, None)
    bpm_punk, _ = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Punk",
        subgenres=["Punk Rock"],
        raw_tags=["punk", "hardcore punk"],
    )
    assert bpm_punk == 108

    # 3. Pixies Tenement Song (112 BPM)
    mock_beat_track.return_value = (112.0, None)
    bpm_pixies, _ = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Alternative Rock", "Post-Punk"],
    )
    assert bpm_pixies == 112

    # 4. Slow Ballad (68 BPM)
    mock_beat_track.return_value = (68.0, None)
    bpm_ballad, _ = detector.detect_bpm("/fake/file.mp3", genre_hint="Soul")
    assert bpm_ballad == 68


@patch("os.path.exists")
def test_bpm_with_preloaded_audio_buffer_essentia(mock_exists):
    """Verify BpmDetector passes preloaded audio array to Essentia RhythmExtractor."""
    mock_exists.return_value = True
    detector = BpmDetector()
    dummy_audio = np.zeros(44100 * 2, dtype=np.float32)

    with patch.dict("sys.modules", _mock_essentia(128.0)):
        bpm, candidates = detector.detect_bpm("/fake/file.mp3", audio=dummy_audio)
        assert bpm == 128
        assert len(candidates) >= 1
        assert any(c.bpm == 128 for c in candidates)


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_with_preloaded_audio_buffer_librosa(mock_beat_track, mock_load, mock_exists):
    """Verify BpmDetector passes preloaded audio array to librosa when Essentia is unavailable."""
    mock_exists.return_value = True
    mock_beat_track.return_value = (128.0, None)

    detector = BpmDetector()
    dummy_audio = np.zeros(44100 * 2, dtype=np.float32)

    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        bpm, candidates = detector.detect_bpm("/fake/file.mp3", audio=dummy_audio)
        assert bpm == 128
        assert len(candidates) >= 1
        assert mock_load.call_count == 0
        mock_beat_track.assert_called_once()
        assert np.array_equal(mock_beat_track.call_args[1]["y"], dummy_audio)
        assert mock_beat_track.call_args[1]["sr"] == 44100


@patch("os.path.exists")
def test_bpm_candidate_extraction_and_tracer(mock_exists):
    """Verify BpmDetector extracts multiple candidates and logs to DecisionTracer."""
    mock_exists.return_value = True
    detector = BpmDetector()
    sr = 22050
    duration = 10
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    synthetic_audio = np.zeros_like(t)
    # Synthetic beat at 120 BPM
    for bt in np.arange(0, duration, 60.0 / 120):
        idx = int(bt * sr)
        if idx < len(synthetic_audio):
            synthetic_audio[idx : min(idx + 200, len(synthetic_audio))] += 1.0

    tracer = DecisionTracer()
    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        bpm, candidates = detector.detect_bpm(
            "/fake/file.mp3",
            audio=synthetic_audio,
            tracer=tracer,
        )
        assert bpm is not None
        assert len(candidates) > 0
        # Check tracer logged candidates and selection
        trace_messages = [e.message for e in tracer.events]
        assert any("BPM candidates evaluated:" in m for m in trace_messages)
        assert any(
            e.action == TraceAction.ACCEPT and "BPM Selector" in e.message for e in tracer.events
        )


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_harmonic_octave_resolution(mock_extract, mock_exists):
    """Verify strong double-time octave candidate promotes BPM and logs trace."""
    from resonate.models import BpmCandidate

    mock_exists.return_value = True
    detector = BpmDetector()

    # Case 1: Strong double-time candidate (All My Life: 84 @ 1.0, 167 @ 0.91)
    mock_extract.return_value = (
        84,
        [
            BpmCandidate(bpm=84, strength=1.0),
            BpmCandidate(bpm=167, strength=0.91),
            BpmCandidate(bpm=42, strength=0.84),
        ],
    )
    tracer = DecisionTracer()
    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        bpm, candidates = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer)
        assert bpm == 167
        assert any(
            "Harmonic octave resolution: promoted 84 BPM to double-time 167 BPM" in e.message
            for e in tracer.events
        )

    # Case 2: Weak double-time candidate (retains 84 BPM)
    mock_extract.return_value = (
        84,
        [
            BpmCandidate(bpm=84, strength=1.0),
            BpmCandidate(bpm=168, strength=0.50),
        ],
    )
    tracer_weak = DecisionTracer()
    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        bpm_weak, _ = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer_weak)
        assert bpm_weak == 84
        assert not any("Harmonic octave resolution" in e.message for e in tracer_weak.events)


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_configurable_octave_resolution(mock_extract, mock_exists):
    """Verify BpmDetector respects custom BpmConfig thresholds and toggles."""
    mock_exists.return_value = True
    cands = [
        BpmCandidate(bpm=84, strength=1.0),
        BpmCandidate(bpm=167, strength=0.91),
    ]
    mock_extract.return_value = (84, cands)

    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        # 1. Disabled octave resolution retains base 84 BPM
        tracer_disabled = DecisionTracer()
        bpm_dis, _ = BpmDetector(config=BpmConfig(octave_resolution=False)).detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_disabled
        )
        assert bpm_dis == 84
        assert not any("Harmonic octave resolution" in e.message for e in tracer_disabled.events)

        # 2. Higher min_strength (0.95) does not promote 0.91
        tracer_high = DecisionTracer()
        bpm_high, _ = BpmDetector(config=BpmConfig(octave_min_strength=0.95)).detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_high
        )
        assert bpm_high == 84
        assert not any("Harmonic octave resolution" in e.message for e in tracer_high.events)


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_max_promoted_bpm_ceiling(mock_extract, mock_exists):
    """Verify harmonic octave promotion respects max_promoted_bpm ceiling."""
    mock_exists.return_value = True
    cands = [
        BpmCandidate(bpm=107, strength=1.0),
        BpmCandidate(bpm=215, strength=0.95),
    ]
    mock_extract.return_value = (107, cands)

    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        # 1. Default ceiling of 190 BPM blocks promotion to 215 BPM
        tracer_default = DecisionTracer()
        bpm_capped, _ = BpmDetector(config=BpmConfig(max_promoted_bpm=190)).detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_default
        )
        assert bpm_capped == 107
        assert not any("promoted" in e.message for e in tracer_default.events)

        # 2. Custom ceiling of 220 BPM allows promotion to 215 BPM
        tracer_raised = DecisionTracer()
        bpm_raised, _ = BpmDetector(config=BpmConfig(max_promoted_bpm=220)).detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_raised
        )
        assert bpm_raised == 215
        assert any(
            "promoted 107 BPM to double-time 215 BPM" in e.message for e in tracer_raised.events
        )


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_consensus_rejects_polyrhythm_override(mock_extract, mock_exists):
    """Verify consensus voting rejects Essentia polyrhythms that contradict fundamental."""
    mock_exists.return_value = True
    cands = [
        BpmCandidate(bpm=72, strength=1.0),
        BpmCandidate(bpm=215, strength=0.95),
        BpmCandidate(bpm=107, strength=0.91),
    ]
    mock_extract.return_value = (72, cands)

    detector = BpmDetector()
    tracer = DecisionTracer()

    with patch.dict("sys.modules", _mock_essentia(107.0)):
        bpm, _ = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer)
        assert bpm == 72
        assert any("rejected: unaligned polyrhythm" in e.message for e in tracer.events)


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_consensus_accepts_corroborated_octave(mock_extract, mock_exists):
    """Verify Essentia can corroborate an octave candidate with moderate Librosa strength."""
    mock_exists.return_value = True
    cands = [
        BpmCandidate(bpm=85, strength=1.0),
        BpmCandidate(bpm=172, strength=0.61),
    ]
    mock_extract.return_value = (85, cands)

    detector = BpmDetector()
    tracer = DecisionTracer()

    with patch.dict("sys.modules", _mock_essentia(172.0)):
        bpm, _ = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer)
        assert bpm == 172
        assert any("corroborated by Essentia 172 BPM" in e.message for e in tracer.events)


@patch("librosa.onset.onset_strength")
@patch("librosa.feature.tempogram")
@patch("librosa.tempo_frequencies")
@patch("librosa.beat.beat_track")
def test_extract_librosa_candidates_temporal_consensus(
    mock_beat_track, mock_tempo_freqs, mock_tempogram, mock_onset
):
    """Verify temporal segment consensus promotes tempo with majority segment votes."""
    from resonate.modules.bpm import _extract_librosa_candidates

    mock_onset.return_value = np.array([1.0] * 60)
    mock_beat_track.return_value = (None, None)

    # 3 BPM bins: [84.0, 111.0, 168.0]
    sub_bpms = np.array([84.0, 111.0, 168.0])
    mock_tempo_freqs.return_value = sub_bpms

    # Tempogram: 3 bins, 60 frames (20 frames per third)
    # Segment 1 (0..20): peaks at 84 (bin 0)
    # Segment 2 (20..40): peaks at 168 (bin 2)
    # Segment 3 (40..60): peaks at 168 (bin 2)
    tg = np.zeros((3, 60), dtype=float)
    tg[0, :20] = 1.0  # Segment 1 peaks at bin 0 (84 BPM)
    tg[2, 20:] = 1.0  # Segments 2 & 3 peak at bin 2 (168 BPM)
    mock_tempogram.return_value = tg

    chosen, candidates = _extract_librosa_candidates(np.zeros(100), 22050)
    assert chosen == 168
    assert candidates[0].bpm == 168
    assert candidates[0].strength == 1.0

