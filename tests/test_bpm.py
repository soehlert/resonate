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
def test_bpm_retain_measured_tempo(mock_beat_track, mock_load, mock_exists):
    """Verify tracks retain their exact measured tempo without artificial doubling."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    detector = BpmDetector()

    # 1. Fast tempo track (144 BPM)
    mock_beat_track.return_value = (143.6, None)
    bpm_fast, _ = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Rock and Roll", "Surf Rock"],
        raw_tags=["surf rock", "rock and roll"],
    )
    assert bpm_fast == 144

    # 2. Mid tempo track (108 BPM)
    mock_beat_track.return_value = (108.0, None)
    bpm_mid, _ = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Punk",
        subgenres=["Punk Rock"],
        raw_tags=["punk", "hardcore punk"],
    )
    assert bpm_mid == 108

    # 3. Moderate tempo track (112 BPM)
    mock_beat_track.return_value = (112.0, None)
    bpm_mod, _ = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Alternative Rock", "Post-Punk"],
    )
    assert bpm_mod == 112

    # 4. Slow tempo track (68 BPM)
    mock_beat_track.return_value = (68.0, None)
    bpm_slow, _ = detector.detect_bpm("/fake/file.mp3", genre_hint="Soul")
    assert bpm_slow == 68


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
    with patch.dict("sys.modules", _mock_essentia(167.0)):
        bpm, candidates = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer)
        assert bpm == 167
        assert any(
            "Harmonic octave resolution: promoted 84 BPM to double-time 167 BPM" in e.message
            for e in tracer.events
        )

    # Case 2: Uncorroborated double-time candidate (Essentia absent/differing retains 84 BPM)
    mock_extract.return_value = (
        84,
        [
            BpmCandidate(bpm=84, strength=1.0),
            BpmCandidate(bpm=168, strength=0.95),
        ],
    )
    detector_weak = BpmDetector()
    tracer_weak = DecisionTracer()
    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        bpm_weak, _ = detector_weak.detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_weak
        )
        assert bpm_weak == 84
        assert not any("Harmonic octave resolution" in e.message for e in tracer_weak.events)


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_configurable_octave_resolution(mock_extract, mock_exists):
    """Verify BpmDetector respects octave_resolution toggle and requires Essentia consensus."""
    mock_exists.return_value = True
    cands = [
        BpmCandidate(bpm=84, strength=1.0),
        BpmCandidate(bpm=167, strength=0.98),
    ]
    mock_extract.return_value = (84, cands)

    # 1. Disabled octave resolution retains base 84 BPM even if Essentia corroborates 167
    tracer_disabled = DecisionTracer()
    with patch.dict("sys.modules", _mock_essentia(167.0)):
        detector_disabled = BpmDetector(config=BpmConfig(octave_resolution=False))
        bpm_dis, _ = detector_disabled.detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_disabled
        )
        assert bpm_dis == 84
        assert not any("Harmonic octave resolution" in e.message for e in tracer_disabled.events)

    # 2. Uncorroborated candidate: Essentia absent/differing retains #1 candidate 84 BPM
    tracer_uncorroborated = DecisionTracer()
    with patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}):
        detector_active = BpmDetector(config=BpmConfig(octave_resolution=True))
        bpm_uncorroborated, _ = detector_active.detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_uncorroborated
        )
        assert bpm_uncorroborated == 84
        assert not any(
            "Harmonic octave resolution" in e.message for e in tracer_uncorroborated.events
        )


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

    with patch.dict("sys.modules", _mock_essentia(215.0)):
        # 1. Ceiling of 190 BPM blocks promotion to 215 BPM even when Essentia votes 215
        tracer_default = DecisionTracer()
        detector_capped = BpmDetector(config=BpmConfig(max_promoted_bpm=190))
        bpm_capped, _ = detector_capped.detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_default
        )
        assert bpm_capped == 107
        assert not any("promoted" in e.message for e in tracer_default.events)

        # 2. Custom ceiling of 220 BPM allows promotion to 215 BPM
        tracer_raised = DecisionTracer()
        detector_raised = BpmDetector(config=BpmConfig(max_promoted_bpm=220))
        bpm_raised, _ = detector_raised.detect_bpm(
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


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_consensus_accepts_corroborated_half_time_octave(mock_extract, mock_exists):
    """Verify Essentia can corroborate a half-time octave candidate."""
    mock_exists.return_value = True
    cands = [
        BpmCandidate(bpm=215, strength=1.0),
        BpmCandidate(bpm=108, strength=0.98),
        BpmCandidate(bpm=72, strength=0.95),
    ]
    mock_extract.return_value = (215, cands)

    detector = BpmDetector()
    tracer = DecisionTracer()

    with patch.dict("sys.modules", _mock_essentia(107.0)):
        bpm, _ = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer)
        assert bpm == 108
        expected_msg = (
            "Harmonic octave resolution: resolved 215 BPM to half-time 108 BPM "
            "(corroborated by Essentia 107 BPM)"
        )
        assert any(expected_msg in e.message for e in tracer.events)
        assert not any("rejected: unaligned polyrhythm" in e.message for e in tracer.events)


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


@patch("librosa.onset.onset_strength")
@patch("librosa.feature.tempogram")
@patch("librosa.tempo_frequencies")
@patch("librosa.beat.beat_track")
def test_extract_librosa_candidates_caps_at_max_bpm(
    mock_beat_track, mock_tempo_freqs, mock_tempogram, mock_onset
):
    """Verify tempogram candidate extraction ignores frequencies above max_bpm."""
    from resonate.modules.bpm import _extract_librosa_candidates

    mock_onset.return_value = np.array([1.0] * 60)
    mock_beat_track.return_value = (None, None)

    # 3 BPM frequencies: 75.0, 112.0, 225.0
    mock_tempo_freqs.return_value = np.array([75.0, 112.0, 225.0])

    # 3 bins, 60 frames. 225 has raw max power 1.0, 112 has power 0.8
    tg = np.zeros((3, 60), dtype=float)
    tg[1, :] = 0.8  # 112 BPM
    tg[2, :] = 1.0  # 225 BPM
    mock_tempogram.return_value = tg

    # With default max_bpm=200, 225 BPM is excluded and 112 BPM is selected
    chosen, candidates = _extract_librosa_candidates(np.zeros(100), 22050, max_bpm=200.0)
    assert chosen == 112
    assert all(c.bpm <= 200 for c in candidates)


@patch("os.path.exists")
@patch("resonate.modules.bpm._extract_librosa_candidates")
def test_bpm_half_time_resolution_only_triggers_above_max_promoted_bpm(mock_extract, mock_exists):
    """Verify half-time demotion only triggers when the initial tempo exceeds max_promoted_bpm."""
    mock_exists.return_value = True

    # 1. Driving rock tempo at 170 BPM <= 190 should NOT be halved to 85 BPM even if Essentia agrees
    cands_rock = [
        BpmCandidate(bpm=170, strength=1.0),
        BpmCandidate(bpm=85, strength=0.95),
    ]
    mock_extract.return_value = (170, cands_rock)
    detector = BpmDetector(config=BpmConfig(max_promoted_bpm=190))
    tracer_rock = DecisionTracer()

    with patch.dict("sys.modules", _mock_essentia(85.0)):
        bpm_rock, _ = detector.detect_bpm("/fake/file.mp3", audio=np.zeros(100), tracer=tracer_rock)
        assert bpm_rock == 170
        assert not any("half-time" in e.message for e in tracer_rock.events)

    # 2. Runaway tempo at 199 BPM > 190 SHOULD be halved to 101 BPM when corroborated by Essentia
    cands_ballad = [
        BpmCandidate(bpm=199, strength=1.0),
        BpmCandidate(bpm=101, strength=0.99),
    ]
    mock_extract.return_value = (199, cands_ballad)
    detector_ballad = BpmDetector(config=BpmConfig(max_promoted_bpm=190))
    tracer_ballad = DecisionTracer()

    with patch.dict("sys.modules", _mock_essentia(101.0)):
        bpm_ballad, _ = detector_ballad.detect_bpm(
            "/fake/file.mp3", audio=np.zeros(100), tracer=tracer_ballad
        )
        assert bpm_ballad == 101
        assert any(
            "resolved 199 BPM to half-time 101 BPM" in e.message for e in tracer_ballad.events
        )

