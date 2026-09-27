"""Unit tests for BpmDetector using mocked and synthetic librosa calls."""

from unittest.mock import MagicMock, patch

import numpy as np

from resonate.engine.tracer import DecisionTracer
from resonate.models import TraceAction
from resonate.modules.bpm import BpmDetector


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
def test_bpm_dnb_octave_disambiguation(mock_beat_track, mock_load, mock_exists):
    """Verify Drum & Bass (87 BPM -> 174 BPM) doubles due to breakbeat convention."""
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
    assert bpm_dnb == 174
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

    mock_extractor = MagicMock(return_value=(128.0, None, None, [64.0], None))
    mock_es = MagicMock()
    mock_es.RhythmExtractor2013.return_value = mock_extractor
    mock_essentia_pkg = MagicMock()
    mock_essentia_pkg.standard = mock_es
    with patch.dict("sys.modules", {"essentia": mock_essentia_pkg, "essentia.standard": mock_es}):
        bpm, candidates = detector.detect_bpm("/fake/file.mp3", audio=dummy_audio)
        assert bpm == 128
        assert len(candidates) == 2
        assert candidates[0].bpm == 128
        assert candidates[1].bpm == 64
        mock_extractor.assert_called_once()
        assert np.array_equal(mock_extractor.call_args[0][0], dummy_audio)


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
