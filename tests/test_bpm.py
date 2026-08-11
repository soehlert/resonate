"""Unit tests for BpmDetector using mocked librosa calls."""

from unittest.mock import patch

import numpy as np

from resonate.modules.bpm import BpmDetector


def test_bpm_file_not_found():
    """Verify that BpmDetector returns None if the target file does not exist."""
    detector = BpmDetector()
    assert detector.detect_bpm("/non/existent/file.mp3") is None


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
    bpm = detector.detect_bpm("/fake/file.mp3")
    assert bpm == 120

    # Test array tempo
    mock_beat_track.return_value = (np.array([135.6]), None)
    bpm = detector.detect_bpm("/fake/file.mp3")
    assert bpm == 136


@patch("os.path.exists")
@patch("librosa.load")
def test_bpm_exception_handling(mock_load, mock_exists):
    """Verify that BpmDetector catches exceptions from librosa and returns None."""
    mock_exists.return_value = True
    mock_load.side_effect = RuntimeError("Failed to decode audio file")

    detector = BpmDetector()
    bpm = detector.detect_bpm("/fake/file.mp3")
    assert bpm is None
