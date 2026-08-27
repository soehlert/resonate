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


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_dnb_octave_disambiguation(mock_beat_track, mock_load, mock_exists):
    """Verify Drum & Bass (87 BPM -> 174 BPM) doubles due to breakbeat convention."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (87.0, None)

    detector = BpmDetector()
    bpm_dnb = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Electronic",
        subgenres=["Drum and Bass"],
        raw_tags=["dnb", "drum and bass", "jungle"],
    )
    assert bpm_dnb == 174


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
    bpm_beach = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Rock and Roll", "Surf Rock"],
        raw_tags=["surf rock", "rock and roll"],
    )
    assert bpm_beach == 144

    # 2. Bad Religion Punk (108 BPM)
    mock_beat_track.return_value = (108.0, None)
    bpm_punk = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Punk",
        subgenres=["Punk Rock"],
        raw_tags=["punk", "hardcore punk"],
    )
    assert bpm_punk == 108

    # 3. Pixies Tenement Song (112 BPM)
    mock_beat_track.return_value = (112.0, None)
    bpm_pixies = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Alternative Rock", "Post-Punk"],
    )
    assert bpm_pixies == 112

    # 4. Slow Ballad (68 BPM)
    mock_beat_track.return_value = (68.0, None)
    bpm_ballad = detector.detect_bpm("/fake/file.mp3", genre_hint="Soul")
    assert bpm_ballad == 68



