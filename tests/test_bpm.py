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
def test_bpm_punk_octave_disambiguation(mock_beat_track, mock_load, mock_exists):
    """Verify that Punk half-tempo detection (92 BPM) is octave-corrected to 184 BPM."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (92.0, None)

    detector = BpmDetector()

    # Without high-tempo genre hint, remains 92 BPM
    bpm_standard = detector.detect_bpm("/fake/file.mp3", genre_hint="Rock")
    assert bpm_standard == 92

    # With Punk genre hint, octave-corrects to 184 BPM
    bpm_punk = detector.detect_bpm("/fake/file.mp3", genre_hint="Punk")
    assert bpm_punk == 184

    # With Punk Rock subgenre, octave-corrects to 184 BPM
    bpm_subgenre = detector.detect_bpm(
        "/fake/file.mp3", genre_hint="Rock", subgenres=["Punk Rock"]
    )
    assert bpm_subgenre == 184


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_rockabilly_raw_tags_octave_disambiguation(mock_beat_track, mock_load, mock_exists):
    """Verify that Rockabilly / Surf raw tags disambiguate 97 BPM to 194 BPM."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (97.0, None)

    detector = BpmDetector()
    bpm_link_wray = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Instrumental Rock", "Instrumental"],
        raw_tags=["rockabilly", "surf rock", "50s", "garage rock"],
    )
    assert bpm_link_wray == 194


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_slow_ballad_retains_slow_tempo(mock_beat_track, mock_load, mock_exists):
    """Verify that relaxing/calm ballads retain their slow tempo without doubling."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (68.0, None)

    detector = BpmDetector()
    bpm_ballad = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Surf Rock"],
        audio_predictions=[("relaxing", 0.25), ("calm", 0.18)],
    )
    assert bpm_ballad == 68


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_pixies_tenement_song_metadata_not_doubled(
    mock_beat_track, mock_load, mock_exists
):
    """Verify Pixies Tenement Song (112 BPM, Rock, Post-Punk) is not falsely doubled."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (112.0, None)

    detector = BpmDetector()
    bpm = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Rock",
        subgenres=["Alternative Rock", "Indie Rock", "Post-Punk"],
        raw_tags=[
            "american",
            "2016 single",
            "bagel",
            "alternative rock",
            "rock",
            "indie rock",
            "alternative",
            "2016",
            "alt rock",
            "indie",
            "post-punk",
            "10s",
            "2010s",
            "playful",
        ],
        audio_predictions=[
            ("love", 0.1595),
            ("energetic", 0.1401),
            ("melodic", 0.1236),
        ],
    )
    assert bpm == 112


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_substring_tags_do_not_trigger_high_tempo(
    mock_beat_track, mock_load, mock_exists
):
    """Verify substring tags (daft punk, steampunk, cyberpunk, surfing) do not trigger doubling."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)
    mock_beat_track.return_value = (95.0, None)

    detector = BpmDetector()
    bpm = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Electronic",
        subgenres=["Synthwave"],
        raw_tags=["daft punk", "steampunk", "cyberpunk", "post-punk", "surfing", "surface"],
        audio_predictions=[("energetic", 0.15)],
    )
    assert bpm == 95


@patch("os.path.exists")
@patch("librosa.load")
@patch("librosa.beat.beat_track")
def test_bpm_dnb_and_skate_punk_disambiguation(mock_beat_track, mock_load, mock_exists):
    """Verify Drum & Bass (87 BPM -> 174 BPM) and Skate Punk (92 BPM -> 184 BPM) octave double."""
    mock_exists.return_value = True
    mock_load.return_value = (np.array([0.0] * 100), 22050)

    detector = BpmDetector()

    # DnB
    mock_beat_track.return_value = (87.0, None)
    bpm_dnb = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Drum and Bass",
        raw_tags=["dnb", "drum and bass", "jungle"],
        audio_predictions=[("energetic", 0.18)],
    )
    assert bpm_dnb == 174

    # Skate Punk
    mock_beat_track.return_value = (92.0, None)
    bpm_punk = detector.detect_bpm(
        "/fake/file.mp3",
        genre_hint="Punk",
        subgenres=["Skate Punk"],
        raw_tags=["skate punk", "punk rock"],
        audio_predictions=[("energetic", 0.16)],
    )
    assert bpm_punk == 184



