"""Tests for PersonalizedMoodTuner anchor head calibration, thresholding, and prediction."""

import numpy as np

from resonate.modules.personalized_tuning import PersonalizedMoodTuner


def test_personalized_tuning_fit_and_predict(tmp_path) -> None:
    """Verify anchor calibration, threshold calibration, and prediction."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "heads.json"))
    assert not tuner.is_trained

    # Generate synthetic 1280-d base vectors for two distinct moods
    np.random.seed(42)
    base_chill = np.random.randn(1280).astype(np.float32)
    base_chill /= np.linalg.norm(base_chill)

    base_energetic = np.random.randn(1280).astype(np.float32)
    base_energetic /= np.linalg.norm(base_energetic)

    # Create small clusters around each base with noise scaled for 1280-d
    chill_anchors = [
        base_chill + np.random.randn(1280).astype(np.float32) * 0.005 for _ in range(10)
    ]
    energetic_anchors = [
        base_energetic + np.random.randn(1280).astype(np.float32) * 0.005 for _ in range(10)
    ]

    tuner.fit(
        {
            "Chill Hang": chill_anchors,
            "Energetic": energetic_anchors,
        }
    )

    assert tuner.is_trained
    assert set(tuner.tuned_moods) == {"Chill Hang", "Energetic"}

    # Chill track prediction
    test_chill = base_chill + np.random.randn(1280).astype(np.float32) * 0.002
    preds = tuner.predict(test_chill)
    assert len(preds) > 0
    assert preds[0][0] == "Chill Hang"
    assert preds[0][1] >= 0.90

    # Energetic track prediction
    test_energetic = base_energetic + np.random.randn(1280).astype(np.float32) * 0.002
    preds_e = tuner.predict(test_energetic)
    assert len(preds_e) > 0
    assert preds_e[0][0] == "Energetic"
    assert preds_e[0][1] >= 0.90

    # Unrelated random vector should not trigger high-confidence match
    random_vec = np.random.randn(1280).astype(np.float32)
    random_preds = tuner.predict(random_vec)
    # Cosine of high-dim random vectors is near 0.0, far below calibrated threshold ~0.70+
    assert random_preds == []


def test_personalized_tuning_save_and_load(tmp_path) -> None:
    """Verify serialization and deserialization of tuned heads to JSON."""
    model_file = str(tmp_path / "custom_heads.json")
    tuner1 = PersonalizedMoodTuner(model_path=model_file)

    np.random.seed(123)
    vec1 = np.random.randn(1280).astype(np.float32)
    vec2 = np.random.randn(1280).astype(np.float32)

    tuner1.fit({"Soulful": [vec1, vec1], "Trippy": [vec2, vec2]})
    tuner1.save_model()

    # Load in new instance
    tuner2 = PersonalizedMoodTuner(model_path=model_file)
    assert tuner2.load_model()
    assert tuner2.is_trained
    assert set(tuner2.tuned_moods) == {"Soulful", "Trippy"}

    # Predictions match between instances
    p1 = tuner1.predict(vec1)
    p2 = tuner2.predict(vec1)
    assert p1 == p2


def test_personalized_tuning_empty_and_corrupt(tmp_path) -> None:
    """Verify tuner handles empty data and corrupted model files gracefully."""
    empty_tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "nonexistent.json"))
    assert not empty_tuner.load_model()
    assert empty_tuner.predict(np.zeros(1280)) == []

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("invalid json content")
    corrupt_tuner = PersonalizedMoodTuner(model_path=str(corrupt_file))
    assert not corrupt_tuner.load_model()
    assert not corrupt_tuner.is_trained


def test_personalized_tuning_small_cluster_calibration(tmp_path) -> None:
    """Verify small cluster (< 3 anchors) threshold fallback calibration."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "small_cluster.json"))

    np.random.seed(99)
    base_v = np.random.randn(1280).astype(np.float32)
    base_v /= np.linalg.norm(base_v)

    # MoodA has 2 anchors (triggers len < 3 branch)
    v_a1 = base_v
    v_a2 = base_v + np.random.randn(1280).astype(np.float32) * 0.001
    v_a2 /= np.linalg.norm(v_a2)

    # MoodB has 1 anchor
    v_b1 = np.random.randn(1280).astype(np.float32)
    v_b1 /= np.linalg.norm(v_b1)

    tuner.fit({"MoodA": [v_a1, v_a2], "MoodB": [v_b1]})
    assert tuner.is_trained

    # Verify thresholds are bounded and correctly calibrated
    thresh_a = tuner.mood_heads["MoodA"]["threshold"]
    thresh_b = tuner.mood_heads["MoodB"]["threshold"]
    assert 0.58 <= thresh_a <= 0.84
    assert 0.58 <= thresh_b <= 0.84
    assert tuner.mood_heads["MoodA"]["track_count"] == 2
    assert tuner.mood_heads["MoodB"]["track_count"] == 1


def test_personalized_tuning_score_all(tmp_path) -> None:
    """Verify score_all returns full diagnostics across all trained heads."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "diag.json"))

    v1 = np.ones(1280, dtype=np.float32) / (1280**0.5)
    v2 = -v1

    tuner.fit({"MoodA": [v1, v1], "MoodB": [v2, v2]})
    assert tuner.is_trained

    scores = tuner.score_all(v1)
    assert len(scores) == 2
    # MoodA should match, MoodB should not match
    mood_dict = {s[0]: s for s in scores}
    assert mood_dict["MoodA"][3] is True
    assert mood_dict["MoodA"][1] >= 0.99
    assert mood_dict["MoodB"][3] is False
    assert mood_dict["MoodB"][1] <= 0.0


def test_personalized_tuning_adaptive_k_selection(tmp_path) -> None:
    """Verify adaptive k default of 3 and custom target_k selection."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "k_test.json"))

    np.random.seed(77)
    v = np.random.randn(1280).astype(np.float32)
    v /= np.linalg.norm(v)

    tracks_25 = [v + np.random.randn(1280).astype(np.float32) * 0.001 for _ in range(25)]
    tracks_2 = [v + np.random.randn(1280).astype(np.float32) * 0.001 for _ in range(2)]
    tracks_1 = [v]

    # Default target_k = 3
    tuner.fit(
        {
            "Standard": tracks_25,
            "Small": tracks_2,
            "Single": tracks_1,
        }
    )

    assert tuner.get_k("Standard") == 3
    assert tuner.get_k("Small") == 2
    assert tuner.get_k("Single") == 1

    # Explicit target_k = 5
    tuner.fit({"Large": tracks_25}, target_k=5)
    assert tuner.get_k("Large") == 5


def test_personalized_tuning_multimodal_cluster_resolution(tmp_path) -> None:
    """Verify k-NN correctly matches a sub-style within a multimodal anchor playlist."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "multimodal.json"))

    np.random.seed(42)
    # Acoustic base vector
    v_acoustic = np.random.randn(1280).astype(np.float32)
    v_acoustic /= np.linalg.norm(v_acoustic)

    # Orthogonal alt-pop base vector
    v_alt = np.random.randn(1280).astype(np.float32)
    v_alt -= np.dot(v_alt, v_acoustic) * v_acoustic
    v_alt /= np.linalg.norm(v_alt)

    # Completely unrelated punk base vector
    v_punk = np.random.randn(1280).astype(np.float32)
    v_punk -= np.dot(v_punk, v_acoustic) * v_acoustic
    v_punk -= np.dot(v_punk, v_alt) * v_alt
    v_punk /= np.linalg.norm(v_punk)

    # Playlist has 20 acoustic tracks and 3 alt-pop tracks (23 total, k=3)
    acoustic_anchors = [
        v_acoustic + np.random.randn(1280).astype(np.float32) * 0.005 for _ in range(20)
    ]
    alt_anchors = [v_alt + np.random.randn(1280).astype(np.float32) * 0.005 for _ in range(3)]
    all_anchors = acoustic_anchors + alt_anchors

    tuner.fit({"Chill Hang": all_anchors}, target_k=3)
    assert tuner.get_k("Chill Hang") == 3

    # Test track resembling the 3 alt-pop anchors
    test_alt_track = v_alt + np.random.randn(1280).astype(np.float32) * 0.002
    test_alt_track /= np.linalg.norm(test_alt_track)

    alt_score_all = tuner.score_all(test_alt_track)
    assert len(alt_score_all) == 1
    mood, score, threshold, is_match = alt_score_all[0]
    assert mood == "Chill Hang"
    assert is_match is True
    assert score >= 0.85
    assert threshold >= 0.75

    # Test track from punk genre (unrelated)
    test_punk_track = v_punk
    punk_score_all = tuner.score_all(test_punk_track)
    assert len(punk_score_all) == 1
    _, punk_score, _, punk_match = punk_score_all[0]
    assert punk_match is False
    assert punk_score < 0.65


def test_personalized_tuning_top_neighbors_diagnostics(tmp_path) -> None:
    """Verify get_top_neighbors returns descending cosine similarities with track titles."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "diag_neighbors.json"))

    v1 = np.array([1.0, 0.0, 0.0] + [0.0] * 1277, dtype=np.float32)
    v2 = np.array([0.9, 0.1, 0.0] + [0.0] * 1277, dtype=np.float32)
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0.5, 0.5, 0.0] + [0.0] * 1277, dtype=np.float32)
    v3 /= np.linalg.norm(v3)

    tuner.fit(
        {"TestMood": [v1, v2, v3]},
        track_names={"TestMood": ["Song Alpha", "Song Beta", "Song Gamma"]},
    )

    neighbors = tuner.get_top_neighbors(v1, "TestMood", n_neighbors=3)
    assert len(neighbors) == 3
    assert neighbors[0][0] == "Song Alpha"
    assert neighbors[0][1] >= 0.99
    assert neighbors[1][0] == "Song Beta"
    assert neighbors[1][1] > neighbors[2][1]
    assert neighbors[2][0] == "Song Gamma"



