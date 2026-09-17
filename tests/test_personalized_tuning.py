"""Tests for PersonalizedMoodTuner centroid calculation, thresholding, and prediction."""

import numpy as np

from resonate.modules.personalized_tuning import PersonalizedMoodTuner


def test_personalized_tuning_fit_and_predict(tmp_path) -> None:
    """Verify centroid calculation, threshold calibration, and prediction."""
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


def test_personalized_tuning_small_cluster_and_contrastive(tmp_path) -> None:
    """Verify small cluster (< 3 anchors) threshold fallback and contrastive margin elevation."""
    tuner = PersonalizedMoodTuner(model_path=str(tmp_path / "contrastive.json"))

    np.random.seed(99)
    base_v = np.random.randn(1280).astype(np.float32)
    base_v /= np.linalg.norm(base_v)

    # MoodA has 2 anchors (triggers len < 3 branch)
    v_a1 = base_v
    v_a2 = base_v + np.random.randn(1280).astype(np.float32) * 0.001
    v_a2 /= np.linalg.norm(v_a2)

    # MoodB is intentionally positioned close to MoodA (high inter-cluster similarity)
    v_b1 = base_v + np.random.randn(1280).astype(np.float32) * 0.01
    v_b1 /= np.linalg.norm(v_b1)

    tuner.fit({"MoodA": [v_a1, v_a2], "MoodB": [v_b1]})
    assert tuner.is_trained

    # Inter-cluster similarity between MoodA and MoodB is high (~0.9+)
    # Tuner must elevate threshold to maintain contrastive margin
    thresh_a = tuner.mood_heads["MoodA"]["threshold"]
    inter_sim = float(
        np.dot(
            tuner.mood_heads["MoodA"]["centroid"],
            tuner.mood_heads["MoodB"]["centroid"],
        )
    )
    assert thresh_a > inter_sim or thresh_a >= 0.90

