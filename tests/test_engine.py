import pytest

from resonate.config import MoodConflictRule
from resonate.engine.mood_rules import (
    apply_bpm_mood_rules,
    get_genre_seeded_moods,
    is_mood_excluded_by_genre,
    is_valid_mood_tag,
    resolve_mood_conflicts,
    synthesize_track_moods,
)
from resonate.engine.taxonomy import (
    deduplicate_subgenres,
    is_valid_subgenre_tag,
    promote_genre_by_subgenres,
    sanitize_subgenres_for_genre,
)
from resonate.engine.tracer import DecisionTracer
from resonate.models import LyricsAnalysisResult


def test_promote_genre_by_subgenres_punk_promotion() -> None:
    """Test Rock/Pop promoted to Punk when punk child subgenres strictly outnumber parent."""
    genre, decision = promote_genre_by_subgenres(
        "Rock", ["Skate Punk", "Pop-Punk", "Hardcore Punk"]
    )
    assert genre == "Punk"
    assert decision is not None
    assert decision.original_genre == "Rock"
    assert decision.promoted_genre == "Punk"
    assert "Skate Punk" in decision.contributing_subgenres


def test_promote_genre_by_subgenres_metal_promotion() -> None:
    """Test Rock promoted to Metal when metal child subgenres strictly outnumber parent."""
    genre, decision = promote_genre_by_subgenres("Rock", ["Heavy Metal", "Thrash Metal"])
    assert genre == "Metal"
    assert decision is not None
    assert decision.original_genre == "Rock"
    assert decision.promoted_genre == "Metal"


def test_promote_genre_by_subgenres_no_promotion_when_parent_dominates() -> None:
    """Test Rock stays Rock when rock subgenres outnumber child metal/punk subgenres."""
    genre, decision = promote_genre_by_subgenres(
        "Rock", ["Alternative Rock", "Classic Rock", "Art Rock", "Heavy Metal"]
    )
    assert genre == "Rock"
    assert decision is None


def test_sanitize_subgenres_for_metal_and_punk() -> None:
    """Test accidental hip-hop subgenres stripped from Metal unless raw tags contain hip-hop."""
    cleaned = sanitize_subgenres_for_genre(
        "Metal",
        ["Heavy Metal", "Trap", "Cloud Rap", "Thrash Metal"],
        raw_tags=["metal", "heavy metal", "metallica"],
    )
    assert cleaned == ["Heavy Metal", "Thrash Metal"]


def test_sanitize_subgenres_for_classical() -> None:
    """Test rock/metal/hip-hop subgenres stripped from Classical."""
    cleaned = sanitize_subgenres_for_genre(
        "Classical",
        ["Chamber Music", "Baroque", "Heavy Metal", "Trap"],
        raw_tags=["classical", "beethoven", "symphony"],
    )
    assert cleaned == ["Chamber Music", "Baroque"]


def test_deduplicate_subgenres_and_filter_conflicts() -> None:
    """Test subgenres deduplicated, primary genre dropped, and mutual style conflicts resolved."""
    res = deduplicate_subgenres(
        primary_genre="Rock",
        subgenres=["Rock", "Alternative Rock", "alternative rock", "Soft Rock", "Hard Rock"],
    )
    # Primary "Rock" dropped, case duplicate dropped, mutual conflict Soft Rock/Hard Rock resolved
    assert "Rock" not in res
    assert len([s for s in res if s.lower() == "alternative rock"]) == 1
    # Only one of Soft Rock or Hard Rock retained
    assert not ("Soft Rock" in res and "Hard Rock" in res)


def test_bpm_mood_rules_no_veto() -> None:
    """Verify BPM does not hard-veto audio predictions on low or mid tempo tracks."""
    # Low tempo: retains Energetic and does not veto
    assert apply_bpm_mood_rules(["Energetic", "Mellow"], detected_bpm=85) == [
        "Energetic",
        "Mellow",
    ]
    # High tempo: retains moods without stripping
    assert apply_bpm_mood_rules(["Energetic", "Happy"], detected_bpm=140) == [
        "Energetic",
        "Happy",
    ]


def test_resolve_mood_conflicts_acoustic_and_heavy() -> None:
    """Test acoustic/mellow drops heavy/aggressive/rowdy."""
    res = resolve_mood_conflicts(["Acoustic", "Heavy", "Aggressive", "Mellow"])
    assert "Acoustic" in res
    assert "Mellow" in res
    assert "Heavy" not in res
    assert "Aggressive" not in res


def test_synthesize_track_moods_with_lyrics_and_bpm() -> None:
    """Test full mood synthesis pipeline combining text, audio, valence, and BPM."""
    lyrics_res = LyricsAnalysisResult(
        lyrics_text="I feel so sad and lonely in the dark",
        source="lrclib",
        valence_score=-0.65,
        mood_scores={"Dark": 0.50, "Melancholic": 0.60},
    )
    moods = synthesize_track_moods(
        text_moods=["Chill Hang", "Happy"],
        seeded_moods=get_genre_seeded_moods(["Slowcore"]),
        essentia_moods=["Melancholic"],
        essentia_top=[("melancholic", 0.45), ("sad", 0.30)],
        detected_bpm=72,
        lyrics_analysis=lyrics_res,
        primary_genre="Rock",
        subgenres=["Slowcore"],
        raw_tags=["slowcore", "sadcore", "indie"],
    )
    # Strong negative valence should knock out Happy and Chill Hang, keeping Melancholic and Dark
    assert "Happy" not in moods
    assert "Chill Hang" not in moods
    assert "Melancholic" in moods


def test_is_valid_subgenre_tag_and_mood_tag() -> None:
    """Test tag validation filters."""
    assert is_valid_subgenre_tag("90s", "Radiohead") is False
    assert is_valid_subgenre_tag("Radiohead", "Radiohead") is False
    assert is_valid_subgenre_tag("singer-songwriter", "Bob Dylan") is True

    assert is_valid_mood_tag("rock", "Radiohead") is False  # Genre keyword
    assert is_valid_mood_tag("melancholic", "Radiohead") is True
    assert is_valid_mood_tag("seen live", "Radiohead") is False  # Boilerplate


def test_synthesize_track_moods_low_bpm_retains_audio_energetic() -> None:
    """Verify audio ML Energetic prediction is preserved on a 99 BPM track."""
    moods = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=["Energetic"],
        essentia_top=[("energetic", 0.29)],
        detected_bpm=99,
        lyrics_analysis=None,
        primary_genre="Hip-Hop",
        subgenres=["Conscious Hip Hop", "Rap"],
        raw_tags=["conscious hip hop", "hip hop", "rap"],
    )
    assert "Energetic" in moods


@pytest.mark.parametrize(
    ("energetic_score", "expected_in_moods"),
    [
        (0.12, False),
        (0.14, False),
        (0.18, False),
        (0.24, False),
        (0.25, True),
        (0.35, True),
    ],
)
def test_synthesize_track_moods_energetic_confidence_threshold(
    energetic_score: float, expected_in_moods: bool
) -> None:
    """Verify Energetic requires genuine confidence (>= 0.25) and rejects acoustic loudness bias."""
    moods = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[("energetic", energetic_score), ("melodic", 0.08)],
        detected_bpm=100,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Alternative Rock"],
        raw_tags=["rock"],
    )
    if expected_in_moods:
        assert "Energetic" in moods
    else:
        assert "Energetic" not in moods


def test_synthesize_track_moods_does_not_force_three_moods() -> None:
    """Verify tracks with confident moods do not pad up to 3 using weak acoustic guesses."""
    moods = synthesize_track_moods(
        text_moods=["Chill Hang"],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[
            ("energetic", 0.14),
            ("love", 0.13),
            ("melodic", 0.07),
            ("dark", 0.05),
        ],
        detected_bpm=83,
        lyrics_analysis=None,
        primary_genre="Blues",
        subgenres=["Blues Rock"],
        raw_tags=["blues", "blues rock"],
    )
    assert moods == ["Chill Hang"]
    assert len(moods) == 1


@pytest.mark.parametrize(
    ("mood", "subgenres", "primary_genre", "raw_tags", "expected"),
    [
        ("Aggressive", ["Southern Rock"], "Rock", ["southern rock", "classic rock"], True),
        ("Aggressive", ["Southern Rock"], "Rock", ["southern rock", "aggressive rock"], False),
        ("Aggressive", ["Heavy Metal"], "Metal", ["heavy metal", "thrash metal"], False),
        ("Mellow", ["Hard Rock"], "Rock", ["hard rock", "classic rock"], True),
        ("Mellow", ["Heavy Metal"], "Rock", ["heavy metal"], True),
        ("Mellow", ["Hard Rock"], "Rock", ["hard rock", "mellow rock"], False),
        ("Mellow", ["Indie Folk"], "Folk", ["indie folk"], False),
        ("Chill Hang", ["Bebop", "Hard Bop"], "Jazz", ["jazz", "bebop"], True),
        ("Chill Hang", ["Punk Rock"], "Punk", ["punk", "rock"], True),
        ("Chill Hang", ["Thrash Metal"], "Metal", ["metal"], True),
        ("Chill Hang", ["Hard Bop"], "Jazz", ["jazz", "chill hang"], False),
        ("Chill Hang", ["Indie Folk"], "Folk", ["indie folk"], False),
    ],
)
def test_is_mood_excluded_by_genre(
    mood: str,
    subgenres: list[str],
    primary_genre: str | None,
    raw_tags: list[str],
    expected: bool,
) -> None:
    """Verify genre-based mood exclusions and explicit tag overrides."""
    assert is_mood_excluded_by_genre(mood, subgenres, primary_genre, raw_tags) is expected


@pytest.mark.parametrize(
    ("mood", "subgenres", "primary_genre", "raw_tags", "should_have_mood"),
    [
        ("Aggressive", ["Southern Rock"], "Rock", ["southern rock", "blues rock"], False),
        ("Aggressive", ["Southern Rock"], "Rock", ["southern rock", "aggressive rock"], True),
        ("Mellow", ["Hard Rock"], "Rock", ["hard rock"], False),
        ("Mellow", ["Hard Rock"], "Rock", ["hard rock", "mellow rock"], True),
        ("Chill Hang", ["Hard Bop"], "Jazz", ["jazz", "hard bop"], False),
        ("Chill Hang", ["Hard Bop"], "Jazz", ["jazz", "chill hang"], True),
        ("Chill Hang", ["Punk Rock"], "Punk", ["punk", "rock"], False),
        ("Acoustic", ["Hard Rock"], "Rock", ["hard rock"], False),
        ("Acoustic", ["Hard Rock"], "Rock", ["hard rock", "acoustic"], True),
    ],
)
def test_synthesize_track_moods_genre_exclusion(
    mood: str,
    subgenres: list[str],
    primary_genre: str,
    raw_tags: list[str],
    should_have_mood: bool,
) -> None:
    """Verify synthesize_track_moods excludes or retains candidate moods based on genre rules."""
    moods = synthesize_track_moods(
        text_moods=[mood],
        seeded_moods=[],
        essentia_moods=[mood],
        essentia_top=[(mood.lower(), 0.35)],
        detected_bpm=120,
        lyrics_analysis=None,
        primary_genre=primary_genre,
        subgenres=subgenres,
        raw_tags=raw_tags,
    )
    assert (mood in moods) is should_have_mood


def test_resolve_mood_conflicts_custom_rules() -> None:
    """Verify resolve_mood_conflicts respects caller-provided conflict rules."""
    custom_conflicts = [
        MoodConflictRule(if_present=["Party"], drop=["Melancholic"]),
    ]
    result = resolve_mood_conflicts(["Party", "Melancholic"], conflicts=custom_conflicts)
    assert result == ["Party"]


def test_synthesize_track_moods_unmapped_acoustic_labels_ignored() -> None:
    """Verify acoustic predictions for unmapped labels do not produce moods."""
    moods = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[("melodic", 0.15), ("film", 0.30)],
        detected_bpm=110,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Alternative Rock"],
        raw_tags=["rock"],
    )
    assert moods == []


def test_synthesize_track_moods_conflict_resolution() -> None:
    """Verify mutual conflict rules drop conflicting moods during synthesis."""
    moods = synthesize_track_moods(
        text_moods=["Chill Hang"],
        seeded_moods=[],
        essentia_moods=["Rowdy"],
        essentia_top=[("rowdy", 0.30)],
        detected_bpm=150,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Alternative Rock"],
        raw_tags=["alternative rock"],
    )
    assert "Chill Hang" not in moods
    assert "Rowdy" in moods


def test_synthesize_track_moods_records_decision_trace() -> None:
    """Verify synthesize_track_moods populates decision_trace with exclusions and conflict drops."""
    trace: list[str] = []
    moods = synthesize_track_moods(
        text_moods=["Melodic"],
        seeded_moods=["Heavy"],
        essentia_moods=[],
        essentia_top=[("acoustic", 0.30), ("love", 0.14)],
        detected_bpm=110,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Hard Rock"],
        raw_tags=["hard rock", "classic rock"],
        decision_trace=trace,
    )
    # Acoustic excluded by Hard Rock genre rule
    assert "Acoustic" not in moods
    # Trace should contain Essentia love threshold skip
    assert any("love" in t.lower() and "threshold" in t.lower() for t in trace)
    # Trace should contain Acoustic genre exclusion drop
    assert any("acoustic" in t.lower() and "drop" in t.lower() for t in trace)
    # Trace should note fallback skipped because candidate moods already exist
    assert any("fallback" in t.lower() for t in trace)
    # Trace should contain final resolved moods
    assert any("final resolved moods" in t.lower() for t in trace)


def test_decision_tracer_methods() -> None:
    """Verify DecisionTracer records formatting and respects enabled flag."""
    tracer = DecisionTracer()
    tracer.record("Arbitrary message")
    tracer.accept("Provider", "Melodic", score=0.85)
    tracer.skip("Essentia", "love", "score below threshold")
    tracer.reject("Dark", "valence conflict")
    tracer.drop("Acoustic", "genre exclusion")

    assert len(tracer.messages) == 5
    assert "Arbitrary message" in tracer.messages[0]
    assert "Provider accepted: 'Melodic' (score=0.85)" in tracer.messages[1]
    assert "Essentia 'love' skipped: score below threshold" in tracer.messages[2]
    assert "Rejected 'Dark': valence conflict" in tracer.messages[3]
    assert "Dropped 'Acoustic': genre exclusion" in tracer.messages[4]

    disabled_tracer = DecisionTracer(enabled=False)
    disabled_tracer.record("Should not be added")
    disabled_tracer.accept("Provider", "Melodic")
    assert len(disabled_tracer.messages) == 0
