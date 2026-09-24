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
    ("raw_tags", "should_have_aggressive"),
    [
        (["southern rock", "blues rock"], False),
        (["southern rock", "aggressive rock"], True),
    ],
)
def test_synthesize_track_moods_genre_exclusion(
    raw_tags: list[str],
    should_have_aggressive: bool,
) -> None:
    """Verify synthesize_track_moods excludes or retains Aggressive based on raw tags."""
    moods = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=["Aggressive", "Energetic"],
        essentia_top=[("aggressive", 0.35), ("energetic", 0.40)],
        detected_bpm=120,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Southern Rock"],
        raw_tags=raw_tags,
    )
    assert ("Aggressive" in moods) is should_have_aggressive
    assert "Energetic" in moods


def test_resolve_mood_conflicts_custom_rules() -> None:
    """Verify resolve_mood_conflicts respects caller-provided conflict rules."""
    custom_conflicts = [
        MoodConflictRule(if_present=["Party"], drop=["Melancholic"]),
    ]
    result = resolve_mood_conflicts(["Party", "Melancholic"], conflicts=custom_conflicts)
    assert result == ["Party"]


def test_synthesize_track_moods_generic_melodic_ignored_and_chill_hang_genre_exclusions() -> None:
    """Verify melodic is ignored as generic and Chill Hang is excluded for Jazz/Punk/Metal."""
    # 1. Melodic is not a mood: score 0.15 should produce no mood
    moods_melodic = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[("melodic", 0.15)],
        detected_bpm=110,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Alternative Rock"],
        raw_tags=["rock"],
    )
    assert "Chill Hang" not in moods_melodic
    assert moods_melodic == []

    # 2. Jazz track with acoustic chill prediction has Chill Hang dropped via genre exclusions
    moods_jazz = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[("chill", 0.20)],
        detected_bpm=120,
        lyrics_analysis=None,
        primary_genre="Jazz",
        subgenres=["Hard Bop", "Post-Bop", "Bebop"],
        raw_tags=["hard bop", "post-bop", "jazz", "bebop"],
    )
    assert "Chill Hang" not in moods_jazz
    assert moods_jazz == []

    # 3. Explicit raw tag overrides genre exclusion (happy path for explicit tagging)
    moods_jazz_explicit = synthesize_track_moods(
        text_moods=["Chill Hang"],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[("chill", 0.20)],
        detected_bpm=120,
        lyrics_analysis=None,
        primary_genre="Jazz",
        subgenres=["Hard Bop"],
        raw_tags=["jazz", "chill hang"],
    )
    assert "Chill Hang" in moods_jazz_explicit

    # 4. Rowdy / Heavy conflict drops Chill Hang
    moods_rowdy = synthesize_track_moods(
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
    assert "Chill Hang" not in moods_rowdy
    assert "Rowdy" in moods_rowdy


def test_synthesize_track_moods_mellow_genre_exclusion_and_retention() -> None:
    """Verify Mellow is excluded for Hard Rock/Metal but retained for valid acoustic genres."""
    # 1. Error path: Hard Rock / Metal excludes Mellow even when proposed by personalized tuning
    hard_rock_moods = synthesize_track_moods(
        text_moods=["Atmospheric"],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[],
        detected_bpm=128,
        lyrics_analysis=None,
        primary_genre="Rock",
        subgenres=["Hard Rock", "Heavy Metal", "Classic Rock"],
        raw_tags=["hard rock", "heavy metal", "classic rock"],
        personalized_moods=[("Mellow", 0.85)],
    )
    assert "Mellow" not in hard_rock_moods
    assert hard_rock_moods == ["Atmospheric"]

    # 2. Happy path: Folk / Acoustic genre retains Mellow from personalized tuning
    folk_moods = synthesize_track_moods(
        text_moods=[],
        seeded_moods=[],
        essentia_moods=[],
        essentia_top=[],
        detected_bpm=95,
        lyrics_analysis=None,
        primary_genre="Folk",
        subgenres=["Indie Folk", "Acoustic Rock"],
        raw_tags=["folk", "indie folk", "acoustic"],
        personalized_moods=[("Mellow", 0.85)],
    )
    assert folk_moods == ["Mellow"]
