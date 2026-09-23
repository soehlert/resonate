"""Unit tests for taxonomy hierarchies, genre promotion rules, and mood synthesis."""

from resonate.engine.mood_rules import (
    apply_bpm_mood_rules,
    get_genre_seeded_moods,
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


def test_bpm_mood_rules_gating() -> None:
    """Test BPM tempo gating for Energetic (>=130), Lively (110-129), and Low (<110)."""
    # High tempo: retains Energetic, drops Lively
    assert apply_bpm_mood_rules(["Energetic", "Lively", "Happy"], detected_bpm=140) == [
        "Energetic",
        "Happy",
    ]

    # Mid tempo: converts Energetic to Lively
    assert apply_bpm_mood_rules(["Energetic", "Happy"], detected_bpm=120) == [
        "Lively",
        "Happy",
    ]

    # Low tempo: drops Energetic and Lively
    assert apply_bpm_mood_rules(["Energetic", "Lively", "Mellow"], detected_bpm=85) == [
        "Mellow"
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
