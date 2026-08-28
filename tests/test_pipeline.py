"""Unit tests for the end-to-end EnrichmentPipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from resonate.engine.pipeline import EnrichmentPipeline
from resonate.models import LyricsAnalysisResult, TrackItem
from resonate.modules.bpm import BpmDetector
from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.lyrics import LyricsFetcher
from resonate.modules.mutagen import MutagenTagger
from resonate.modules.tag_mapper import TagMapper
from resonate.providers.manager import ProviderManager


@pytest.fixture
def mock_mappers() -> tuple[TagMapper, TagMapper, TagMapper]:
    """Fixture providing mocked genre, subgenre, and mood mappers."""
    genre_mapper = MagicMock(spec=TagMapper)
    subgenre_mapper = MagicMock(spec=TagMapper)
    mood_mapper = MagicMock(spec=TagMapper)
    mood_mapper.target_moods = ["Party", "Chill Hang", "Energetic", "Melancholic"]

    return genre_mapper, subgenre_mapper, mood_mapper


def test_pipeline_enrich_track_rock_promotion(
    mock_mappers: tuple[TagMapper, TagMapper, TagMapper]
) -> None:
    """Test track enriched with tags and promoted from generic Rock to Punk."""
    genre_mapper, subgenre_mapper, mood_mapper = mock_mappers
    provider_mgr = MagicMock(spec=ProviderManager)
    provider_mgr.get_tags_for_track.return_value = (
        ["rock", "skate punk", "pop-punk"],
        ["skate punk", "pop-punk"],
        True,
        "Blink-182",
    )

    genre_mapper.match_genre_consensus.return_value = [("Rock", "rock", 0.95, 0)]
    subgenre_mapper.match_subgenre_consensus.return_value = [
        ("Skate Punk", "skate punk", 0.92),
        ("Pop-Punk", "pop-punk", 0.90),
    ]
    mood_mapper.match_multiple_tags.return_value = [("Rowdy", "skate punk", 0.85)]

    pipeline = EnrichmentPipeline(
        provider_manager=provider_mgr,
        genre_mapper=genre_mapper,
        subgenre_mapper=subgenre_mapper,
        mood_mapper=mood_mapper,
    )

    track = TrackItem(
        rating_key="101",
        title="Dammit",
        artist="Blink-182",
        album="Dude Ranch",
    )

    result = pipeline.enrich_track(track, do_genre=True, do_subgenre=True, do_mood=True)

    assert result.rating_key == "101"
    assert result.artist == "Blink-182"
    assert result.primary_genre == "Punk"  # Promoted from Rock to Punk!
    assert "Skate Punk" in result.subgenres
    assert "Pop-Punk" in result.subgenres
    assert result.has_verified_tags is True


def test_pipeline_audio_genre_override_when_unverified_tags(
    mock_mappers: tuple[TagMapper, TagMapper, TagMapper],
    tmp_path: Path,
) -> None:
    """Test audio classifier overrides unverified artist tags when audio file is present."""
    genre_mapper, subgenre_mapper, mood_mapper = mock_mappers
    provider_mgr = MagicMock(spec=ProviderManager)
    # Unverified fallback tags
    provider_mgr.get_tags_for_track.return_value = (
        ["electronic"],
        [],
        False,
        "Unknown Metal Band",
    )
    genre_mapper.match_genre_consensus.return_value = [("Electronic", "electronic", 0.8, 0)]

    essentia_analyzer = MagicMock(spec=EssentiaAnalyzer)
    essentia_analyzer.enabled = True
    essentia_analyzer.analyze_genre_waveform.return_value = ("Metal", ["Heavy Metal"])
    essentia_analyzer.analyze_waveform.return_value = (["Heavy"], 0.88, [("heavy", 0.9)])

    audio_file = tmp_path / "test.flac"
    audio_file.write_bytes(b"dummy audio")

    pipeline = EnrichmentPipeline(
        provider_manager=provider_mgr,
        genre_mapper=genre_mapper,
        subgenre_mapper=subgenre_mapper,
        mood_mapper=mood_mapper,
        essentia_analyzer=essentia_analyzer,
    )

    track = TrackItem(rating_key="102", title="Riff", artist="Unknown Metal Band")
    result = pipeline.enrich_track(track, resolved_path=str(audio_file))

    assert result.primary_genre == "Metal"
    assert result.subgenres == ["Heavy Metal"]
    assert result.has_verified_tags is False


def test_pipeline_bpm_and_lyrics_synthesis(
    mock_mappers: tuple[TagMapper, TagMapper, TagMapper],
    tmp_path: Path,
) -> None:
    """Test high tempo BPM gating and lyrics sentiment integration in mood synthesis."""
    genre_mapper, subgenre_mapper, mood_mapper = mock_mappers
    provider_mgr = MagicMock(spec=ProviderManager)
    provider_mgr.get_tags_for_track.return_value = (
        ["alternative rock"],
        ["alternative rock"],
        True,
        "Radiohead",
    )
    genre_mapper.match_genre_consensus.return_value = [("Rock", "alternative rock", 0.9, 0)]
    subgenre_mapper.match_subgenre_consensus.return_value = [
        ("Alternative Rock", "alternative rock", 0.9)
    ]
    mood_mapper.match_multiple_tags.return_value = [("Chill Hang", "alternative rock", 0.8)]

    bpm_detector = MagicMock(spec=BpmDetector)
    bpm_detector.enabled = True
    bpm_detector.detect_bpm.return_value = 142

    lyrics_fetcher = MagicMock(spec=LyricsFetcher)
    lyrics_fetcher.enabled = True
    lyrics_fetcher.get_lyrics.return_value = ("I am a creep", "lrclib")
    lyrics_fetcher.analyze_lyrics.return_value = LyricsAnalysisResult(
        lyrics_text="I am a creep",
        source="lrclib",
        valence_score=-0.75,
        mood_scores={"Dark": 0.65, "Melancholic": 0.55},
    )

    audio_file = tmp_path / "creep.mp3"
    audio_file.write_bytes(b"dummy mp3")

    pipeline = EnrichmentPipeline(
        provider_manager=provider_mgr,
        genre_mapper=genre_mapper,
        subgenre_mapper=subgenre_mapper,
        mood_mapper=mood_mapper,
        bpm_detector=bpm_detector,
        lyrics_fetcher=lyrics_fetcher,
    )

    track = TrackItem(rating_key="103", title="Creep", artist="Radiohead")
    result = pipeline.enrich_track(track, resolved_path=str(audio_file))

    assert result.bpm == 142
    assert result.lyrics_valence == -0.75
    # Negative lyrics valence drops Chill Hang; high BPM (>130) adds Energetic
    assert "Chill Hang" not in result.moods
    assert "Dark" in result.moods or "Melancholic" in result.moods


def test_pipeline_mutagen_tag_writer(
    mock_mappers: tuple[TagMapper, TagMapper, TagMapper],
    tmp_path: Path,
) -> None:
    """Test Mutagen tag writer called with complete genre, mood, and BPM data."""
    genre_mapper, subgenre_mapper, mood_mapper = mock_mappers
    provider_mgr = MagicMock(spec=ProviderManager)
    provider_mgr.get_tags_for_track.return_value = (["jazz"], ["jazz"], True, "Miles Davis")
    genre_mapper.match_genre_consensus.return_value = [("Jazz", "jazz", 0.95, 0)]
    subgenre_mapper.match_subgenre_consensus.return_value = [("Cool Jazz", "jazz", 0.9)]
    mood_mapper.match_multiple_tags.return_value = [("Mellow", "jazz", 0.85)]

    mutagen_tagger = MagicMock(spec=MutagenTagger)
    mutagen_tagger.enabled = True
    mutagen_tagger.update_file_tags.return_value = True

    audio_file = tmp_path / "so_what.flac"
    audio_file.write_bytes(b"dummy flac")

    pipeline = EnrichmentPipeline(
        provider_manager=provider_mgr,
        genre_mapper=genre_mapper,
        subgenre_mapper=subgenre_mapper,
        mood_mapper=mood_mapper,
        mutagen_tagger=mutagen_tagger,
    )

    track = TrackItem(rating_key="104", title="So What", artist="Miles Davis")
    result = pipeline.enrich_track(
        track,
        resolved_path=str(audio_file),
        write_tags=True,
        overwrite_tags=True,
        dry_run=False,
    )

    assert result.mutagen_updated is True
    mutagen_tagger.update_file_tags.assert_called_once_with(
        file_path=str(audio_file),
        genres=["Jazz", "Cool Jazz"],
        moods=["Mellow", "Relaxed", "Atmospheric"],
        bpm=None,
        overwrite_tags=True,
        dry_run=False,
    )
