"""Unit tests for Resonate domain models in models.py."""

from resonate.models import (
    BatchProcessingStats,
    ProviderConfig,
    ProviderResult,
    TaxonomyDecision,
    TrackEnrichmentResult,
    TrackItem,
)


def test_track_item_defaults() -> None:
    """Test TrackItem creation and default field values."""
    item = TrackItem(rating_key="101", title="Karma Police", artist="Radiohead")
    assert item.rating_key == "101"
    assert item.title == "Karma Police"
    assert item.artist == "Radiohead"
    assert item.album is None
    assert item.raw_tags == []
    assert item.current_moods == []


def test_provider_result_all_tags_deduplication_and_order() -> None:
    """Test ProviderResult all_tags consolidates tags with case-insensitive dedup."""
    res = ProviderResult(
        provider_name="lastfm",
        track_tags=["Alternative Rock", "90s"],
        album_tags=["rock", "ART ROCK", "90s"],
        artist_tags=["Radiohead", "experimental", "Rock"],
        canonical_artist="Radiohead",
        status="success",
    )
    all_tags = res.all_tags
    # Track tags first, then album tags, then artist tags
    assert all_tags == ["Alternative Rock", "90s", "rock", "ART ROCK", "Radiohead", "experimental"]
    # Check that "Rock" (artist) was deduplicated because "rock" was in album tags
    assert len([t for t in all_tags if t.lower() == "rock"]) == 1


def test_provider_result_empty_status() -> None:
    """Test ProviderResult default initialization."""
    res = ProviderResult(provider_name="musicbrainz")
    assert res.provider_name == "musicbrainz"
    assert res.all_tags == []
    assert res.canonical_artist is None
    assert res.status == "success"


def test_taxonomy_decision_model() -> None:
    """Test TaxonomyDecision serialization and field values."""
    decision = TaxonomyDecision(
        original_genre="Pop",
        promoted_genre="Punk",
        reason="Child subgenres strictly outnumber parent genre",
        contributing_subgenres=["pop-punk", "skate punk"],
        confidence=0.95,
    )
    assert decision.original_genre == "Pop"
    assert decision.promoted_genre == "Punk"
    assert len(decision.contributing_subgenres) == 2
    assert decision.confidence == 0.95


def test_track_enrichment_result_model() -> None:
    """Test TrackEnrichmentResult initialization and field values."""
    result = TrackEnrichmentResult(
        rating_key="555",
        title="Paranoid Android",
        artist="Radiohead",
        album="OK Computer",
        primary_genre="Rock",
        subgenres=["Art Rock", "Alternative Rock"],
        moods=["Melancholic", "Energetic"],
        bpm=82,
        lyrics_valence=-0.45,
        raw_tags=["alternative", "art rock", "90s"],
        has_verified_tags=True,
        mutagen_updated=True,
        plex_updated=True,
    )
    assert result.rating_key == "555"
    assert result.primary_genre == "Rock"
    assert result.subgenres == ["Art Rock", "Alternative Rock"]
    assert result.bpm == 82
    assert result.has_verified_tags is True
    assert result.mutagen_updated is True


def test_provider_config_model() -> None:
    """Test ProviderConfig default initialization."""
    config = ProviderConfig(api_key="secret-key", rate_limit_delay=5.0)
    assert config.enabled is True
    assert config.api_key == "secret-key"
    assert config.rate_limit_delay == 5.0
    assert config.timeout_seconds == 10.0


def test_batch_processing_stats_model() -> None:
    """Test BatchProcessingStats initialization."""
    stats = BatchProcessingStats(total_tracks=100, processed_tracks=95, genre_matches=90)
    assert stats.total_tracks == 100
    assert stats.processed_tracks == 95
    assert stats.genre_matches == 90
    assert stats.bpm_detected == 0
