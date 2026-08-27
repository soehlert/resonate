"""Core Pydantic data schemas for Resonate."""

from pydantic import BaseModel, Field


class TrackItem(BaseModel):
    """Representation of a music track item from library or file source."""

    rating_key: str
    title: str
    artist: str
    album: str | None = None
    file_path: str | None = None
    raw_tags: list[str] = Field(default_factory=list)
    current_moods: list[str] = Field(default_factory=list)


class ProcessingResult(BaseModel):
    """Result of processing mood mapping for a track."""

    rating_key: str
    title: str
    artist: str
    mapped_mood: str | None = None
    confidence: float = 0.0
    source: str = "hybrid"
    timestamp: float = 0.0


class LyricsAnalysisResult(BaseModel):
    """Result of lyrics retrieval and mood analysis."""

    lyrics_text: str | None = None
    source: str = "none"
    valence_score: float = 0.0
    mood_scores: dict[str, float] = Field(default_factory=dict)


class BatchProcessingStats(BaseModel):
    """Statistics for a batch processing run."""

    total_tracks: int = 0
    processed_tracks: int = 0
    skipped_tracks: int = 0
    genre_matches: int = 0
    subgenre_matches: int = 0
    mood_matches: int = 0
    bpm_detected: int = 0
    mutagen_writes: int = 0
    plex_syncs: int = 0
    failed_tracks: int = 0


class ProviderResult(BaseModel):
    """Standardized metadata result payload returned by any metadata provider."""

    provider_name: str
    track_tags: list[str] = Field(default_factory=list)
    album_tags: list[str] = Field(default_factory=list)
    artist_tags: list[str] = Field(default_factory=list)
    canonical_artist: str | None = None
    release_year: int | None = None
    duration_ms: float = 0.0
    status: str = "success"

    @property
    def all_tags(self) -> list[str]:
        """Consolidate all tags preserving track, album, and artist precedence."""
        seen: set[str] = set()
        consolidated: list[str] = []
        for tag in self.track_tags + self.album_tags + self.artist_tags:
            clean = tag.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                consolidated.append(clean)
        return consolidated


class TaxonomyDecision(BaseModel):
    """Record of a taxonomy hierarchy promotion or rule-based override."""

    original_genre: str | None = None
    promoted_genre: str | None = None
    reason: str
    contributing_subgenres: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class TrackEnrichmentResult(BaseModel):
    """Comprehensive typed result of enriching a track through the pipeline."""

    rating_key: str
    title: str
    artist: str
    album: str | None = None
    resolved_path: str | None = None
    primary_genre: str | None = None
    subgenres: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)
    bpm: int | None = None
    lyrics_valence: float | None = None
    raw_tags: list[str] = Field(default_factory=list)
    has_verified_tags: bool = False
    mutagen_updated: bool = False
    plex_updated: bool = False
    skipped: bool = False
    duration_ms: float = 0.0


class ProviderConfig(BaseModel):
    """Configuration options for an individual metadata provider."""

    enabled: bool = True
    api_key: str = ""
    api_secret: str = ""
    rate_limit_delay: float = 0.0
    timeout_seconds: float = 10.0
