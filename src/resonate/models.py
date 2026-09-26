from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MoodSource(IntEnum):
    """Authority tiers for candidate mood origins (higher value = higher authority)."""

    GENRE_SEED = 1
    PROVIDER_FALLBACK = 2
    LYRICS = 3
    ACOUSTIC = 4
    CLASSIFIER = 5
    PERSONALIZED_ANCHOR = 6
    TEXT_TAG = 7


class MoodEvidence(BaseModel):
    """Evidence record for a candidate mood including source tier and raw score."""

    source: MoodSource = MoodSource.GENRE_SEED
    score: float = 0.0

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, MoodEvidence):
            return NotImplemented
        if self.source != other.source:
            return self.source < other.source
        return self.score < other.score

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, MoodEvidence):
            return NotImplemented
        if self.source != other.source:
            return self.source < other.source
        return self.score <= other.score

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, MoodEvidence):
            return NotImplemented
        if self.source != other.source:
            return self.source > other.source
        return self.score > other.score

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, MoodEvidence):
            return NotImplemented
        if self.source != other.source:
            return self.source > other.source
        return self.score >= other.score

    def __str__(self) -> str:
        if self.score > 0.0:
            return f"{self.source.name} score={self.score:.2f}"
        return self.source.name


class TraceAction(StrEnum):
    """Action category for diagnostic decision events."""

    ACCEPT = "accept"
    REJECT = "reject"
    DROP = "drop"
    SKIP = "skip"
    INFO = "info"


class TraceEvent(BaseModel):
    """Structured event capturing a diagnostic decision step."""

    action: TraceAction = TraceAction.INFO
    message: str

    def __str__(self) -> str:
        """Format trace event as message string."""
        return self.message


class TrackItem(BaseModel):
    """Representation of a music track item from library or file source."""

    rating_key: str
    title: str
    artist: str
    album_artist: str | None = None
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
    track_specific_tags: list[str] = Field(default_factory=list)
    essentia_predictions: list[tuple[str, float]] = Field(default_factory=list)
    has_verified_tags: bool = False
    mutagen_updated: bool = False
    plex_updated: bool = False
    skipped: bool = False
    duration_ms: float = 0.0
    phase_timings: dict[str, float] = Field(default_factory=dict)
    decision_trace: list[TraceEvent] = Field(default_factory=list)

    @field_validator("decision_trace", mode="before")
    @classmethod
    def _coerce_trace_events(cls, val: Any) -> list[TraceEvent]:
        """Coerce strings or dicts into typed TraceEvent objects."""
        if not isinstance(val, list):
            return []
        coerced: list[TraceEvent] = []
        for item in val:
            if isinstance(item, str):
                coerced.append(TraceEvent(action=TraceAction.INFO, message=item))
            elif isinstance(item, TraceEvent):
                coerced.append(item)
            elif isinstance(item, dict):
                coerced.append(TraceEvent(**item))
        return coerced


class ProviderConfig(BaseModel):
    """Configuration options for an individual metadata provider."""

    enabled: bool = True
    api_key: str = ""
    rate_limit_delay: float = 0.0
    timeout_seconds: float = 10.0
