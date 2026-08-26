"""Core Pydantic data schemas for Resonate."""

from pydantic import BaseModel, Field


class TrackItem(BaseModel):
    """Representation of a music track item."""

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
    confidence: float
    source: str
    timestamp: float


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
    lastfm_matches: int = 0
    essentia_matches: int = 0
    failed_tracks: int = 0
