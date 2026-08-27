"""Abstract base metadata provider contract for Resonate."""

from abc import ABC, abstractmethod

from resonate.models import ProviderResult


class BaseMetadataProvider(ABC):
    """Abstract base class for all metadata provider integrations."""

    name: str = "base"
    enabled: bool = True
    rate_limit_delay: float = 0.0
    timeout_seconds: float = 10.0

    @abstractmethod
    def fetch_track_tags(
        self, artist: str, title: str, album: str | None = None
    ) -> list[str]:
        """Fetch track-level tags for a specific song."""
        ...

    @abstractmethod
    def fetch_album_tags(self, artist: str, album: str) -> list[str]:
        """Fetch album/release-level tags for an album."""
        ...

    @abstractmethod
    def fetch_artist_tags(self, artist: str) -> list[str]:
        """Fetch artist-level tags."""
        ...

    def resolve_canonical_artist(self, artist: str) -> str | None:
        """Resolve artist alias or rebrand to canonical name if supported."""
        return None

    def query_all(
        self, artist: str, title: str, album: str | None = None
    ) -> ProviderResult:
        """Fetch track, album, and artist tags into a standardized ProviderResult."""
        if not self.enabled:
            return ProviderResult(provider_name=self.name, status="disabled")

        track_tags = self.fetch_track_tags(artist, title, album=album) if title else []
        album_tags = self.fetch_album_tags(artist, album) if album else []
        artist_tags = self.fetch_artist_tags(artist) if artist else []
        canonical = self.resolve_canonical_artist(artist)

        return ProviderResult(
            provider_name=self.name,
            track_tags=track_tags,
            album_tags=album_tags,
            artist_tags=artist_tags,
            canonical_artist=canonical,
            status="success",
        )
