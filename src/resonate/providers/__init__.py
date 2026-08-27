"""Metadata providers package for Resonate."""

from resonate.providers.base import BaseMetadataProvider
from resonate.providers.discogs import DiscogsProvider
from resonate.providers.lastfm import LastFmProvider
from resonate.providers.manager import ProviderManager
from resonate.providers.musicbrainz import MusicBrainzProvider

__all__ = [
    "BaseMetadataProvider",
    "DiscogsProvider",
    "LastFmProvider",
    "MusicBrainzProvider",
    "ProviderManager",
]
