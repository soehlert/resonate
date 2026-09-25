"""Processing pipeline modules for Resonate."""

from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.lyrics import LyricsFetcher
from resonate.modules.plex import PlexSync
from resonate.modules.tag_mapper import TagMapper

__all__ = [
    "EssentiaAnalyzer",
    "LyricsFetcher",
    "PlexSync",
    "TagMapper",
]

