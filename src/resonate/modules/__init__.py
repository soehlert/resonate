"""Processing pipeline modules for Resonate."""

from resonate.modules.beets import BeetsTagger
from resonate.modules.essentia import EssentiaAnalyzer
from resonate.modules.lastfm import LastFmFetcher
from resonate.modules.lyrics import LyricsFetcher
from resonate.modules.plex import PlexSync
from resonate.modules.tag_mapper import TagMapper

__all__ = [
    "BeetsTagger",
    "EssentiaAnalyzer",
    "LastFmFetcher",
    "LyricsFetcher",
    "PlexSync",
    "TagMapper",
]

