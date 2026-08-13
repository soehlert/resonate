"""Plex Media Server integration module for sync and mood tag updates."""

import logging
from typing import Any

import requests
import urllib3
from plexapi.server import PlexServer

from resonate.models import TrackItem

# Squelch unverified HTTPS warnings for local Plex connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class PlexSync:
    """Synchronize track metadata with a Plex Media Server instance."""

    def __init__(self, url: str, token: str, library_name: str = "Music") -> None:
        """Initialize PlexSync with server URL, token, and library name."""
        self.url = url
        self.token = token
        self.library_name = library_name
        self.server: Any = None
        self.library: Any = None

    def connect(self) -> bool:
        """Connect to Plex server and load specified music library."""
        try:
            session = requests.Session()
            session.verify = False
            self.server = PlexServer(self.url, self.token, session=session)
            self.library = self.server.library.section(self.library_name)
            return True
        except Exception as err:
            logger.warning(f"Failed to connect to Plex server at {self.url}: {err}")
            self.server = None
            self.library = None
            return False

    def fetch_audio_tracks(
        self, limit: int | None = None, artist: str | None = None
    ) -> list[TrackItem]:
        """Fetch audio tracks from target Plex music library."""
        if self.library is None:
            if not self.connect():
                return []

        try:
            # Pass limit directly to searchTracks if available to speed up query
            kwargs = {}
            if limit is not None and not artist:
                kwargs["limit"] = limit
            tracks = self.library.searchTracks(**kwargs)
            result: list[TrackItem] = []

            for track in tracks:
                rating_key = str(getattr(track, "ratingKey", ""))
                title = getattr(track, "title", "")
                artist_name = getattr(track, "grandparentTitle", "") or getattr(
                    track, "originalTitle", ""
                )
                if artist and artist.lower() not in artist_name.lower():
                    continue

                album = getattr(track, "parentTitle", "")
                moods = [m.tag for m in getattr(track, "moods", []) if hasattr(m, "tag")]

                media = getattr(track, "media", [])
                path = ""
                if media and len(media) > 0:
                    parts = getattr(media[0], "parts", [])
                    if parts and len(parts) > 0:
                        path = getattr(parts[0], "file", "")

                result.append(
                    TrackItem(
                        rating_key=rating_key,
                        title=title,
                        artist=artist_name,
                        album=album,
                        file_path=path,
                        current_moods=moods,
                    )
                )

                # Apply local limit after filtering if artist was specified
                if artist and limit is not None and len(result) >= limit:
                    break

            return result
        except Exception as err:
            logger.warning(
                f"Failed to fetch audio tracks from Plex library '{self.library_name}': {err}"
            )
            return []

    def update_track_metadata(
        self,
        rating_key: str,
        genres: list[str] | None = None,
        moods: list[str] | None = None,
        bpm: int | None = None,
        overwrite_tags: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Update genres, moods, and BPM in the Plex track item."""
        if self.server is None:
            if not self.connect():
                return False

        try:
            track = self.server.fetchItem(int(rating_key) if rating_key.isdigit() else rating_key)
            if track is None:
                logger.warning(f"Plex track not found for rating_key: {rating_key}")
                return False

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would update Plex track '{rating_key}': "
                    f"genres={genres}, moods={moods}, bpm={bpm} (overwrite={overwrite_tags})"
                )
                return True

            # 1. Update Genres
            if genres:
                existing_genres = [g.tag for g in getattr(track, "genres", []) if hasattr(g, "tag")]
                if overwrite_tags:
                    if existing_genres:
                        track.removeGenre(existing_genres)
                    track.addGenre(genres)
                else:
                    if not existing_genres:
                        track.addGenre(genres)

            # 2. Update Moods
            if moods:
                existing_moods = [m.tag for m in getattr(track, "moods", []) if hasattr(m, "tag")]
                if overwrite_tags:
                    if existing_moods:
                        track.removeMood(existing_moods)
                    track.addMood(moods)
                    track.lockMood()
                else:
                    if not existing_moods:
                        track.addMood(moods)
                        track.lockMood()

            # 3. Update BPM
            if bpm is not None:
                existing_bpm = getattr(track, "bpm", None)
                if overwrite_tags or existing_bpm is None or existing_bpm == 0:
                    track.edit(**{"bpm.value": bpm})

            return True
        except Exception as err:
            logger.warning(
                f"Failed to update Plex track metadata for rating_key '{rating_key}': {err}"
            )
            return False

    def update_track_mood(self, rating_key: str, mood: str, dry_run: bool = False) -> bool:
        """Scrub existing mood tags, add new mood tag, and lock field on Plex track."""
        return self.update_track_metadata(
            rating_key=rating_key,
            moods=[mood],
            overwrite_tags=True,
            dry_run=dry_run,
        )
