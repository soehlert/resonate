"""Mutagen tagger module to embed genre, mood, and BPM metadata directly in files."""

import logging
import os

logger = logging.getLogger(__name__)


class MutagenTagger:
    """Read and write metadata tags directly in audio files using mutagen."""

    def __init__(self, enabled: bool = True) -> None:
        """Initialize MutagenTagger."""
        self.enabled = enabled

    def update_file_tags(
        self,
        file_path: str,
        genres: list[str] | None = None,
        moods: list[str] | None = None,
        bpm: int | None = None,
        overwrite_tags: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Update genre, mood, and BPM tags in the target file."""
        if not self.enabled:
            logger.info("Mutagen tag writing is disabled.")
            return False

        if not os.path.exists(file_path):
            logger.warning(f"File not found for Mutagen tagging: {file_path}")
            return False

        _, ext = os.path.splitext(file_path.lower())

        if dry_run:
            logger.info(
                f"[DRY RUN] Would write to '{file_path}': "
                f"genres={genres}, moods={moods}, bpm={bpm} (overwrite={overwrite_tags})"
            )
            return True

        try:
            if ext == ".mp3":
                return self._update_mp3(file_path, genres, moods, bpm, overwrite_tags)
            elif ext == ".flac":
                return self._update_flac(file_path, genres, moods, bpm, overwrite_tags)
            elif ext in (".m4a", ".mp4"):
                return self._update_mp4(file_path, genres, moods, bpm, overwrite_tags)
            else:
                logger.warning(f"Unsupported file extension for tag writing: {ext}")
                return False
        except Exception as err:
            logger.warning(f"Failed to write tags to '{file_path}': {err}")
            return False

    def _update_mp3(
        self,
        file_path: str,
        genres: list[str] | None,
        moods: list[str] | None,
        bpm: int | None,
        overwrite_tags: bool,
    ) -> bool:
        """Write ID3 tags to MP3 file."""
        from mutagen import MutagenError
        from mutagen.id3 import ID3, TBPM, TCON, TMOO, TXXX, ID3NoHeaderError

        try:
            audio = ID3(file_path)
        except ID3NoHeaderError:
            # If no ID3 header exists, create a clean one
            audio = ID3()
        except MutagenError as e:
            logger.warning(f"Error opening ID3 tag for '{file_path}': {e}")
            return False

        modified = False

        # 1. Write TCON (Genres)
        if genres:
            existing = audio.get("TCON")
            existing_val = existing.text if existing else []
            # Check if we should write
            if overwrite_tags or not existing_val:
                audio["TCON"] = TCON(encoding=3, text=genres)
                modified = True

        # 2. Write TMOO / TXXX:MOOD (Moods)
        if moods:
            # TMOO (Official mood tag)
            existing_tmoo = audio.get("TMOO")
            existing_tmoo_val = existing_tmoo.text if existing_tmoo else []
            if overwrite_tags or not existing_tmoo_val:
                audio["TMOO"] = TMOO(encoding=3, text=moods)
                modified = True

            # TXXX:MOOD (Common fallback mood tag)
            existing_txxx = audio.get("TXXX:MOOD")
            existing_txxx_val = existing_txxx.text if existing_txxx else []
            if overwrite_tags or not existing_txxx_val:
                audio["TXXX:MOOD"] = TXXX(encoding=3, desc="MOOD", text=moods)
                modified = True

        # 3. Write TBPM (BPM)
        if bpm is not None:
            existing_tbpm = audio.get("TBPM")
            existing_tbpm_val = existing_tbpm.text if existing_tbpm else []
            if overwrite_tags or not existing_tbpm_val:
                audio["TBPM"] = TBPM(encoding=3, text=[str(bpm)])
                modified = True

        if modified:
            audio.save(file_path)
            logger.info(f"Successfully wrote ID3 tags to '{file_path}'")
            return True
        return False

    def _update_flac(
        self,
        file_path: str,
        genres: list[str] | None,
        moods: list[str] | None,
        bpm: int | None,
        overwrite_tags: bool,
    ) -> bool:
        """Write Vorbis comments to FLAC file."""
        from mutagen.flac import FLAC

        audio = FLAC(file_path)
        modified = False

        # 1. Write genres
        if genres:
            existing = audio.get("genre", [])
            if overwrite_tags or not existing:
                audio["genre"] = genres
                modified = True

        # 2. Write moods
        if moods:
            existing = audio.get("mood", [])
            if overwrite_tags or not existing:
                audio["mood"] = moods
                modified = True

        # 3. Write BPM
        if bpm is not None:
            existing = audio.get("bpm", [])
            if overwrite_tags or not existing:
                audio["bpm"] = [str(bpm)]
                modified = True

        if modified:
            audio.save()
            logger.info(f"Successfully wrote FLAC tags to '{file_path}'")
            return True
        return False

    def _update_mp4(
        self,
        file_path: str,
        genres: list[str] | None,
        moods: list[str] | None,
        bpm: int | None,
        overwrite_tags: bool,
    ) -> bool:
        """Write MP4 atoms to M4A/MP4 file."""
        from mutagen.mp4 import MP4

        audio = MP4(file_path)
        modified = False

        # 1. Write genres (\xa9gen atom)
        if genres:
            existing = audio.get("\xa9gen", [])
            if overwrite_tags or not existing:
                audio["\xa9gen"] = genres
                modified = True

        # 2. Write moods (----:com.apple.iTunes:mood custom tag)
        if moods:
            mood_key = "----:com.apple.iTunes:mood"
            existing = audio.get(mood_key, [])
            if overwrite_tags or not existing:
                audio[mood_key] = [m.encode("utf-8") for m in moods]
                modified = True

        # 3. Write BPM (tmpo atom)
        if bpm is not None:
            existing = audio.get("tmpo", [])
            if overwrite_tags or not existing:
                audio["tmpo"] = [int(bpm)]
                modified = True

        if modified:
            audio.save()
            logger.info(f"Successfully wrote MP4 tags to '{file_path}'")
            return True
        return False
