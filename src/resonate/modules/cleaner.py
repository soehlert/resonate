"""Modular audio file tag cleaning engine."""

import logging
import os
import re

from pydantic import BaseModel, Field

from resonate.modules.external_metadata import clean_retailer_noise, uncensor_title

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".flac",
    ".m4a",
    ".mp4",
    ".ogg",
    ".opus",
    ".aif",
    ".aiff",
    ".wav",
}


class TagChange(BaseModel):
    """Record of a tag field change."""

    field: str
    old_value: str
    new_value: str


class FileCleanResult(BaseModel):
    """Result of cleaning tags on an audio file."""

    file_path: str
    changed: bool = False
    changes: list[TagChange] = Field(default_factory=list)
    error: str | None = None


def normalize_track_number(track_str: str | None) -> tuple[str | None, str | None]:
    """Normalize track numbers, leading zeros, and extract disc prefixes."""
    if not track_str or not track_str.strip():
        return (None, None)

    val = track_str.strip()
    extracted_disc: str | None = None

    # Handle multi-disc hyphenated syntax: "1-04" -> disc="1", track="4"
    if "-" in val:
        parts = val.split("-", 1)
        if parts[0].isdigit() and parts[1].isdigit():
            extracted_disc = str(int(parts[0]))
            val = str(int(parts[1]))

    # Handle fractional track formatting: "01/12", "1/0", "01/00"
    if "/" in val:
        parts = val.split("/", 1)
        track_part = parts[0].strip()
        total_part = parts[1].strip()

        track_num = int(track_part) if track_part.isdigit() else track_part
        total_num = int(total_part) if total_part.isdigit() else total_part

        # If total is 0 or missing, omit total
        if isinstance(total_num, int) and total_num <= 0:
            val = str(track_num)
        else:
            val = f"{track_num}/{total_num}"
    elif val.isdigit():
        val = str(int(val))

    return (val, extracted_disc)


def clean_whitespace_and_artifacts(text: str | None) -> str | None:
    """Trim double spaces, null characters, and empty dangling brackets."""
    if not text:
        return text

    # Remove null byte characters
    cleaned = text.replace("\x00", "").strip()

    # Clean empty brackets like (), [], ( ), [ ]
    cleaned = re.sub(r"[\(\[]\s*[\)\]]", "", cleaned)

    # Collapse multiple consecutive spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class TagCleaner:
    """Sanitize and clean noisy audio file tags."""

    def __init__(
        self,
        clean_retailer: bool = True,
        uncensor: bool = True,
        normalize_track_numbers: bool = True,
        clean_whitespace: bool = True,
    ) -> None:
        """Initialize TagCleaner with enabled rule toggles."""
        self.clean_retailer = clean_retailer
        self.uncensor = uncensor
        self.normalize_track_numbers = normalize_track_numbers
        self.clean_whitespace = clean_whitespace

    def clean_file(self, file_path: str, dry_run: bool = False) -> FileCleanResult:
        """Clean tags for a single audio file."""
        if not os.path.exists(file_path):
            return FileCleanResult(file_path=file_path, error="File not found")

        _, ext = os.path.splitext(file_path.lower())
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return FileCleanResult(
                file_path=file_path, error=f"Unsupported file extension: {ext}"
            )

        try:
            import mutagen

            audio = mutagen.File(file_path, easy=True)
            if audio is None:
                return FileCleanResult(
                    file_path=file_path, error="Failed to parse audio metadata"
                )

            changes: list[TagChange] = []

            # 1. Clean Album
            if "album" in audio and audio["album"]:
                old_album = str(audio["album"][0])
                new_album = old_album
                if self.clean_retailer:
                    cleaned = clean_retailer_noise(new_album)
                    if cleaned is not None:
                        new_album = cleaned
                if self.clean_whitespace:
                    cleaned_ws = clean_whitespace_and_artifacts(new_album)
                    if cleaned_ws is not None:
                        new_album = cleaned_ws

                if new_album != old_album:
                    changes.append(
                        TagChange(field="Album", old_value=old_album, new_value=new_album)
                    )
                    audio["album"] = [new_album]

            # 2. Clean Title
            if "title" in audio and audio["title"]:
                old_title = str(audio["title"][0])
                new_title = old_title
                if self.uncensor:
                    new_title = uncensor_title(new_title)
                if self.clean_whitespace:
                    cleaned_ws = clean_whitespace_and_artifacts(new_title)
                    if cleaned_ws is not None:
                        new_title = cleaned_ws

                if new_title != old_title:
                    changes.append(
                        TagChange(field="Title", old_value=old_title, new_value=new_title)
                    )
                    audio["title"] = [new_title]

            # 3. Clean Track & Disc Numbers
            if self.normalize_track_numbers and "tracknumber" in audio and audio["tracknumber"]:
                old_track = str(audio["tracknumber"][0])
                new_track, extracted_disc = normalize_track_number(old_track)
                if new_track and new_track != old_track:
                    changes.append(
                        TagChange(field="Track", old_value=old_track, new_value=new_track)
                    )
                    audio["tracknumber"] = [new_track]

                if extracted_disc and (
                    "discnumber" not in audio or not audio["discnumber"]
                ):
                    changes.append(
                        TagChange(field="Disc", old_value="(None)", new_value=extracted_disc)
                    )
                    audio["discnumber"] = [extracted_disc]

            # 4. Clean Artist & AlbumArtist whitespace
            if self.clean_whitespace:
                for art_key in ("artist", "albumartist"):
                    if art_key in audio and audio[art_key]:
                        old_art = str(audio[art_key][0])
                        new_art = clean_whitespace_and_artifacts(old_art)
                        if new_art and new_art != old_art:
                            changes.append(
                                TagChange(
                                    field=art_key.capitalize(),
                                    old_value=old_art,
                                    new_value=new_art,
                                )
                            )
                            audio[art_key] = [new_art]

            if changes and not dry_run:
                audio.save()

            return FileCleanResult(
                file_path=file_path, changed=bool(changes), changes=changes
            )
        except Exception as err:
            logger.warning(f"Error cleaning file '{file_path}': {err}")
            return FileCleanResult(file_path=file_path, error=str(err))

    def clean_path(
        self, target_path: str, recursive: bool = True, dry_run: bool = False
    ) -> list[FileCleanResult]:
        """Clean an audio file or all files in a directory."""
        if not os.path.exists(target_path):
            return [FileCleanResult(file_path=target_path, error="Path does not exist")]

        if os.path.isfile(target_path):
            return [self.clean_file(target_path, dry_run=dry_run)]

        results: list[FileCleanResult] = []
        if recursive:
            for root, _, files in os.walk(target_path):
                for fname in sorted(files):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_AUDIO_EXTENSIONS:
                        fpath = os.path.join(root, fname)
                        results.append(self.clean_file(fpath, dry_run=dry_run))
        else:
            for fname in sorted(os.listdir(target_path)):
                fpath = os.path.join(target_path, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_AUDIO_EXTENSIONS:
                        results.append(self.clean_file(fpath, dry_run=dry_run))

        return results
