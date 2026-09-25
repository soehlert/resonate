"""Modular audio file tag cleaning engine."""

import logging
import os
import re
from typing import Any

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


def sanitize_filename_component(text: str) -> str:
    """Sanitize a title or string component for safe filesystem filename usage."""
    if not text:
        return ""
    s = text.replace("\x00", "")
    s = s.replace(":", " -")
    s = re.sub(r"[\/\\]+", "-", s)
    s = s.replace('"', "'")
    s = s.replace("|", "-")
    s = re.sub(r"[\*\?\<\>]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"-{2,}", "-", s).strip(" .-_")
    return s


def parse_audio_path_fallback(file_path: str) -> dict[str, str | None]:
    """Extract fallback metadata (disc, track, title) from file path and directory structure."""
    dir_name, fname = os.path.split(file_path)
    stem, _ = os.path.splitext(fname)

    disc: str | None = None
    track: str | None = None
    title: str | None = None

    # Check directory for Disc/CD: e.g. 'CD 1', 'Disc 2', 'Disk 02'
    parent_dir = os.path.basename(dir_name)
    disc_match = re.search(r"(?:cd|disc|disk)\s*0*([1-9]\d*)", parent_dir, re.IGNORECASE)
    if disc_match:
        disc = disc_match.group(1)

    # Convert underscore separators to spaces if no ' - ' hyphen separator exists
    clean_stem = stem
    if " - " not in clean_stem and "_" in clean_stem:
        clean_stem = clean_stem.replace("_", " ")

    # 1. Matches leading Disc-Track format: e.g. '1-03 - Chapters', '1.03 Chapters', '1-03 Chapters'
    m_dt = re.match(r"^([1-9]\d*)[-.](\d{1,3})[\.\s\-]+(.+)$", clean_stem)
    if m_dt:
        disc = m_dt.group(1)
        track = str(int(m_dt.group(2)))
        title = m_dt.group(3).strip(" .-_")
        return {"disc": disc, "track": track, "title": title}

    # 2. Split on hyphens/dashes flanked by whitespace: ' - ', ' – ', ' — '
    parts = [p.strip() for p in re.split(r"\s+[\-–—]\s+", clean_stem) if p.strip()]

    if len(parts) >= 2:
        # Check if first part and second part are disc & track: e.g. '01 - 02 - Title'
        if (
            len(parts) >= 3
            and re.match(r"^\d{1,2}$", parts[0])
            and re.match(r"^\d{1,3}$", parts[1])
        ):
            disc = str(int(parts[0]))
            track = str(int(parts[1]))
            title = " - ".join(parts[2:])
        elif re.match(r"^\d{1,3}$", parts[0]):
            track = str(int(parts[0]))
            # e.g. '01 - Artist - Title' (3 parts) vs '01 - Title' (2 parts)
            if len(parts) == 3:
                title = parts[2]
            else:
                title = " - ".join(parts[1:])
        elif re.match(r"^\d{1,3}$", parts[-1]):
            # 'Artist - Title - 01'
            track = str(int(parts[-1]))
            title = " - ".join(parts[:-1])
        else:
            # Look for a track number in middle parts: 'Artist - 01 - Title'
            track_idx = -1
            for i, p in enumerate(parts):
                if re.match(r"^\d{1,3}$", p):
                    track_idx = i
                    track = str(int(p))
                    break
            if track_idx != -1:
                title = " - ".join(parts[track_idx + 1 :])
            else:
                # No numerical part: 'Artist - Title' -> title is last part
                title = parts[-1]
    else:
        # Single chunk: check '01. Title', '01 Title', '01-Title'
        m = re.match(r"^(\d{1,3})[\.\s\-]+(.+)$", clean_stem)
        if m:
            track = str(int(m.group(1)))
            title = m.group(2).strip(" .-_")
        else:
            # Pure track number without title like '01'
            if re.match(r"^\d{1,3}$", clean_stem.strip(" .-_")):
                track = str(int(clean_stem.strip(" .-_")))
                title = None
            else:
                title = clean_stem.strip(" .-_")

    return {"disc": disc, "track": track, "title": title}


def format_audio_filename(
    track: str | int | None,
    title: str | None,
    ext: str,
    disc: str | int | None = None,
) -> str | None:
    """Generate clean filename: '{track} - {title}{ext}' with no leading zeros."""
    if not title or not title.strip():
        return None

    clean_title = sanitize_filename_component(title.strip())
    if not clean_title:
        return None

    track_num_str = ""
    if track:
        t_clean, disc_extracted = normalize_track_number(str(track))
        if t_clean:
            track_num_str = t_clean.split("/")[0].strip()
        if not disc and disc_extracted:
            disc = disc_extracted

    if disc and str(disc).isdigit() and int(disc) > 1:
        prefix = f"{int(disc)}-{track_num_str}" if track_num_str else str(disc)
    else:
        prefix = track_num_str

    if prefix:
        return f"{prefix} - {clean_title}{ext}"
    return f"{clean_title}{ext}"


class TagCleaner:
    """Sanitize and clean noisy audio file tags."""

    def __init__(
        self,
        clean_retailer: bool = True,
        uncensor: bool = True,
        normalize_track_numbers: bool = True,
        clean_whitespace: bool = True,
        rename_files: bool = False,
    ) -> None:
        """Initialize TagCleaner with enabled rule toggles."""
        self.clean_retailer = clean_retailer
        self.uncensor = uncensor
        self.normalize_track_numbers = normalize_track_numbers
        self.clean_whitespace = clean_whitespace
        self.rename_files = rename_files

    def clean_file(
        self,
        file_path: str,
        dry_run: bool = False,
        claimed_targets: set[str] | None = None,
    ) -> FileCleanResult:
        """Clean tags for a single audio file."""
        if not os.path.exists(file_path):
            return FileCleanResult(file_path=file_path, error="File not found")

        _, ext = os.path.splitext(file_path.lower())
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return FileCleanResult(file_path=file_path, error=f"Unsupported file extension: {ext}")

        try:
            import mutagen

            audio = mutagen.File(file_path, easy=True)
            if audio is None:
                return FileCleanResult(file_path=file_path, error="Failed to parse audio metadata")

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

                if extracted_disc and ("discnumber" not in audio or not audio["discnumber"]):
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

            # 5. Optional file renaming
            if self.rename_files:
                curr_track = (
                    audio.get("tracknumber")[0]
                    if "tracknumber" in audio and audio["tracknumber"]
                    else None
                )
                curr_title = audio.get("title")[0] if "title" in audio and audio["title"] else None
                curr_disc = (
                    audio.get("discnumber")[0]
                    if "discnumber" in audio and audio["discnumber"]
                    else None
                )

                # Fallback to path / filename if metadata is missing
                if not curr_title or not curr_track or not curr_disc:
                    path_meta = parse_audio_path_fallback(file_path)
                    if not curr_title and path_meta.get("title"):
                        curr_title = path_meta["title"]
                    if not curr_track and path_meta.get("track"):
                        curr_track = path_meta["track"]
                    if not curr_disc and path_meta.get("disc"):
                        curr_disc = path_meta["disc"]

                dir_name, old_fname = os.path.split(file_path)
                _, file_ext = os.path.splitext(old_fname)
                new_fname = format_audio_filename(
                    track=curr_track,
                    title=curr_title,
                    ext=file_ext,
                    disc=curr_disc,
                )

                if new_fname and new_fname != old_fname:
                    target_fpath = os.path.join(dir_name, new_fname)
                    target_key = os.path.abspath(target_fpath).lower()
                    is_case_only = target_fpath.lower() == file_path.lower()

                    if not is_case_only and os.path.exists(target_fpath):
                        logger.warning(
                            f"Cannot rename '{file_path}' to '{target_fpath}': Target exists."
                        )
                    elif claimed_targets is not None and target_key in claimed_targets:
                        logger.warning(
                            f"Cannot rename '{file_path}' to '{target_fpath}': "
                            "Target conflicts with another file in batch."
                        )
                    else:
                        if claimed_targets is not None:
                            claimed_targets.add(target_key)
                        changes.append(
                            TagChange(
                                field="Filename",
                                old_value=old_fname,
                                new_value=new_fname,
                            )
                        )
                        if not dry_run:
                            # Save ID3 tags before renaming file
                            audio.save()
                            os.rename(file_path, target_fpath)
                            file_path = target_fpath

            if changes and not dry_run and not self.rename_files:
                audio.save()

            return FileCleanResult(file_path=file_path, changed=bool(changes), changes=changes)
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
        claimed_targets: set[str] = set()
        if recursive:
            for root, _, files in os.walk(target_path):
                for fname in sorted(files):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_AUDIO_EXTENSIONS:
                        fpath = os.path.join(root, fname)
                        results.append(
                            self.clean_file(fpath, dry_run=dry_run, claimed_targets=claimed_targets)
                        )
        else:
            for fname in sorted(os.listdir(target_path)):
                fpath = os.path.join(target_path, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_AUDIO_EXTENSIONS:
                        results.append(
                            self.clean_file(fpath, dry_run=dry_run, claimed_targets=claimed_targets)
                        )

        return results


TAG_NAME_ALIASES: dict[str, list[str]] = {
    "album": ["album", "talb", "\xa9alb"],
    "title": ["title", "tit2", "\xa9nam"],
    "artist": ["artist", "tpe1", "\xa9art"],
    "albumartist": ["albumartist", "album_artist", "tpe2", "aart"],
    "genre": ["genre", "genres", "tcon", "\xa9gen"],
    "mood": ["mood", "moods", "tmoo", "txxx:mood"],
    "bpm": ["bpm", "tbpm", "tmpo"],
    "track": ["track", "tracknumber", "trck", "trkn"],
    "tracknumber": ["tracknumber", "track", "trck", "trkn"],
    "disc": ["disc", "discnumber", "tpos", "disk"],
    "discnumber": ["discnumber", "disc", "tpos", "disk"],
    "date": ["date", "year", "tdrc", "tyer", "\xa9day"],
    "year": ["year", "date", "tdrc", "tyer", "\xa9day"],
    "comment": ["comment", "comm", "\xa9cmt"],
    "composer": ["composer", "tcom", "\xa9wrt"],
}


class FileInspectResult(BaseModel):
    """Inspection result for an audio file."""

    file_path: str
    tags: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def inspect_file_tags(
    file_path: str,
    raw: bool = False,
    tag_filters: list[str] | None = None,
) -> FileInspectResult:
    """Read metadata tags from an audio file."""
    if not os.path.exists(file_path):
        return FileInspectResult(file_path=file_path, error="File not found")

    _, ext = os.path.splitext(file_path.lower())
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        return FileInspectResult(
            file_path=file_path, error=f"Unsupported audio file extension: {ext}"
        )

    try:
        import mutagen

        if raw:
            audio = mutagen.File(file_path)
            if audio is None:
                return FileInspectResult(file_path=file_path, error="Could not read metadata tags")

            raw_dict: dict[str, str] = {}
            for k in sorted(audio.keys()):
                val = audio.get(k)
                if val is not None:
                    if hasattr(val, "text"):
                        raw_dict[str(k)] = (
                            ", ".join(str(x) for x in val.text)
                            if isinstance(val.text, list)
                            else str(val.text)
                        )
                    else:
                        raw_dict[str(k)] = str(val)

            if tag_filters:
                filter_set = {f.lower().strip() for f in tag_filters if f.strip()}
                expanded_keys = set()
                for f in filter_set:
                    if f in TAG_NAME_ALIASES:
                        expanded_keys.update(TAG_NAME_ALIASES[f])
                    else:
                        expanded_keys.add(f)

                filtered_dict = {
                    k: v
                    for k, v in raw_dict.items()
                    if k.lower() in expanded_keys
                    or any(k.lower().startswith(f) for f in expanded_keys)
                }
                return FileInspectResult(file_path=file_path, tags=filtered_dict)

            return FileInspectResult(file_path=file_path, tags=raw_dict)

        # Standard easy mode
        audio = mutagen.File(file_path, easy=True)
        if audio is None:
            return FileInspectResult(file_path=file_path, error="Could not read metadata tags")

        standard_tags: dict[str, Any] = {}
        key_order = [
            "artist",
            "album",
            "title",
            "tracknumber",
            "discnumber",
            "genre",
            "mood",
            "bpm",
            "date",
            "albumartist",
        ]

        for k in key_order:
            val = audio.get(k)
            if val:
                standard_tags[k] = val[0] if isinstance(val, list) and len(val) == 1 else val

        if ext == ".mp3" and ("mood" not in standard_tags or "bpm" not in standard_tags):
            try:
                raw_audio = mutagen.File(file_path)
                if raw_audio:
                    if "mood" not in standard_tags:
                        tmoo = raw_audio.get("TMOO") or raw_audio.get("TXXX:MOOD")
                        if tmoo and hasattr(tmoo, "text") and tmoo.text:
                            standard_tags["mood"] = (
                                tmoo.text[0] if len(tmoo.text) == 1 else list(tmoo.text)
                            )
                    if "bpm" not in standard_tags:
                        tbpm = raw_audio.get("TBPM")
                        if tbpm and hasattr(tbpm, "text") and tbpm.text:
                            standard_tags["bpm"] = str(tbpm.text[0])
            except Exception:
                pass

        if tag_filters:
            filter_set = {f.lower().strip() for f in tag_filters if f.strip()}
            expanded_keys = set()
            for f in filter_set:
                if f in TAG_NAME_ALIASES:
                    expanded_keys.update(TAG_NAME_ALIASES[f])
                else:
                    expanded_keys.add(f)

            filtered_tags = {k: v for k, v in standard_tags.items() if k.lower() in expanded_keys}
            return FileInspectResult(file_path=file_path, tags=filtered_tags)

        return FileInspectResult(file_path=file_path, tags=standard_tags)
    except Exception as err:
        logger.warning(f"Error inspecting file '{file_path}': {err}")
        return FileInspectResult(file_path=file_path, error=str(err))


def inspect_path(
    target_path: str,
    raw: bool = False,
    tag_filters: list[str] | None = None,
    recursive: bool = True,
) -> list[FileInspectResult]:
    """Inspect metadata tags for a file or directory."""
    if not os.path.exists(target_path):
        return [FileInspectResult(file_path=target_path, error="Path does not exist")]

    if os.path.isfile(target_path):
        return [inspect_file_tags(target_path, raw=raw, tag_filters=tag_filters)]

    results: list[FileInspectResult] = []
    if recursive:
        for root, _, files in os.walk(target_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_AUDIO_EXTENSIONS:
                    fpath = os.path.join(root, fname)
                    results.append(inspect_file_tags(fpath, raw=raw, tag_filters=tag_filters))
    else:
        for fname in sorted(os.listdir(target_path)):
            fpath = os.path.join(target_path, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_AUDIO_EXTENSIONS:
                    results.append(inspect_file_tags(fpath, raw=raw, tag_filters=tag_filters))

    return results
