"""Audio file inspection and validation utilities."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_valid_audio_header(file_path: str) -> bool:
    """Validate whether file begins with supported audio container magic bytes."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 64:
            return False
        with open(file_path, "rb") as f:
            header = f.read(64)
        if len(header) < 12:
            return False
        # FLAC
        if header.startswith(b"fLaC"):
            return True
        # MP3 (ID3v2)
        if header.startswith(b"ID3"):
            return True
        # MP3 (sync frames: 0xFF followed by MPEG audio sync bits)
        if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            return True
        # OGG
        if header.startswith(b"OggS"):
            return True
        # WAV / WAVE
        if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
            return True
        # MP4 / M4A / AAC
        if header[4:8] == b"ftyp":
            return True
        # AIFF
        if header.startswith(b"FORM") and header[8:12] in (b"AIFF", b"AIFC"):
            return True
        return False
    except Exception:
        return False


def get_audio_duration(file_path: str) -> float | None:
    """Read stream duration in seconds from audio file header using mutagen."""
    try:
        import mutagen

        tag_data = mutagen.File(file_path)
        if tag_data is not None and tag_data.info and hasattr(tag_data.info, "length"):
            length = float(tag_data.info.length)
            if length > 0.0:
                return length
    except Exception as err:
        logger.debug(f"Failed to read audio duration from '{file_path}': {err}")
    return None


def calculate_audio_window(
    file_path: str | None = None,
    duration: float | None = None,
    target_duration: float = 90.0,
) -> tuple[float, float]:
    """Calculate symmetrical midpoint audio window (start_sec, end_sec) in seconds.

    Centers a target_duration slice (default: 90s) directly in the middle of the track,
    skipping long intros and outros. For tracks shorter than target_duration, loads from
    0 to duration (or target_duration if duration is unknown).
    """
    if duration is None and file_path is not None:
        duration = get_audio_duration(file_path)

    if duration is None or duration <= 0.0:
        return 0.0, target_duration

    if duration <= target_duration:
        return 0.0, float(duration)

    start_sec = (duration - target_duration) / 2.0
    end_sec = start_sec + target_duration
    return round(start_sec, 2), round(end_sec, 2)


__all__ = [
    "calculate_audio_window",
    "get_audio_duration",
    "is_valid_audio_header",
]
