"""Audio file inspection and validation utilities."""

import os


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
