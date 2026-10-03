"""Audio file inspection and validation utilities."""

from __future__ import annotations

import logging
import os
import subprocess

import numpy as np

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
        # MP3 / AAC ADTS (sync frames: 0xFF followed by MPEG/AAC audio sync bits)
        if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            return True
        if header[0] == 0xFF and (header[1] & 0xF0) == 0xF0:
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
        # ASF / WMA
        if header.startswith(b"\x30\x26\xb2\x75"):
            return True
        # Monkey's Audio (APE)
        if header.startswith(b"MAC "):
            return True
        # Musepack (MPC)
        if header.startswith(b"MP+") or header.startswith(b"MPC"):
            return True
        # WavPack
        if header.startswith(b"wvpk"):
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
    """Calculate symmetrical midpoint audio window (start_sec, end_sec) in seconds."""
    if duration is None and file_path is not None:
        duration = get_audio_duration(file_path)

    if duration is None or duration <= 0.0:
        return 0.0, target_duration

    if duration <= target_duration:
        return 0.0, float(duration)

    start_sec = (duration - target_duration) / 2.0
    end_sec = start_sec + target_duration
    return round(start_sec, 2), round(end_sec, 2)


def decode_audio_stream(
    file_path: str,
    sample_rate: int = 44100,
    start_sec: float = 0.0,
    duration: float = 90.0,
) -> np.ndarray | None:
    """Decode audio slice using FFmpeg directly to raw float32 PCM numpy array."""
    if not file_path or not os.path.exists(file_path):
        return None

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "1",
        "-v",
        "error",
        "-ss",
        str(max(0.0, start_sec)),
        "-t",
        str(max(1.0, duration)),
        "-i",
        file_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15.0,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        return np.frombuffer(proc.stdout, dtype=np.float32)
    except Exception as err:
        logger.debug(f"FFmpeg audio decode failed for '{file_path}': {err}")
        return None


def decode_audio(
    file_path: str,
    start_sec: float = 0.0,
    end_sec: float = 90.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Decode 44.1kHz and 16kHz audio buffers using FFmpeg for MIR and ML analysis."""
    if not file_path or not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return None, None

    duration = max(1.0, end_sec - start_sec)
    audio_44k = decode_audio_stream(
        file_path, sample_rate=44100, start_sec=start_sec, duration=duration
    )
    if audio_44k is None or len(audio_44k) == 0:
        return None, None

    audio_16k: np.ndarray | None = None
    try:
        import essentia.standard as es

        audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_44k)
    except Exception:
        audio_16k = decode_audio_stream(
            file_path, sample_rate=16000, start_sec=start_sec, duration=duration
        )

    return audio_44k, audio_16k


# Direct alias for any legacy callers
decode_audio_isolated = decode_audio

__all__ = [
    "calculate_audio_window",
    "decode_audio",
    "decode_audio_isolated",
    "decode_audio_stream",
    "get_audio_duration",
    "is_valid_audio_header",
]
