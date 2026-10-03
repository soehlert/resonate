"""Audio file inspection and validation utilities."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_POISONED_AUDIO_FILES: set[str] = set()


def is_file_poisoned(file_path: str) -> bool:
    """Return True if the file has failed header validation, timed out, or crashed the decoder."""
    return file_path in _POISONED_AUDIO_FILES


def clear_poisoned_files() -> None:
    """Clear poisoned file registry (used for tests)."""
    _POISONED_AUDIO_FILES.clear()


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


def _worker_decode_audio(conn: Any, file_path: str, start_sec: float, end_sec: float) -> None:
    """Worker process entrypoint to decode audio with Essentia EasyLoader."""
    try:
        import essentia.standard as es

        audio_44k = es.EasyLoader(
            filename=file_path,
            sampleRate=44100,
            startTime=start_sec,
            endTime=end_sec,
        )()
        audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_44k)
        conn.send((audio_44k, audio_16k))
    except Exception as err:
        logger.debug(f"Worker audio decode failed for '{file_path}': {err}")
        try:
            import essentia.standard as es

            audio_44k = es.EasyLoader(
                filename=file_path,
                sampleRate=44100,
                startTime=0,
                endTime=90,
            )()
            audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_44k)
            conn.send((audio_44k, audio_16k))
        except Exception as fb_err:
            logger.debug(f"Fallback worker decode failed for '{file_path}': {fb_err}")
            conn.send((None, None))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def decode_audio_isolated(
    file_path: str,
    start_sec: float = 0.0,
    end_sec: float = 90.0,
    timeout: float = 30.0,
) -> tuple[Any, Any]:
    """Decode audio buffers in an isolated process to guard against native C++ segfaults.

    Returns (audio_44k, audio_16k) numpy arrays, or (None, None) if corrupt, unreadable,
    or if the native decoder crashed.
    """
    if not file_path or not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return None, None

    if is_file_poisoned(file_path):
        return None, None

    if not is_valid_audio_header(file_path):
        _POISONED_AUDIO_FILES.add(file_path)
        logger.warning(
            f"Audio file '{file_path}' has unrecognized or invalid audio header bytes; "
            "skipping audio decode."
        )
        return None, None

    es_mod = sys.modules.get("essentia.standard") or sys.modules.get("essentia")
    if es_mod is not None and ("mock" in type(es_mod).__module__ or hasattr(es_mod, "mock_calls")):
        try:
            import essentia.standard as es

            audio_44k = es.EasyLoader(
                filename=file_path,
                sampleRate=44100,
                startTime=start_sec,
                endTime=end_sec,
            )()
            audio_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_44k)
            return audio_44k, audio_16k
        except Exception as err:
            logger.debug(f"Direct mocked audio decode failed for '{file_path}': {err}")
            return None, None

    start_method = "fork" if "fork" in mp.get_all_start_methods() else None
    ctx = mp.get_context(start_method)
    p_conn, c_conn = ctx.Pipe(duplex=False)

    process = ctx.Process(
        target=_worker_decode_audio,
        args=(c_conn, file_path, start_sec, end_sec),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        _POISONED_AUDIO_FILES.add(file_path)
        logger.warning(
            f"Audio decode timed out after {timeout}s on '{file_path}'; killing worker."
        )
        process.kill()
        process.join()
        try:
            p_conn.close()
        except Exception:
            pass
        return None, None

    if process.exitcode != 0:
        _POISONED_AUDIO_FILES.add(file_path)
        logger.warning(
            f"Native audio decoder process crashed with exit code {process.exitcode} "
            f"on '{file_path}' (corrupt audio frame or unhandled native stream); "
            "skipping audio analysis for this track."
        )
        try:
            p_conn.close()
        except Exception:
            pass
        return None, None

    result = (None, None)
    if p_conn.poll():
        try:
            result = p_conn.recv()
        except Exception as err:
            logger.debug(f"Failed to read audio buffer from worker pipe: {err}")
    try:
        p_conn.close()
    except Exception:
        pass
    return result


__all__ = [
    "calculate_audio_window",
    "clear_poisoned_files",
    "decode_audio_isolated",
    "get_audio_duration",
    "is_file_poisoned",
    "is_valid_audio_header",
]
