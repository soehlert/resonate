"""Lyrics retrieval and sentiment/mood analysis module for Resonate."""

import logging
import os
import re
from typing import Any

import numpy as np
import requests

from resonate.models import LyricsAnalysisResult
from resonate.modules.external_metadata import clean_retailer_noise, uncensor_title
from resonate.utils.state import StateManager

logger = logging.getLogger(__name__)

# Basic valence dictionary for rapid polarity scoring
POSITIVE_WORDS = {
    "happy",
    "joy",
    "joyful",
    "love",
    "loving",
    "beautiful",
    "peace",
    "peaceful",
    "celebrate",
    "celebration",
    "party",
    "dance",
    "dancing",
    "bright",
    "sun",
    "sunshine",
    "smile",
    "smiling",
    "laugh",
    "laughter",
    "cheer",
    "cheerful",
    "good",
    "wonderful",
    "paradise",
    "heaven",
    "blessed",
    "delight",
    "fun",
    "free",
    "freedom",
    "alive",
    "shine",
    "glowing",
    "sweet",
    "hope",
    "hopeful",
}

NEGATIVE_WORDS = {
    "kill",
    "killing",
    "killer",
    "die",
    "died",
    "dead",
    "death",
    "gun",
    "guns",
    "bullet",
    "bullets",
    "shoot",
    "shooting",
    "shot",
    "suicide",
    "blood",
    "bleed",
    "bleeding",
    "bloody",
    "grave",
    "murder",
    "corpse",
    "pain",
    "hate",
    "hating",
    "hell",
    "gloom",
    "gloomy",
    "sorrow",
    "agony",
    "depressed",
    "depression",
    "cry",
    "crying",
    "tears",
    "lonely",
    "alone",
    "hopeless",
    "broken",
    "grief",
    "darkness",
    "suffer",
    "suffering",
    "wound",
    "tragedy",
    "poison",
    "choke",
    "drown",
}

LYRICAL_MOOD_DESCRIPTIONS: dict[str, str] = {
    "Dark": "dark, violent, deadly, tragic, suicidal, bleeding, gun, kill, painful death, horror",
    "Melancholic": "sad, melancholic, weeping, heartbreak, lonely, crying, sorrow, grief, breakup",
    "Romantic": "romantic, love, devotion, kiss, embrace, sweetheart, forever in love, passion",
    "Party": "party, club, dancing, celebration, drinks, weekend, nightlife, energetic fun",
    "Happy": "happy, joyful, sunny, cheerful, feeling good, wonderful, smiling, laughing, delight",
    "Calm": "calm, peaceful, serene, gentle, quiet, tranquil, softly sleeping, meditation",
    "Energetic": "energetic, powerhouse, fast driving, rebellious, screaming, running, wild",
}


def clean_lyrics_text(text: str) -> str:
    """Clean and normalize lyrics text by removing timestamp tags and section markers."""
    if not text:
        return ""

    # Remove synced lyrics timestamps like [01:23.45] or [01:23]
    cleaned = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]", "", text)
    # Remove standard section headers like [Chorus], [Verse 1], (Bridge), etc.
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    cleaned = re.sub(
        r"\((?:Chorus|Verse|Bridge|Outro|Intro|Guitar Solo)[^\)]*\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace
    lines = [line.strip() for line in cleaned.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def calculate_valence_score(lyrics_text: str) -> float:
    """Calculate normalized sentiment valence polarity between -1.0 (dark) and +1.0 (bright)."""
    if not lyrics_text:
        return 0.0

    words = re.findall(r"\b[a-z]{3,}\b", lyrics_text.lower())
    if not words:
        return 0.0

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    total_hits = pos_count + neg_count
    if total_hits == 0:
        return 0.0

    # Net polarity scaled between -1.0 and 1.0
    return (pos_count - neg_count) / total_hits


class LyricsFetcher:
    """Multi-tier lyrics retrieval engine supporting local embedded tags and LRCLIB."""

    def __init__(
        self,
        state_manager: StateManager | None = None,
        prefer_embedded: bool = True,
        lrclib_url: str = "https://lrclib.net",
        request_timeout: float = 5.0,
    ) -> None:
        """Initialize LyricsFetcher."""
        self.state_manager = state_manager
        self.prefer_embedded = prefer_embedded
        self.lrclib_url = lrclib_url.rstrip("/")
        self.request_timeout = request_timeout

    def extract_embedded_lyrics(self, file_path: str) -> str | None:
        """Extract lyrics from embedded audio file metadata or sidecar files."""
        if not file_path or not os.path.exists(file_path):
            return None

        # 1. Check sidecar .lrc or .txt files first
        base_path, _ = os.path.splitext(file_path)
        for ext in (".lrc", ".txt"):
            sidecar_path = base_path + ext
            if os.path.exists(sidecar_path):
                try:
                    with open(sidecar_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if content:
                            return content
                except Exception as err:
                    logger.debug(f"Failed to read sidecar file '{sidecar_path}': {err}")

        # 2. Extract embedded tags using mutagen
        try:
            from mutagen import File
            from mutagen.flac import FLAC
            from mutagen.id3 import ID3
            from mutagen.mp4 import MP4

            _, ext = os.path.splitext(file_path.lower())
            if ext == ".mp3":
                try:
                    id3 = ID3(file_path)
                    for frame in id3.values():
                        if frame.FrameID == "USLT":
                            if hasattr(frame, "text") and frame.text:
                                return str(frame.text)
                        elif frame.FrameID == "TXXX":
                            if hasattr(frame, "desc") and frame.desc.upper() in (
                                "LYRICS",
                                "UNSYNCED LYRICS",
                            ):
                                if hasattr(frame, "text") and frame.text:
                                    t_val = (
                                        frame.text[0]
                                        if isinstance(frame.text, list)
                                        else frame.text
                                    )
                                    return str(t_val)
                except Exception:
                    pass
            elif ext == ".flac":
                try:
                    flac = FLAC(file_path)
                    for tag in ("lyrics", "unsyncedlyrics", "unsynced lyrics", "lyric"):
                        if tag in flac:
                            vals = flac[tag]
                            if vals and vals[0].strip():
                                return vals[0].strip()
                except Exception:
                    pass
            elif ext in (".m4a", ".mp4"):
                try:
                    mp4 = MP4(file_path)
                    if "\xa9lyr" in mp4:
                        vals = mp4["\xa9lyr"]
                        if vals and vals[0].strip():
                            return vals[0].strip()
                except Exception:
                    pass
            else:
                # Generic Mutagen handler
                audio = File(file_path)
                if audio and hasattr(audio, "tags") and audio.tags:
                    for key, val in audio.tags.items():
                        if "lyrics" in key.lower():
                            if isinstance(val, list) and val:
                                return str(val[0])
                            return str(val)
        except Exception as err:
            logger.debug(f"Mutagen extraction failed for '{file_path}': {err}")

        return None

    def _query_lrclib_api(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        duration: int | None = None,
    ) -> str | None:
        """Internal helper for LRCLIB /api/get and /api/search."""
        artist_clean = artist.strip()
        title_clean = title.strip()

        # Step 1: Try exact lookup via /api/get
        try:
            params: dict[str, Any] = {
                "artist_name": artist_clean,
                "track_name": title_clean,
            }
            if album and album.strip():
                params["album_name"] = album.strip()
            if duration:
                params["duration"] = int(duration)

            resp = requests.get(
                f"{self.lrclib_url}/api/get",
                params=params,
                timeout=self.request_timeout,
                headers={"User-Agent": "Resonate/0.1.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                plain = data.get("plainLyrics")
                if plain and plain.strip():
                    return plain.strip()
                synced = data.get("syncedLyrics")
                if synced and synced.strip():
                    return clean_lyrics_text(synced)
        except Exception as err:
            logger.debug(f"LRCLIB /api/get failed for '{artist} - {title}': {err}")

        # Step 2: Fallback to /api/search
        try:
            resp = requests.get(
                f"{self.lrclib_url}/api/search",
                params={"q": f"{artist_clean} {title_clean}"},
                timeout=self.request_timeout,
                headers={"User-Agent": "Resonate/0.1.0"},
            )
            if resp.status_code == 200:
                results = resp.json()
                if isinstance(results, list) and len(results) > 0:
                    first = results[0]
                    plain = first.get("plainLyrics")
                    if plain and plain.strip():
                        return plain.strip()
                    synced = first.get("syncedLyrics")
                    if synced and synced.strip():
                        return clean_lyrics_text(synced)
        except Exception as err:
            logger.debug(f"LRCLIB /api/search failed for '{artist} - {title}': {err}")

        return None

    def fetch_lrclib_lyrics(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        duration: int | None = None,
    ) -> str | None:
        """Fetch lyrics from LRCLIB with uncensored title and cleaned album fallbacks."""
        if not artist or not title:
            return None

        # 1. Primary lookup with exact title/album
        result = self._query_lrclib_api(artist, title, album=album, duration=duration)
        if result:
            return result

        # 2. Try uncensored title and retailer-cleaned album variants if different
        uncensored = uncensor_title(title)
        cleaned_album = clean_retailer_noise(album) if album else None

        if (uncensored and uncensored.lower() != title.lower()) or (
            cleaned_album and album and cleaned_album.lower() != album.lower()
        ):
            target_title = uncensored if uncensored else title
            target_album = cleaned_album if cleaned_album else album
            result = self._query_lrclib_api(
                artist, target_title, album=target_album, duration=duration
            )
            if result:
                return result

        return None

    def get_lyrics(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        file_path: str | None = None,
        duration: int | None = None,
    ) -> tuple[str | None, str]:
        """Retrieve lyrics coordinating cache -> local embedded -> LRCLIB remote."""
        if not artist or not title:
            return (None, "none")

        # 1. Check SQLite state cache
        if self.state_manager:
            cached = self.state_manager.get_cached_lyrics(artist, title)
            if cached and cached.get("lyrics_text"):
                return (cached["lyrics_text"], f"cached:{cached.get('source', 'unknown')}")

        lyrics_text: str | None = None
        source: str = "none"

        # 2. Local embedded check
        if self.prefer_embedded and file_path:
            lyrics_text = self.extract_embedded_lyrics(file_path)
            if lyrics_text:
                source = "embedded"

        # 3. LRCLIB remote fetch
        if not lyrics_text:
            lyrics_text = self.fetch_lrclib_lyrics(artist, title, album=album, duration=duration)
            if lyrics_text:
                source = "lrclib"

        # 4. Fallback to embedded if not preferred earlier
        if not lyrics_text and not self.prefer_embedded and file_path:
            lyrics_text = self.extract_embedded_lyrics(file_path)
            if lyrics_text:
                source = "embedded"

        # Save to SQLite cache if found
        if lyrics_text and self.state_manager:
            self.state_manager.save_cached_lyrics(artist, title, lyrics_text, source)

        return (lyrics_text, source)

    def analyze_lyrics(
        self,
        lyrics_text: str | None,
        source: str = "none",
        tag_mapper: Any | None = None,
    ) -> LyricsAnalysisResult:
        """Analyze lyrics text to compute continuous valence and semantic mood scores."""
        if not lyrics_text:
            return LyricsAnalysisResult(
                lyrics_text=None,
                source=source,
                valence_score=0.0,
                mood_scores={},
            )

        cleaned = clean_lyrics_text(lyrics_text)
        valence = calculate_valence_score(cleaned)
        mood_scores: dict[str, float] = {}

        # Semantic mood mapping using SentenceTransformer embeddings via TagMapper
        if tag_mapper is not None and cleaned:
            model = tag_mapper._get_model() if hasattr(tag_mapper, "_get_model") else None
            if model is not None:
                try:
                    # Take representative sample (first 1000 chars) to keep inference fast
                    sample_text = cleaned[:1200]
                    target_moods = list(LYRICAL_MOOD_DESCRIPTIONS.keys())
                    descriptions = list(LYRICAL_MOOD_DESCRIPTIONS.values())

                    lyric_emb = model.encode([sample_text], convert_to_tensor=False)
                    desc_emb = model.encode(descriptions, convert_to_tensor=False)

                    lyric_arr = np.asarray(lyric_emb, dtype=np.float32)
                    desc_arr = np.asarray(desc_emb, dtype=np.float32)

                    lyric_norm = lyric_arr / np.maximum(
                        np.linalg.norm(lyric_arr, axis=1, keepdims=True), 1e-9
                    )
                    desc_norm = desc_arr / np.maximum(
                        np.linalg.norm(desc_arr, axis=1, keepdims=True), 1e-9
                    )

                    sims = np.dot(lyric_norm, desc_norm.T)[0]

                    for mood_name, sim in zip(target_moods, sims, strict=False):
                        score = float(sim)
                        # Modulate score with lexical valence for polarization
                        if mood_name == "Dark" and valence < -0.2:
                            score = min(1.0, score + abs(valence) * 0.25)
                        elif mood_name == "Happy" and valence > 0.2:
                            score = min(1.0, score + valence * 0.25)
                        elif mood_name == "Happy" and valence < -0.3:
                            score = max(0.0, score - abs(valence) * 0.40)
                        elif mood_name == "Melancholic" and valence < -0.2:
                            score = min(1.0, score + abs(valence) * 0.20)

                        mood_scores[mood_name] = round(score, 4)
                except Exception as err:
                    logger.warning(f"Failed semantic lyrics embedding analysis: {err}")

        # If no model is available, provide basic lexical fallback scores
        if not mood_scores and cleaned:
            if valence < -0.3:
                mood_scores["Dark"] = round(abs(valence) * 0.8, 4)
                mood_scores["Melancholic"] = round(abs(valence) * 0.6, 4)
            elif valence > 0.3:
                mood_scores["Happy"] = round(valence * 0.8, 4)

        return LyricsAnalysisResult(
            lyrics_text=lyrics_text,
            source=source,
            valence_score=round(valence, 4),
            mood_scores=mood_scores,
        )
