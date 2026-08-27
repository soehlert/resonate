"""External metadata fetchers for MusicBrainz and Discogs integrations."""

import json
import logging
import re
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


ARTIST_ALIASES: dict[str, list[str]] = {
    "ye": ["kanye west"],
    "kanye west": ["ye"],
    "kanye": ["kanye west", "ye"],
    "yasiin bey": ["mos def"],
    "mos def": ["yasiin bey"],
    "childish gambino": ["donald glover"],
    "donald glover": ["childish gambino"],
    "2pac": ["tupac", "tupac shakur"],
    "tupac": ["2pac", "tupac shakur"],
    "tupac shakur": ["2pac", "tupac"],
    "snoop dogg": ["snoop lion"],
    "snoop lion": ["snoop dogg"],
    "mf doom": ["doom", "viktor vaughn", "king geedorah"],
    "doom": ["mf doom"],
}


def get_artist_aliases(artist: str) -> list[str]:
    """Return all known alias names and grammatical variants for an artist."""
    clean = artist.lower().strip()
    aliases = [artist]

    # 1. Known dictionary rebrands (e.g. Ye -> Kanye West)
    if clean in ARTIST_ALIASES:
        for alias in ARTIST_ALIASES[clean]:
            if alias.lower() != clean and alias not in aliases:
                aliases.append(alias)

    # 2. Compound band name variants (e.g. 'Jay & Americans' <-> 'Jay & The Americans')
    if " & " in artist and " & the " not in clean:
        alt = artist.replace(" & ", " & The ")
        if alt not in aliases:
            aliases.append(alt)
    elif " & the " in clean:
        idx = clean.index(" & the ")
        alt = artist[:idx] + " & " + artist[idx + 7 :]
        if alt not in aliases:
            aliases.append(alt)

    if " and " in clean and " and the " not in clean:
        alt = artist.replace(" and ", " and The ").replace(" AND ", " AND The ")
        if alt not in aliases:
            aliases.append(alt)
    elif " and the " in clean:
        idx = clean.index(" and the ")
        alt = artist[:idx] + " and " + artist[idx + 9 :]
        if alt not in aliases:
            aliases.append(alt)

    # 3. Leading 'The' variants (e.g. 'The Beatles' <-> 'Beatles')
    if clean.startswith("the ") and len(clean) > 4:
        without_the = artist[4:].strip()
        if without_the and without_the not in aliases:
            aliases.append(without_the)
    elif not clean.startswith("the ") and len(clean) > 2:
        with_the = f"The {artist}"
        if with_the not in aliases:
            aliases.append(with_the)

    return aliases


def _normalize_band_name(name: str) -> str:
    """Normalize artist name by removing leading and compound articles (the)."""
    s = name.lower().strip()
    if s.startswith("the "):
        s = s[4:].strip()
    s = s.replace(" & the ", " & ")
    s = s.replace(" and the ", " & ")
    s = s.replace(" and ", " & ")
    s = s.replace(" + the ", " + ")
    return s.strip()


def artist_matches(expected: str, candidate: str) -> bool:
    """Verify that candidate artist name matches expected artist (preventing Ye matching Yes)."""
    exp = expected.lower().strip()
    cand = candidate.lower().strip()
    if not exp or not cand:
        return True
    if exp == cand:
        return True
    if exp.startswith("the ") and exp[4:].strip() == cand:
        return True
    if cand.startswith("the ") and cand[4:].strip() == exp:
        return True

    # Compound band name normalization (e.g. 'Jay & Americans' vs 'Jay & The Americans')
    norm_exp = _normalize_band_name(exp)
    norm_cand = _normalize_band_name(cand)
    if norm_exp and norm_cand and norm_exp == norm_cand:
        return True

    for sep in [" feat", " ft.", " with ", " & ", " and ", " / ", ", ", " x ", " vs ", " vs. "]:
        if cand.startswith(f"{exp}{sep}") or norm_cand.startswith(f"{norm_exp}{sep}"):
            return True
        if exp in ARTIST_ALIASES:
            for alias in ARTIST_ALIASES[exp]:
                if cand.startswith(f"{alias}{sep}"):
                    return True
    if exp in ARTIST_ALIASES and cand in ARTIST_ALIASES[exp]:
        return True
    if cand in ARTIST_ALIASES and exp in ARTIST_ALIASES[cand]:
        return True
    return False


def _preserve_case_replace(match: re.Match[str], word: str) -> str:
    """Helper to preserve title casing (e.g. F**k -> Fuck, f**k -> fuck)."""
    matched = match.group(0)
    if matched.isupper():
        return word.upper()
    if matched and matched[0].isupper():
        return word.capitalize()
    return word.lower()


def uncensor_title(title: str) -> str:
    """Normalize common profanity censorship masks (e.g. Hatef--k -> Hatefuck) for API queries."""
    if not title:
        return ""
    s = title

    # f--k / f**k / f*ck / f*k
    s = re.sub(
        r"(?i)\bf[\*\-\_\.]{1,3}k\b",
        lambda m: _preserve_case_replace(m, "fuck"),
        s,
    )
    s = re.sub(
        r"(?i)\bf\*ck\b",
        lambda m: _preserve_case_replace(m, "fuck"),
        s,
    )
    s = re.sub(
        r"(?i)(?<=[\w])f[\*\-\_\.]{2,3}k\b",
        lambda m: _preserve_case_replace(m, "fuck"),
        s,
    )
    s = re.sub(
        r"(?i)(?<=[\w])f\*ck\b",
        lambda m: _preserve_case_replace(m, "fuck"),
        s,
    )

    # sh*t / sh!t / s**t / s--t
    s = re.sub(
        r"(?i)\bsh[\*\-\_\.!]{1,2}t\b",
        lambda m: _preserve_case_replace(m, "shit"),
        s,
    )
    s = re.sub(
        r"(?i)\bs[\*\-\_\.]{2}t\b",
        lambda m: _preserve_case_replace(m, "shit"),
        s,
    )
    s = re.sub(
        r"(?i)(?<=[\w])sh[\*\-\_\.!]{1,2}t\b",
        lambda m: _preserve_case_replace(m, "shit"),
        s,
    )

    # b*tch / b**ch
    s = re.sub(
        r"(?i)\bb[\*\-\_\.]{1,2}tch\b",
        lambda m: _preserve_case_replace(m, "bitch"),
        s,
    )
    s = re.sub(
        r"(?i)\bb[\*\-\_\.]{2,3}h\b",
        lambda m: _preserve_case_replace(m, "bitch"),
        s,
    )

    # a**hole / a**
    s = re.sub(
        r"(?i)\ba[\*\-\_\.]{2}hole\b",
        lambda m: _preserve_case_replace(m, "asshole"),
        s,
    )
    s = re.sub(
        r"(?i)\ba[\*\-\_\.]{2}\b",
        lambda m: _preserve_case_replace(m, "ass"),
        s,
    )

    # d*ck / d**k
    s = re.sub(
        r"(?i)\bd[\*\-\_\.]{1,2}ck\b",
        lambda m: _preserve_case_replace(m, "dick"),
        s,
    )
    s = re.sub(
        r"(?i)\bd[\*\-\_\.]{2}k\b",
        lambda m: _preserve_case_replace(m, "dick"),
        s,
    )

    # p*ssy / p***y
    s = re.sub(
        r"(?i)\bp[\*\-\_\.]{1,2}ssy\b",
        lambda m: _preserve_case_replace(m, "pussy"),
        s,
    )
    s = re.sub(
        r"(?i)\bp[\*\-\_\.]{3}y\b",
        lambda m: _preserve_case_replace(m, "pussy"),
        s,
    )

    return s


RETAILER_EXCLUSIVE_PATTERNS = [
    r"(?i)\s*[\(\[\-–—]\s*best\s*buy(?:\s+(?:exclusive|edition|bonus|version|track[s]?|digital|release))*\s*[\)\]]?",
    r"(?i)\s*[\(\[\-–—]\s*target(?:\s+(?:exclusive|edition|bonus|version|track[s]?|digital|release))*\s*[\)\]]?",
    r"(?i)\s*[\(\[\-–—]\s*walmart(?:\s+(?:exclusive|edition|bonus|version|track[s]?|digital|release))*\s*[\)\]]?",
    r"(?i)\s*[\(\[\-–—]\s*itunes(?:\s+(?:exclusive|edition|bonus|version|track[s]?|digital|release))*\s*[\)\]]?",
    r"(?i)\s*[\(\[\-–—]\s*amazon(?:\s+(?:exclusive|edition|bonus|version|track[s]?|digital|release))*\s*[\)\]]?",
    r"(?i)\s*[\(\[\-–—]\s*spotify(?:\s+(?:exclusive|edition|bonus|version|track[s]?|session[s]?))*\s*[\)\]]?",
    r"(?i)\s*[\(\[\-–—]\s*circuit\s*city(?:\s+(?:exclusive|edition|bonus|version))*\s*[\)\]]?",
]


def clean_retailer_noise(album: str | None) -> str | None:
    """Strip retailer marketing noise (e.g. Best Buy Exclusive) while keeping musical editions."""
    if not album:
        return album
    cleaned = album
    for pattern in RETAILER_EXCLUSIVE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    # Strip any trailing hyphens, commas, or empty brackets left over
    cleaned = re.sub(r"[\(\[]\s*[\)\]]", "", cleaned)
    cleaned = re.sub(r"\s*[\-–—,]\s*$", "", cleaned)
    return cleaned.strip()


class MusicBrainzFetcher:
    """Fetch tags and genres from the MusicBrainz API."""

    def __init__(self) -> None:
        """Initialize MusicBrainzFetcher."""
        self.headers = {"User-Agent": "Resonate/1.0.0 (https://github.com/soehlert/resonate)"}
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce MusicBrainz API rate limit of 3.5 seconds per request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 3.5:
            time.sleep(3.5 - elapsed)
        self._last_request_time = time.time()

    def resolve_canonical_artist(self, artist: str) -> str | None:
        """Query MusicBrainz artist search to resolve aliases/rebrands to canonical name."""
        if not artist:
            return None
        clean_artist = artist.replace('"', '\\"')
        query = f'artist:"{clean_artist}"'
        encoded_query = urllib.parse.quote(query)
        url = f"https://musicbrainz.org/ws/2/artist/?query={encoded_query}&fmt=json"

        self._rate_limit()
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    return None
                data = json.loads(response.read().decode("utf-8"))
                artists = data.get("artists", [])
                if not artists:
                    return None
                top_match = artists[0]
                score = int(top_match.get("score", 0))
                canonical_name = top_match.get("name", "")
                if score >= 90 and canonical_name and canonical_name.lower() != artist.lower():
                    aliases = [
                        a.get("name", "").lower()
                        for a in top_match.get("aliases", [])
                        if isinstance(a, dict)
                    ]
                    if artist.lower() in aliases or score == 100:
                        return str(canonical_name)
        except Exception as err:
            logger.debug(f"MusicBrainz artist alias query failed for '{artist}': {err}")
        return None

    def get_recording_tags(self, artist: str, title: str, album: str | None = None) -> list[str]:
        """Search for a recording on MusicBrainz and return its tags and genres."""
        cleaned_album = clean_retailer_noise(album) if album else None
        uncensored_title = uncensor_title(title)

        title_variants = [title]
        if uncensored_title and uncensored_title.lower() != title.lower():
            title_variants.append(uncensored_title)

        album_variants: list[str | None] = []
        if album:
            if cleaned_album and cleaned_album.lower() != album.lower():
                album_variants.append(cleaned_album)
            album_variants.append(album)
        else:
            album_variants.append(None)

        for art in get_artist_aliases(artist):
            for alb in album_variants:
                for tit in title_variants:
                    tags = self._fetch_recording_tags_for_artist(
                        art, tit, album=alb, expected_artist=artist
                    )
                    if tags:
                        return tags
        return []

    def _execute_query(self, query: str, log_context: str) -> dict | None:
        """Execute a MusicBrainz search query with retries and rate limiting."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://musicbrainz.org/ws/2/recording/?query={encoded_query}&fmt=json"

        data = None
        max_retries = 2
        for attempt in range(max_retries + 1):
            self._rate_limit()
            req = urllib.request.Request(url, headers=self.headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    if response.status != 200:
                        return None
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as http_err:
                if http_err.code == 503 and attempt < max_retries:
                    logger.info(
                        f"MusicBrainz 503 for '{log_context}', "
                        f"retrying in 4s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(4.0)
                    continue
                logger.warning(f"MusicBrainz API query failed for '{log_context}': {http_err}")
                return None
            except Exception as err:
                logger.warning(f"MusicBrainz API query failed for '{log_context}': {err}")
                return None
        return data

    def _fetch_recording_tags_for_artist(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        expected_artist: str | None = None,
    ) -> list[str]:
        """Query MusicBrainz for a specific artist name, title, and optional album."""
        target_artist = expected_artist or artist
        clean_artist = artist.replace('"', '\\"')
        clean_title = title.replace('"', '\\"')

        data = None
        # Step 1: If album is provided, try album-filtered search first for release disambiguation
        if album and album.strip():
            clean_album = album.strip().replace('"', '\\"')
            query = (
                f'artist:"{clean_artist}" AND release:"{clean_album}" '
                f'AND (recording:"{clean_title}" OR track:"{clean_title}")'
            )
            data = self._execute_query(query, f"{artist} - {album} - {title}")

        # Step 2: Fall back to recording-only search if no results or album was not provided
        if not data or not data.get("recordings"):
            query = (
                f'artist:"{clean_artist}" AND (recording:"{clean_title}" OR track:"{clean_title}")'
            )
            data = self._execute_query(query, f"{artist} - {title}")

        if not data:
            return []

        try:
            recordings = data.get("recordings", [])
            if not recordings:
                return []

            # Find matching recording whose artist credit aligns with query artist
            # If album is supplied, prefer recording with matching release name
            matching_rec = None
            for r in recordings:
                artist_credits = r.get("artist-credit", [])
                artist_credit_name = "".join(
                    ac.get("name", "") for ac in artist_credits if isinstance(ac, dict)
                )
                if not artist_credit_name and artist_credits:
                    first_ac = artist_credits[0]
                    if isinstance(first_ac, dict):
                        artist_credit_name = first_ac.get("artist", {}).get("name", "")
                if artist_credit_name and not artist_matches(target_artist, artist_credit_name):
                    continue

                if album and album.strip():
                    releases = r.get("releases", [])
                    if any(
                        album.strip().lower() in rel.get("title", "").lower()
                        for rel in releases
                        if isinstance(rel, dict)
                    ):
                        matching_rec = r
                        break

                if matching_rec is None:
                    matching_rec = r

            if not matching_rec:
                logger.debug(
                    f"No MusicBrainz recording matched artist '{target_artist}' for title '{title}'"
                )
                return []

            rec = matching_rec
            tags: list[str] = []

            # 1. Recording level tags & genres
            for t in rec.get("tags", []):
                if name := t.get("name"):
                    tags.append(name)
            for g in rec.get("genres", []):
                if name := g.get("name"):
                    tags.append(name)

            # 2. Artist level tags & genres
            for ac in rec.get("artist-credit", []):
                artist_obj = ac.get("artist", {})
                for t in artist_obj.get("tags", []):
                    if name := t.get("name"):
                        tags.append(name)
                for g in artist_obj.get("genres", []):
                    if name := g.get("name"):
                        tags.append(name)

            # 3. Release group tags & genres
            for rel in rec.get("releases", []):
                rg = rel.get("release-group", {})
                for t in rg.get("tags", []):
                    if name := t.get("name"):
                        tags.append(name)
                for g in rg.get("genres", []):
                    if name := g.get("name"):
                        tags.append(name)

            return list(set(tags))
        except Exception as err:
            logger.warning(f"MusicBrainz API query failed for '{artist} - {title}': {err}")
            return []


class DiscogsFetcher:
    """Fetch release genres and styles from the Discogs API."""

    def __init__(self, api_token: str | None = None) -> None:
        """Initialize DiscogsFetcher with optional API token."""
        self.api_token = api_token
        self.headers = {"User-Agent": "Resonate/0.1.0"}
        if self.api_token:
            self.headers["Authorization"] = f"Discogs token={self.api_token}"

    def get_release_genres(self, artist: str, title: str, album: str | None = None) -> list[str]:
        """Search Discogs for a release and return its genres and styles."""
        if not self.api_token:
            logger.debug("Discogs API token not configured. Skipping Discogs lookup.")
            return []

        cleaned_album = clean_retailer_noise(album) if album else None
        target_album = cleaned_album or album
        uncensored_title = uncensor_title(title)

        # Prefer album title if available, otherwise search track title
        query = (
            f"{artist} - {target_album}"
            if target_album and target_album.strip()
            else f"{artist} - {uncensored_title}"
        )
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.discogs.com/database/search?q={encoded_query}&type=release"

        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status != 200:
                    return []
                data = json.loads(response.read().decode("utf-8"))

            results = data.get("results", [])
            if not results:
                # If album search failed, try fallback with artist - title
                if target_album and target_album.strip():
                    fallback_query = f"{artist} - {uncensored_title}"
                    encoded_fallback = urllib.parse.quote(fallback_query)
                    url = (
                        f"https://api.discogs.com/database/search?q={encoded_fallback}&type=release"
                    )
                    req = urllib.request.Request(url, headers=self.headers)
                    with urllib.request.urlopen(req, timeout=15) as response:
                        if response.status != 200:
                            return []
                        data = json.loads(response.read().decode("utf-8"))
                    results = data.get("results", [])

            if not results:
                return []

            first_result = results[0]
            genres = first_result.get("genre", [])
            styles = first_result.get("style", [])

            # Ensure return is flat list of strings
            all_tags = []
            for g in genres:
                if isinstance(g, str):
                    all_tags.append(g)
            for s in styles:
                if isinstance(s, str):
                    all_tags.append(s)

            return list(set(all_tags))
        except Exception as err:
            logger.warning(f"Discogs API query failed for '{artist} - {title}': {err}")
            return []
