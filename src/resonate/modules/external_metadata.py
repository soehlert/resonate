"""External metadata fetchers for MusicBrainz and Discogs integrations."""

import logging
import re

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
    "snoop lion": ["snoop dogg"],
    "mf doom": ["doom", "viktor vaughn", "king geedorah"],
    "doom": ["mf doom"],
}


def get_artist_aliases(artist: str) -> list[str]:
    """Return all known alias names and grammatical variants for an artist."""
    clean = artist.lower().strip()
    aliases: list[str] = []

    # If the queried artist is a known shorthand/rebrand (e.g. 'Ye' -> 'Kanye West'),
    # prioritize primary canonical name first to avoid third-party collisions (e.g. Ye -> Yes)
    if clean in ARTIST_ALIASES:
        for alias in ARTIST_ALIASES[clean]:
            if len(alias) > len(clean) and alias not in aliases:
                aliases.append(alias)

    if artist not in aliases:
        aliases.append(artist)

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

    # 3. Leading 'The' variants (strip-only, e.g. 'The Beatles' -> 'Beatles')
    if clean.startswith("the ") and len(clean) > 4:
        without_the = artist[4:].strip()
        if without_the and without_the not in aliases:
            aliases.append(without_the)

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

    # Alphanumeric equivalence for spacing/punctuation variants (e.g. 'Raymen' vs 'Ray Men')
    alnum_exp = re.sub(r"[^a-z0-9]", "", exp)
    alnum_cand = re.sub(r"[^a-z0-9]", "", cand)
    if alnum_exp and alnum_cand and alnum_exp == alnum_cand:
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


__all__ = [
    "ARTIST_ALIASES",
    "artist_matches",
    "clean_retailer_noise",
    "get_artist_aliases",
    "uncensor_title",
]
