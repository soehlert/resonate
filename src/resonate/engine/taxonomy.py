"""Genre taxonomy hierarchy, family promotion rules, and subgenre filtering engine."""

from __future__ import annotations

import logging
from collections import Counter

from resonate.models import TaxonomyDecision

logger = logging.getLogger(__name__)

NATIONALITY_STRINGS: set[str] = {
    "american",
    "british",
    "australian",
    "canadian",
    "german",
    "french",
    "japanese",
    "english",
}

GENERIC_MODIFIERS: set[str] = {
    "indie",
    "rock",
    "pop",
    "metal",
    "punk",
    "folk",
    "country",
    "alternative",
    "post",
    "garage",
    "soft",
    "hard",
    "hardcore",
    "emo",
    "cloud",
    "gangsta",
    "conscious",
    "instrumental",
    "acoustic",
    "vocal",
    "electric",
    "heavy",
    "ambient",
    "experimental",
    "classic",
    "modern",
    "industrial",
    "southern",
    "roots",
    "progressive",
    "psychedelic",
    "psych",
    "surf",
    "stoner",
    "space",
    "glam",
    "gothic",
    "goth",
    "sludge",
    "drone",
    "chamber",
    "symphonic",
    "dance",
    "wave",
    "funk",
    "latin",
    "nu",
}

PRIMARY_GENRE_STEMS: dict[str, list[str]] = {
    "Punk": ["punk", "hardcore punk", "street punk", "skate punk", "post-punk"],
    "Metal": ["metal", "heavy metal", "thrash", "death metal", "black metal", "doom metal"],
    "Rock": ["rock", "rock and roll", "rock n roll", "rockabilly", "classic rock"],
    "Pop": ["pop", "dance-pop", "synthpop", "electropop"],
    "Hip-Hop": ["hip-hop", "hip hop", "rap", "hiphop", "trap", "gangsta rap"],
    "Electronic": [
        "electronic",
        "electronica",
        "techno",
        "house",
        "industrial",
        "trance",
        "edm",
        "synthwave",
        "dubstep",
        "dnb",
        "drum and bass",
        "ambient",
        "club",
    ],
    "Jazz": [
        "jazz",
        "bebop",
        "hard bop",
        "cool jazz",
        "modal jazz",
        "big band",
        "swing",
        "jazz fusion",
        "soul jazz",
        "smooth jazz",
        "vocal jazz",
        "dixieland",
        "gypsy jazz",
        "free jazz",
        "bossa nova",
        "latin jazz",
    ],
    "Blues": ["blues", "delta blues", "chicago blues"],
    "Country": ["country", "alt-country", "bluegrass", "americana"],
    "Folk": ["folk", "indie folk", "americana"],
    "R&B": ["r&b", "rnb", "rhythm and blues", "rhythm & blues"],
    "Soul": ["soul", "motown", "neo-soul"],
    "Reggae": ["reggae", "dub", "ska"],
    "Latin": ["latin", "reggaeton", "salsa", "bossa nova"],
    "Dance": ["dance", "club", "edm", "house"],
    "Classical": ["classical", "symphonic", "chamber music", "baroque", "opera"],
    "Indie": ["indie", "indie rock", "indie pop", "indie folk", "lo-fi"],
}

SUB_GENRE_STEMS: dict[str, list[str]] = {
    "Punk Rock": ["punk rock", "punk"],
    "Rap": ["rap", "hip hop", "hip-hop", "hiphop"],
    "East Coast Hip Hop": ["east coast hip hop", "east coast rap"],
    "West Coast Hip Hop": ["west coast hip hop", "west coast rap"],
    "G-Funk": ["g-funk", "g funk"],
    "Boom Bap": ["boom bap"],
    "Trap": ["trap", "southern rap", "dirty south"],
    "Gangsta Rap": ["gangsta rap", "gangsta"],
    "Conscious Hip Hop": ["conscious hip hop", "conscious rap", "political hip hop"],
    "Cloud Rap": ["cloud rap"],
    "Emo Rap": ["emo rap"],
    "Hardcore Hip Hop": ["hardcore hip hop", "hardcore rap"],
    "Alternative Hip Hop": ["alternative hip hop", "alternative rap"],
    "Thrash Metal": ["thrash", "thrash metal"],
    "Alternative Metal": ["alternative metal", "alt metal", "alt-metal"],
    "Funk Metal": ["funk metal", "metal funk"],
    "Nu-Metal": ["nu metal", "nu-metal", "numetal"],
    "Industrial": ["industrial", "ebm", "industrial dance", "aggrotech"],
    "Industrial Metal": ["industrial metal", "cyber metal"],
    "Sludge Metal": ["sludge metal", "sludge"],
    "Hardcore Punk": ["hardcore punk"],
    "Rockabilly": ["rockabilly", "psychobilly"],
    "Rock and Roll": [
        "rock and roll",
        "rock n roll",
        "rock & roll",
        "rock'n'roll",
        "rock'n roll",
        "rock 'n' roll",
    ],
    "Oldies": ["oldies"],
    "R&B": ["r&b", "rnb", "rhythm and blues", "rhythm & blues"],
    "Contemporary R&B": ["contemporary r&b", "contemporary rnb", "r&b", "rnb"],
    "Prog Rock": ["prog rock", "progressive rock", "prog"],
    "Progressive Metal": ["progressive metal", "prog metal"],
    "Big Band": ["big band", "big-band", "swing orchestra", "jazz orchestra"],
    "Swing": ["swing", "swing music", "big band swing"],
    "Bebop": ["bebop", "bop"],
    "Hard Bop": ["hard bop", "hardbop"],
    "Cool Jazz": ["cool jazz", "west coast jazz"],
    "Modal Jazz": ["modal jazz", "modal"],
    "Jazz Fusion": ["jazz fusion", "fusion", "jazz-rock", "jazz rock"],
    "Chamber Music": [
        "chamber music",
        "string quartet",
        "chamber orchestra",
        "string ensemble",
        "trio",
        "quartet",
        "quintet",
    ],
    "Symphony": [
        "symphony",
        "symphonic",
        "orchestral",
        "orchestra",
        "philharmonic",
        "symphony orchestra",
    ],
    "Baroque": ["baroque", "early music", "harpsichord"],
    "Opera": ["opera", "operatic", "aria", "soprano", "tenor", "libretto"],
    "Soul Jazz": ["soul jazz", "soul-jazz"],
    "Smooth Jazz": ["smooth jazz", "contemporary jazz"],
    "Vocal Jazz": ["vocal jazz", "jazz vocals", "standards", "traditional pop"],
    "Latin Jazz": ["latin jazz", "afro-cuban jazz"],
    "Bossa Nova": ["bossa nova", "bossa", "samba jazz"],
    "Free Jazz": ["free jazz", "avant-garde jazz", "avant garde jazz"],
    "Dixieland": ["dixieland", "trad jazz", "traditional jazz", "new orleans jazz"],
    "Gypsy Jazz": ["gypsy jazz", "jazz manouche"],
    "Shoegaze": ["shoegaze", "shoe gaze", "noise pop", "wall of sound"],
    "Post-Rock": ["post-rock", "post rock", "crescendo rock"],
    "Dream Pop": ["dream pop", "dreampop", "ethereal wave", "ethereal pop"],
    "Bluegrass": ["bluegrass", "progressive bluegrass", "newgrass"],
    "Country Rock": ["country rock", "southern rock", "country-rock"],
    "Alt-Country": ["alt-country", "alternative country", "alt country"],
    "Alternative Rock": ["alternative rock", "alt rock", "alt-rock", "alternative"],
    "Hard Rock": ["hard rock"],
    "Roots Rock": ["roots rock", "roots-rock"],
    "Americana": ["americana"],
    "Southern Rock": ["southern rock", "southern-rock"],
    "Outlaw Country": ["outlaw country", "outlaw"],
    "Roots Reggae": ["roots reggae", "reggae roots", "roots"],
    "Dub": ["dub", "dub reggae", "king tubby"],
    "Reggae Rock": ["reggae rock", "ska rock", "sublime"],
    "Post-Punk": ["post-punk", "post punk", "dark post-punk"],
    "Gothic": ["goth", "gothic", "gothic rock", "goth rock"],
    "Darkwave": ["darkwave", "dark wave", "synth goth"],
    "Slowcore": ["slowcore", "slow core", "sadcore"],
    "Sadcore": ["sadcore", "slowcore"],
    "Instrumental": ["instrumental"],
    "Instrumental Rock": ["instrumental rock"],
}

SUBGENRE_TO_FAMILY: dict[str, str] = {
    # Metal
    "heavy metal": "Metal",
    "thrash metal": "Metal",
    "death metal": "Metal",
    "black metal": "Metal",
    "doom metal": "Metal",
    "power metal": "Metal",
    "sludge metal": "Metal",
    "industrial metal": "Metal",
    "progressive metal": "Metal",
    "alternative metal": "Metal",
    "funk metal": "Metal",
    "nu-metal": "Metal",
    # Punk
    "punk rock": "Punk",
    "hardcore punk": "Punk",
    "post-hardcore": "Punk",
    "skate punk": "Punk",
    "pop-punk": "Punk",
    "crossover thrash": "Punk",
    # Rock / Alt
    "alternative rock": "Rock",
    "stoner rock": "Rock",
    "space rock": "Rock",
    "shoegaze": "Rock",
    "indie rock": "Rock",
    "classic rock": "Rock",
    "hard rock": "HardRock",
    "psychedelic rock": "Rock",
    "post-rock": "Rock",
    "progressive rock": "Rock",
    "prog rock": "Rock",
    "garage rock": "Rock",
    "art rock": "Rock",
    "grunge": "Rock",
    "glam rock": "Rock",
    "soft rock": "Rock",
    "acoustic rock": "Rock",
    "pop rock": "Rock",
    "rock and roll": "Rock",
    "rockabilly": "Rock",
    "southern rock": "Rock",
    # Country / Roots / Americana
    "alt-country": "Roots",
    "country rock": "Roots",
    "americana": "Roots",
    "roots rock": "Roots",
    "bluegrass": "Roots",
    "outlaw country": "Roots",
    "folk rock": "Roots",
    "indie folk": "Roots",
    "folk": "Roots",
    # Hip-Hop / Rap
    "rap": "Hip-Hop",
    "hip-hop": "Hip-Hop",
    "east coast hip hop": "Hip-Hop",
    "west coast hip hop": "Hip-Hop",
    "g-funk": "Hip-Hop",
    "boom bap": "Hip-Hop",
    "trap": "Hip-Hop",
    "southern rap": "Hip-Hop",
    "gangsta rap": "Hip-Hop",
    "conscious hip hop": "Hip-Hop",
    "cloud rap": "Hip-Hop",
    "emo rap": "Hip-Hop",
    "hardcore hip hop": "Hip-Hop",
    "alternative hip hop": "Hip-Hop",
    # Electronic
    "house": "Electronic",
    "techno": "Electronic",
    "edm": "Electronic",
    "industrial": "Electronic",
    "trance": "Electronic",
    "synthwave": "Electronic",
    "ambient": "Electronic",
    "synthpop": "Electronic",
    "electropop": "Electronic",
    "trip-hop": "Electronic",
    "idm": "Electronic",
    "drum and bass": "Electronic",
    "dubstep": "Electronic",
    # R&B / Soul
    "r&b": "R&B",
    "contemporary r&b": "R&B",
    "soul": "Soul",
    "motown": "Soul",
    "neo-soul": "Soul",
    "funk": "Funk",
    "disco": "Disco",
    # Jazz
    "big band": "Jazz",
    "swing": "Jazz",
    "bebop": "Jazz",
    "hard bop": "Jazz",
    "cool jazz": "Jazz",
    "modal jazz": "Jazz",
    "jazz fusion": "Jazz",
    "soul jazz": "Jazz",
    "smooth jazz": "Jazz",
    "vocal jazz": "Jazz",
    "latin jazz": "Jazz",
    "free jazz": "Jazz",
    "dixieland": "Jazz",
    "gypsy jazz": "Jazz",
}

DEFAULT_PRIMARY_GENRES: list[str] = [
    "Rock",
    "Pop",
    "Indie",
    "Hip-Hop",
    "Electronic",
    "Jazz",
    "Blues",
    "Classical",
    "Country",
    "Folk",
    "R&B",
    "Metal",
    "Punk",
    "Reggae",
    "Latin",
    "Soul",
    "Dance",
]

DEFAULT_SUB_GENRES: list[str] = [
    "Americana",
    "Southern Rock",
    "Country Rock",
    "Alt-Country",
    "Outlaw Country",
    "Roots Rock",
    "Alternative Rock",
    "Hard Rock",
    "Heavy Metal",
    "Thrash Metal",
    "Hardcore Punk",
    "Crossover Thrash",
    "Pop-Punk",
    "Post-Hardcore",
    "Death Metal",
    "Black Metal",
    "Power Metal",
    "Doom Metal",
    "Stoner Rock",
    "Indie Folk",
    "Lo-Fi",
    "Trip-Hop",
    "Grunge",
    "Indie Rock",
    "Indie Pop",
    "Classic Rock",
    "Rockabilly",
    "Oldies",
    "Rock and Roll",
    "Folk Rock",
    "Pop Rock",
    "Psychedelic Rock",
    "British Invasion",
    "Prog Rock",
    "Punk Rock",
    "Art Rock",
    "Glam Rock",
    "New Wave",
    "Post-Punk",
    "Acoustic Rock",
    "Soft Rock",
    "Skate Punk",
    "Garage Rock",
    "Disco",
    "Funk",
    "House",
    "EDM",
    "Techno",
    "Club",
    "Dance-Pop",
    "Rap",
    "Hip-Hop",
    "East Coast Hip Hop",
    "West Coast Hip Hop",
    "G-Funk",
    "Boom Bap",
    "Trap",
    "Southern Rap",
    "Gangsta Rap",
    "Conscious Hip Hop",
    "Cloud Rap",
    "Emo Rap",
    "Hardcore Hip Hop",
    "Alternative Hip Hop",
    "Reggaeton",
    "Ska",
    "Synthpop",
    "Synthwave",
    "Neo-Soul",
    "Motown",
    "R&B",
    "Contemporary R&B",
    "Afrobeat",
    "Reggae",
    "Roots Reggae",
    "Dub",
    "Reggae Rock",
    "Groove",
    "Bluegrass",
    "Blues Rock",
    "Electric Blues",
    "Chicago Blues",
    "Delta Blues",
    "Chamber Music",
    "Symphonic",
    "Symphony",
    "Baroque",
    "Opera",
    "Singer-Songwriter",
    "Electropop",
    "Trance",
    "Electronica",
    "IDM",
    "Ambient",
    "Dubstep",
    "Drum and Bass",
    "Slowcore",
    "Post-Rock",
    "Sadcore",
    "Darkwave",
    "Gothic",
    "Shoegaze",
    "Dream Pop",
    "Emo",
    "Progressive Metal",
    "Alternative Metal",
    "Funk Metal",
    "Nu-Metal",
    "Industrial",
    "Industrial Metal",
    "Sludge Metal",
    "Big Band",
    "Swing",
    "Bebop",
    "Hard Bop",
    "Cool Jazz",
    "Modal Jazz",
    "Jazz Fusion",
    "Soul Jazz",
    "Smooth Jazz",
    "Vocal Jazz",
    "Latin Jazz",
    "Bossa Nova",
    "Free Jazz",
    "Dixieland",
    "Gypsy Jazz",
    "Instrumental",
    "Instrumental Rock",
]

MUTUALLY_EXCLUSIVE_STYLES: list[set[str]] = [
    {"Soft Rock", "Hard Rock"},
    {"Soft Rock", "Heavy Metal"},
    {"Soft Rock", "Punk Rock"},
    {"Acoustic Rock", "Hard Rock"},
    {"Acoustic Rock", "Heavy Metal"},
    {"Acoustic Rock", "Punk Rock"},
    {"Pop Rock", "Heavy Metal"},
    {"Alt-Country", "Hard Rock"},
    {"Country Rock", "Hard Rock"},
    {"Americana", "Hard Rock"},
    {"Indie Folk", "Hard Rock"},
]

COMPOUND_SUBGENRE_WHITELIST: set[str] = {
    "singer-songwriter",
    "singer songwriter",
    "indie rock",
    "indie pop",
    "indie folk",
    "folk rock",
    "pop rock",
    "punk rock",
    "hard rock",
    "soft rock",
    "acoustic rock",
    "country rock",
    "southern rock",
    "roots rock",
    "garage rock",
    "psychedelic rock",
    "progressive rock",
    "prog rock",
    "heavy metal",
    "progressive metal",
    "thrash metal",
    "death metal",
    "black metal",
    "doom metal",
    "power metal",
    "hardcore punk",
    "skate punk",
    "pop-punk",
    "post-hardcore",
    "post-punk",
    "post-rock",
    "synth-pop",
    "synthpop",
    "synthwave",
    "dance-pop",
    "trip-hop",
    "hip-hop",
    "hip hop",
    "lo-fi",
    "contemporary r&b",
    "alt-country",
    "blues rock",
    "electric blues",
    "chicago blues",
    "delta blues",
    "chamber music",
    "alternative metal",
    "alt-metal",
    "funk metal",
    "nu-metal",
    "nu metal",
    "industrial metal",
    "sludge metal",
    "big band",
    "cool jazz",
    "modal jazz",
    "jazz fusion",
    "soul jazz",
    "smooth jazz",
    "vocal jazz",
    "latin jazz",
    "bossa nova",
    "free jazz",
    "gypsy jazz",
    "trad jazz",
    "dream pop",
    "dreampop",
    "shoegaze",
    "noise pop",
    "roots reggae",
    "reggae rock",
    "dub reggae",
    "progressive bluegrass",
    "newgrass",
    "outlaw country",
}


def is_valid_subgenre_tag(tag: str, artist: str, album: str | None = None) -> bool:
    """Filter out non-genre tags, playlists, TV shows, and decades from subgenre candidates."""
    tag_lower = tag.lower().strip()

    if any(c.isdigit() for c in tag_lower):
        return False

    artist_lower = artist.lower().strip()
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return False
    artist_words = [w.strip() for w in artist_lower.split() if len(w.strip()) > 3]
    if any(w in tag_lower for w in artist_words):
        return False

    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return False
        album_words = [w.strip() for w in album_lower.split() if len(w.strip()) > 3]
        if any(w in tag_lower for w in album_words):
            return False

    if tag_lower in COMPOUND_SUBGENRE_WHITELIST:
        return True

    # 4. Skip common non-genre/boilerplate/playlist/TV descriptors
    boilerplate = {
        "fav",
        "favorites",
        "favourite",
        "favorite",
        "personal favourites",
        "seen live",
        "live",
        "heard on",
        "pandora",
        "spotify",
        "playlist",
        "track",
        "song",
        "album",
        "albums",
        "artist",
        "music",
        "singer",
        "songwriter",
        "band",
        "great",
        "nice",
        "awesome",
        "good",
        "mp3",
        "tag",
        "recommend",
        "soundtrack",
        "ost",
        "theme",
        "version",
        "remix",
        "cover",
        "gtst",
        "ludo sanders",
        "series",
        "tv",
        "label",
        "catfish",
        "radio",
        "bagel",
        "nachspiel",
        "gr last",
    }
    if any(b in tag_lower for b in boilerplate):
        return False

    return True


def promote_genre_by_subgenres(
    mapped_genre: str | None, mapped_subgenres: list[str]
) -> tuple[str | None, TaxonomyDecision | None]:
    """Elevate generic Rock or Pop if child subgenres strictly outnumber parent."""
    if not mapped_genre or mapped_genre not in {"Rock", "Pop"} or not mapped_subgenres:
        return mapped_genre, None

    subgenre_family_counts: Counter[str] = Counter()
    for sg in mapped_subgenres:
        fam = SUBGENRE_TO_FAMILY.get(sg.lower())
        if fam:
            subgenre_family_counts[fam] += 1

    parent_family = mapped_genre
    parent_count = subgenre_family_counts.get(parent_family, 0)

    top_candidates = [
        (fam, cnt)
        for fam, cnt in subgenre_family_counts.most_common()
        if fam != parent_family
    ]
    if top_candidates:
        top_child_family, top_child_count = top_candidates[0]
        if top_child_count > parent_count:
            decision = TaxonomyDecision(
                original_genre=mapped_genre,
                promoted_genre=top_child_family,
                reason=(
                    f"Child family '{top_child_family}' subgenres ({top_child_count}) "
                    f"strictly outnumber parent '{parent_family}' subgenres ({parent_count})"
                ),
                contributing_subgenres=[
                    sg for sg in mapped_subgenres
                    if SUBGENRE_TO_FAMILY.get(sg.lower()) == top_child_family
                ],
                confidence=1.0,
            )
            return top_child_family, decision

    return mapped_genre, None


def sanitize_subgenres_for_genre(
    mapped_genre: str | None, mapped_subgenres: list[str], raw_tags: list[str]
) -> list[str]:
    """Apply cross-family sanity guards to strip incompatible subgenres."""
    if not mapped_genre or not mapped_subgenres:
        return mapped_subgenres

    raw_clean_set = {r.lower().strip() for r in raw_tags}

    # 1. Punk / Metal / Rock: strip Hip-Hop subgenres unless explicit hip-hop tags are present
    if mapped_genre in {"Punk", "Metal", "Rock"}:
        hiphop_subgenres = {
            "hardcore hip hop",
            "conscious hip hop",
            "alternative hip hop",
            "east coast hip hop",
            "west coast hip hop",
            "cloud rap",
            "emo rap",
            "trap",
            "gangsta rap",
            "g-funk",
            "boom bap",
        }
        if not any(r in {"hip-hop", "hip hop", "rap", "hiphop"} for r in raw_clean_set):
            mapped_subgenres = [
                s for s in mapped_subgenres if s.lower() not in hiphop_subgenres
            ]

    # 2. Hip-Hop / Rap: strip Metal / Punk subgenres unless explicit metal/punk tags are present
    elif mapped_genre in {"Hip-Hop", "Rap"}:
        metal_punk_subgenres = {
            "hardcore punk",
            "post-hardcore",
            "skate punk",
            "pop-punk",
            "heavy metal",
            "thrash metal",
            "death metal",
            "black metal",
            "doom metal",
            "sludge metal",
            "industrial metal",
        }
        if not any(
            r in {"metal", "heavy metal", "punk", "punk rock"} for r in raw_clean_set
        ):
            mapped_subgenres = [
                s for s in mapped_subgenres if s.lower() not in metal_punk_subgenres
            ]

    # 3. Classical: strip incompatible modern rock/pop/metal keywords
    elif mapped_genre == "Classical":
        incompatible_classical_keywords = {
            "rock",
            "metal",
            "punk",
            "hip-hop",
            "hip hop",
            "rap",
            "trap",
            "country",
            "funk",
        }
        mapped_subgenres = [
            s for s in mapped_subgenres
            if not any(k in s.lower() for k in incompatible_classical_keywords)
        ]

    return mapped_subgenres


def deduplicate_subgenres(
    primary_genre: str | None, subgenres: list[str]
) -> list[str]:
    """Deduplicate subgenres and remove primary genre exact matches and mutual style conflicts."""
    if not subgenres:
        return []

    seen: set[str] = set()
    cleaned: list[str] = []
    primary_lower = primary_genre.lower().strip() if primary_genre else ""

    for s in subgenres:
        s_clean = s.strip()
        s_lower = s_clean.lower()
        if not s_clean:
            continue
        if s_lower == primary_lower:
            continue
        if s_lower not in seen:
            seen.add(s_lower)
            cleaned.append(s_clean)

    # Filter mutually exclusive styles
    final: list[str] = []
    for item in cleaned:
        conflict = False
        for group in MUTUALLY_EXCLUSIVE_STYLES:
            if item in group and any(e in group for e in final):
                conflict = True
                break
        if not conflict:
            final.append(item)

    return final
