"""Genre taxonomy hierarchy, family promotion rules, and subgenre filtering engine."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

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

GENRE_STOP_WORDS: set[str] = {
    "rock",
    "pop",
    "metal",
    "punk",
    "jazz",
    "blues",
    "folk",
    "soul",
    "funk",
    "country",
    "electronic",
    "dance",
    "classical",
    "music",
    "song",
    "songs",
    "album",
    "hits",
    "greatest",
    "best",
    "collection",
    "years",
    "volume",
    "vol",
    "anthology",
    "edition",
    "live",
}

PRIMARY_GENRE_STEMS: dict[str, list[str]] = {
    "Punk": [
        "punk",
        "hardcore punk",
        "street punk",
        "skate punk",
        "post-punk",
        "ska punk",
        "third wave ska",
        "pop-punk",
    ],
    "Metal": [
        "metal",
        "heavy metal",
        "thrash",
        "death metal",
        "black metal",
        "doom metal",
        "rap metal",
        "rap-metal",
        "rapcore",
    ],
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
    "Reggae": ["reggae", "dub", "roots reggae", "rocksteady", "lovers rock"],
    "Latin": ["latin", "reggaeton", "salsa", "bossa nova"],
    "Dance": ["dance", "club", "edm", "house"],
    "Classical": ["classical", "symphonic", "chamber music", "baroque", "opera"],
    "Indie": ["indie", "indie rock", "indie pop", "indie folk", "lo-fi"],
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


@dataclass(frozen=True)
class SubgenreSpec:
    """Specification for a canonical subgenre in the taxonomy."""

    name: str
    family: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


SUBGENRE_REGISTRY: list[SubgenreSpec] = [
    SubgenreSpec(
        name="Americana",
        family="Roots",
        aliases=("americana",),
        description="Americana music, roots rock, alt-country, folk americana",
    ),
    SubgenreSpec(
        name="Southern Rock",
        family="Rock",
        aliases=("southern rock", "southern-rock"),
        description="Southern rock music, country rock, blues rock, americana rock",
    ),
    SubgenreSpec(
        name="Country Rock",
        family="Roots",
        aliases=("country rock", "country-rock"),
        description="Country rock music, southern rock, country guitar rock",
    ),
    SubgenreSpec(
        name="Alt-Country",
        family="Roots",
        aliases=("alt-country", "alternative country", "alt country"),
        description="Alt-country music, alternative country, americana roots rock",
    ),
    SubgenreSpec(
        name="Outlaw Country",
        family="Roots",
        aliases=("outlaw country", "outlaw"),
        description="Outlaw country music, raw acoustic rebel country rock guitar",
    ),
    SubgenreSpec(
        name="Roots Rock",
        family="Roots",
        aliases=("roots rock", "roots-rock"),
        description="Roots rock music, americana, southern rock, classic roots rock",
    ),
    SubgenreSpec(
        name="Alternative Rock",
        family="Rock",
        aliases=("alternative rock", "alt rock", "alt-rock", "alternative"),
        description="Alternative rock music, 90s alt-rock, indie alternative",
    ),
    SubgenreSpec(
        name="Hard Rock",
        family="HardRock",
        aliases=("hard rock",),
        description="Hard rock music, heavy guitar riffs, driving loud rock",
    ),
    SubgenreSpec(
        name="Heavy Metal",
        family="Metal",
        aliases=(),
        description="Heavy metal music, aggressive metal, heavy distortion headbanging",
    ),
    SubgenreSpec(
        name="Thrash Metal", family="Metal", aliases=("thrash", "thrash metal"), description=""
    ),
    SubgenreSpec(name="Hardcore Punk", family="Punk", aliases=("hardcore punk",), description=""),
    SubgenreSpec(name="Crossover Thrash", family="Punk", aliases=(), description=""),
    SubgenreSpec(name="Pop-Punk", family="Punk", aliases=(), description=""),
    SubgenreSpec(name="Post-Hardcore", family="Punk", aliases=(), description=""),
    SubgenreSpec(name="Death Metal", family="Metal", aliases=(), description=""),
    SubgenreSpec(name="Black Metal", family="Metal", aliases=(), description=""),
    SubgenreSpec(name="Power Metal", family="Metal", aliases=(), description=""),
    SubgenreSpec(name="Doom Metal", family="Metal", aliases=(), description=""),
    SubgenreSpec(name="Stoner Rock", family="Rock", aliases=(), description=""),
    SubgenreSpec(name="Indie Folk", family="Roots", aliases=(), description=""),
    SubgenreSpec(name="Lo-Fi", family="Indie", aliases=(), description=""),
    SubgenreSpec(name="Trip-Hop", family="Electronic", aliases=(), description=""),
    SubgenreSpec(
        name="Grunge",
        family="Rock",
        aliases=(),
        description="Grunge music, 90s seattle grunge, distorted heavy alt-rock",
    ),
    SubgenreSpec(
        name="Indie Rock",
        family="Rock",
        aliases=(),
        description="Indie rock music, independent rock band, alt-indie guitar",
    ),
    SubgenreSpec(
        name="Indie Pop",
        family="Indie",
        aliases=(),
        description="Indie pop music, catchy melody indie pop, cheerful alt-pop",
    ),
    SubgenreSpec(
        name="Classic Rock",
        family="Rock",
        aliases=(),
        description="Classic rock music, 60s 70s vintage rock, classic album rock",
    ),
    SubgenreSpec(
        name="Rockabilly", family="Rock", aliases=("rockabilly", "psychobilly"), description=""
    ),
    SubgenreSpec(name="Oldies", family="Rock", aliases=("oldies",), description=""),
    SubgenreSpec(
        name="Rock and Roll",
        family="Rock",
        aliases=(
            "rock and roll",
            "rock n roll",
            "rock & roll",
            "rock'n'roll",
            "rock'n roll",
            "rock 'n' roll",
        ),
        description="",
    ),
    SubgenreSpec(
        name="Folk Rock",
        family="Roots",
        aliases=(),
        description="Folk rock music, acoustic guitar folk rock, 60s folk rock",
    ),
    SubgenreSpec(
        name="Pop Rock",
        family="Rock",
        aliases=(),
        description="Pop rock music, mainstream commercial radio pop rock",
    ),
    SubgenreSpec(
        name="Psychedelic Rock",
        family="Rock",
        aliases=(),
        description="Psychedelic rock music, trippy 60s psych rock, acid rock",
    ),
    SubgenreSpec(
        name="British Invasion",
        family="Rock",
        aliases=(),
        description="British invasion music, 60s UK rock, Beatlemania garage pop",
    ),
    SubgenreSpec(
        name="Prog Rock",
        family="Rock",
        aliases=("prog rock", "progressive rock", "prog"),
        description="Progressive rock music, prog rock, complex synth art rock",
    ),
    SubgenreSpec(
        name="Punk Rock",
        family="Punk",
        aliases=("punk rock", "punk"),
        description="Punk rock music, fast energetic DIY underground punk rock",
    ),
    SubgenreSpec(
        name="Art Rock",
        family="Rock",
        aliases=(),
        description="Art rock music, experimental avant-garde art rock",
    ),
    SubgenreSpec(
        name="Glam Rock",
        family="Rock",
        aliases=(),
        description="Glam rock music, 70s glam rock, theatrical glitter rock",
    ),
    SubgenreSpec(
        name="New Wave",
        family="Rock",
        aliases=(),
        description="New wave music, 80s new wave, synth-pop post-punk",
    ),
    SubgenreSpec(
        name="Post-Punk",
        family="Punk",
        aliases=("post-punk", "post punk", "dark post-punk"),
        description="Post-punk music, dark post-punk, gothic goth post-punk",
    ),
    SubgenreSpec(
        name="Acoustic Rock",
        family="Rock",
        aliases=(),
        description="Acoustic rock music, unplugged acoustic guitar rock",
    ),
    SubgenreSpec(
        name="Soft Rock",
        family="Rock",
        aliases=(),
        description="Soft rock music, mellow gentle soft rock ballad",
    ),
    SubgenreSpec(
        name="Skate Punk",
        family="Punk",
        aliases=(),
        description="Skate punk music, fast melodic skate punk, pop-punk",
    ),
    SubgenreSpec(
        name="Garage Rock",
        family="Rock",
        aliases=(),
        description="Garage rock music, raw garage rock, 60s garage punk",
    ),
    SubgenreSpec(
        name="Disco",
        family="Disco",
        aliases=(),
        description="Disco music, 70s dance disco, funky disco groove",
    ),
    SubgenreSpec(
        name="Funk",
        family="Funk",
        aliases=(),
        description="Funk music, groovy bass funk, rhythm and blues funk",
    ),
    SubgenreSpec(
        name="House",
        family="Electronic",
        aliases=(),
        description="House music, electronic 4/4 dance house beat",
    ),
    SubgenreSpec(
        name="EDM",
        family="Electronic",
        aliases=(),
        description="EDM music, electronic dance music, festival synth drop",
    ),
    SubgenreSpec(
        name="Techno",
        family="Electronic",
        aliases=(),
        description="Techno music, dark underground club techno beat",
    ),
    SubgenreSpec(name="Club", family="Dance", aliases=(), description=""),
    SubgenreSpec(name="Dance-Pop", family="Dance", aliases=(), description=""),
    SubgenreSpec(
        name="Rap",
        family="Hip-Hop",
        aliases=("rap", "hip hop", "hip-hop", "hiphop"),
        description="Rap music, hip hop rap verses, rhyming rap track",
    ),
    SubgenreSpec(
        name="Hip-Hop",
        family="Hip-Hop",
        aliases=(),
        description="Hip-hop music, 90s hip hop beats rap music groove",
    ),
    SubgenreSpec(
        name="East Coast Hip Hop",
        family="Hip-Hop",
        aliases=("east coast hip hop", "east coast rap"),
        description="East Coast hip hop music, 90s NYC boom bap rap beats",
    ),
    SubgenreSpec(
        name="West Coast Hip Hop",
        family="Hip-Hop",
        aliases=("west coast hip hop", "west coast rap"),
        description="West Coast hip hop music, California g-funk synth rap",
    ),
    SubgenreSpec(
        name="G-Funk",
        family="Hip-Hop",
        aliases=("g-funk", "g funk"),
        description="G-funk music, smooth funk synthesizer West Coast g-funk",
    ),
    SubgenreSpec(
        name="Boom Bap",
        family="Hip-Hop",
        aliases=("boom bap",),
        description="Boom bap music, 90s drum break jazz sample boom bap rap",
    ),
    SubgenreSpec(
        name="Trap",
        family="Hip-Hop",
        aliases=("trap", "southern rap", "dirty south"),
        description="Trap music, 808 bass hi-hat rolls southern trap beat",
    ),
    SubgenreSpec(name="Southern Rap", family="Hip-Hop", aliases=(), description=""),
    SubgenreSpec(
        name="Gangsta Rap",
        family="Hip-Hop",
        aliases=("gangsta rap", "gangsta"),
        description="Gangsta rap music, gritty street rap hardcore hip hop",
    ),
    SubgenreSpec(
        name="Conscious Hip Hop",
        family="Hip-Hop",
        aliases=("conscious hip hop", "conscious rap", "political hip hop"),
        description="Conscious hip hop music, thoughtful lyrical conscious rap",
    ),
    SubgenreSpec(
        name="Cloud Rap",
        family="Hip-Hop",
        aliases=("cloud rap",),
        description="Cloud rap music, hazy atmospheric reverb lo-fi rap",
    ),
    SubgenreSpec(
        name="Emo Rap",
        family="Hip-Hop",
        aliases=("emo rap",),
        description="Emo rap music, melancholic guitar trap beat sad rap",
    ),
    SubgenreSpec(
        name="Hardcore Hip Hop",
        family="Hip-Hop",
        aliases=("hardcore hip hop", "hardcore rap"),
        description="Hardcore hip hop music, aggressive loud hardcore rap",
    ),
    SubgenreSpec(
        name="Alternative Hip Hop",
        family="Hip-Hop",
        aliases=("alternative hip hop", "alternative rap"),
        description="Alternative hip hop music, experimental creative indie rap",
    ),
    SubgenreSpec(
        name="Reggaeton",
        family="Latin",
        aliases=(),
        description="Reggaeton music, Latin urban reggaeton beat",
    ),
    SubgenreSpec(
        name="Ska",
        family="Reggae",
        aliases=("ska", "traditional ska", "2 tone", "two tone"),
        description="Ska music, upbeat ska punk, brass horn ska dance",
    ),
    SubgenreSpec(
        name="Ska Punk",
        family="Punk",
        aliases=("ska punk", "third wave ska", "ska-punk"),
        description="Ska punk music, upbeat ska brass punk rock, fast horns punk, third wave ska",
    ),
    SubgenreSpec(
        name="Synthpop",
        family="Electronic",
        aliases=(),
        description="Synthpop music, 80s synthesizer pop, synth-pop",
    ),
    SubgenreSpec(name="Synthwave", family="Electronic", aliases=(), description=""),
    SubgenreSpec(
        name="Neo-Soul",
        family="Soul",
        aliases=(),
        description="Neo-soul music, smooth modern R&B neo-soul groove",
    ),
    SubgenreSpec(
        name="Motown",
        family="Soul",
        aliases=(),
        description="Motown music, 60s Detroit soul Motown R&B",
    ),
    SubgenreSpec(
        name="R&B",
        family="R&B",
        aliases=("r&b", "rnb", "rhythm and blues", "rhythm & blues"),
        description="R&B music, rhythm and blues, soulful smooth groove vocals",
    ),
    SubgenreSpec(
        name="Contemporary R&B",
        family="R&B",
        aliases=("contemporary r&b", "contemporary rnb"),
        description="Contemporary R&B music, modern pop R&B, smooth melodic groove",
    ),
    SubgenreSpec(
        name="Afrobeat",
        family="Roots",
        aliases=(),
        description="Afrobeat music, West African rhythmic afrobeat groove",
    ),
    SubgenreSpec(name="Reggae", family="Reggae", aliases=(), description=""),
    SubgenreSpec(
        name="Roots Reggae",
        family="Reggae",
        aliases=("roots reggae", "reggae roots", "roots"),
        description="",
    ),
    SubgenreSpec(
        name="Dub", family="Reggae", aliases=("dub", "dub reggae", "king tubby"), description=""
    ),
    SubgenreSpec(
        name="Reggae Rock",
        family="Reggae",
        aliases=("reggae rock", "ska rock", "sublime"),
        description="",
    ),
    SubgenreSpec(name="Groove", family="Funk", aliases=(), description=""),
    SubgenreSpec(
        name="Bluegrass",
        family="Roots",
        aliases=("bluegrass", "progressive bluegrass", "newgrass"),
        description="Bluegrass music, acoustic banjo acoustic bluegrass",
    ),
    SubgenreSpec(
        name="Blues Rock",
        family="Rock",
        aliases=("blues rock", "blues-rock", "boogie rock", "slide guitar blues"),
        description="Blues rock music, electric guitar blues rock riff",
    ),
    SubgenreSpec(
        name="Electric Blues",
        family="Blues",
        aliases=(),
        description="Electric blues music, Chicago electric blues guitar",
    ),
    SubgenreSpec(
        name="Chicago Blues",
        family="Blues",
        aliases=(),
        description="Chicago blues music, harmonica electric blues",
    ),
    SubgenreSpec(
        name="Delta Blues",
        family="Blues",
        aliases=(),
        description="Delta blues music, acoustic slide guitar country blues",
    ),
    SubgenreSpec(
        name="Chamber Music",
        family="Classical",
        aliases=(
            "chamber music",
            "string quartet",
            "chamber orchestra",
            "string ensemble",
            "trio",
            "quartet",
            "quintet",
        ),
        description="Chamber music, classical string quartet acoustic chamber ensemble",
    ),
    SubgenreSpec(
        name="Symphonic",
        family="Classical",
        aliases=(),
        description="Symphonic music, orchestral classical symphony ensemble",
    ),
    SubgenreSpec(
        name="Symphony",
        family="Classical",
        aliases=(
            "symphony",
            "symphonic",
            "orchestral",
            "orchestra",
            "philharmonic",
            "symphony orchestra",
        ),
        description="Symphony music, orchestral classical symphony philharmonic ensemble",
    ),
    SubgenreSpec(
        name="Baroque",
        family="Classical",
        aliases=("baroque", "early music", "harpsichord"),
        description="Baroque music, classical early music harpsichord baroque ensemble",
    ),
    SubgenreSpec(
        name="Opera",
        family="Classical",
        aliases=("opera", "operatic", "aria", "soprano", "tenor", "libretto"),
        description="Opera music, classical operatic vocal aria soprano orchestra",
    ),
    SubgenreSpec(
        name="Singer-Songwriter",
        family="Folk",
        aliases=(),
        description="Singer-songwriter music, acoustic guitar vocal ballad",
    ),
    SubgenreSpec(name="Electropop", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="Trance", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="Electronica", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="IDM", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="Ambient", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="Dubstep", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="Drum and Bass", family="Electronic", aliases=(), description=""),
    SubgenreSpec(name="Slowcore", family="Rock", aliases=("slowcore", "slow core"), description=""),
    SubgenreSpec(
        name="Post-Rock",
        family="Rock",
        aliases=("post-rock", "post rock", "crescendo rock"),
        description="Post-rock music, instrumental cinematic crescendo ambient dynamic rock",
    ),
    SubgenreSpec(name="Sadcore", family="Rock", aliases=("sadcore",), description=""),
    SubgenreSpec(
        name="Darkwave",
        family="Electronic",
        aliases=("darkwave", "dark wave", "synth goth"),
        description="",
    ),
    SubgenreSpec(
        name="Gothic",
        family="Rock",
        aliases=("goth", "gothic", "gothic rock", "goth rock"),
        description="",
    ),
    SubgenreSpec(
        name="Shoegaze",
        family="Rock",
        aliases=("shoegaze", "shoe gaze", "noise pop", "wall of sound"),
        description="Shoegaze music, wall of sound distorted guitar reverb feedback dream pop",
    ),
    SubgenreSpec(
        name="Dream Pop",
        family="Rock",
        aliases=("dream pop", "dreampop", "ethereal wave", "ethereal pop"),
        description="Dream pop music, ethereal reverb guitar lush synthesizer gentle pop",
    ),
    SubgenreSpec(name="Emo", family="Rock", aliases=(), description=""),
    SubgenreSpec(
        name="Progressive Metal",
        family="Metal",
        aliases=("progressive metal", "prog metal"),
        description="Progressive metal music, prog metal, complex heavy metal guitar riff",
    ),
    SubgenreSpec(
        name="Alternative Metal",
        family="Metal",
        aliases=("alternative metal", "alt metal", "alt-metal"),
        description="Alternative metal music, alt-metal, heavy 90s alternative metal riff",
    ),
    SubgenreSpec(
        name="Funk Metal",
        family="Metal",
        aliases=("funk metal", "metal funk"),
        description="Funk metal music, slap bass heavy funk metal, aggressive groove",
    ),
    SubgenreSpec(
        name="Nu-Metal",
        family="Metal",
        aliases=("nu metal", "nu-metal", "numetal"),
        description="Nu-metal music, 90s 2000s nu metal, downtuned heavy riff",
    ),
    SubgenreSpec(
        name="Rap Metal",
        family="Metal",
        aliases=("rap metal", "rap-metal", "rap rock", "rap-rock", "rapcore"),
        description=(
            "Rap metal music, alternative metal rap rock rapcore heavy riff aggressive vocal"
        ),
    ),
    SubgenreSpec(
        name="Industrial",
        family="Electronic",
        aliases=("industrial", "ebm", "industrial dance", "aggrotech"),
        description="",
    ),
    SubgenreSpec(
        name="Industrial Metal",
        family="Metal",
        aliases=("industrial metal", "cyber metal"),
        description="Industrial metal music, machine electronic synth heavy metal",
    ),
    SubgenreSpec(
        name="Sludge Metal",
        family="Metal",
        aliases=("sludge metal", "sludge"),
        description="Sludge metal music, slow heavy distorted sludge doom riff",
    ),
    SubgenreSpec(
        name="Big Band",
        family="Jazz",
        aliases=("big band", "big-band", "swing orchestra", "jazz orchestra"),
        description="Big band music, swing orchestra horn section big band jazz",
    ),
    SubgenreSpec(
        name="Swing",
        family="Jazz",
        aliases=("swing", "swing music", "big band swing"),
        description="Swing music, 30s 40s swing jazz, upbeat dancing swing band",
    ),
    SubgenreSpec(
        name="Bebop",
        family="Jazz",
        aliases=("bebop", "bop"),
        description="Bebop music, fast tempo complex harmony jazz improvisation",
    ),
    SubgenreSpec(
        name="Hard Bop",
        family="Jazz",
        aliases=("hard bop", "hardbop"),
        description="Hard bop music, soulful bluesy energetic modern jazz",
    ),
    SubgenreSpec(
        name="Cool Jazz",
        family="Jazz",
        aliases=("cool jazz", "west coast jazz"),
        description="Cool jazz music, relaxed mellow modal west coast jazz",
    ),
    SubgenreSpec(
        name="Modal Jazz",
        family="Jazz",
        aliases=("modal jazz", "modal"),
        description="Modal jazz music, atmospheric modal harmony jazz masterpiece",
    ),
    SubgenreSpec(
        name="Jazz Fusion",
        family="Jazz",
        aliases=("jazz fusion", "fusion", "jazz-rock", "jazz rock"),
        description="Jazz fusion music, electric jazz-rock, virtuosic fusion groove",
    ),
    SubgenreSpec(
        name="Soul Jazz",
        family="Jazz",
        aliases=("soul jazz", "soul-jazz"),
        description="Soul jazz music, groovy organ blues soul jazz rhythm",
    ),
    SubgenreSpec(
        name="Smooth Jazz",
        family="Jazz",
        aliases=("smooth jazz", "contemporary jazz"),
        description="Smooth jazz music, polished mellow contemporary radio jazz",
    ),
    SubgenreSpec(
        name="Vocal Jazz",
        family="Jazz",
        aliases=("vocal jazz", "jazz vocals", "standards", "traditional pop"),
        description="Vocal jazz music, classic jazz standards singer vocal ballad",
    ),
    SubgenreSpec(
        name="Latin Jazz",
        family="Jazz",
        aliases=("latin jazz", "afro-cuban jazz"),
        description="Latin jazz music, afro-cuban percussion brass latin groove",
    ),
    SubgenreSpec(
        name="Bossa Nova",
        family="Latin",
        aliases=("bossa nova", "bossa", "samba jazz"),
        description="Bossa nova music, brazilian acoustic guitar gentle bossa rhythm",
    ),
    SubgenreSpec(
        name="Free Jazz",
        family="Jazz",
        aliases=("free jazz", "avant-garde jazz", "avant garde jazz"),
        description="Free jazz music, avant-garde experimental jazz improvisation",
    ),
    SubgenreSpec(
        name="Dixieland",
        family="Jazz",
        aliases=("dixieland", "trad jazz", "traditional jazz", "new orleans jazz"),
        description="Dixieland music, traditional new orleans brass jazz band",
    ),
    SubgenreSpec(
        name="Gypsy Jazz",
        family="Jazz",
        aliases=("gypsy jazz", "jazz manouche"),
        description="Gypsy jazz music, acoustic guitar swing jazz manouche",
    ),
    SubgenreSpec(
        name="Instrumental",
        family="Rock",
        aliases=("instrumental",),
        description="Instrumental music, no vocals, melodic guitar instrumental",
    ),
    SubgenreSpec(
        name="Instrumental Rock",
        family="Rock",
        aliases=("instrumental rock",),
        description="Instrumental rock music, guitar virtuoso rock, melodic instrumental rock",
    ),
]

DEFAULT_SUB_GENRES: list[str] = [spec.name for spec in SUBGENRE_REGISTRY]

SUB_GENRE_STEMS: dict[str, list[str]] = {
    spec.name: list(spec.aliases) for spec in SUBGENRE_REGISTRY if spec.aliases
}

SUBGENRE_TO_FAMILY: dict[str, str] = {
    alias.lower(): spec.family
    for spec in SUBGENRE_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
}

COMPOUND_SUBGENRE_WHITELIST: set[str] = {
    alias.lower()
    for spec in SUBGENRE_REGISTRY
    for alias in (spec.name.lower(), *spec.aliases)
    if " " in alias or "-" in alias
}

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


def is_valid_subgenre_tag(tag: str, artist: str, album: str | None = None) -> bool:
    """Filter out non-genre tags, playlists, TV shows, and decades from subgenre candidates."""
    tag_lower = tag.lower().strip()

    if any(c.isdigit() for c in tag_lower):
        return False

    if tag_lower in COMPOUND_SUBGENRE_WHITELIST:
        return True

    artist_lower = artist.lower().strip()
    if artist_lower in tag_lower or tag_lower in artist_lower:
        return False
    artist_words = [
        w.strip(" \t\n\r:;,.!?()[]{}\"'")
        for w in artist_lower.split()
        if len(w.strip(" \t\n\r:;,.!?()[]{}\"'")) > 3
        and w.strip(" \t\n\r:;,.!?()[]{}\"'") not in GENRE_STOP_WORDS
    ]
    if any(w in tag_lower for w in artist_words):
        return False

    if album:
        album_lower = album.lower().strip()
        if album_lower in tag_lower or tag_lower in album_lower:
            return False
        album_words = [
            w.strip(" \t\n\r:;,.!?()[]{}\"'")
            for w in album_lower.split()
            if len(w.strip(" \t\n\r:;,.!?()[]{}\"'")) > 3
            and w.strip(" \t\n\r:;,.!?()[]{}\"'") not in GENRE_STOP_WORDS
        ]
        if any(w in tag_lower for w in album_words):
            return False

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
        "alternative and punk",
        "alternative and rock",
        "rock and punk",
        "rock & punk",
        "punk and rock",
        "pop and rock",
        "rock and pop",
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
    if not mapped_genre or mapped_genre not in {"Rock", "Pop", "Reggae"} or not mapped_subgenres:
        return mapped_genre, None

    subgenre_family_counts: Counter[str] = Counter()
    for sg in mapped_subgenres:
        fam = SUBGENRE_TO_FAMILY.get(sg.lower())
        if fam == "HardRock":
            fam = "Rock"
        if fam and fam in DEFAULT_PRIMARY_GENRES:
            subgenre_family_counts[fam] += 1

    parent_family = mapped_genre
    parent_count = subgenre_family_counts.get(parent_family, 0)

    top_candidates = [
        (fam, cnt) for fam, cnt in subgenre_family_counts.most_common() if fam != parent_family
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
                    sg
                    for sg in mapped_subgenres
                    if (
                        SUBGENRE_TO_FAMILY.get(sg.lower()) == top_child_family
                        or (
                            top_child_family == "Rock"
                            and SUBGENRE_TO_FAMILY.get(sg.lower()) == "HardRock"
                        )
                    )
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
            mapped_subgenres = [s for s in mapped_subgenres if s.lower() not in hiphop_subgenres]

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
        if not any(r in {"metal", "heavy metal", "punk", "punk rock"} for r in raw_clean_set):
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
            s
            for s in mapped_subgenres
            if not any(k in s.lower() for k in incompatible_classical_keywords)
        ]

    return mapped_subgenres


def deduplicate_subgenres(primary_genre: str | None, subgenres: list[str]) -> list[str]:
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
