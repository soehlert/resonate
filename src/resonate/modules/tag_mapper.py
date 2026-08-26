import logging
import os
from typing import Any

import numpy as np

# Squelch Hugging Face Hub token warnings and progress bars
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MOODS = [
    "party",
    "chill hang",
    "energetic",
    "groovy",
    "acoustic",
    "electronic",
    "melancholic",
    "upbeat",
    "dark",
    "happy",
    "relaxed",
    "aggressive",
    "romantic",
    "calm",
    "mellow",
    "lively",
    "funky",
    "intense",
    "hypnotic",
    "atmospheric",
    "bittersweet",
    "intimate",
]


GENRE_MOOD_SEEDS: dict[str, list[str]] = {
    "Punk Rock": ["Rowdy", "Aggressive"],
    "Skate Punk": ["Rowdy", "Aggressive"],
    "Pop-Punk": ["Rowdy", "Upbeat"],
    "Hardcore": ["Aggressive", "Heavy"],
    "Hard Rock": ["Heavy"],
    "Heavy Metal": ["Heavy", "Aggressive", "Intense"],
    "Grunge": ["Heavy", "Dark"],
    "Dance-Pop": ["Party", "Upbeat"],
    "Disco": ["Party", "Groovy"],
    "Funk": ["Groovy", "Funky"],
    "Funk Rock": ["Groovy", "Funky"],
    "Ska": ["Upbeat"],
    "Ska Punk": ["Rowdy", "Upbeat"],
    "Classical": ["Atmospheric", "Calm"],
    "Baroque": ["Atmospheric", "Calm"],
    "Chamber Music": ["Calm", "Mellow"],
    "Symphonic": ["Atmospheric", "Intense"],
    "Opera": ["Atmospheric", "Intense", "Romantic"],
    "Soft Rock": ["Mellow", "Relaxed"],
    "Acoustic Rock": ["Acoustic", "Mellow"],
    "Singer-Songwriter": ["Acoustic", "Melancholic"],
    "Slowcore": ["Melancholic", "Dark"],
    "Sadcore": ["Melancholic", "Dark"],
    "Emo": ["Melancholic"],
    "Motown": ["Soulful", "Groovy"],
    "Neo-Soul": ["Soulful", "Groovy"],
    "Psychedelic Rock": ["Trippy", "Atmospheric"],
    "Indie Rock": ["Chill Hang"],
    "Indie Pop": ["Chill Hang"],
    "Indie Folk": ["Chill Hang", "Acoustic"],
    "Lo-Fi": ["Chill Hang", "Mellow"],
    "Americana": ["Chill Hang"],
    "Alternative Rock": ["Chill Hang"],
    "Hip-Hop": ["Groovy", "Soulful"],
    "Rap": ["Groovy", "Rowdy"],
    "East Coast Hip Hop": ["Groovy", "Soulful"],
    "West Coast Hip Hop": ["Groovy", "Mellow"],
    "G-Funk": ["Groovy", "Mellow"],
    "Boom Bap": ["Groovy", "Soulful"],
    "Trap": ["Intense", "Dark", "Groovy"],
    "Southern Rap": ["Intense", "Dark", "Groovy"],
    "Gangsta Rap": ["Intense", "Dark", "Aggressive", "Rowdy"],
    "Hardcore Hip Hop": ["Intense", "Dark", "Aggressive", "Rowdy"],
    "Conscious Hip Hop": ["Soulful", "Mellow"],
    "Alternative Hip Hop": ["Groovy", "Soulful"],
    "Cloud Rap": ["Melancholic", "Mellow", "Atmospheric"],
    "Emo Rap": ["Melancholic", "Mellow", "Atmospheric"],
    "R&B": ["Soulful", "Groovy"],
    "Contemporary R&B": ["Soulful", "Groovy"],
    "Rockabilly": ["Upbeat", "Chill Hang"],
    "Progressive Metal": ["Heavy", "Intense", "Atmospheric"],
    "Alternative Metal": ["Heavy", "Intense"],
    "Funk Metal": ["Funky", "Heavy"],
    "Nu-Metal": ["Heavy", "Intense"],
    "Industrial Metal": ["Heavy", "Intense", "Dark"],
    "Sludge Metal": ["Heavy", "Dark"],
    "Thrash Metal": ["Heavy", "Aggressive", "Intense"],
    "Death Metal": ["Heavy", "Aggressive", "Intense"],
    "Black Metal": ["Heavy", "Dark", "Intense"],
    "Doom Metal": ["Heavy", "Dark"],
    "Blues": ["Soulful", "Melancholic"],
    "Delta Blues": ["Melancholic", "Soulful", "Acoustic"],
    "Chicago Blues": ["Soulful", "Groovy", "Melancholic"],
    "Electric Blues": ["Soulful", "Groovy", "Melancholic"],
    "Blues Rock": ["Soulful", "Groovy"],
    "Jazz": ["Soulful", "Mellow"],
    "Big Band": ["Upbeat", "Lively"],
    "Swing": ["Upbeat", "Lively"],
    "Cool Jazz": ["Mellow", "Relaxed", "Atmospheric"],
    "Modal Jazz": ["Mellow", "Atmospheric"],
    "Bebop": ["Energetic", "Intense"],
    "Hard Bop": ["Soulful", "Energetic"],
    "Soul Jazz": ["Soulful", "Groovy"],
    "Vocal Jazz": ["Soulful", "Romantic"],
    "Bossa Nova": ["Chill Hang", "Relaxed"],
    "Latin Jazz": ["Lively", "Groovy"],
    "Smooth Jazz": ["Mellow", "Relaxed"],
    "Jazz Fusion": ["Intense", "Energetic"],
    "Free Jazz": ["Intense", "Experimental"],
    "Dixieland": ["Upbeat", "Lively"],
    "Gypsy Jazz": ["Upbeat", "Lively"],
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
    "Industrial Metal": ["industrial metal", "cyber metal"],
    "Sludge Metal": ["sludge metal", "sludge"],
    "Hardcore Punk": ["hardcore punk"],
    "Rockabilly": ["rockabilly", "rock and roll", "rock n roll"],
    "Oldies": ["oldies"],
    "R&B": ["r&b", "rnb", "rhythm and blues", "rhythm & blues"],
    "Contemporary R&B": ["contemporary r&b", "contemporary rnb", "r&b", "rnb"],
    "Prog Rock": ["prog rock", "progressive rock", "prog"],
    "Progressive Metal": ["progressive metal", "prog metal"],
    "Big Band": ["big band", "big-band", "orchestra"],
    "Swing": ["swing", "swing music", "big band swing"],
    "Bebop": ["bebop", "bop"],
    "Hard Bop": ["hard bop", "hardbop"],
    "Cool Jazz": ["cool jazz", "west coast jazz"],
    "Modal Jazz": ["modal jazz", "modal"],
    "Jazz Fusion": ["jazz fusion", "fusion", "jazz-rock", "jazz rock"],
    "Soul Jazz": ["soul jazz", "soul-jazz"],
    "Smooth Jazz": ["smooth jazz", "contemporary jazz"],
    "Vocal Jazz": ["vocal jazz", "jazz vocals", "standards", "traditional pop"],
    "Latin Jazz": ["latin jazz", "afro-cuban jazz"],
    "Bossa Nova": ["bossa nova", "bossa", "samba jazz"],
    "Free Jazz": ["free jazz", "avant-garde jazz", "avant garde jazz"],
    "Dixieland": ["dixieland", "trad jazz", "traditional jazz", "new orleans jazz"],
    "Gypsy Jazz": ["gypsy jazz", "jazz manouche"],
    "Instrumental": ["instrumental"],
    "Instrumental Rock": ["instrumental rock"],
}

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


def get_genre_seeded_moods(subgenres: list[str]) -> list[str]:
    """Get natural acoustic mood seeds based on mapped sub-genres/styles."""
    seeded = []
    for sg in subgenres:
        if sg in GENRE_MOOD_SEEDS:
            for mood in GENRE_MOOD_SEEDS[sg]:
                if mood not in seeded:
                    seeded.append(mood)
    return seeded


CONTEXTUAL_DESCRIPTIONS: dict[str, str] = {
    # Sub-Genres / Styles
    "Americana": "Americana music, roots rock, alt-country, folk americana",
    "Southern Rock": "Southern rock music, country rock, blues rock, americana rock",
    "Country Rock": "Country rock music, southern rock, country guitar rock",
    "Alt-Country": "Alt-country music, alternative country, americana roots rock",
    "Roots Rock": "Roots rock music, americana, southern rock, classic roots rock",
    "Alternative Rock": "Alternative rock music, 90s alt-rock, indie alternative",
    "Hard Rock": "Hard rock music, heavy guitar riffs, driving loud rock",
    "Heavy Metal": "Heavy metal music, aggressive metal, heavy distortion headbanging",
    "Grunge": "Grunge music, 90s seattle grunge, distorted heavy alt-rock",
    "Indie Rock": "Indie rock music, independent rock band, alt-indie guitar",
    "Indie Pop": "Indie pop music, catchy melody indie pop, cheerful alt-pop",
    "Classic Rock": "Classic rock music, 60s 70s vintage rock, classic album rock",
    "Folk Rock": "Folk rock music, acoustic guitar folk rock, 60s folk rock",
    "Pop Rock": "Pop rock music, mainstream commercial radio pop rock",
    "Psychedelic Rock": "Psychedelic rock music, trippy 60s psych rock, acid rock",
    "British Invasion": "British invasion music, 60s UK rock, Beatlemania garage pop",
    "Prog Rock": "Progressive rock music, prog rock, complex synth art rock",
    "Punk Rock": "Punk rock music, fast energetic DIY underground punk rock",
    "Art Rock": "Art rock music, experimental avant-garde art rock",
    "Glam Rock": "Glam rock music, 70s glam rock, theatrical glitter rock",
    "New Wave": "New wave music, 80s new wave, synth-pop post-punk",
    "Post-Punk": "Post-punk music, dark post-punk, gothic goth post-punk",
    "Acoustic Rock": "Acoustic rock music, unplugged acoustic guitar rock",
    "Soft Rock": "Soft rock music, mellow gentle soft rock ballad",
    "Skate Punk": "Skate punk music, fast melodic skate punk, pop-punk",
    "Garage Rock": "Garage rock music, raw garage rock, 60s garage punk",
    "Disco": "Disco music, 70s dance disco, funky disco groove",
    "Funk": "Funk music, groovy bass funk, rhythm and blues funk",
    "House": "House music, electronic 4/4 dance house beat",
    "EDM": "EDM music, electronic dance music, festival synth drop",
    "Techno": "Techno music, dark underground club techno beat",
    "Rap": "Rap music, hip hop rap verses, rhyming rap track",
    "Hip-Hop": "Hip-hop music, 90s hip hop beats rap music groove",
    "East Coast Hip Hop": "East Coast hip hop music, 90s NYC boom bap rap beats",
    "West Coast Hip Hop": "West Coast hip hop music, California g-funk synth rap",
    "G-Funk": "G-funk music, smooth funk synthesizer West Coast g-funk",
    "Boom Bap": "Boom bap music, 90s drum break jazz sample boom bap rap",
    "Trap": "Trap music, 808 bass hi-hat rolls southern trap beat",
    "Gangsta Rap": "Gangsta rap music, gritty street rap hardcore hip hop",
    "Conscious Hip Hop": "Conscious hip hop music, thoughtful lyrical conscious rap",
    "Cloud Rap": "Cloud rap music, hazy atmospheric reverb lo-fi rap",
    "Emo Rap": "Emo rap music, melancholic guitar trap beat sad rap",
    "Hardcore Hip Hop": "Hardcore hip hop music, aggressive loud hardcore rap",
    "Alternative Hip Hop": "Alternative hip hop music, experimental creative indie rap",
    "Reggaeton": "Reggaeton music, Latin urban reggaeton beat",
    "Ska": "Ska music, upbeat ska punk, brass horn ska dance",
    "Synthpop": "Synthpop music, 80s synthesizer pop, synth-pop",
    "Neo-Soul": "Neo-soul music, smooth modern R&B neo-soul groove",
    "Motown": "Motown music, 60s Detroit soul Motown R&B",
    "R&B": "R&B music, rhythm and blues, soulful smooth groove vocals",
    "Contemporary R&B": "Contemporary R&B music, modern pop R&B, smooth melodic groove",
    "Afrobeat": "Afrobeat music, West African rhythmic afrobeat groove",
    "Bluegrass": "Bluegrass music, acoustic banjo acoustic bluegrass",
    "Singer-Songwriter": "Singer-songwriter music, acoustic guitar vocal ballad",
    "Blues Rock": "Blues rock music, electric guitar blues rock riff",
    "Electric Blues": "Electric blues music, Chicago electric blues guitar",
    "Chicago Blues": "Chicago blues music, harmonica electric blues",
    "Delta Blues": "Delta blues music, acoustic slide guitar country blues",
    "Chamber Music": "Chamber music, classical string quartet acoustic chamber ensemble",
    "Symphonic": "Symphonic music, orchestral classical symphony ensemble",
    "Progressive Metal": "Progressive metal music, prog metal, complex heavy metal guitar riff",
    "Alternative Metal": "Alternative metal music, alt-metal, heavy 90s alternative metal riff",
    "Funk Metal": "Funk metal music, slap bass heavy funk metal, aggressive groove",
    "Nu-Metal": "Nu-metal music, 90s 2000s nu metal, downtuned heavy riff",
    "Industrial Metal": "Industrial metal music, machine electronic synth heavy metal",
    "Sludge Metal": "Sludge metal music, slow heavy distorted sludge doom riff",
    "Big Band": "Big band music, swing orchestra horn section big band jazz",
    "Swing": "Swing music, 30s 40s swing jazz, upbeat dancing swing band",
    "Bebop": "Bebop music, fast tempo complex harmony jazz improvisation",
    "Hard Bop": "Hard bop music, soulful bluesy energetic modern jazz",
    "Cool Jazz": "Cool jazz music, relaxed mellow modal west coast jazz",
    "Modal Jazz": "Modal jazz music, atmospheric modal harmony jazz masterpiece",
    "Jazz Fusion": "Jazz fusion music, electric jazz-rock, virtuosic fusion groove",
    "Soul Jazz": "Soul jazz music, groovy organ blues soul jazz rhythm",
    "Smooth Jazz": "Smooth jazz music, polished mellow contemporary radio jazz",
    "Vocal Jazz": "Vocal jazz music, classic jazz standards singer vocal ballad",
    "Latin Jazz": "Latin jazz music, afro-cuban percussion brass latin groove",
    "Bossa Nova": "Bossa nova music, brazilian acoustic guitar gentle bossa rhythm",
    "Free Jazz": "Free jazz music, avant-garde experimental jazz improvisation",
    "Dixieland": "Dixieland music, traditional new orleans brass jazz band",
    "Gypsy Jazz": "Gypsy jazz music, acoustic guitar swing jazz manouche",
    "Instrumental": "Instrumental music, no vocals, melodic guitar instrumental",
    "Instrumental Rock": "Instrumental rock music, guitar virtuoso rock, melodic instrumental rock",
    # Moods / Vibes
    "Party": "Party music, energetic celebration fun club dance party",
    "Chill Hang": (
        "Chill hang music, millennial indie rock, indie pop, indie folk, "
        "laid-back americana, relaxed mellow easygoing listening, lo-fi chill"
    ),
    "Energetic": "Energetic music, driving high-intensity powerful energetic energy",
    "Groovy": "Groovy music, rhythmic funk bass dance groove",
    "Acoustic": "Acoustic music, unplugged acoustic guitar organic sound",
    "Electronic": "Electronic music, synthesizer electronic beat synth sound",
    "Melancholic": "Melancholic music, sad bittersweet somber melancholic ballad",
    "Lively": "Lively music, bright upbeat active lively animated pop",
    "Relaxed": "Relaxed music, calm peaceful gentle relaxed quiet sound",
    "Romantic": "Romantic music, intimate passionate love romantic ballad",
    "Calm": "Calm music, peaceful quiet meditative calm sound",
    "Upbeat": "Upbeat music, happy cheerful feel-good upbeat pop",
    "Dark": "Dark music, brooding minor key heavy dark atmospheric",
    "Happy": "Happy music, joyful bright happy feel-good song",
    "Mellow": "Mellow music, soft gentle relaxed mellow acoustic",
    "Heavy": "Heavy music, aggressive heavy guitar distortion loud rock",
    "Aggressive": "Aggressive music, intense rowdy loud aggressive metal punk",
    "Soulful": "Soulful music, smooth vocal R&B soulful emotional blues",
    "Trippy": "Trippy music, hypnotic psychedelic spacey trippy sound",
}


class TagMapper:
    """Map raw music tags to target moods via vector embeddings."""

    def __init__(
        self,
        target_moods: list[str] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        model: Any | None = None,
        threshold: float = 0.45,
    ) -> None:
        """Initialize TagMapper with target moods and SentenceTransformer model."""
        self.target_moods = target_moods if target_moods is not None else DEFAULT_TARGET_MOODS
        self.model_name = model_name
        self._model = model
        self.threshold = threshold
        self.target_embeddings: Any = None

        if self._model is not None and self.target_moods:
            self._init_embeddings()

    def _get_model(self) -> Any:
        """Lazy load or return existing SentenceTransformer model."""
        if self._model is None:
            try:
                import os

                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                from sentence_transformers import SentenceTransformer

                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name)
            except Exception as err:
                logger.warning(
                    f"Failed to load SentenceTransformer model '{self.model_name}': {err}"
                )
                self._model = None

        if self._model is not None and self.target_embeddings is None and self.target_moods:
            self._init_embeddings()
        return self._model

    def _encode(self, model: Any, texts: list[str]) -> Any:
        """Encode texts using model, handling convert_to_tensor parameter gracefully."""
        try:
            return model.encode(texts, convert_to_tensor=False)
        except TypeError:
            return model.encode(texts)

    def _init_embeddings(self) -> None:
        """Pre-compute embeddings for target moods using contextual descriptions."""
        if self._model is not None and self.target_moods:
            try:
                descriptions = [
                    CONTEXTUAL_DESCRIPTIONS.get(tm, f"{tm} music") for tm in self.target_moods
                ]
                self.target_embeddings = self._encode(self._model, descriptions)
            except Exception as err:
                logger.warning(f"Failed to pre-compute embeddings for target moods: {err}")
                self.target_embeddings = None

    def match_tags(
        self, raw_tags: list[str], threshold: float | None = None
    ) -> tuple[str | None, str | None, str | None, float]:
        """Match raw tags against target moods using cosine similarity."""
        cutoff = threshold if threshold is not None else self.threshold
        if not raw_tags or not self.target_moods:
            return (None, None, None, 0.0)

        model = self._get_model()
        if model is None or self.target_embeddings is None:
            return (None, None, None, 0.0)

        try:
            raw_embeddings = self._encode(model, raw_tags)
        except Exception as err:
            logger.warning(f"Failed to encode raw tags: {err}")
            return (None, None, None, 0.0)

        raw_arr = np.asarray(raw_embeddings, dtype=np.float32)
        target_arr = np.asarray(self.target_embeddings, dtype=np.float32)

        raw_norms = np.linalg.norm(raw_arr, axis=1, keepdims=True)
        raw_norms = np.maximum(raw_norms, 1e-9)
        target_norms = np.linalg.norm(target_arr, axis=1, keepdims=True)
        target_norms = np.maximum(target_norms, 1e-9)

        raw_norm = raw_arr / raw_norms
        target_norm = target_arr / target_norms

        sim_matrix = np.dot(raw_norm, target_norm.T)

        max_flat_idx = int(np.argmax(sim_matrix))
        best_tag_idx, best_mood_idx = np.unravel_index(max_flat_idx, sim_matrix.shape)
        max_score = float(sim_matrix[best_tag_idx, best_mood_idx])
        best_mood = self.target_moods[best_mood_idx]
        best_raw_tag = raw_tags[best_tag_idx]

        if max_score >= cutoff:
            return (best_mood, best_mood, best_raw_tag, max_score)
        return (None, best_mood, best_raw_tag, max_score)

    def map_tags(
        self, raw_tags: list[str], threshold: float | None = None
    ) -> tuple[str | None, str | None, str | None, float]:
        """Alias for match_tags to map raw tags to target moods."""
        return self.match_tags(raw_tags, threshold=threshold)

    def match_multiple_tags(
        self, raw_tags: list[str], threshold: float | None = None, max_matches: int = 3
    ) -> list[tuple[str, str, float]]:
        """Match raw tags against target tags using configured threshold and capping top results."""
        cutoff = threshold if threshold is not None else self.threshold
        if not raw_tags or not self.target_moods:
            return []

        model = self._get_model()
        if model is None or self.target_embeddings is None:
            return []

        try:
            raw_embeddings = self._encode(model, raw_tags)
        except Exception as err:
            logger.warning(f"Failed to encode raw tags: {err}")
            return []

        raw_arr = np.asarray(raw_embeddings, dtype=np.float32)
        target_arr = np.asarray(self.target_embeddings, dtype=np.float32)

        raw_norms = np.linalg.norm(raw_arr, axis=1, keepdims=True)
        raw_norms = np.maximum(raw_norms, 1e-9)
        target_norms = np.linalg.norm(target_arr, axis=1, keepdims=True)
        target_norms = np.maximum(target_norms, 1e-9)

        raw_norm = raw_arr / raw_norms
        target_norm = target_arr / target_norms

        sim_matrix = np.dot(raw_norm, target_norm.T)

        matched_results = []
        # Track candidates discovered from top consensus tags (raw_tags[:5])
        top_consensus_candidates: set[str] = set()

        for col_idx, target_tag in enumerate(self.target_moods):
            target_clean = target_tag.lower().strip()
            target_words = set(target_clean.replace("-", " ").split())
            is_compound = len(target_words) > 1

            best_raw: str | None = None
            best_score: float = 0.0
            best_raw_idx: int = -1

            for raw_idx, raw in enumerate(raw_tags):
                # Top-5 candidate gating: only raw_tags[:5] can introduce new candidates
                if raw_idx >= 5 and target_tag not in top_consensus_candidates:
                    continue

                raw_clean = raw.lower().strip()
                raw_words = set(raw_clean.replace("-", " ").split())
                rank_factor = max(0.50, 1.0 - (raw_idx * 0.04))

                # 1. Exact string match
                if raw_clean == target_clean or (
                    len(raw_clean) > 3 and raw_clean == target_clean.replace("-", " ")
                ):
                    score = 1.0 * rank_factor
                    if score > best_score:
                        best_score = score
                        best_raw = raw
                        best_raw_idx = raw_idx
                    continue

                # 2. Word-stem substring inclusion
                generic_modifiers = {
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
                }
                if is_compound and raw_clean in generic_modifiers:
                    is_substring = False
                else:
                    is_substring = len(raw_clean) >= 3 and (
                        raw_clean in target_clean
                        or (raw_words and raw_words.issubset(target_words))
                    )

                # Contextual Indie Disambiguation
                if target_tag == "Indie Rock" and "indie" in raw_words:
                    if any(w in r.lower() for r in raw_tags[:3] for w in ["rock", "garage"]):
                        is_substring = True
                elif target_tag == "Indie Pop" and "indie" in raw_words:
                    if any(w in r.lower() for r in raw_tags[:3] for w in ["pop", "dance"]):
                        is_substring = True
                elif target_tag == "Indie Folk" and "indie" in raw_words:
                    if any(w in r.lower() for r in raw_tags[:3] for w in ["folk", "acoustic"]):
                        is_substring = True

                # 3. Data-driven taxonomy stem matching
                if target_tag in PRIMARY_GENRE_STEMS:
                    for stem in PRIMARY_GENRE_STEMS[target_tag]:
                        if stem in raw_clean:
                            # Prevent generic Rock from matching if raw tag belongs to Punk or Metal
                            if target_tag == "Rock" and any(
                                p in raw_clean for p in ["punk", "metal"]
                            ):
                                continue
                            is_substring = True
                            break

                if not is_substring and target_tag in SUB_GENRE_STEMS:
                    for stem in SUB_GENRE_STEMS[target_tag]:
                        if stem in raw_clean:
                            is_substring = True
                            break

                # Block nationality strings from matching Americana
                if target_tag == "Americana" and raw_clean in NATIONALITY_STRINGS:
                    is_substring = False

                if is_substring:
                    score = 0.95 * rank_factor
                    if score > best_score:
                        best_score = score
                        best_raw = raw
                        best_raw_idx = raw_idx

            if best_raw is not None and best_score > 0:
                matched_results.append((target_tag, best_raw, best_score))
                if best_raw_idx < 3:
                    top_consensus_candidates.add(target_tag)
                continue

            # Disable fuzzy vector similarity for sub-genres and primary genres
            if (
                self.target_moods == DEFAULT_SUB_GENRES
                or self.target_moods == DEFAULT_PRIMARY_GENRES
            ):
                continue

            # Instrumentation tags (acoustic, electronic) require explicit keyword hits
            if target_tag.lower() in ["acoustic", "electronic"]:
                continue
            row_idx = int(np.argmax(sim_matrix[:, col_idx]))
            matched_raw = raw_tags[row_idx].lower().strip()
            # Skip fuzzy matching if the matched raw tag is a generic primary genre name
            generic_primary_words = {
                "rock",
                "pop",
                "metal",
                "jazz",
                "blues",
                "country",
                "folk",
                "rap",
                "hip hop",
                "hiphop",
                "electronic",
                "dance",
            }
            if matched_raw in generic_primary_words:
                continue
            score = float(sim_matrix[row_idx, col_idx])
            # Rank-weighted scoring based on Last.fm community consensus order
            rank_factor = max(0.40, 1.0 - (row_idx * 0.05))
            effective_score = score * rank_factor
            if effective_score >= cutoff * 0.80:
                matched_results.append((target_tag, raw_tags[row_idx], effective_score))

        # Deduplicate and keep highest effective_score for each target_tag
        unique_matches: dict[str, tuple[str, float]] = {}
        for tgt, raw, score in matched_results:
            if tgt not in unique_matches or score > unique_matches[tgt][1]:
                unique_matches[tgt] = (raw, score)

        sorted_results = sorted(
            [(k, v[0], v[1]) for k, v in unique_matches.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        # Filter mutually exclusive style pairs (keep higher-scoring/preferred style)
        final_results = []
        for item in sorted_results:
            tag, raw, score = item
            conflict = False
            for group in MUTUALLY_EXCLUSIVE_STYLES:
                if tag in group:
                    for existing in final_results:
                        if existing[0] in group:
                            conflict = True
                            break
            if not conflict:
                final_results.append(item)

        # For subgenres, gate matches strictly to candidates introduced in the top 3 tags
        if self.target_moods == DEFAULT_SUB_GENRES and top_consensus_candidates:
            final_results = [item for item in final_results if item[0] in top_consensus_candidates]

        return final_results[:max_matches]

    def match_genre_consensus(
        self, raw_tags: list[str]
    ) -> list[tuple[str, str, float, int]]:
        """Match all raw tags against primary genres without single-match deduplication.

        Returns a list of (target_genre, raw_tag, score, raw_idx) for every matching raw tag.
        """
        if not raw_tags or not self.target_moods:
            return []

        matched_results: list[tuple[str, str, float, int]] = []
        for raw_idx, raw in enumerate(raw_tags):
            raw_clean = raw.lower().strip()
            raw_words = set(raw_clean.replace("-", " ").split())
            rank_factor = max(0.50, 1.0 - (raw_idx * 0.04))

            for target_tag in self.target_moods:
                target_clean = target_tag.lower().strip()
                target_words = set(target_clean.replace("-", " ").split())
                is_compound = len(target_words) > 1

                # 1. Exact string match
                if raw_clean == target_clean or (
                    len(raw_clean) > 3 and raw_clean == target_clean.replace("-", " ")
                ):
                    score = 1.0 * rank_factor
                    matched_results.append((target_tag, raw, score, raw_idx))
                    continue

                # 2. Word-stem / taxonomy stem matching
                is_stem_match = False
                if target_tag in PRIMARY_GENRE_STEMS:
                    for stem in PRIMARY_GENRE_STEMS[target_tag]:
                        if stem in raw_clean:
                            # Prevent generic Rock from matching if raw tag belongs to Punk or Metal
                            if target_tag == "Rock" and any(
                                p in raw_clean for p in ["punk", "metal"]
                            ):
                                continue
                            is_stem_match = True
                            break

                if is_stem_match:
                    score = 0.95 * rank_factor
                    matched_results.append((target_tag, raw, score, raw_idx))
                    continue

                # 3. Substring inclusion for non-generic modifiers
                generic_modifiers = {
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
                }
                if is_compound and raw_clean in generic_modifiers:
                    continue

                if len(raw_clean) >= 3 and (
                    raw_clean in target_clean
                    or (raw_words and raw_words.issubset(target_words))
                ):
                    # Block nationality strings from matching Americana / Country / Folk
                    if (
                        target_tag in {"Americana", "Country", "Folk"}
                        and raw_clean in NATIONALITY_STRINGS
                    ):
                        continue
                    score = 0.90 * rank_factor
                    matched_results.append((target_tag, raw, score, raw_idx))

        return matched_results


DEFAULT_PRIMARY_GENRES = [
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

MUTUALLY_EXCLUSIVE_STYLES: list[set[str]] = [
    {"Soft Rock", "Hard Rock"},
    {"Soft Rock", "Heavy Metal"},
    {"Soft Rock", "Punk Rock"},
    {"Acoustic Rock", "Hard Rock"},
    {"Acoustic Rock", "Heavy Metal"},
    {"Acoustic Rock", "Punk Rock"},
    {"Pop Rock", "Heavy Metal"},
]

MUTUALLY_EXCLUSIVE_MOODS: list[set[str]] = [
    {"Upbeat", "Dark"},
    {"Upbeat", "Melancholic"},
    {"Upbeat", "Heavy"},
    {"Romantic", "Heavy"},
    {"Romantic", "Aggressive"},
    {"Romantic", "Dark"},
    {"Acoustic", "Heavy"},
    {"Acoustic", "Aggressive"},
    {"Mellow", "Heavy"},
    {"Mellow", "Aggressive"},
]


def resolve_mood_conflicts(moods: list[str]) -> list[str]:
    """Resolve mutually exclusive mood conflicts (e.g. Dark vs Upbeat, Heavy vs Chill Hang)."""
    if not moods:
        return []

    mood_lower_set = {m.lower() for m in moods}

    # If Acoustic or Mellow is present, drop Heavy, Aggressive, and Rowdy
    if any(m in mood_lower_set for m in {"acoustic", "mellow", "meditative", "calm"}):
        moods = [m for m in moods if m.lower() not in {"heavy", "aggressive", "rowdy"}]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Rowdy, Intense, or Dark is present, drop mellow/chill/groovy/happy moods
    if any(
        m in mood_lower_set for m in {"heavy", "aggressive", "rowdy", "intense", "hardcore", "dark"}
    ):
        moods = [
            m
            for m in moods
            if m.lower()
            not in {
                "chill hang",
                "groovy",
                "relaxed",
                "calm",
                "mellow",
                "romantic",
                "happy",
                "upbeat",
            }
        ]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Dark, or Melancholic is present, drop Happy and Upbeat
    if any(m in mood_lower_set for m in {"heavy", "aggressive", "dark", "melancholic"}):
        moods = [m for m in moods if m.lower() not in {"happy", "upbeat"}]
        mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Dark, Rowdy, or Hardcore is present, drop Romantic
    if any(m in mood_lower_set for m in {"heavy", "aggressive", "dark", "rowdy", "hardcore"}):
        moods = [m for m in moods if m.lower() != "romantic"]
        mood_lower_set = {m.lower() for m in moods}

    # Energetic and Lively mutual exclusion (keep Energetic, drop Lively)
    if "energetic" in mood_lower_set and "lively" in mood_lower_set:
        moods = [m for m in moods if m.lower() != "lively"]

    return moods


DEFAULT_SUB_GENRES = [
    "Americana",
    "Southern Rock",
    "Country Rock",
    "Alt-Country",
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
    "Dub",
    "Groove",
    "Bluegrass",
    "Blues Rock",
    "Electric Blues",
    "Chicago Blues",
    "Delta Blues",
    "Chamber Music",
    "Symphonic",
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
    "Emo",
    "Progressive Metal",
    "Alternative Metal",
    "Funk Metal",
    "Nu-Metal",
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

DEFAULT_MOOD_TAGS = [
    "Party",
    "Chill Hang",
    "Energetic",
    "Groovy",
    "Acoustic",
    "Electronic",
    "Melancholic",
    "Lively",
    "Relaxed",
    "Romantic",
    "Calm",
    "Upbeat",
    "Dark",
    "Happy",
    "Fun",
    "Celebration",
    "Festive",
    "Mellow",
    "Feel-Good",
    "Friendly",
    "Intense",
    "Driving",
    "Powerful",
    "Aggressive",
    "Rowdy",
    "Funky",
    "Rhythmic",
    "Soulful",
    "Smooth",
    "Unplugged",
    "Intimate",
    "Organic",
    "Warm",
    "Hypnotic",
    "Futuristic",
    "Atmospheric",
    "Sad",
    "Bittersweet",
    "Somber",
    "Brooding",
    "Gloomy",
    "Emotional",
]
