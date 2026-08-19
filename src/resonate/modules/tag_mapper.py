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
    "Emo": ["Melancholic", "Chill Hang"],
    "Motown": ["Soulful", "Groovy"],
    "Neo-Soul": ["Soulful", "Groovy"],
    "Psychedelic Rock": ["Trippy", "Atmospheric"],
    "Indie Rock": ["Chill Hang"],
    "Indie Pop": ["Chill Hang"],
    "Indie Folk": ["Acoustic", "Chill Hang"],
    "Alternative Rock": ["Chill Hang"],
    "Americana": ["Chill Hang"],
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
    "Dance-Pop": "Dance-pop music, upbeat electronic dance pop single",
    "Rap": "Rap music, hip hop rap verses, rhyming rap track",
    "Reggaeton": "Reggaeton music, Latin urban reggaeton beat",
    "Ska": "Ska music, upbeat ska punk, brass horn ska dance",
    "Synthpop": "Synthpop music, 80s synthesizer pop, synth-pop",
    "Neo-Soul": "Neo-soul music, smooth modern R&B neo-soul groove",
    "Motown": "Motown music, 60s Detroit soul Motown R&B",
    "Afrobeat": "Afrobeat music, West African rhythmic afrobeat groove",
    "Bluegrass": "Bluegrass music, acoustic banjo acoustic bluegrass",
    "Singer-Songwriter": "Singer-songwriter music, acoustic guitar vocal ballad",
    "Blues Rock": "Blues rock music, electric guitar blues rock riff",
    "Electric Blues": "Electric blues music, Chicago electric blues guitar",
    "Chicago Blues": "Chicago blues music, harmonica electric blues",
    "Delta Blues": "Delta blues music, acoustic slide guitar country blues",
    "Chamber Music": "Chamber music, classical string quartet acoustic chamber ensemble",
    "Symphonic": "Symphonic music, orchestral classical symphony ensemble",
    "Classical": "Classical music, acoustic orchestra piano classical composition",
    # Moods / Vibes
    "Party": "Party music, energetic celebration fun club dance party",
    "Chill Hang": (
        "Chill hang music, millennial indie rock, indie pop, indie folk, 2000s alt-rock, nostalgic"
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

            # Check for exact string match first
            exact_hit = False
            for raw_idx, raw in enumerate(raw_tags):
                # Top-5 candidate gating: only raw_tags[:5] can introduce new candidates
                if raw_idx >= 5 and target_tag not in top_consensus_candidates:
                    continue

                raw_clean = raw.lower().strip()
                if raw_clean == target_clean or (
                    len(raw_clean) > 3 and raw_clean == target_clean.replace("-", " ")
                ):
                    rank_factor = max(0.50, 1.0 - (raw_idx * 0.04))
                    matched_results.append((target_tag, raw, 1.0 * rank_factor))
                    if raw_idx < 5:
                        top_consensus_candidates.add(target_tag)
                    exact_hit = True
                    break

            if not exact_hit:
                # Check for word-stem substring inclusion
                stem_hit = False
                for raw_idx, raw in enumerate(raw_tags):
                    # Top-5 candidate gating: only raw_tags[:5] can introduce new candidates
                    if raw_idx >= 5 and target_tag not in top_consensus_candidates:
                        continue

                    raw_clean = raw.lower().strip()
                    raw_words = set(raw_clean.replace("-", " ").split())

                    # Generic single modifier words must not match compound sub-genres blindly
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
                        if any(w in r.lower() for r in raw_tags[:5] for w in ["rock", "garage"]):
                            is_substring = True
                    elif target_tag == "Indie Pop" and "indie" in raw_words:
                        if any(w in r.lower() for r in raw_tags[:5] for w in ["pop", "dance"]):
                            is_substring = True
                    elif target_tag == "Indie Folk" and "indie" in raw_words:
                        if any(w in r.lower() for r in raw_tags[:5] for w in ["folk", "acoustic"]):
                            is_substring = True

                    # Stem matches for primary genre targets with multi-word raw tags
                    if target_tag == "Rock" and (
                        "rock" in raw_words
                        or "rock" in raw_clean
                        or any(
                            r in raw_clean for r in ["rock n roll", "rock and roll", "rockabilly"]
                        )
                    ):
                        is_substring = True
                    elif target_tag == "Pop" and ("pop" in raw_words or "pop" in raw_clean):
                        is_substring = True
                    elif target_tag == "Metal" and ("metal" in raw_words or "metal" in raw_clean):
                        is_substring = True
                    elif target_tag == "Punk" and ("punk" in raw_words or "punk" in raw_clean):
                        is_substring = True
                    elif target_tag == "Hip-Hop" and any(
                        kw in raw_clean for kw in ["hip-hop", "hip hop", "rap", "hiphop"]
                    ):
                        is_substring = True
                    elif target_tag == "Jazz" and ("jazz" in raw_words or "jazz" in raw_clean):
                        is_substring = True
                    elif target_tag == "Blues" and ("blues" in raw_words or "blues" in raw_clean):
                        is_substring = True
                    elif target_tag == "Country" and (
                        "country" in raw_words or "country" in raw_clean
                    ):
                        is_substring = True
                    elif target_tag == "Folk" and ("folk" in raw_words or "folk" in raw_clean):
                        is_substring = True
                    elif target_tag == "Electronic" and (
                        "electronic" in raw_words
                        or "electronic" in raw_clean
                        or any(
                            e in raw_clean
                            for e in [
                                "electronica",
                                "techno",
                                "house",
                                "trance",
                                "edm",
                                "synthpop",
                                "electropop",
                                "synthwave",
                                "dubstep",
                                "dnb",
                                "drum and bass",
                                "ambient",
                                "club",
                            ]
                        )
                    ):
                        is_substring = True
                    elif target_tag == "R&B" and any(
                        kw in raw_clean for kw in ["r&b", "rnb", "rhythm and blues"]
                    ):
                        is_substring = True
                    elif target_tag == "Soul" and ("soul" in raw_words or "soul" in raw_clean):
                        is_substring = True
                    elif target_tag == "Reggae" and (
                        "reggae" in raw_words or "reggae" in raw_clean
                    ):
                        is_substring = True
                    elif target_tag == "Latin" and (
                        "latin" in raw_words or "latin" in raw_clean or "reggaeton" in raw_clean
                    ):
                        is_substring = True
                    elif target_tag == "Dance" and ("dance" in raw_words or "dance" in raw_clean):
                        is_substring = True
                    elif target_tag == "Classical" and (
                        "classical" in raw_words or "classical" in raw_clean
                    ):
                        is_substring = True
                    elif target_tag == "Indie" and ("indie" in raw_words or "indie" in raw_clean):
                        is_substring = True
                    elif target_tag == "Punk Rock" and (
                        "punk rock" in raw_clean or ("punk" in raw_words and len(raw_words) == 1)
                    ):
                        is_substring = True
                    elif target_tag == "Rap" and any(
                        w in raw_clean for w in ["rap", "hip hop", "hip-hop", "hiphop"]
                    ):
                        is_substring = True
                    elif target_tag == "Thrash Metal" and "thrash" in raw_clean:
                        is_substring = True
                    elif target_tag == "Hardcore Punk" and (
                        "hardcore punk" in raw_clean
                        or ("hardcore" in raw_clean and "punk" in raw_clean)
                    ):
                        is_substring = True
                    elif target_tag == "Rockabilly" and (
                        "rockabilly" in raw_clean or "rock n roll" in raw_clean
                    ):
                        is_substring = True
                    elif target_tag == "Oldies" and "oldies" in raw_clean:
                        is_substring = True

                    # Block nationality strings from matching Americana
                    if target_tag == "Americana" and raw_clean in {
                        "american",
                        "british",
                        "australian",
                        "canadian",
                        "german",
                        "french",
                        "japanese",
                        "english",
                    }:
                        is_substring = False

                    if is_substring:
                        rank_factor = max(0.50, 1.0 - (raw_idx * 0.04))
                        matched_results.append((target_tag, raw, 0.95 * rank_factor))
                        if raw_idx < 5:
                            top_consensus_candidates.add(target_tag)
                        stem_hit = True
                        break
                if stem_hit:
                    continue

                # Disable fuzzy vector similarity for sub-genres to prevent sub-genre hallucinations
                # (e.g. 'pop rock' mapping to 'Post-Rock' or 'chillout' mapping to 'Dubstep')
                if self.target_moods == DEFAULT_SUB_GENRES:
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

        return final_results[:max_matches]


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
    {"Acoustic Rock", "Heavy Metal"},
    {"Pop Rock", "Heavy Metal"},
]

MUTUALLY_EXCLUSIVE_MOODS: list[set[str]] = [
    {"Upbeat", "Dark"},
    {"Upbeat", "Melancholic"},
    {"Upbeat", "Heavy"},
    {"Romantic", "Heavy"},
    {"Romantic", "Aggressive"},
    {"Romantic", "Dark"},
]


def resolve_mood_conflicts(moods: list[str]) -> list[str]:
    """Resolve mutually exclusive mood conflicts (e.g. Dark vs Upbeat, Heavy vs Chill Hang)."""
    if not moods:
        return []

    mood_lower_set = {m.lower() for m in moods}

    # If Heavy, Aggressive, Rowdy, or Intense is present, drop mellow/chill/groovy/happy moods
    if any(m in mood_lower_set for m in {"heavy", "aggressive", "rowdy", "intense", "hardcore"}):
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

    # If Heavy, Aggressive, or Dark is present, drop Romantic
    if any(m in mood_lower_set for m in {"heavy", "aggressive", "dark"}):
        moods = [m for m in moods if m.lower() != "romantic"]
        mood_lower_set = {m.lower() for m in moods}

    # If Energetic or Lively is present, drop Melancholic
    if any(m in mood_lower_set for m in {"energetic", "lively"}):
        moods = [m for m in moods if m.lower() != "melancholic"]
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
    "Reggaeton",
    "Ska",
    "Synthpop",
    "Synthwave",
    "Neo-Soul",
    "Motown",
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
    "Ballad",
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
