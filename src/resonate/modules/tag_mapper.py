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


CONTEXTUAL_DESCRIPTIONS: dict[str, str] = {
    # Sub-Genres / Styles
    "Americana": "Americana music, roots rock, alt-country, folk americana",
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
    "Chill Hang": "Chill hang music, relaxed mellow background chillout lounge",
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
        for col_idx, target_tag in enumerate(self.target_moods):
            # Check for exact string match first
            exact_hit = False
            for raw in raw_tags:
                raw_clean = raw.lower().strip()
                target_clean = target_tag.lower().strip()
                if raw_clean == target_clean or (
                    len(raw_clean) > 3 and raw_clean == target_clean.replace("-", " ")
                ):
                    matched_results.append((target_tag, raw, 1.0))
                    exact_hit = True
                    break

            if not exact_hit:
                # Check for word-stem substring inclusion (e.g. "punk" in "Punk Rock")
                stem_hit = False
                for raw in raw_tags:
                    raw_clean = raw.lower().strip()
                    target_clean = target_tag.lower().strip()
                    raw_words = set(raw_clean.replace("-", " ").split())
                    target_words = set(target_clean.replace("-", " ").split())
                    is_substring = len(raw_clean) >= 3 and (
                        raw_clean in target_clean
                        or (raw_words and raw_words.issubset(target_words))
                    )
                    if is_substring:
                        matched_results.append((target_tag, raw, 0.95))
                        stem_hit = True
                        break
                if stem_hit:
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
                if score >= cutoff:
                    matched_results.append((target_tag, raw_tags[row_idx], score))

        # Deduplicate and keep highest score for each target_tag
        unique_matches: dict[str, tuple[str, float]] = {}
        for tgt, raw, score in matched_results:
            if tgt not in unique_matches or score > unique_matches[tgt][1]:
                unique_matches[tgt] = (raw, score)

        sorted_results = sorted(
            [(k, v[0], v[1]) for k, v in unique_matches.items()],
            key=lambda x: x[2],
            reverse=True,
        )
        # Filter mutually exclusive style pairs (keep higher-scoring style)
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

DEFAULT_SUB_GENRES = [
    "Americana",
    "Alternative Rock",
    "Hard Rock",
    "Heavy Metal",
    "Grunge",
    "Indie Rock",
    "Indie Pop",
    "Classic Rock",
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
