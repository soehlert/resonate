# Brainstorm: Incorporating Lyrics into the Mood Detection Pipeline

## Goal
Enrich Resonate's mood detection pipeline by integrating lyrics fetching and sentiment/theme analysis as an auxiliary, low-weight modifier and guardrail/veto filter. This ensures musical vibe (e.g., upbeat, chill hang) remains primary while preventing glaring mood mismatches on lyrically dark or explicit tracks (e.g., "Pumped Up Kicks" remaining "Chill Hang" but being disqualified from "Happy/Feel-Good").

## Constraints
- **Zero-Cost & Keyless API**: Lyrics retrieval must be free, robust, and require no user registration, API keys, or accounts.
- **Low Weight / Non-Destructive**: Lyrical sentiment must not override acoustic energy or core genres (e.g., an upbeat indie-pop song must not become a doom ballad simply because the lyrics are sarcastic or dark).
- **Offline / Local First**: Must prioritize existing local embedded audio tags (`USLT`, `LYRICS`) and sidecar files before making external network calls.
- **Performant & Lightweight**: Sentiment analysis must run efficiently during batch processing without introducing massive LLM dependencies or heavy GPU requirements.
- **Graceful Degradation**: Instrumental songs, foreign language tracks, or tracks with missing lyrics must proceed through the pipeline without latency penalties or mood distortion.

## Known context
- **Current Pipeline**:
  - Phase 1: Tag collection from Last.fm, Beets, MusicBrainz, Discogs, and raw file tags.
  - Phase 2: BPM detection via Librosa/Aubio.
  - Phase 2.5: Essentia waveform deep learning analysis (`mtg_jamendo_moodtheme-discogs-effnet-1.pb`) for acoustic mood classification.
  - Phase 3: Consolidation via `TagMapper` (`all-MiniLM-L6-v2` embeddings) and rule-based mutual exclusion (`resolve_mood_conflicts`).
- **Storage & State**: Resonate already maintains a local SQLite database (`data/state.sqlite`) for caching processed state and track results.
- **Embeddings**: `sentence-transformers` is already bundled and loaded in memory for `TagMapper`.

## Risks
1. **The "Pumped Up Kicks" False-Negative Trap**: Overweighting dark lyrics could cause catchy, upbeat indie/pop songs to be classified as depressing/gloomy music, ruining casual playlists.
2. **Lyrics Provider Rate Limiting & Outages**: External lyrics providers can be rate-limited, blocked, or return incorrect lyrics for live/remix versions.
3. **Sarcasm and Metaphor Blindness**: Standard NLP sentiment models often misinterpret irony, slang, or metaphorical dark themes.
4. **Processing Overhead**: Analyzing full lyric texts across thousands of library tracks could slow down batch runs if heavyweight NLP models are introduced.

## Options (2–4)

### Option 1: Multi-Tier Lyrics Provider (Local + LRCLIB) with Dual-Track Scoring (Lexicon + MiniLM Semantic Moods & Veto Filter)
- **Lyrics Source**:
  - Tier 1: Local embedded tags via Mutagen (`USLT` frame in ID3, `LYRICS`/`UNSYNCEDLYRICS` in Vorbis/FLAC, `©lyr` in MP4) and `.lrc` sidecars.
  - Tier 2: LRCLIB API (`https://lrclib.net/api/get` or `/api/search`) — free, open, fast JSON REST API, no auth required.
  - Caching: Persist retrieved lyrics in `state.sqlite`.
- **Sentiment & Mood Analysis**:
  - Split lyric evaluation into two dimensions:
    1. **Valence/Polarity Score** (via VADER / lightweight valence lexicon): produces a -1.0 to +1.0 polarity score.
    2. **Thematic Embeddings** (via existing `all-MiniLM-L6-v2`): embed key lyric verses/choruses and measure cosine similarity against lyrical mood seeds (`romantic`, `melancholic`, `dark`, `angsty`, `party/celebratory`).
- **Pipeline Integration**:
  - **Weighting (15-20%)**: Lyrical mood candidates contribute low-weight additive score to `combined_moods`.
  - **Veto Filter (Polarity Guardrail)**: If lyrics have extreme negative polarity / dark thematic score (> 0.85):
    - Suppress purely joyful moods: `Happy`, `Feel-Good`, `Celebration`.
    - Retain acoustic/energy moods: `Chill Hang`, `Energetic`, `Upbeat`, `Groovy`.
    - Add descriptive nuance tag: `Dark` or `Bittersweet`.

### Option 2: Dedicated Transformer Emotion Classifier (e.g. RoBERTa Emotion / GoEmotions)
- **Lyrics Source**: LRCLIB API + local tags.
- **Sentiment Analysis**:
  - Download a dedicated HuggingFace classification pipeline (e.g., `bhadresh-psav/distilbert-base-uncased-emotion` or `j-hartmann/emotion-english-distilroberta-base` predicting fine-grained emotions like joy, sadness, anger, fear, surprise, love).
- **Pros**: High classification accuracy on complex text.
- **Cons**: High memory overhead (another ~300MB-500MB PyTorch model loaded alongside Essentia and MiniLM), significantly slower inference on large libraries.

### Option 3: Keyword / Blacklist Pattern Matching & Strict Rule Engine
- **Lyrics Source**: LRCLIB API + local tags.
- **Sentiment Analysis**:
  - Simple regex / keyword matching against curated dictionaries (e.g., words associated with violence, suicide, despair vs party, dancing, love).
- **Pros**: Zero dependencies, instantaneous evaluation (0.1ms).
- **Cons**: Brittle, prone to high false positives on harmless word usage or metaphors, lacks semantic subtlety.

## Recommendation
**Adopt Option 1 (Multi-Tier Local + LRCLIB with Dual-Track MiniLM + Valence Veto Engine).**

### Why this approach fits best:
1. **Zero New Heavy Dependencies**: It leverages the existing `all-MiniLM-L6-v2` model already in memory in `TagMapper`, keeping RAM low and speed fast.
2. **Solves the "Pumped Up Kicks" Dilemma**: By separating *acoustic vibe* (`Chill Hang`, `Upbeat`) from *valence vetoes*, a dark lyrical theme safely removes "Happy" without disqualifying the song from casual "Chill Hang" playlists.
3. **Rock-Solid Free Source**: LRCLIB (`lrclib.net`) is the current gold standard for open-source lyrics (backed by the community, REST JSON, no API key needed), complemented by instant local tag reading via Mutagen.
4. **Cached in SQLite**: Once fetched, lyrics are stored in `data/state.sqlite`, ensuring instant subsequent runs and zero repeated network requests.

## Acceptance criteria
1. **Lyrics Module (`lyrics.py`)**:
   - Reads local embedded metadata tags (`mutagen`) and checks for sidecar `.lrc`/`.txt` files.
   - Falls back to `lrclib.net` REST API with clean error handling and rate-limit friendliness.
   - Caches fetched lyrics in SQLite `state.sqlite`.
2. **Lyric Mood / Sentiment Analyzer**:
   - Computes lyrical valence and maps semantic theme embeddings using `all-MiniLM-L6-v2`.
   - Distinguishes between light melancholy, romantic themes, celebratory themes, and extreme dark/violent themes.
3. **Pipeline Integration & Weighting in `main.py` & `tag_mapper.py`**:
   - Lyrical scores are integrated at a configurable weight (default: 0.15 - 0.20).
   - Veto logic successfully removes `Happy` / `Feel-Good` when lyrical darkness exceeds threshold, while preserving `Chill Hang` / `Energetic` for upbeat/groove songs.
4. **Configuration & CLI Control**:
   - `config.yaml` includes a `lyrics` section (`enabled: bool`, `weight: float`, `veto_happy_on_dark: bool`, `prefer_embedded: bool`).
5. **Testing**:
   - Unit tests covering local tag extraction, LRCLIB API client (mocked), lyric sentiment scoring, and pipeline veto behavior (including test cases for "Pumped Up Kicks" style songs).
