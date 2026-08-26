# Resonate Backlog & Roadmap

## Completed Features
- [x] **Primary Genre & Sub-Genre Classification**: Taxonomy-guided primary genre and granular style mapping.
- [x] **Essentia Waveform Audio Deep Learning Analysis**: Deep learning acoustic mood classification via Discogs-EffNet graph models.
- [x] **BPM Audio Detection**: Offline tempo estimation directly from audio waveforms using `librosa`.
- [x] **Direct Metadata Tagging**: Embedded file tagging via Mutagen for MP3, FLAC, and M4A/MP4 files.
- [x] **Plex Media Server Sync**: Automatic genre, mood, and BPM synchronization.
- [x] **Multi-Tier Lyrics Enrichment**:
  - Embedded metadata (`USLT`, `LYRICS`, `©lyr`) and sidecar `.lrc`/`.txt` extraction.
  - Zero-cost, keyless LRCLIB REST API client.
  - SQLite persistent caching in `state.sqlite`.
  - Valence polarity scoring and semantic mood classification with `TagMapper` embeddings.
  - Mood conflict resolution and guardrail filtering (preventing dark lyrics from receiving cheerful tags while preserving acoustic vibes like `Chill Hang`).

## Upcoming Ideas & Enhancements
- [ ] Spotify playlist sync integration.
- [ ] Multi-lingual lyrics translation / cross-lingual sentiment mapping.
- [ ] ReplayGain / EBU R128 loudness normalization analysis.
