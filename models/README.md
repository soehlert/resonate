# Essentia Models Directory

This directory stores pre-trained TensorFlow models (`.pb` files) used by Essentia for local audio waveform mood classification and feature extraction.

## Downloading Pre-trained Models

To enable local audio waveform fallback analysis, download the required Discogs EffNet backbone model and mood classification head into the `models/` directory:

### 1. Download Model Files
Run the following commands on your host server inside your `resonate` project directory:

```bash
cd models
curl -O https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -O https://essentia.upf.edu/models/classification-heads/music_mood/music_mood-discogs-effnet-1.pb
curl -O https://essentia.upf.edu/models/classification-heads/music_mood/music_mood-discogs-effnet-1.json
```

### 2. Verify Files
Ensure the `models/` directory contains:
- `discogs-effnet-bs64-1.pb` (Feature Extractor)
- `music_mood-discogs-effnet-1.pb` (Mood Classification Head)
- `music_mood-discogs-effnet-1.json` (Class Labels Metadata)
