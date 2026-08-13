# Essentia Models Directory

This directory stores pre-trained TensorFlow models (`.pb` files) used by Essentia for local audio waveform mood/genre classification and feature extraction.

## Downloading Pre-trained Models

To enable local audio waveform fallback analysis, download the required Discogs EffNet backbone model and classification head into the `models/` directory:

### 1. Download Model Files
Run the following commands in your terminal inside your `resonate` project directory:

```bash
cd models

# 1. Feature Extractor Backbone (~70 MB)
curl -O https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb

# 2. Jamendo Mood/Theme Classification Head Weights (~2.7 MB) & Metadata
curl -O https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb
curl -O https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json

# 3. Optional: Discogs 400 Genre/Style Classification Head (~2 MB)
curl -O https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -O https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json
```

### 2. Verify Files
Ensure the `models/` directory contains:
- `discogs-effnet-bs64-1.pb` (Feature Extractor)
- `mtg_jamendo_moodtheme-discogs-effnet-1.pb` (Jamendo Mood & Theme Model)
- `mtg_jamendo_moodtheme-discogs-effnet-1.json` (Mood Class Labels Metadata)
