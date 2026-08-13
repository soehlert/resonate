# Essentia Models Directory

This directory stores pre-trained TensorFlow models (`.pb` files) used by Essentia for local audio waveform mood/genre classification and feature extraction.

## Downloading Pre-trained Models

To enable local audio waveform fallback analysis, download the required Discogs EffNet backbone model and classification head into the `models/` directory:

### 1. Download Model Files
Run the following commands in your terminal inside your `resonate` project directory:

```bash
cd models

# 1. Feature Extractor Backbone
curl -O https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb

# 2. Classification Head Weights & Metadata
curl -O https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -O https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json
```

### 2. Verify Files
Ensure the `models/` directory contains:
- `discogs-effnet-bs64-1.pb` (Feature Extractor)
- `genre_discogs400-discogs-effnet-1.pb` (Classification Head)
- `genre_discogs400-discogs-effnet-1.json` (Class Labels Metadata)
