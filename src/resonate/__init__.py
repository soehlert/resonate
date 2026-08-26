"""Resonate package initialization."""

import logging
import os

__version__ = "0.1.0"

# Silence TensorFlow C++ and oneDNN backend logging before importing models
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    import essentia

    essentia.log.warningActive = False
    essentia.log.infoActive = False
    essentia.log.active = False
except Exception:
    pass
