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

# Restrict BLAS, OpenMP, and TensorFlow intra/inter-op threads to 1 per worker
# to prevent thread contention when running multiple concurrent workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    import essentia

    essentia.log.warningActive = False
    essentia.log.infoActive = False
    essentia.log.active = False
except Exception:
    pass
