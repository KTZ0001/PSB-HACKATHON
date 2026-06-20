"""Central path + runtime configuration for Aegis.

Single source of truth for where data and model artifacts live, so nothing
hardcodes paths. Values can be overridden via environment variables (see
.env.example), but every setting has a working default.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Data locations ---
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

NEURIPS_BAF_DIR = DATA_RAW / "neurips_baf"
SYNTHETIC_FIN_DIR = DATA_RAW / "synthetic_financial"

# The NeurIPS BAF suite ships 6 variants (Base + Variant I..V). Aegis trains on
# the Base variant; the others exist for fairness/bias testing and are NOT used
# yet (see build prompt Dataset 1).
NEURIPS_BAF_BASE_FILE = "Base.csv"

# --- Model artifacts ---
MODEL_DIR = Path(os.environ.get("AEGIS_MODEL_DIR", REPO_ROOT / "models_store"))

# --- API ---
API_HOST = os.environ.get("AEGIS_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("AEGIS_API_PORT", "8000"))
