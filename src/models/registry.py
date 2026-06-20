"""Persistence for trained Aegis model artifacts.

One bundle = the feature pipeline + IsolationForest + XGBoost classifier +
anomaly-score normalisation stats + metadata (version, trained_at). Saved/loaded
atomically so the API always loads a self-consistent set.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib

from src import config

BUNDLE_FILENAME = "aegis_models.joblib"
MODEL_VERSION = "0.1.0"


@dataclass
class ModelBundle:
    feature_pipeline: Any
    isolation_forest: Any
    anomaly_decision_min: float  # training-set min of decision_function
    anomaly_decision_max: float  # training-set max of decision_function
    xgb_classifier: Any
    class_order: list[str]
    model_version: str = MODEL_VERSION
    trained_at: str = ""
    metrics: dict | None = None

    def normalise_anomaly(self, decision_value: float) -> float:
        """Map IsolationForest decision_function -> anomaly risk in [0,100].

        decision_function: higher = more normal. We invert and min-max scale
        against the training distribution so 100 = most anomalous.
        """
        lo, hi = self.anomaly_decision_min, self.anomaly_decision_max
        if hi <= lo:
            return 0.0
        norm = (decision_value - lo) / (hi - lo)  # 0 = anomalous, 1 = normal
        norm = max(0.0, min(1.0, norm))
        return float((1.0 - norm) * 100.0)


def save_bundle(bundle: ModelBundle, model_dir: Path | None = None) -> Path:
    model_dir = Path(model_dir or config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not bundle.trained_at:
        bundle.trained_at = dt.datetime.now(dt.timezone.utc).isoformat()
    path = model_dir / BUNDLE_FILENAME
    joblib.dump(bundle, path)
    return path


def load_bundle(model_dir: Path | None = None) -> ModelBundle:
    model_dir = Path(model_dir or config.MODEL_DIR)
    path = model_dir / BUNDLE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model bundle at {path}. Run `python scripts/train.py` first."
        )
    return joblib.load(path)


def bundle_exists(model_dir: Path | None = None) -> bool:
    model_dir = Path(model_dir or config.MODEL_DIR)
    return (model_dir / BUNDLE_FILENAME).exists()
