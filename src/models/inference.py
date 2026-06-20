"""Behavioral risk engine inference.

Wraps a trained ModelBundle and turns a single record (dict or DataFrame row)
into the behavioral signals the score combiner (Phase 3) consumes:
  - anomaly_risk  (0-100, IsolationForest, higher = more anomalous)
  - class_proba   ({legitimate, account_takeover, mule_network} -> prob)
  - predicted_type / confidence

This is intentionally pure with respect to the model: no graph, no social-
engineering rules here (those are separate layers combined downstream).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features import baf_features as bf
from src.models.registry import ModelBundle, load_bundle


@dataclass
class BehavioralResult:
    anomaly_risk: float          # 0-100
    class_proba: dict            # class name -> probability
    predicted_type: str
    confidence: float            # probability of the predicted class
    xgb_fraud_proba: float       # P(not legitimate)


class RiskEngine:
    """Loads a ModelBundle once and scores records."""

    def __init__(self, bundle: ModelBundle | None = None):
        self.bundle = bundle or load_bundle()

    def score_frame(self, df: pd.DataFrame) -> list[BehavioralResult]:
        X = self.bundle.feature_pipeline.transform(df)

        decision = self.bundle.isolation_forest.decision_function(X)
        proba = self.bundle.xgb_classifier.predict_proba(X)

        results = []
        legit_idx = bf.CLASS_TO_IDX[bf.CLASS_LEGIT]
        for i in range(len(X)):
            anomaly_risk = self.bundle.normalise_anomaly(float(decision[i]))
            class_proba = {
                cls: float(proba[i][bf.CLASS_TO_IDX[cls]]) for cls in self.bundle.class_order
            }
            pred_idx = int(proba[i].argmax())
            predicted_type = self.bundle.class_order[pred_idx]
            results.append(
                BehavioralResult(
                    anomaly_risk=anomaly_risk,
                    class_proba=class_proba,
                    predicted_type=predicted_type,
                    confidence=float(proba[i][pred_idx]),
                    xgb_fraud_proba=float(1.0 - proba[i][legit_idx]),
                )
            )
        return results

    def score_record(self, record: dict) -> BehavioralResult:
        return self.score_frame(pd.DataFrame([record]))[0]
