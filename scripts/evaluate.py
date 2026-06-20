"""Evaluate the trained behavioral engine and print sample scored examples.

Produces the Phase 1 stop-point artefact: stored test metrics + a sample of
5 scored examples (2 legitimate, 2 account_takeover, 1 mule_network) drawn from
the temporal test split, scored through the real inference path.

Usage:  python scripts/evaluate.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config
from src.features import baf_features as bf
from src.models.inference import RiskEngine
from src.models.registry import load_bundle

SHOW_SIGNALS = [
    "name_email_similarity", "credit_risk_score", "bank_months_count",
    "device_distinct_emails_8w", "foreign_request", "velocity_6h",
    "proposed_credit_limit", "customer_age",
]


def main() -> int:
    bundle = load_bundle()
    print(f"Model version: {bundle.model_version}  trained_at: {bundle.trained_at}")
    m = bundle.metrics or {}
    print(f"\nTest size: {m.get('n_test'):,}  |  IsolationForest anomaly AUC: "
          f"{m.get('iso_anomaly_auc'):.4f}  |  XGBoost P(fraud) AUC: {m.get('xgb_fraud_auc'):.4f}")
    print("\nDerived label distribution (full dataset):")
    for k, v in (m.get("label_distribution") or {}).items():
        print(f"  {k:<18} {v:,}")
    print("\n=== XGBoost classification report (test) ===")
    print(m.get("xgb_report", "(missing)"))

    # Load test split and pick representative rows by derived label.
    df = pd.read_csv(config.NEURIPS_BAF_DIR / config.NEURIPS_BAF_BASE_FILE)
    df = df.drop(columns=[c for c in bf.CONSTANT_COLS if c in df.columns])
    df["fraud_type"] = bf.derive_fraud_type_label(df)
    _, test = bf.temporal_split(df)

    picks = pd.concat([
        test[test.fraud_type == bf.CLASS_LEGIT].head(2),
        test[test.fraud_type == bf.CLASS_ATO].head(2),
        test[test.fraud_type == bf.CLASS_MULE].head(1),
    ])

    engine = RiskEngine(bundle=bundle)
    results = engine.score_frame(picks)

    print("\n" + "=" * 78)
    print("SAMPLE SCORED EXAMPLES (from temporal test split)")
    print("=" * 78)
    for (_, row), res in zip(picks.iterrows(), results):
        print(f"\n--- true_label={row['fraud_type']} ---")
        print(f"  anomaly_risk (0-100):  {res.anomaly_risk:6.2f}")
        print(f"  predicted_type:        {res.predicted_type}  (confidence {res.confidence:.3f})")
        probs = "  ".join(f"{k}={v:.3f}" for k, v in res.class_proba.items())
        print(f"  class_proba:           {probs}")
        print(f"  P(fraud):              {res.xgb_fraud_proba:.3f}")
        sig = "  ".join(f"{s}={row[s]}" for s in SHOW_SIGNALS if s in row)
        print(f"  raw signals:           {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
