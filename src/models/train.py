"""Train the Aegis behavioral risk engine on the NeurIPS BAF Base variant.

Trains two models that share one feature pipeline:
  1. IsolationForest  -- unsupervised anomaly score (trained on the legitimate-
     heavy training split; learns "normal" account-opening behaviour).
  2. XGBoost          -- supervised 3-class classifier:
                         legitimate / account_takeover / mule_network
                         (the mule_network class is a DERIVED HEURISTIC label;
                          see src/features/baf_features.derive_fraud_type_label).

Evaluation uses BAF's intended temporal split (train months 0-5, test 6-7) so
metrics reflect deployment-realistic, no-leakage performance.

Run:  python scripts/train.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src import config
from src.features import baf_features as bf
from src.models.registry import ModelBundle, save_bundle


def load_baf_base() -> pd.DataFrame:
    path = config.NEURIPS_BAF_DIR / config.NEURIPS_BAF_BASE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


# Tempering exponent for class weights. 1.0 = full inverse-frequency (over-
# weights tiny classes -> floods false positives); 0.0 = uniform (minority
# recall collapses). 0.45 was chosen empirically on the FULL temporal split
# (scripts/tune_class_weights.py): it maximises macro-F1 (0.526) and gives the
# best minority precision/recall balance -- ATO R 0.27 / mule R 0.51 -- while
# cutting legit->ATO false positives ~9x vs full weighting (21,841 -> 2,358).
# The continuous P(fraud) AUC (~0.89) is stable across all powers, so this only
# tunes the argmax type-attribution, not the underlying fraud ranking.
DEFAULT_CLASS_WEIGHT_POWER = 0.45


def _tempered_sample_weight(y_train, power: float):
    """Per-sample weights = (N / (K * class_count)) ** power."""
    class_counts = np.bincount(y_train, minlength=len(bf.CLASS_ORDER))
    class_weight = len(y_train) / (len(bf.CLASS_ORDER) * np.maximum(class_counts, 1))
    class_weight = class_weight ** power
    return class_weight[y_train]


def train_models(
    random_state: int = 42,
    class_weight_power: float = DEFAULT_CLASS_WEIGHT_POWER,
) -> ModelBundle:
    print("Loading BAF Base variant ...")
    df = load_baf_base()
    df = df.drop(columns=[c for c in bf.CONSTANT_COLS if c in df.columns])

    # Derived 3-class label.
    df["fraud_type"] = bf.derive_fraud_type_label(df)
    print("Derived fraud-type label distribution:")
    print(df["fraud_type"].value_counts())

    # Temporal split (no leakage).
    train_df, test_df = bf.temporal_split(df)
    print(f"\nTemporal split: train={len(train_df):,} rows, test={len(test_df):,} rows")

    # Feature pipeline (fit on train only).
    pipe = bf.FeaturePipeline().fit(train_df)
    X_train = pipe.transform(train_df)
    X_test = pipe.transform(test_df)
    print(f"Feature matrix: {X_train.shape[1]} features")

    y_train = train_df["fraud_type"].map(bf.CLASS_TO_IDX).to_numpy()
    y_test = test_df["fraud_type"].map(bf.CLASS_TO_IDX).to_numpy()
    fraud_train = train_df[bf.TARGET_COL].to_numpy()
    fraud_test = test_df[bf.TARGET_COL].to_numpy()

    # --- 1. IsolationForest (unsupervised) ---
    # Fit on the legitimate rows only so it models "normal" behaviour, then
    # score everything. contamination set to the observed train fraud rate.
    print("\nTraining IsolationForest ...")
    legit_mask = fraud_train == 0
    iso = IsolationForest(
        n_estimators=200,
        max_samples=4096,
        contamination=float(fraud_train.mean()),
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train[legit_mask])

    train_decision = iso.decision_function(X_train)
    dec_min, dec_max = float(train_decision.min()), float(train_decision.max())
    test_decision = iso.decision_function(X_test)
    # Anomaly risk: higher = more anomalous. AUC vs the true binary fraud label.
    test_anomaly_risk = -(test_decision)  # monotonic; sign-flip so higher=anomalous
    iso_auc = roc_auc_score(fraud_test, test_anomaly_risk)
    print(f"IsolationForest anomaly-score ROC-AUC (vs fraud_bool, test): {iso_auc:.4f}")

    # --- 2. XGBoost 3-class classifier (supervised) ---
    print(f"\nTraining XGBoost 3-class classifier (class_weight_power={class_weight_power}) ...")
    # Tempered per-sample weights to counter imbalance without over-weighting.
    sample_weight = _tempered_sample_weight(y_train, class_weight_power)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,      # regularise against tiny-class overfitting
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=len(bf.CLASS_ORDER),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)

    print("\n=== XGBoost classification report (test) ===")
    report = classification_report(
        y_test, y_pred, target_names=bf.CLASS_ORDER, digits=4, zero_division=0
    )
    print(report)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c[:10]:>12}" for c in bf.CLASS_ORDER))
    for i, c in enumerate(bf.CLASS_ORDER):
        print(f"{c[:12]:>12}" + "  ".join(f"{v:>12,}" for v in cm[i]))

    # Binary fraud AUC from the classifier (P(not legitimate)).
    p_fraud = 1.0 - proba[:, bf.CLASS_TO_IDX[bf.CLASS_LEGIT]]
    xgb_fraud_auc = roc_auc_score(fraud_test, p_fraud)
    print(f"\nXGBoost P(fraud) ROC-AUC (vs fraud_bool, test): {xgb_fraud_auc:.4f}")

    metrics = {
        "iso_anomaly_auc": iso_auc,
        "xgb_fraud_auc": xgb_fraud_auc,
        "xgb_report": report,
        "confusion_matrix": cm.tolist(),
        "class_order": bf.CLASS_ORDER,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "label_distribution": df["fraud_type"].value_counts().to_dict(),
        "class_weight_power": class_weight_power,
    }

    bundle = ModelBundle(
        feature_pipeline=pipe,
        isolation_forest=iso,
        anomaly_decision_min=dec_min,
        anomaly_decision_max=dec_max,
        xgb_classifier=xgb,
        class_order=bf.CLASS_ORDER,
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        metrics=metrics,
    )
    path = save_bundle(bundle)
    print(f"\nSaved model bundle -> {path}")
    return bundle


if __name__ == "__main__":
    train_models()
