"""Phase 1 model tests.

These train a small, fast model on a sample of the real BAF data (if present)
and assert behavioural ordering: fraud rows should, on average, receive a higher
anomaly risk and a higher P(fraud) than legitimate rows. This is a real signal
assertion, not a "does it run" smoke test.

If the BAF dataset hasn't been downloaded, the tests skip with a clear message
rather than failing.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from src import config
from src.features import baf_features as bf
from src.models.registry import ModelBundle
from src.models.inference import RiskEngine

BASE_CSV = config.NEURIPS_BAF_DIR / config.NEURIPS_BAF_BASE_FILE


@pytest.fixture(scope="module")
def small_bundle():
    if not BASE_CSV.exists():
        pytest.skip(f"BAF data not found at {BASE_CSV}; run scripts/download_data.py")

    # Stratified-ish sample: keep all fraud rows from a chunk + a legit sample,
    # to get a trainable, balanced-enough mini set quickly.
    df = pd.read_csv(BASE_CSV, nrows=120_000)
    df = df.drop(columns=[c for c in bf.CONSTANT_COLS if c in df.columns])
    df["fraud_type"] = bf.derive_fraud_type_label(df)

    train = df.sample(frac=0.7, random_state=0)
    test = df.drop(train.index)

    pipe = bf.FeaturePipeline().fit(train)
    X_train, X_test = pipe.transform(train), pipe.transform(test)
    y_train = train["fraud_type"].map(bf.CLASS_TO_IDX).to_numpy()

    iso = IsolationForest(n_estimators=60, max_samples=2048, random_state=0, n_jobs=-1)
    iso.fit(X_train[train[bf.TARGET_COL].to_numpy() == 0])
    dec = iso.decision_function(X_train)

    counts = np.bincount(y_train, minlength=len(bf.CLASS_ORDER))
    w = len(y_train) / (len(bf.CLASS_ORDER) * np.maximum(counts, 1))
    xgb = XGBClassifier(
        n_estimators=80, max_depth=5, learning_rate=0.2, tree_method="hist",
        objective="multi:softprob", num_class=len(bf.CLASS_ORDER),
        eval_metric="mlogloss", random_state=0, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, sample_weight=w[y_train])

    bundle = ModelBundle(
        feature_pipeline=pipe, isolation_forest=iso,
        anomaly_decision_min=float(dec.min()), anomaly_decision_max=float(dec.max()),
        xgb_classifier=xgb, class_order=bf.CLASS_ORDER,
    )
    return bundle, test


def test_fraud_rows_score_higher_than_legit(small_bundle):
    bundle, test = small_bundle
    engine = RiskEngine(bundle=bundle)

    legit = test[test[bf.TARGET_COL] == 0].head(400)
    fraud = test[test[bf.TARGET_COL] == 1].head(400)
    if len(fraud) < 20:
        pytest.skip("not enough fraud rows in sample")

    legit_res = engine.score_frame(legit)
    fraud_res = engine.score_frame(fraud)

    mean_legit_fraud_proba = np.mean([r.xgb_fraud_proba for r in legit_res])
    mean_fraud_fraud_proba = np.mean([r.xgb_fraud_proba for r in fraud_res])
    assert mean_fraud_fraud_proba > mean_legit_fraud_proba

    mean_legit_anom = np.mean([r.anomaly_risk for r in legit_res])
    mean_fraud_anom = np.mean([r.anomaly_risk for r in fraud_res])
    assert mean_fraud_anom > mean_legit_anom


def test_anomaly_risk_is_bounded(small_bundle):
    bundle, test = small_bundle
    engine = RiskEngine(bundle=bundle)
    res = engine.score_frame(test.head(50))
    for r in res:
        assert 0.0 <= r.anomaly_risk <= 100.0
        assert abs(sum(r.class_proba.values()) - 1.0) < 1e-5
        assert r.predicted_type in bf.CLASS_ORDER
