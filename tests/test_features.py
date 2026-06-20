"""Phase 1 feature-layer tests: label derivation, pipeline, device graph."""
import pandas as pd
import pytest

from src.features import baf_features as bf
from src.features.device_graph import DeviceAccountGraph


def _mini_baf_frame() -> pd.DataFrame:
    """A tiny frame with all columns the pipeline touches."""
    base = {col: 0.0 for col in bf.NUMERIC_COLS}
    base.update({c: "AA" for c in bf.CATEGORICAL_COLS})
    base["device_os"] = "windows"
    base["source"] = "INTERNET"
    rows = []
    for fb, dde in [(0, 1), (1, 1), (1, 3), (0, 5)]:
        r = dict(base)
        r[bf.TARGET_COL] = fb
        r["device_distinct_emails_8w"] = dde
        r[bf.SPLIT_COL] = 0
        rows.append(r)
    return pd.DataFrame(rows)


def test_label_derivation_maps_three_classes():
    df = _mini_baf_frame()
    labels = bf.derive_fraud_type_label(df).tolist()
    # (fraud=0,dde=1)->legit, (fraud=1,dde=1)->ATO,
    # (fraud=1,dde=3)->mule, (fraud=0,dde=5)->legit (not fraud, so still legit)
    assert labels == [bf.CLASS_LEGIT, bf.CLASS_ATO, bf.CLASS_MULE, bf.CLASS_LEGIT]


def test_pipeline_is_deterministic_and_column_stable():
    df = _mini_baf_frame()
    pipe = bf.FeaturePipeline().fit(df)
    a = pipe.transform(df)
    b = pipe.transform(df)
    assert list(a.columns) == pipe.feature_names
    assert a.equals(b)
    # missing-indicator columns exist for sentinel cols
    for col in bf.SENTINEL_MISSING_COLS:
        assert f"{col}__is_missing" in a.columns


def test_pipeline_handles_unseen_categorical_level():
    df = _mini_baf_frame()
    pipe = bf.FeaturePipeline().fit(df)
    novel = df.head(1).copy()
    novel["device_os"] = "haiku_os_9000"  # never seen in fit
    out = pipe.transform(novel)
    # Column set unchanged; all device_os one-hots are 0 for the unknown level.
    assert list(out.columns) == pipe.feature_names
    dev_cols = [c for c in out.columns if c.startswith("device_os=")]
    assert out[dev_cols].sum(axis=1).iloc[0] == 0


def test_sentinel_missing_flag_fires():
    df = _mini_baf_frame()
    df.loc[0, "bank_months_count"] = -1
    pipe = bf.FeaturePipeline().fit(df)
    out = pipe.transform(df)
    assert out.loc[0, "bank_months_count__is_missing"] == 1
    assert out.loc[1, "bank_months_count__is_missing"] == 0


def test_device_graph_risk_flag_and_fanout():
    g = DeviceAccountGraph(risk_threshold=3)
    # One device shared across 3 accounts -> risky.
    for acct in ["u1", "u2", "u3"]:
        g.add_event(acct, "shared_device")
    info = g.get_device_risk("shared_device")
    assert info["account_count"] == 3
    assert info["risk_flag"] is True
    assert sorted(info["linked_accounts"]) == ["u1", "u2", "u3"]

    # A normal single-account device -> not risky, score 0.
    g.add_event("u9", "personal_device")
    normal = g.get_device_risk("personal_device")
    assert normal["account_count"] == 1
    assert normal["risk_flag"] is False
    assert g.device_risk_score("personal_device") == 0.0
    assert g.device_risk_score("shared_device") > 0.0


def test_device_graph_unknown_device_is_safe_default():
    g = DeviceAccountGraph()
    info = g.get_device_risk("never_seen")
    assert info == {
        "device_id": "never_seen",
        "account_count": 0,
        "risk_flag": False,
        "linked_accounts": [],
    }


def test_device_graph_persistence_roundtrip(tmp_path):
    g = DeviceAccountGraph(risk_threshold=2)
    g.add_event("a", "d1")
    g.add_event("b", "d1")
    p = tmp_path / "graph.json"
    g.save(p)
    g2 = DeviceAccountGraph.load(p)
    assert g2.get_device_risk("d1")["account_count"] == 2
    assert g2.risk_threshold == 2
