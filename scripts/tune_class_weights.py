"""Sweep class_weight_power to pick a precision/recall balance for the XGBoost
type classifier.

Trains a fast XGBoost (fewer trees) per power on the real temporal-train split
(all fraud kept, legit downsampled for speed) and evaluates on the FULL temporal
test split so per-class metrics are honest. Used to justify
DEFAULT_CLASS_WEIGHT_POWER in src/models/train.py.

Usage:  python scripts/tune_class_weights.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config
from src.features import baf_features as bf

df = pd.read_csv(config.NEURIPS_BAF_DIR / config.NEURIPS_BAF_BASE_FILE)
df = df.drop(columns=[c for c in bf.CONSTANT_COLS if c in df.columns])
df["fraud_type"] = bf.derive_fraud_type_label(df)
train, test = bf.temporal_split(df)

# Use the FULL temporal-train split (no legit downsampling) so per-class recall
# numbers reflect the real deployed class balance. (An earlier version
# downsampled legit for speed, which over-stated minority recall.)
pipe = bf.FeaturePipeline().fit(train)
X_tr, X_te = pipe.transform(train), pipe.transform(test)
y_tr = train["fraud_type"].map(bf.CLASS_TO_IDX).to_numpy()
y_te = test["fraud_type"].map(bf.CLASS_TO_IDX).to_numpy()
fraud_te = test.fraud_bool.to_numpy()

counts = np.bincount(y_tr, minlength=3)
base_w = len(y_tr) / (3 * np.maximum(counts, 1))
ato_i, mule_i = bf.CLASS_TO_IDX[bf.CLASS_ATO], bf.CLASS_TO_IDX[bf.CLASS_MULE]

print(f"{'power':>6} | {'ATO P':>7} {'ATO R':>7} | {'mule P':>7} {'mule R':>7} | "
      f"{'macroF1':>8} | {'fraudAUC':>8} | legit_FP_as_ATO")
print("-" * 78)
for power in [0.3, 0.45, 0.6, 0.75]:
    w = (base_w ** power)[y_tr]
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
        objective="multi:softprob", num_class=3, tree_method="hist",
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr, sample_weight=w)
    proba = clf.predict_proba(X_te)
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=[0, 1, 2], zero_division=0)
    macro_f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    auc = roc_auc_score(fraud_te, 1 - proba[:, bf.CLASS_TO_IDX[bf.CLASS_LEGIT]])
    legit_as_ato = int(((y_te == 0) & (pred == ato_i)).sum())
    print(f"{power:>6.1f} | {p[ato_i]:>7.3f} {r[ato_i]:>7.3f} | {p[mule_i]:>7.3f} {r[mule_i]:>7.3f} | "
          f"{macro_f1:>8.3f} | {auc:>8.3f} | {legit_as_ato:>12,}")
