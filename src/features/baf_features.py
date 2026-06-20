"""Feature engineering for the NeurIPS Bank Account Fraud (BAF) dataset.

This module is the single source of truth for how a BAF-style record becomes a
numeric feature vector. The same `FeaturePipeline` is used at training time and
at API inference time, so the model never sees a differently-shaped vector than
it was trained on.

Every feature below is derivable from data a bank already logs at account
opening / session time. Column meanings are documented in FEATURE_DOCS and come
from the dataset's published documentation (Jesus et al., NeurIPS 2022,
"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML
Evaluation"). We do not invent meanings.

Derived label note
------------------
BAF ships a single binary target `fraud_bool`. Aegis needs a 3-class taxonomy
(legitimate / account_takeover / mule_network). The `mule_network` class is a
DERIVED HEURISTIC label, not ground truth — see `derive_fraud_type_label`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Column groups -----------------------------------------------------------

TARGET_COL = "fraud_bool"
SPLIT_COL = "month"  # BAF's intended temporal train/test split key (not a feature)

# Dropped: constant in the Base variant (all zeros) -> zero information.
CONSTANT_COLS = ["device_fraud_count"]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# Columns where -1 is a documented "missing/unknown" sentinel, not a real value.
SENTINEL_MISSING_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
]

# Numeric features = everything except target, split key, constants, categoricals.
NUMERIC_COLS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
]

# Human-readable meaning of each signal (used in README + explanations).
FEATURE_DOCS = {
    "income": "Applicant annual income (quantile-normalised in [0,1]).",
    "name_email_similarity": "String similarity between full name and email (low = mismatch, a known fraud signal).",
    "prev_address_months_count": "Months at previous registered address (-1 = unknown).",
    "current_address_months_count": "Months at current registered address (-1 = unknown).",
    "customer_age": "Applicant age in years (rounded to decade by the dataset).",
    "days_since_request": "Days elapsed since the application/request was made.",
    "intended_balcon_amount": "Initial transferred/balance-consolidation amount (can be negative).",
    "zip_count_4w": "Count of applications from the same zip in the last 4 weeks (velocity).",
    "velocity_6h": "Average application velocity over the last 6 hours.",
    "velocity_24h": "Average application velocity over the last 24 hours.",
    "velocity_4w": "Average application velocity over the last 4 weeks.",
    "bank_branch_count_8w": "Applications at the same bank branch in the last 8 weeks.",
    "date_of_birth_distinct_emails_4w": "Distinct emails sharing this DOB in the last 4 weeks (identity-sharing signal).",
    "credit_risk_score": "Internal credit risk score for the application.",
    "email_is_free": "1 if the email is from a free provider.",
    "phone_home_valid": "1 if the supplied home phone is valid.",
    "phone_mobile_valid": "1 if the supplied mobile phone is valid.",
    "bank_months_count": "Age of the applicant's existing bank account in months (-1 = unknown).",
    "has_other_cards": "1 if the applicant holds other cards with the bank.",
    "proposed_credit_limit": "Requested credit limit.",
    "foreign_request": "1 if the request originated from a country other than the bank's.",
    "session_length_in_minutes": "Length of the online application session.",
    "keep_alive_session": "1 if the user kept the session alive.",
    "device_distinct_emails_8w": "Distinct emails seen from this device in 8 weeks (>=2 = shared-device / mule signal).",
}

# Predicted-type taxonomy shared across the whole engine.
CLASS_LEGIT = "legitimate"
CLASS_ATO = "account_takeover"
CLASS_MULE = "mule_network"
CLASS_ORDER = [CLASS_LEGIT, CLASS_ATO, CLASS_MULE]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


# --- Label derivation --------------------------------------------------------

# Threshold for the derived mule heuristic: a device linked to >= this many
# distinct emails in 8 weeks looks like a shared cash-out device.
MULE_DISTINCT_EMAILS_THRESHOLD = 2


def derive_fraud_type_label(df: pd.DataFrame) -> pd.Series:
    """Derive the 3-class fraud-type label from BAF's binary `fraud_bool`.

    Rules (DERIVED HEURISTIC — documented as such, NOT dataset ground truth):
      * fraud_bool == 0                      -> legitimate
      * fraud_bool == 1 and device shared    -> mule_network
      * fraud_bool == 1 and not shared       -> account_takeover

    "device shared" = device_distinct_emails_8w >= MULE_DISTINCT_EMAILS_THRESHOLD,
    i.e. the device opening/using this account has been seen with multiple
    distinct email identities recently — the signature of a mule cash-out node
    rather than a single hijacked owner. `device_fraud_count` would be the ideal
    signal but is constant (0) in the Base variant, so it is unusable here.
    """
    is_fraud = df[TARGET_COL] == 1
    device_shared = df["device_distinct_emails_8w"] >= MULE_DISTINCT_EMAILS_THRESHOLD

    label = pd.Series(CLASS_LEGIT, index=df.index, dtype=object)
    label[is_fraud & ~device_shared] = CLASS_ATO
    label[is_fraud & device_shared] = CLASS_MULE
    return label


# --- Feature pipeline --------------------------------------------------------

@dataclass
class FeaturePipeline:
    """Deterministic BAF record -> numeric feature matrix transform.

    Fit once on training data (to freeze the one-hot column order), then reused
    unchanged at inference time. Persist/restore with joblib.
    """

    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    _fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        self.categorical_levels = {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in CATEGORICAL_COLS
        }
        # Build feature_names by transforming a one-row frame.
        sample = self._transform_frame(df.head(1))
        self.feature_names = list(sample.columns)
        self._fitted = True
        return self

    def _transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # Numeric features (kept as-is; tree models + IsolationForest handle the
        # -1 sentinels fine, and we additionally expose explicit missing flags).
        for col in NUMERIC_COLS:
            out[col] = pd.to_numeric(df[col], errors="coerce")

        # Explicit missing-indicator flags for sentinel columns.
        for col in SENTINEL_MISSING_COLS:
            out[f"{col}__is_missing"] = (pd.to_numeric(df[col], errors="coerce") == -1).astype(int)

        # One-hot categoricals against frozen levels (unknown level -> all zeros).
        for col in CATEGORICAL_COLS:
            levels = self.categorical_levels.get(col) or sorted(df[col].astype(str).unique().tolist())
            values = df[col].astype(str)
            for lvl in levels:
                out[f"{col}={lvl}"] = (values == lvl).astype(int)

        return out.fillna(0.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit().")
        out = self._transform_frame(df)
        # Guarantee exact training column order (add missing, drop extras).
        return out.reindex(columns=self.feature_names, fill_value=0.0)

    def transform_record(self, record: dict) -> pd.DataFrame:
        """Transform a single record dict (API inference path)."""
        return self.transform(pd.DataFrame([record]))


def temporal_split(df: pd.DataFrame, train_max_month: int = 5):
    """Split BAF by `month` as the dataset authors intend (no leakage).

    Train on months 0..train_max_month, test on the later months. This mirrors
    the dataset's deployment-realistic temporal evaluation.
    """
    train = df[df[SPLIT_COL] <= train_max_month].copy()
    test = df[df[SPLIT_COL] > train_max_month].copy()
    return train, test
