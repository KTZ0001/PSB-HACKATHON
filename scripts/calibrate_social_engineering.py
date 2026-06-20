"""Calibrate the social-engineering risk-policy engine on Dataset 2.

Dataset 2 (synthetic financial fraud) has the closest analogues to the
transaction-context signals the SE engine uses. This script prints, for each
*derivable* signal, the distribution among fraud vs non-fraud rows so the
engine's thresholds/weights are justified by data rather than guessed.

HONEST SCOPE: Dataset 2 has columns
    [transaction_id, user_id, amount, transaction_type, merchant_category,
     country, hour, device_risk_score, ip_risk_score, is_fraud]
so it can calibrate:
    - amount_vs_user_baseline_ratio  (amount vs the user's own mean)
    - night_time_transaction         (from `hour`)
    - is_new_high_risk_country_or_geo (from `country` + `ip_risk_score`)
    - txn_velocity_short_window      (weak proxy: per-user transaction count;
                                      no timestamp exists, only `hour`)
It CANNOT calibrate (no analogous columns -> documented as expert-set priors):
    - first_transfer_to_payee, payee_age_hours, failed_attempts_before_success

Usage:  python scripts/calibrate_social_engineering.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config

NIGHT_HOURS = set(range(0, 6))  # 00:00-05:59


def _fraud_rate_by_bucket(df, col, bins, labels=None):
    cats = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby(cats, observed=False)["is_fraud"].agg(["mean", "count"])
    return g


def main() -> int:
    path = next((config.SYNTHETIC_FIN_DIR).glob("**/*.csv"))
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df):,} rows, fraud rate {df.is_fraud.mean():.4%}\n")

    # --- amount vs user baseline ---
    user_mean = df.groupby("user_id")["amount"].transform("mean")
    df["amount_ratio"] = df["amount"] / user_mean.replace(0, np.nan)
    print("=== amount_vs_user_baseline_ratio ===")
    print(df.groupby("is_fraud")["amount_ratio"].describe()[["mean", "50%", "75%", "max"]])
    print("\nfraud rate by amount_ratio bucket:")
    print(_fraud_rate_by_bucket(df, "amount_ratio", [0, 1, 1.5, 2, 3, 5, np.inf]))

    # --- night-time ---
    df["is_night"] = df["hour"].isin(NIGHT_HOURS)
    print("\n=== night_time_transaction (hour in 0..5) ===")
    print(df.groupby("is_night")["is_fraud"].agg(["mean", "count"]))

    # --- geo / ip risk ---
    print("\n=== ip_risk_score ===")
    print(df.groupby("is_fraud")["ip_risk_score"].describe()[["mean", "50%", "75%"]])
    print("fraud rate by ip_risk_score bucket:")
    print(_fraud_rate_by_bucket(df, "ip_risk_score", [0, 0.3, 0.5, 0.7, 0.85, 1.0]))

    print("\n=== device_risk_score ===")
    print(df.groupby("is_fraud")["device_risk_score"].describe()[["mean", "50%", "75%"]])

    print("\n=== fraud rate by country (top 12 by volume) ===")
    cc = df.groupby("country")["is_fraud"].agg(["mean", "count"]).sort_values("count", ascending=False)
    print(cc.head(12))

    # --- velocity proxy (per-user count) ---
    df["user_txn_count"] = df.groupby("user_id")["transaction_id"].transform("count")
    print("\n=== txn_velocity proxy: per-user transaction count ===")
    print(_fraud_rate_by_bucket(df, "user_txn_count", [0, 1, 2, 3, 5, np.inf]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
