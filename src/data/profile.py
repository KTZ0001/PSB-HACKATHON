"""Data profiling for the two Aegis datasets.

Produces, for each dataset, the summary required at the Phase 0 stop-point:
  - shape (rows x cols)
  - column list with dtypes and null rates
  - class balance on the fraud label (auto-detected)

This module is intentionally dependency-light (pandas only) and is runnable as
a script:

    python -m src.data.profile

It also tries to answer one specific question for the NeurIPS BAF dataset:
does it contain any social-engineering / victim-authorized signal? It scans
column names for tell-tale tokens and reports findings honestly rather than
assuming.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src import config

# Tokens that would hint at a social-engineering / victim-authorized signal.
SOCIAL_ENG_TOKENS = (
    "social",
    "scam",
    "victim",
    "authorized",
    "authorised",
    "self_initiated",
    "self-initiated",
    "payee",
    "beneficiary",
    "transfer_to",
    "coached",
    "app_fraud",  # "authorised push payment" fraud
)

# Candidate fraud-label column names across the two datasets.
FRAUD_LABEL_CANDIDATES = (
    "fraud_bool",
    "isfraud",
    "is_fraud",
    "fraud",
    "label",
    "class",
    "target",
)


def _find_csv(directory: Path, preferred: str | None = None) -> Path | None:
    """Find a CSV in a directory, preferring an exact filename if given."""
    if preferred:
        exact = directory / preferred
        if exact.exists():
            return exact
    candidates = sorted(directory.glob("**/*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def _detect_fraud_label(df: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in FRAUD_LABEL_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None


def profile_csv(path: Path, name: str, label_hint: str | None = None) -> dict:
    """Load a CSV and print a profiling summary. Returns a small dict of facts."""
    print("=" * 78)
    print(f"DATASET: {name}")
    print(f"FILE:    {path}")
    print("=" * 78)

    # Read a capped sample first for very large files to keep profiling fast,
    # then read full for accurate null/class stats if it's reasonably sized.
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    print(f"\nShape: {n_rows:,} rows x {n_cols} columns\n")

    # Column / dtype / null-rate table.
    null_rates = df.isna().mean()
    print(f"{'column':<32}{'dtype':<12}{'null_rate':>10}")
    print("-" * 54)
    for col in df.columns:
        print(f"{col[:31]:<32}{str(df[col].dtype):<12}{null_rates[col]:>10.4f}")

    # Class balance.
    label = label_hint if (label_hint and label_hint in df.columns) else _detect_fraud_label(df)
    facts = {"name": name, "rows": n_rows, "cols": n_cols, "label": label}
    if label is not None:
        counts = df[label].value_counts(dropna=False)
        total = counts.sum()
        print(f"\nClass balance on fraud label '{label}':")
        for value, cnt in counts.items():
            print(f"  {str(value):<10} {cnt:>12,}  ({cnt / total:.4%})")
        # Positive (fraud) rate if binary-ish.
        if df[label].nunique(dropna=True) <= 2:
            try:
                facts["fraud_rate"] = float(pd.to_numeric(df[label], errors="coerce").mean())
            except Exception:
                facts["fraud_rate"] = None
    else:
        print("\n[!] No fraud-label column auto-detected.")

    # Social-engineering signal scan.
    hits = [
        c for c in df.columns
        if any(tok in c.lower() for tok in SOCIAL_ENG_TOKENS)
    ]
    facts["social_eng_columns"] = hits
    print("\nSocial-engineering / victim-authorized signal scan:")
    if hits:
        print(f"  Possible related columns found: {hits}")
        print("  -> Inspect these manually; presence of a name token is not proof of a label.")
    else:
        print("  No columns matching social-engineering tokens "
              f"({', '.join(SOCIAL_ENG_TOKENS[:6])}, ...).")
        print("  -> Consistent with this dataset having NO social-engineering ground truth.")

    print()
    return facts


def main() -> int:
    targets = [
        (
            config.NEURIPS_BAF_DIR,
            config.NEURIPS_BAF_BASE_FILE,
            "NeurIPS Bank Account Fraud (Base variant)",
            "fraud_bool",
        ),
        (
            config.SYNTHETIC_FIN_DIR,
            None,
            "Synthetic Financial Fraud",
            None,
        ),
    ]

    any_missing = False
    summaries = []
    for directory, preferred, name, label_hint in targets:
        csv = _find_csv(directory, preferred)
        if csv is None:
            any_missing = True
            print(f"[missing] No CSV found under {directory}. "
                  "Run `python scripts/download_data.py` first.\n")
            continue
        summaries.append(profile_csv(csv, name, label_hint))

    if summaries:
        print("=" * 78)
        print("ONE-LINE SUMMARIES")
        print("=" * 78)
        for s in summaries:
            fr = s.get("fraud_rate")
            fr_str = f", fraud_rate={fr:.4%}" if isinstance(fr, float) else ""
            print(f"- {s['name']}: {s['rows']:,}x{s['cols']}, "
                  f"label={s['label']}{fr_str}, "
                  f"social_eng_cols={s['social_eng_columns'] or 'none'}")

    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
