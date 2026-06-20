"""Download the two Kaggle datasets Aegis depends on.

Datasets (see README / build prompt Data section):
  1. Bank Account Fraud (NeurIPS 2022)  -> data/raw/neurips_baf
     sgpjesus/bank-account-fraud-dataset-neurips-2022
  2. Synthetic Financial Fraud          -> data/raw/synthetic_financial
     umitka/synthetic-financial-fraud-dataset

Auth model
----------
Kaggle CLI auth has moved to OAuth / single-token auth. This script does NOT
look for a kaggle.json credentials file. Instead it verifies that the locally
configured `kaggle` CLI is authenticated (by making a lightweight API call) and
fails with an actionable message pointing at `kaggle auth login` if not.

Terms acceptance
----------------
The NeurIPS BAF dataset requires accepting its terms on the Kaggle website
(the "I Understand and Accept" button on the dataset page) before the API will
serve it, even with valid auth. If a download returns 403 / "you must accept
this dataset's terms", this script surfaces that exact message and the dataset
URL rather than retrying silently.

Usage:
    python scripts/download_data.py            # download both
    python scripts/download_data.py --check    # only verify auth, no download
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve repo root relative to this file so the script works from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

DATASETS = [
    {
        "ref": "sgpjesus/bank-account-fraud-dataset-neurips-2022",
        "dest": RAW_DIR / "neurips_baf",
        "url": "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022",
        "requires_terms": True,
    },
    {
        "ref": "umitka/synthetic-financial-fraud-dataset",
        "dest": RAW_DIR / "synthetic_financial",
        "url": "https://www.kaggle.com/datasets/umitka/synthetic-financial-fraud-dataset",
        "requires_terms": False,
    },
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout+stderr as text. Never raises on non-zero."""
    return subprocess.run(cmd, capture_output=True, text=True)


def check_auth() -> bool:
    """Return True if the kaggle CLI is installed and authenticated.

    We make a cheap authenticated call (`kaggle datasets list`) and inspect the
    result rather than probing for a credentials file. Prints an actionable
    message on failure.
    """
    # 1. Is the CLI installed / importable as an executable?
    probe = _run(["kaggle", "--version"])
    if probe.returncode != 0 and not probe.stdout:
        print(
            "ERROR: the `kaggle` CLI is not available on PATH.\n"
            "       Install it with `pip install -r requirements.txt` and ensure\n"
            "       your Python Scripts directory is on PATH.",
            file=sys.stderr,
        )
        return False

    # 2. Is it authenticated? A lightweight authed call is the reliable probe.
    # NOTE: avoid version-specific flags (e.g. --page-size); a bare search-list
    # call is supported across CLI versions and still requires valid auth.
    result = _run(["kaggle", "datasets", "list", "-s", "fraud"])
    combined = (result.stdout or "") + (result.stderr or "")
    authed = result.returncode == 0 or "ref" in combined.lower()

    unauth_markers = ("401", "unauthorized", "credentials", "authenticate", "sign in")
    if not authed or any(m in combined.lower() for m in unauth_markers):
        print(
            "ERROR: the Kaggle CLI is not authenticated.\n"
            "       Run `kaggle auth login` (OAuth browser flow) or set the\n"
            "       KAGGLE_API_TOKEN environment variable for this shell, then retry.\n"
            f"       (Kaggle said: {combined.strip()[:300]})",
            file=sys.stderr,
        )
        return False

    print("[ok] Kaggle CLI is authenticated.")
    return True


def download_one(ds: dict) -> bool:
    """Download and unzip one dataset. Returns True on success."""
    ref, dest, url = ds["ref"], ds["dest"], ds["url"]
    dest.mkdir(parents=True, exist_ok=True)

    # Skip if it already looks populated (any CSV present).
    if any(dest.glob("*.csv")) or any(dest.glob("**/*.csv")):
        print(f"[skip] {ref} already present in {dest.relative_to(REPO_ROOT)}")
        return True

    print(f"[download] {ref} -> {dest.relative_to(REPO_ROOT)}")
    result = _run(
        ["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip"]
    )
    combined = (result.stdout or "") + (result.stderr or "")

    # Authoritative success check: did CSV(s) actually land on disk? The Kaggle
    # CLI can print progress/warnings to stderr and return a non-zero code even
    # on a successful --unzip, so we trust the filesystem over the exit code.
    landed = any(dest.glob("*.csv")) or any(dest.glob("**/*.csv"))
    if landed:
        print(f"[ok] {ref} downloaded and unzipped.")
        return True

    # No CSV landed -> report the most likely cause (terms not accepted is the
    # usual culprit for the NeurIPS BAF dataset).
    lowered = combined.lower()
    if "403" in combined or "accept" in lowered or "terms" in lowered or ds["requires_terms"]:
        print(
            f"\nERROR: Kaggle refused the download of `{ref}` (no data landed).\n"
            "       This dataset likely requires accepting its terms on the\n"
            "       website first. Open this page, click\n"
            '       "I Understand and Accept", then re-run this script:\n'
            f"           {url}\n"
            f"       (Kaggle said: {combined.strip()[:400]})",
            file=sys.stderr,
        )
    else:
        print(
            f"\nERROR: failed to download `{ref}`.\n"
            f"       (Kaggle said: {combined.strip()[:400]})",
            file=sys.stderr,
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Aegis Kaggle datasets.")
    parser.add_argument(
        "--check", action="store_true", help="Only verify Kaggle auth; do not download."
    )
    args = parser.parse_args()

    if not check_auth():
        return 1
    if args.check:
        return 0

    ok = True
    for ds in DATASETS:
        ok = download_one(ds) and ok

    if not ok:
        print("\nOne or more datasets failed to download. See messages above.", file=sys.stderr)
        return 1

    print("\nAll datasets ready under data/raw/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
