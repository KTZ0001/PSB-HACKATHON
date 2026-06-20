"""Phase 2 stop-point artefact: 5 social-engineering example scores.

Includes the headline case the whole thesis rests on: a transaction that looks
COMPLETELY NORMAL on device/geo (no high-risk country, not night-time, no
velocity spike) but is still flagged because the money-movement shape is a
classic coached-victim / APP-scam pattern.

Usage:  python scripts/se_examples.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features.user_profile import UserProfileStore
from src.scoring.social_engineering import RuleWeightedSocialEngineeringEngine

engine = RuleWeightedSocialEngineeringEngine()

EXAMPLES = [
    (
        "1. Normal: known payee, amount ~ baseline, daytime",
        {
            "first_transfer_to_payee": False,
            "payee_age_hours": 4200.0,
            "amount_vs_user_baseline_ratio": 1.05,
            "txn_velocity_short_window": 1,
            "failed_attempts_before_success": 0,
            "is_new_high_risk_country_or_geo": False,
            "night_time_transaction": False,
        },
    ),
    (
        "2. *** SE HEADLINE *** normal device/geo, but coached-victim money shape",
        {
            "first_transfer_to_payee": True,        # brand-new payee
            "payee_age_hours": 0.2,                  # added ~12 min ago
            "amount_vs_user_baseline_ratio": 5.2,    # far above this user's norm
            "txn_velocity_short_window": 1,
            "failed_attempts_before_success": 2,     # fumbled auth (being coached)
            "is_new_high_risk_country_or_geo": False,  # <-- geo looks totally normal
            "night_time_transaction": False,           # <-- not night either
        },
    ),
    (
        "3. Mildly elevated: newish payee (2 days), 2x amount",
        {
            "first_transfer_to_payee": False,
            "payee_age_hours": 48.0,
            "amount_vs_user_baseline_ratio": 2.0,
            "txn_velocity_short_window": 1,
            "failed_attempts_before_success": 0,
            "is_new_high_risk_country_or_geo": False,
            "night_time_transaction": False,
        },
    ),
    (
        "4. ATO-overlap: high-risk geo + night + high amount (also trips device risk)",
        {
            "first_transfer_to_payee": True,
            "payee_age_hours": 1.0,
            "amount_vs_user_baseline_ratio": 4.0,
            "txn_velocity_short_window": 1,
            "failed_attempts_before_success": 0,
            "is_new_high_risk_country_or_geo": True,
            "night_time_transaction": True,
        },
    ),
    (
        "5. Coached urgency: rapid transfers to a new payee",
        {
            "first_transfer_to_payee": True,
            "payee_age_hours": 0.5,
            "amount_vs_user_baseline_ratio": 3.1,
            "txn_velocity_short_window": 4,
            "failed_attempts_before_success": 1,
            "is_new_high_risk_country_or_geo": False,
            "night_time_transaction": False,
        },
    ),
]


def show(title: str, signals: dict) -> None:
    res = engine.score(signals)
    print(f"\n{title}")
    print(f"  social_engineering_score : {res.social_engineering_score:6.2f}  "
          f"is_likely_social_engineering = {res.is_likely_social_engineering}")
    fired = {k: v for k, v in res.contributions.items() if v > 0}
    fired = dict(sorted(fired.items(), key=lambda kv: kv[1], reverse=True))
    print("  contributions (points)   : " + (
        "  ".join(f"{k}={v}" for k, v in fired.items()) or "(none)"))


def main() -> int:
    print("=" * 80)
    print("SOCIAL-ENGINEERING RISK-POLICY ENGINE — example scores")
    print("=" * 80)
    for title, signals in EXAMPLES:
        show(title, signals)

    # Demonstrate the same headline case emerging from raw transaction events via
    # the UserProfileStore (signals COMPUTED, not hand-fed).
    print("\n" + "=" * 80)
    print("Headline case derived from raw events via UserProfileStore")
    print("=" * 80)
    store = UserProfileStore()
    store.seed_baseline("user_42", [100, 120, 95, 110, 105])  # normal spending ~£106
    # genuine owner, normal device, transfers £560 to a payee added minutes ago,
    # after 2 failed auth attempts; daytime, domestic.
    signals = store.compute_signals(
        user_id="user_42", amount=560.0, payee_id="payee_NEW",
        timestamp=1000.0, country="UK", hour=14, failed_attempts_before_success=2,
    )
    print(f"  computed signals: {signals}")
    show("  scored:", signals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
