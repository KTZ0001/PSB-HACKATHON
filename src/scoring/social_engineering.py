"""Social-engineering risk-policy engine (Phase 2).

This is the heart of the Aegis thesis: catching fraud the genuine, authenticated
account owner is *coached* into authorising (scam/APP fraud). Device, geo and IP
all look normal -- so device/behavioural risk scoring has no signal. What gives
it away is the *shape of the money movement*: a large, first-ever transfer to a
freshly-added payee, often after a few fumbled auth attempts (the victim being
talked through it on the phone).

It is a RULE-WEIGHTED policy engine, NOT a supervised ML classifier, because
neither available dataset has social-engineering ground-truth labels. Each
signal produces an activation in [0,1]; the score is a weighted sum scaled to
[0,100]. Every weight is documented as either:
  * DATA-CALIBRATED on Dataset 2 (scripts/calibrate_social_engineering.py), or
  * an EXPERT-SET PRIOR (no dataset analogue; to be re-fit on labelled cases
    post-deployment).

Transition-to-ML design: the engine consumes a canonical signal dict (see
SE_SIGNAL_NAMES) and exposes `signal_vector()`. A future supervised model
implements the same `SocialEngineeringScorer` interface over the same vector, so
swapping the policy engine for an ML model is a contained change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np

# Canonical signal order — shared by the policy engine and any future ML model.
SE_SIGNAL_NAMES = [
    "first_transfer_to_payee",        # bool: never transferred to this payee before
    "payee_age_hours",                # float: hours since payee was added
    "amount_vs_user_baseline_ratio",  # float: txn amount / user's rolling avg
    "txn_velocity_short_window",      # int: transfers in the last short window
    "failed_attempts_before_success", # int: failed auth attempts before this txn
    "is_new_high_risk_country_or_geo",# bool: new/high-risk country or geo
    "night_time_transaction",         # bool: local-time night transaction
]


@dataclass
class SocialEngineeringConfig:
    """Configurable weights (max points) and thresholds for each signal.

    Defaults reflect the Phase 2 calibration. `source` of each weight:
      amount_ratio_*            -> DATA-CALIBRATED (Dataset 2: ratio>=3 ~near-certain)
      high_risk_geo_weight      -> DATA-CALIBRATED (Dataset 2: NG / ip_risk>0.7 = 100%)
      night_weight              -> calibrated DIRECTION, TEMPERED down (synthetic
                                   data shows 100% but real night txns are mostly legit)
      first_transfer_to_payee_* -> EXPERT PRIOR (classic APP-scam signal; no analogue)
      payee_age_*               -> EXPERT PRIOR
      failed_attempts_*         -> EXPERT PRIOR
      velocity_*                -> EXPERT PRIOR (no analogue; Dataset 2 velocity flat)
    """

    # amount ratio: linear ramp between these two ratios -> activation 0..1
    amount_ratio_low: float = 1.5
    amount_ratio_high: float = 3.0
    amount_ratio_weight: float = 28.0

    # first transfer to this payee (boolean)
    first_transfer_weight: float = 22.0

    # payee freshness: newer than payee_age_full_hours -> activation 1.0,
    # older than payee_age_zero_hours -> 0.0
    payee_age_full_hours: float = 0.0
    payee_age_zero_hours: float = 72.0
    payee_age_weight: float = 16.0

    # failed auth attempts: saturates at failed_attempts_cap
    failed_attempts_cap: int = 3
    failed_attempts_weight: float = 12.0

    # high-risk / new geo (boolean)
    high_risk_geo_weight: float = 14.0

    # night-time (boolean) -- intentionally low
    night_weight: float = 6.0

    # velocity: 1 txn -> 0, saturates at velocity_cap
    velocity_cap: int = 4
    velocity_weight: float = 10.0

    # score at/above which is_likely_social_engineering = True
    flag_threshold: float = 50.0

    @classmethod
    def from_dict(cls, d: dict) -> "SocialEngineeringConfig":
        base = cls()
        return replace(base, **{k: v for k, v in d.items() if hasattr(base, k)})


@dataclass
class SocialEngineeringResult:
    social_engineering_score: float          # 0-100
    is_likely_social_engineering: bool
    contributions: dict                      # signal -> points contributed
    signals: dict                            # the raw signal dict scored


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def signal_vector(signals: dict) -> np.ndarray:
    """Canonical numeric vector for the signal dict (future-ML feature parity)."""
    return np.array(
        [
            float(bool(signals.get("first_transfer_to_payee", False))),
            float(signals.get("payee_age_hours", 1e9) or 0.0),
            float(signals.get("amount_vs_user_baseline_ratio", 1.0) or 0.0),
            float(signals.get("txn_velocity_short_window", 1) or 0),
            float(signals.get("failed_attempts_before_success", 0) or 0),
            float(bool(signals.get("is_new_high_risk_country_or_geo", False))),
            float(bool(signals.get("night_time_transaction", False))),
        ],
        dtype=float,
    )


class SocialEngineeringScorer(ABC):
    """Interface shared by the rule engine and any future supervised model."""

    @abstractmethod
    def score(self, signals: dict) -> SocialEngineeringResult: ...


class RuleWeightedSocialEngineeringEngine(SocialEngineeringScorer):
    """Default, deterministic, inspectable policy engine."""

    def __init__(self, config: SocialEngineeringConfig | None = None):
        self.config = config or SocialEngineeringConfig()

    def score(self, signals: dict) -> SocialEngineeringResult:
        c = self.config
        contrib: dict[str, float] = {}

        # amount vs baseline (DATA-CALIBRATED)
        ratio = float(signals.get("amount_vs_user_baseline_ratio", 1.0) or 1.0)
        act = _clip01((ratio - c.amount_ratio_low) / max(1e-9, c.amount_ratio_high - c.amount_ratio_low))
        contrib["amount_vs_user_baseline_ratio"] = act * c.amount_ratio_weight

        # first transfer to payee (EXPERT)
        contrib["first_transfer_to_payee"] = (
            c.first_transfer_weight if signals.get("first_transfer_to_payee") else 0.0
        )

        # payee freshness (EXPERT)
        age = float(signals.get("payee_age_hours", 1e9))
        span = max(1e-9, c.payee_age_zero_hours - c.payee_age_full_hours)
        act = _clip01((c.payee_age_zero_hours - age) / span)
        # only meaningful if a payee age was actually supplied
        contrib["payee_age_hours"] = act * c.payee_age_weight if age < 1e8 else 0.0

        # failed attempts (EXPERT)
        attempts = float(signals.get("failed_attempts_before_success", 0) or 0)
        act = _clip01(attempts / max(1, c.failed_attempts_cap))
        contrib["failed_attempts_before_success"] = act * c.failed_attempts_weight

        # high-risk / new geo (DATA-CALIBRATED)
        contrib["is_new_high_risk_country_or_geo"] = (
            c.high_risk_geo_weight if signals.get("is_new_high_risk_country_or_geo") else 0.0
        )

        # night-time (TEMPERED)
        contrib["night_time_transaction"] = (
            c.night_weight if signals.get("night_time_transaction") else 0.0
        )

        # velocity (EXPERT)
        vel = float(signals.get("txn_velocity_short_window", 1) or 1)
        act = _clip01((vel - 1) / max(1, c.velocity_cap - 1))
        contrib["txn_velocity_short_window"] = act * c.velocity_weight

        score = min(100.0, sum(contrib.values()))
        return SocialEngineeringResult(
            social_engineering_score=round(score, 2),
            is_likely_social_engineering=score >= c.flag_threshold,
            contributions={k: round(v, 2) for k, v in contrib.items()},
            signals=dict(signals),
        )
