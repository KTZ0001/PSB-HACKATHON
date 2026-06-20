"""Score combiner + tiered action decision (Phase 3).

Combines the three risk layers into a single trust_score (0-100, HIGHER = more
trustworthy) plus a predicted_type, confidence, score_breakdown and a recommended
action.

Why noisy-OR instead of a plain weighted average
-------------------------------------------------
The Aegis thesis is that a *clean device/geo must not rescue a suspicious money
movement* (social-engineering fraud looks normal on the session). A linear
weighted average of risks would let a near-zero behavioural risk dilute a high
social-engineering risk back below the action threshold — silently re-opening
the exact blind spot we exist to close. Noisy-OR instead lets any single strong
layer pull trust down on its own; the per-layer weights act as reliability
multipliers (how much a layer is allowed to contribute at most). A `weighted_avg`
method is still available via config for comparison.

All weights and thresholds live in CombinerConfig (configurable, not magic
numbers buried in logic).
"""
from __future__ import annotations

from dataclasses import dataclass

# Action labels (also used by the API contract).
ACTION_ALLOW = "allow"
ACTION_STEP_UP = "step_up_auth"
ACTION_COOLING_OFF = "cooling_off"
ACTION_BLOCK = "block_and_flag_analyst"

# Predicted-type labels (extends the Phase 1 taxonomy with social_engineering).
TYPE_LEGIT = "legitimate"
TYPE_ATO = "account_takeover"
TYPE_MULE = "mule_network"
TYPE_SE = "social_engineering"


@dataclass
class CombinerConfig:
    # --- behavioural sub-combo (IsolationForest is weak -> low weight) ---
    w_anomaly: float = 0.2
    w_xgb_fraud: float = 0.8

    # --- combination method + per-layer reliability (noisy-OR) ---
    combine_method: str = "noisy_or"          # "noisy_or" | "weighted_avg"
    reliability_behavioral: float = 0.85
    reliability_mule: float = 0.85
    reliability_social_engineering: float = 0.90

    # --- weights for the weighted_avg method (sum ~1) ---
    w_behavioral: float = 0.45
    w_mule: float = 0.20
    w_social_engineering: float = 0.35

    # --- action thresholds on trust_score ---
    allow_threshold: float = 70.0
    step_up_threshold: float = 35.0
    cooling_off_threshold: float = 15.0

    # --- action parameters ---
    step_up_method: str = "otp"               # "otp" | "face" | "push"
    cooling_off_minutes: int = 20

    @classmethod
    def from_dict(cls, d: dict) -> "CombinerConfig":
        from dataclasses import replace
        base = cls()
        return replace(base, **{k: v for k, v in d.items() if hasattr(base, k)})


@dataclass
class CombinedScore:
    trust_score: float
    predicted_type: str
    confidence: float
    recommended_action: str
    score_breakdown: dict        # {behavioral, mule_graph, social_engineering} as RISK 0-100
    action_detail: dict          # method / delay etc.


def _noisy_or(risks_weighted: list[float]) -> float:
    """Combine independent risks in [0,1] -> [0,1]. Any strong risk dominates."""
    prod = 1.0
    for r in risks_weighted:
        prod *= (1.0 - max(0.0, min(1.0, r)))
    return 1.0 - prod


class ScoreCombiner:
    def __init__(self, config: CombinerConfig | None = None):
        self.config = config or CombinerConfig()

    def combine(
        self,
        *,
        anomaly_risk: float,          # 0-100 (IsolationForest)
        xgb_fraud_proba: float,       # 0-1   (P(not legitimate))
        xgb_mule_proba: float,        # 0-1   (P(mule_network))
        mule_graph_risk: float,       # 0-1   (device fan-out)
        social_engineering_score: float,  # 0-100
    ) -> CombinedScore:
        c = self.config

        # --- per-layer risk components (0-100) ---
        behavioral_risk = 100.0 * (
            c.w_anomaly * (anomaly_risk / 100.0) + c.w_xgb_fraud * xgb_fraud_proba
        )
        mule_graph_risk_pct = 100.0 * max(mule_graph_risk, xgb_mule_proba)
        se_risk = float(social_engineering_score)

        # --- overall risk ---
        if c.combine_method == "weighted_avg":
            overall_risk = (
                c.w_behavioral * behavioral_risk
                + c.w_mule * mule_graph_risk_pct
                + c.w_social_engineering * se_risk
            )
        else:  # noisy_or (default)
            overall_risk = 100.0 * _noisy_or([
                c.reliability_behavioral * behavioral_risk / 100.0,
                c.reliability_mule * mule_graph_risk_pct / 100.0,
                c.reliability_social_engineering * se_risk / 100.0,
            ])

        trust_score = max(0.0, min(100.0, 100.0 - overall_risk))

        # --- predicted type + confidence ---
        breakdown = {
            "behavioral": round(behavioral_risk, 2),
            "mule_graph": round(mule_graph_risk_pct, 2),
            "social_engineering": round(se_risk, 2),
        }
        if trust_score >= c.allow_threshold:
            predicted_type = TYPE_LEGIT
            confidence = round(trust_score / 100.0, 3)
        else:
            candidates = {
                TYPE_ATO: behavioral_risk,
                TYPE_MULE: mule_graph_risk_pct,
                TYPE_SE: se_risk,
            }
            predicted_type = max(candidates, key=candidates.get)
            total = sum(candidates.values()) or 1.0
            confidence = round(candidates[predicted_type] / total, 3)

        action, detail = self.decide_action(trust_score)
        return CombinedScore(
            trust_score=round(trust_score, 2),
            predicted_type=predicted_type,
            confidence=confidence,
            recommended_action=action,
            score_breakdown=breakdown,
            action_detail=detail,
        )

    def decide_action(self, trust_score: float) -> tuple[str, dict]:
        """Map trust_score -> tiered action (thresholds configurable)."""
        c = self.config
        if trust_score >= c.allow_threshold:
            return ACTION_ALLOW, {}
        if trust_score >= c.step_up_threshold:
            return ACTION_STEP_UP, {"method": c.step_up_method}
        if trust_score >= c.cooling_off_threshold:
            return ACTION_COOLING_OFF, {
                "delay_minutes": c.cooling_off_minutes,
                "alert_analyst": True,
            }
        return ACTION_BLOCK, {"alert_analyst": True}
