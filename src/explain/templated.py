"""Default templated explanation generator — deterministic, no network/LLM.

Produces a fraud-analyst-audit-log-style explanation by inspecting which signals
fired in each layer. Pure string templating; reproducible for the same input.
"""
from __future__ import annotations

from src.explain.base import ExplanationGenerator
from src.scoring.combiner import (
    ACTION_ALLOW,
    ACTION_STEP_UP,
    ACTION_COOLING_OFF,
    ACTION_BLOCK,
    TYPE_LEGIT,
    TYPE_ATO,
    TYPE_MULE,
    TYPE_SE,
)

_ACTION_PHRASE = {
    ACTION_ALLOW: "Allow transaction",
    ACTION_STEP_UP: "Require step-up authentication",
    ACTION_COOLING_OFF: "Apply cooling-off hold and alert an analyst",
    ACTION_BLOCK: "Block transaction and flag to an analyst",
}

_TYPE_PHRASE = {
    TYPE_LEGIT: "consistent with the customer's normal behaviour",
    TYPE_ATO: "consistent with account takeover (anomalous session/behaviour)",
    TYPE_MULE: "consistent with a mule / coordinated cash-out network",
    TYPE_SE: "consistent with social engineering (a coached, victim-authorised transfer)",
}

# Human phrasing for the SE signals when they fire.
_SE_SIGNAL_PHRASE = {
    "first_transfer_to_payee": "first-ever transfer to this payee",
    "payee_age_hours": "payee was added very recently",
    "amount_vs_user_baseline_ratio": "amount far above the customer's baseline",
    "txn_velocity_short_window": "rapid repeated transfers in a short window",
    "failed_attempts_before_success": "multiple failed auth attempts beforehand",
    "is_new_high_risk_country_or_geo": "new or high-risk country/geo",
    "night_time_transaction": "night-time transaction",
}


class TemplatedExplanationGenerator(ExplanationGenerator):
    def explain(self, features: dict, score_breakdown: dict) -> str:
        trust = features.get("trust_score")
        ptype = features.get("predicted_type", TYPE_LEGIT)
        action = features.get("recommended_action", ACTION_ALLOW)
        detail = features.get("action_detail", {}) or {}

        lead = (
            f"Trust score {trust:.1f}/100 — predicted '{ptype}', "
            f"{_TYPE_PHRASE.get(ptype, ptype)}. "
            f"Recommended action: {_ACTION_PHRASE.get(action, action)}"
        )
        if action == ACTION_STEP_UP and detail.get("method"):
            lead += f" via {detail['method'].upper()}"
        elif action == ACTION_COOLING_OFF and detail.get("delay_minutes"):
            lead += f" ({detail['delay_minutes']} min hold)"
        lead += "."

        reasons = self._reasons(features, score_breakdown)
        if reasons:
            return lead + " Key drivers: " + "; ".join(reasons) + "."
        return lead + " No risk signals fired across any layer."

    def _reasons(self, features: dict, breakdown: dict) -> list[str]:
        reasons: list[str] = []

        # --- behavioural layer ---
        beh = features.get("behavioral", {}) or {}
        b_risk = breakdown.get("behavioral", 0.0)
        if b_risk >= 30:
            fp = beh.get("xgb_fraud_proba")
            ar = beh.get("anomaly_risk")
            bits = []
            if fp is not None:
                bits.append(f"model fraud probability {fp:.2f}")
            if ar is not None:
                bits.append(f"anomaly risk {ar:.0f}/100")
            reasons.append(f"behavioural risk {b_risk:.0f}/100 ({', '.join(bits)})")

        # --- mule graph layer ---
        mg = features.get("mule_graph", {}) or {}
        if mg.get("risk_flag"):
            n = mg.get("account_count")
            reasons.append(
                f"device linked to {n} distinct accounts (mule fan-out)"
            )

        # --- social-engineering layer ---
        se = features.get("social_engineering", {}) or {}
        if se.get("is_likely") or breakdown.get("social_engineering", 0) >= 35:
            contribs = se.get("contributions", {}) or {}
            fired = [k for k, v in contribs.items() if v and v > 0]
            # order by contribution descending
            fired.sort(key=lambda k: contribs.get(k, 0), reverse=True)
            phrases = [_SE_SIGNAL_PHRASE.get(k, k) for k in fired[:4]]
            if phrases:
                reasons.append(
                    f"social-engineering pattern ({se.get('score', 0):.0f}/100): "
                    + ", ".join(phrases)
                )
        return reasons
