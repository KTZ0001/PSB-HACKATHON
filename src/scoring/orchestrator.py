"""Aegis orchestrator: one entry point that runs all layers and returns the
full scored response (Phase 3). Used by both the demo script and the REST API.

Pipeline:
    request -> [behavioural engine] + [device graph] + [SE engine]
            -> [score combiner] -> [explanation] -> response dict
"""
from __future__ import annotations

from src.features.device_graph import DeviceAccountGraph
from src.features.user_profile import UserProfileStore
from src.models.inference import RiskEngine
from src.scoring.combiner import ScoreCombiner, CombinerConfig
from src.scoring.social_engineering import (
    RuleWeightedSocialEngineeringEngine,
    SocialEngineeringScorer,
)
from src.explain.base import ExplanationGenerator
from src.explain.templated import TemplatedExplanationGenerator

# Benign default account-opening/behavioural context (values near BAF legit
# medians). A request's `behavioral_features` overrides any of these keys, so a
# caller only needs to supply the fields that differ from "normal".
DEFAULT_BAF_RECORD = {
    "income": 0.5,
    "name_email_similarity": 0.6,
    "prev_address_months_count": 12,
    "current_address_months_count": 60,
    "customer_age": 35,
    "days_since_request": 0.01,
    "intended_balcon_amount": 0.0,
    "zip_count_4w": 1200,
    "velocity_6h": 5000.0,
    "velocity_24h": 4800.0,
    "velocity_4w": 4500.0,
    "bank_branch_count_8w": 10,
    "date_of_birth_distinct_emails_4w": 9,
    "credit_risk_score": 120,
    "email_is_free": 0,
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "bank_months_count": 24,
    "has_other_cards": 1,
    "proposed_credit_limit": 500.0,
    "foreign_request": 0,
    "session_length_in_minutes": 6.0,
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
    "payment_type": "AA",
    "employment_status": "CA",
    "housing_status": "BC",
    "source": "INTERNET",
    "device_os": "windows",
}


class AegisOrchestrator:
    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        device_graph: DeviceAccountGraph | None = None,
        profile_store: UserProfileStore | None = None,
        se_engine: SocialEngineeringScorer | None = None,
        combiner: ScoreCombiner | None = None,
        explainer: ExplanationGenerator | None = None,
    ):
        self.risk_engine = risk_engine or RiskEngine()
        self.device_graph = device_graph or DeviceAccountGraph()
        self.profile_store = profile_store or UserProfileStore()
        self.se_engine = se_engine or RuleWeightedSocialEngineeringEngine()
        self.combiner = combiner or ScoreCombiner(CombinerConfig())
        self.explainer = explainer or TemplatedExplanationGenerator()

    def score(self, request: dict) -> dict:
        user_id = str(request.get("user_id", "unknown"))
        device_id = str(request.get("device_id", "unknown"))

        # --- 1. behavioural layer ---
        record = dict(DEFAULT_BAF_RECORD)
        record.update(request.get("behavioral_features", {}) or {})
        beh = self.risk_engine.score_record(record)

        # --- 2. mule device-graph layer (register this session, then score) ---
        self.device_graph.add_event(user_id, device_id)
        device_info = self.device_graph.get_device_risk(device_id)
        mule_graph_risk = self.device_graph.device_risk_score(device_id)

        # --- 3. social-engineering layer ---
        signals = self.profile_store.compute_signals(
            user_id=user_id,
            amount=float(request.get("amount", 0.0)),
            payee_id=request.get("payee_id"),
            timestamp=float(request.get("timestamp", 0.0)),
            country=request.get("country"),
            hour=request.get("hour"),
            failed_attempts_before_success=int(request.get("failed_attempts", 0)),
            is_high_risk_geo=request.get("is_high_risk_geo"),
        )
        se = self.se_engine.score(signals)
        # fold this transaction into the user's profile for future calls
        self.profile_store.update(
            user_id=user_id,
            amount=float(request.get("amount", 0.0)),
            payee_id=request.get("payee_id"),
            timestamp=float(request.get("timestamp", 0.0)),
        )

        # --- 4. combine ---
        combined = self.combiner.combine(
            anomaly_risk=beh.anomaly_risk,
            xgb_fraud_proba=beh.xgb_fraud_proba,
            xgb_mule_proba=beh.class_proba.get("mule_network", 0.0),
            mule_graph_risk=mule_graph_risk,
            social_engineering_score=se.social_engineering_score,
        )

        # --- 5. explanation ---
        explain_features = {
            "trust_score": combined.trust_score,
            "predicted_type": combined.predicted_type,
            "recommended_action": combined.recommended_action,
            "action_detail": combined.action_detail,
            "behavioral": {
                "anomaly_risk": round(beh.anomaly_risk, 2),
                "xgb_fraud_proba": round(beh.xgb_fraud_proba, 4),
                "predicted_type": beh.predicted_type,
                "class_proba": {k: round(v, 4) for k, v in beh.class_proba.items()},
            },
            "mule_graph": device_info,
            "social_engineering": {
                "score": se.social_engineering_score,
                "is_likely": se.is_likely_social_engineering,
                "contributions": se.contributions,
                "signals": se.signals,
            },
        }
        explanation = self.explainer.explain(explain_features, combined.score_breakdown)

        # --- 6. response (matches the API contract) ---
        return {
            "trust_score": combined.trust_score,
            "predicted_type": combined.predicted_type,
            "confidence": combined.confidence,
            "recommended_action": combined.recommended_action,
            "action_detail": combined.action_detail,
            "explanation": explanation,
            "score_breakdown": combined.score_breakdown,
            "raw_signals": {
                "behavioral": explain_features["behavioral"],
                "mule_graph": device_info,
                "social_engineering": explain_features["social_engineering"],
            },
        }
