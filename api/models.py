"""Pydantic request/response models — the versioned API contract.

These models ARE the contract the frontend depends on. They drive the
auto-generated OpenAPI schema at /docs, so keep them complete and accurate.
Swapping the entire UI must require zero backend changes; changing this contract
is what /api/v2 would be for.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

PredictedType = Literal["legitimate", "account_takeover", "mule_network", "social_engineering"]
RecommendedAction = Literal["allow", "step_up_auth", "cooling_off", "block_and_flag_analyst"]


class BehavioralFeatures(BaseModel):
    """Optional account-opening / session features (BAF schema).

    Every field is optional; omitted fields fall back to a benign default, so a
    caller only sends what differs from a normal session. Field meanings are
    documented in src/features/baf_features.FEATURE_DOCS.
    """
    model_config = {"extra": "ignore"}

    income: Optional[float] = None
    name_email_similarity: Optional[float] = None
    prev_address_months_count: Optional[float] = None
    current_address_months_count: Optional[float] = None
    customer_age: Optional[float] = None
    days_since_request: Optional[float] = None
    intended_balcon_amount: Optional[float] = None
    zip_count_4w: Optional[float] = None
    velocity_6h: Optional[float] = None
    velocity_24h: Optional[float] = None
    velocity_4w: Optional[float] = None
    bank_branch_count_8w: Optional[float] = None
    date_of_birth_distinct_emails_4w: Optional[float] = None
    credit_risk_score: Optional[float] = None
    email_is_free: Optional[int] = None
    phone_home_valid: Optional[int] = None
    phone_mobile_valid: Optional[int] = None
    bank_months_count: Optional[float] = None
    has_other_cards: Optional[int] = None
    proposed_credit_limit: Optional[float] = None
    foreign_request: Optional[int] = None
    session_length_in_minutes: Optional[float] = None
    keep_alive_session: Optional[int] = None
    device_distinct_emails_8w: Optional[float] = None
    payment_type: Optional[str] = None
    employment_status: Optional[str] = None
    housing_status: Optional[str] = None
    source: Optional[str] = None
    device_os: Optional[str] = None


class ScoreRequest(BaseModel):
    user_id: str = Field(..., description="Stable customer/account identifier.")
    device_id: str = Field(..., description="Device identifier for this session.")
    amount: float = Field(0.0, ge=0, description="Transaction amount in account currency.")
    payee_id: Optional[str] = Field(None, description="Payee/beneficiary identifier, if a transfer.")
    timestamp: float = Field(0.0, description="Event time in epoch seconds (used for payee age / velocity).")
    country: Optional[str] = Field(None, description="ISO country of the transaction/session.")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Local hour 0-23 (for night-time signal).")
    failed_attempts: int = Field(0, ge=0, description="Failed auth attempts before this transaction.")
    is_high_risk_geo: Optional[bool] = Field(
        None, description="Override: force the high-risk-geo flag (else derived from country)."
    )
    behavioral_features: Optional[BehavioralFeatures] = Field(
        None, description="Optional session/account features; omitted fields use benign defaults."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "carol", "device_id": "carol_usual_phone",
                    "amount": 600.0, "payee_id": "brand_new_payee",
                    "country": "UK", "hour": 15, "failed_attempts": 2,
                }
            ]
        }
    }


class ScoreBreakdown(BaseModel):
    behavioral: float = Field(..., description="Behavioural risk component (0-100).")
    mule_graph: float = Field(..., description="Mule device-graph risk component (0-100).")
    social_engineering: float = Field(..., description="Social-engineering risk component (0-100).")


class ScoreResponse(BaseModel):
    trust_score: float = Field(..., description="0-100, HIGHER = more trustworthy.")
    predicted_type: PredictedType
    confidence: float
    recommended_action: RecommendedAction
    action_detail: dict = Field(default_factory=dict)
    explanation: str
    score_breakdown: ScoreBreakdown
    raw_signals: dict = Field(default_factory=dict, description="All computed features across layers.")


class DeviceRiskResponse(BaseModel):
    device_id: str
    linked_account_count: int
    risk_flag: bool
    linked_accounts: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_version: Optional[str] = None
    loaded_at: Optional[str] = None
