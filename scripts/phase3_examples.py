"""Phase 3 stop-point artefact: full combined output for the thesis scenarios.

Each scenario runs through the complete pipeline (behavioural + mule graph + SE
-> combiner -> tiered action -> templated explanation) and prints the full
response a frontend would receive.

Usage:  python scripts/phase3_examples.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.features.device_graph import DeviceAccountGraph
from src.features.user_profile import UserProfileStore
from src.models.inference import RiskEngine
from src.scoring.orchestrator import AegisOrchestrator

# Load the trained behavioural model once; reuse across scenarios.
SHARED_ENGINE = RiskEngine()


def fresh_orchestrator() -> AegisOrchestrator:
    return AegisOrchestrator(
        risk_engine=SHARED_ENGINE,
        device_graph=DeviceAccountGraph(),
        profile_store=UserProfileStore(),
    )


def show(title: str, resp: dict) -> None:
    b = resp["score_breakdown"]
    print(f"\n{'-' * 78}\n{title}")
    print(f"  trust_score      : {resp['trust_score']:.1f}/100   "
          f"predicted_type: {resp['predicted_type']}   confidence: {resp['confidence']}")
    print(f"  recommended_action: {resp['recommended_action']}  {resp.get('action_detail') or ''}")
    print(f"  breakdown (risk) : behavioral={b['behavioral']}  "
          f"mule_graph={b['mule_graph']}  social_engineering={b['social_engineering']}")
    print(f"  explanation      : {resp['explanation']}")


def main() -> int:
    # 1. Normal legitimate transaction. Alice has an established spending
    # baseline AND a long-known payee (her landlord), as in real recurring rent.
    orch = fresh_orchestrator()
    orch.profile_store.seed_baseline("alice", [100, 120, 95, 110])
    orch.profile_store.update("alice", 100.0, "landlord", timestamp=0.0)  # payee known since t=0
    resp = orch.score({
        "user_id": "alice", "device_id": "alice_phone",
        "amount": 105.0, "payee_id": "landlord", "country": "UK", "hour": 13,
        "timestamp": 600_000.0,  # ~167h later -> landlord is an established payee
    })
    show("1. NORMAL — known payee, ~baseline amount, normal device", resp)

    # 2. Account takeover: anomalous behavioural/session features.
    orch = fresh_orchestrator()
    resp = orch.score({
        "user_id": "bob", "device_id": "unknown_new_device",
        "amount": 900.0, "payee_id": "new_payee_x", "country": "UK", "hour": 3,
        "behavioral_features": {
            "name_email_similarity": 0.07, "credit_risk_score": 330,
            "bank_months_count": -1, "phone_mobile_valid": 0,
            "foreign_request": 1, "proposed_credit_limit": 1900,
            "customer_age": 50, "device_distinct_emails_8w": 1,
            "keep_alive_session": 0, "has_other_cards": 0,
        },
    })
    show("2. ACCOUNT TAKEOVER — anomalous device/session/behaviour", resp)

    # 3. Mule network: one device appears across many user_ids in succession.
    orch = fresh_orchestrator()
    last = None
    for i, uid in enumerate(["mule_a", "mule_b", "mule_c", "mule_d"]):
        last = orch.score({
            "user_id": uid, "device_id": "shared_cashout_device",
            "amount": 480.0, "payee_id": f"drop_{i}", "country": "UK", "hour": 2,
        })
    show("3. MULE NETWORK — same device cashing out across 4 accounts (final call)", last)

    # 4. Social engineering: clean device + clean geo, suspicious money movement.
    orch = fresh_orchestrator()
    orch.profile_store.seed_baseline("carol", [90, 110, 100, 105, 95])  # ~£100 norm
    resp = orch.score({
        "user_id": "carol", "device_id": "carol_usual_phone",  # her real device
        "amount": 600.0,             # 6x her baseline
        "payee_id": "brand_new_payee",  # never seen
        "timestamp": 1000.0,
        "country": "UK",             # domestic, NOT high-risk
        "hour": 15,                  # afternoon, NOT night
        "failed_attempts": 2,        # fumbled auth (being coached)
    })
    show("4. SOCIAL ENGINEERING — clean device/geo, coached-victim money shape", resp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
