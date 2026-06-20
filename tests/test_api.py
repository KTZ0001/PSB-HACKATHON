"""Phase 4 API tests — all three endpoints + one case per fraud type.

Uses FastAPI's TestClient with the lifespan active (so the model bundle loads).
Skips cleanly if the model hasn't been trained yet. Each test uses unique
user/device ids so the shared device graph + profile store don't interfere.
"""
import pytest

# FastAPI's TestClient needs httpx. It is a test-only dependency and not in the
# core requirements, so skip these HTTP-level tests cleanly if it's absent (the
# scoring logic itself is covered by test_combiner / orchestrator tests, and the
# live API is verified via scripts against a running uvicorn).
pytest.importorskip("httpx", reason="install httpx to run HTTP-level API tests")

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from src.models.registry import bundle_exists  # noqa: E402

pytestmark = pytest.mark.skipif(
    not bundle_exists(), reason="model bundle missing; run scripts/train.py"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/api/v1/score" in r.json()["endpoints"]


def test_score_normal_is_allow(client):
    # Prime a baseline + known payee, then a normal transaction.
    for _ in range(3):
        client.post("/api/v1/score", json={
            "user_id": "api_alice", "device_id": "api_alice_dev",
            "amount": 100.0, "payee_id": "alice_landlord", "country": "UK",
            "hour": 12, "timestamp": 0.0,
        })
    r = client.post("/api/v1/score", json={
        "user_id": "api_alice", "device_id": "api_alice_dev",
        "amount": 105.0, "payee_id": "alice_landlord", "country": "UK",
        "hour": 12, "timestamp": 500000.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_type"] == "legitimate"
    assert body["recommended_action"] == "allow"
    assert body["trust_score"] >= 70
    assert set(body["score_breakdown"]) == {"behavioral", "mule_graph", "social_engineering"}


def test_score_account_takeover(client):
    r = client.post("/api/v1/score", json={
        "user_id": "api_bob", "device_id": "api_bob_newdev",
        "amount": 900.0, "payee_id": "bob_new", "country": "UK", "hour": 3,
        "behavioral_features": {
            "name_email_similarity": 0.07, "credit_risk_score": 330,
            "bank_months_count": -1, "phone_mobile_valid": 0,
            "foreign_request": 1, "proposed_credit_limit": 1900,
            "customer_age": 50, "keep_alive_session": 0, "has_other_cards": 0,
        },
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_type"] == "account_takeover"
    assert body["trust_score"] < 70
    assert body["recommended_action"] != "allow"


def test_score_social_engineering(client):
    # Build a spending baseline on a known payee, then a large first-ever
    # transfer to a brand-new payee with failed attempts -> SE, clean device/geo.
    for _ in range(3):
        client.post("/api/v1/score", json={
            "user_id": "api_carol", "device_id": "api_carol_phone",
            "amount": 100.0, "payee_id": "carol_known", "country": "UK", "hour": 14,
        })
    r = client.post("/api/v1/score", json={
        "user_id": "api_carol", "device_id": "api_carol_phone",  # her real device
        "amount": 600.0, "payee_id": "carol_scam_payee",
        "country": "UK", "hour": 15, "failed_attempts": 2, "timestamp": 1000.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_type"] == "social_engineering"
    assert body["recommended_action"] != "allow"
    # device/geo were clean -> those components are ~0; SE drives the decision.
    assert body["score_breakdown"]["mule_graph"] == 0.0
    assert body["score_breakdown"]["social_engineering"] >= 50


def test_score_mule_and_device_endpoint(client):
    # One device cashing out across 4 distinct accounts.
    last = None
    for i in range(4):
        last = client.post("/api/v1/score", json={
            "user_id": f"api_mule_{i}", "device_id": "api_shared_dev",
            "amount": 480.0, "payee_id": f"drop_{i}", "country": "UK", "hour": 2,
        }).json()
    assert last["predicted_type"] == "mule_network"
    assert last["recommended_action"] in ("cooling_off", "block_and_flag_analyst")

    r = client.get("/api/v1/device/api_shared_dev/risk")
    assert r.status_code == 200
    info = r.json()
    assert info["linked_account_count"] == 4
    assert info["risk_flag"] is True
    assert len(info["linked_accounts"]) == 4


def test_device_endpoint_unknown_device(client):
    r = client.get("/api/v1/device/never_seen_dev/risk")
    assert r.status_code == 200
    info = r.json()
    assert info["linked_account_count"] == 0
    assert info["risk_flag"] is False


def test_score_validation_error(client):
    # Missing required user_id/device_id -> 422.
    r = client.post("/api/v1/score", json={"amount": 100})
    assert r.status_code == 422
