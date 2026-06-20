"""Aegis REST API (FastAPI) — the only thing a frontend talks to.

Endpoints (versioned under /api/v1):
  POST /api/v1/score                  -> full trust score + action + explanation
  GET  /api/v1/device/{device_id}/risk -> mule device-graph fan-out
  GET  /api/v1/health                 -> liveness + loaded model version

OpenAPI docs are auto-generated at /docs. CORS is permissive for local dev so
different frontends can point at this during the hackathon.

Run:  uvicorn api.app:app --reload     (or python -m api.app)
"""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    DeviceRiskResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from src.scoring.orchestrator import AegisOrchestrator

API_PREFIX = "/api/v1"

# Module-level shared state (one orchestrator = one device graph + profile store
# + loaded model, shared across all requests so the mule graph accumulates).
_state: dict = {"orchestrator": None, "model_version": None, "loaded_at": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        orch = AegisOrchestrator()
        _state["orchestrator"] = orch
        _state["model_version"] = orch.risk_engine.bundle.model_version
        _state["loaded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    except Exception as exc:  # model bundle missing/unreadable -> degraded mode
        _state["error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(
    title="Aegis Identity-Trust Scoring API",
    version="1.0.0",
    description="Continuous identity-trust scoring: one engine, three fraud disguises.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_orchestrator() -> AegisOrchestrator:
    orch = _state.get("orchestrator")
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {_state.get('error') or 'run scripts/train.py first'}",
        )
    return orch


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    if _state.get("orchestrator") is None:
        return HealthResponse(status="model_not_loaded")
    return HealthResponse(
        status="ok",
        model_version=_state.get("model_version"),
        loaded_at=_state.get("loaded_at"),
    )


@app.post(f"{API_PREFIX}/score", response_model=ScoreResponse, tags=["scoring"])
def score(req: ScoreRequest) -> ScoreResponse:
    orch = _get_orchestrator()
    request = req.model_dump(exclude_none=True)
    # Flatten behavioral_features (drop None overrides) for the orchestrator.
    bf = request.pop("behavioral_features", None)
    if bf:
        request["behavioral_features"] = {k: v for k, v in bf.items() if v is not None}
    result = orch.score(request)
    return ScoreResponse(**result)


@app.get(
    f"{API_PREFIX}/device/{{device_id}}/risk",
    response_model=DeviceRiskResponse,
    tags=["scoring"],
)
def device_risk(device_id: str) -> DeviceRiskResponse:
    orch = _get_orchestrator()
    info = orch.device_graph.get_device_risk(device_id)
    return DeviceRiskResponse(
        device_id=info["device_id"],
        linked_account_count=info["account_count"],
        risk_flag=info["risk_flag"],
        linked_accounts=info["linked_accounts"],
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "Aegis Identity-Trust Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [f"{API_PREFIX}/score", f"{API_PREFIX}/device/{{device_id}}/risk", f"{API_PREFIX}/health"],
    }


if __name__ == "__main__":
    import uvicorn

    from src import config

    uvicorn.run("api.app:app", host=config.API_HOST, port=config.API_PORT, reload=False)
