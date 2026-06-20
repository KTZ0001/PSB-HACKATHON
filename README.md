# Aegis — Continuous Identity-Trust Scoring Engine

> Bank of Baroda Hackathon 2026 · *Identity Trust, Protection & Safety*
> A working prototype, not a toy demo and not a production system.

## The one question, three disguises

Every existing identity-trust system asks: **"does this login look suspicious?"**
— device fingerprint, geo-location, IP reputation. That question has a structural
blind spot: it has *no signal left* when the real, legitimate account holder is
the one authorising the fraud (a scam-coached transfer).

Aegis asks a different question: **"is this money movement legitimate, regardless
of how the session looks?"** One trust engine scores the three disguises the same
underlying fraud wears:

| Disguise | What it is | How Aegis catches it |
|---|---|---|
| **Stolen identity** (account takeover) | New device, anomalous geo, recovery abuse | Behavioural / anomaly risk scoring (XGBoost + IsolationForest on BAF) |
| **Coordinated network** (mule accounts) | Cash-out through accounts sharing devices | Device-to-account graph fan-out (networkx) |
| **Willing victim** (social engineering) | Genuine owner talked into transferring — device & geo look normal | Transaction-context **risk-policy engine** on signals from standard logs |

The third is the differentiator. In demo scenario 4, the behavioural risk is **4/100**
and the mule risk is **0** — the device, geo and IP are perfectly normal — yet
Aegis flags the transfer, because the *shape of the money movement* (first-ever
transfer to a payee added minutes ago, far above baseline, after fumbled auth)
gives it away.

## Architecture

```
[Session / Transaction Event]
        |
        v
[Feature Engineering]  device-graph lookup · amount-vs-baseline · payee novelty
        |              · geo / night context · BAF behavioural features
        v
[Risk Layers]
  - IsolationForest      unsupervised anomaly score      (BAF)        \
  - XGBoost 3-class      legitimate / ATO / mule_network  (BAF)        }  behavioural
  - Device-account graph mule fan-out risk                (runtime)   ---  mule_graph
  - SE risk-policy engine rule-weighted, calibrated       (Dataset 2) ---  social_engineering
        |
        v
[Score Combiner]   noisy-OR (a clean device must NOT dilute a suspicious
        |          money movement) -> trust_score 0-100 + predicted_type + confidence
        v
[Tiered Action]    allow / step_up_auth / cooling_off / block_and_flag_analyst
        |
        v
[Explanation]      pluggable; default = deterministic templated (no LLM, no network)
        |
        v
[REST API]         /api/v1/...  <-- the ONLY thing the frontend talks to
```

Backend and frontend are fully decoupled: all logic sits behind a versioned REST
contract (Pydantic-typed, OpenAPI at `/docs`). Swapping the entire UI requires
zero backend changes.

## Quickstart

Requires **Python 3.11+** (developed on 3.12).

```bash
pip install -r requirements.txt

# 1. Download data (see "Kaggle setup" below for auth)
python scripts/download_data.py

# 2. Train the behavioural engine (~1-2 min; writes models_store/aegis_models.joblib)
python scripts/train.py

# 3. Run the live end-to-end demo (auto-launches the API, drives it, shuts it down)
python scripts/demo.py
```

### Other entry points

```bash
python scripts/evaluate.py                 # test metrics + sample scored examples
python scripts/calibrate_social_engineering.py   # SE signal distributions (Dataset 2)
python scripts/se_examples.py              # social-engineering engine example scores
python scripts/phase3_examples.py          # full combined output for the 4 scenarios
uvicorn api.app:app --reload               # run the API standalone; docs at /docs
pytest -q                                  # 31 tests
```

### Kaggle setup

Kaggle CLI auth has moved to OAuth / single-token auth — Aegis does **not** look
for a `kaggle.json` file. Authenticate once with `kaggle auth login` (or set
`KAGGLE_API_TOKEN`). `scripts/download_data.py` verifies auth before downloading
and prints an actionable message if it's missing.

> The NeurIPS BAF dataset requires accepting its terms on the Kaggle website
> (the "I Understand and Accept" button on the dataset page) before the API will
> serve it, even with valid auth. If you hit a 403, the download script tells you
> exactly which page to open.

## API contract (v1)

| Endpoint | Returns |
|---|---|
| `POST /api/v1/score` | `trust_score`, `predicted_type`, `confidence`, `recommended_action`, `action_detail`, `explanation`, `score_breakdown {behavioral, mule_graph, social_engineering}`, `raw_signals` |
| `GET /api/v1/device/{device_id}/risk` | `device_id`, `linked_account_count`, `risk_flag`, `linked_accounts[]` |
| `GET /api/v1/health` | `status`, `model_version`, `loaded_at` |

Minimal request body: `{"user_id", "device_id", "amount", "payee_id", "country", "hour", "failed_attempts"}`.
Optional `behavioral_features` overrides any BAF session field (omitted fields use
benign defaults). Full schema at `/docs`.

## Data sources — the honest notes

**Dataset 1 — Bank Account Fraud (NeurIPS 2022), `Base` variant.** 1,000,000 rows ×
32 cols, fraud rate **1.10%**, no nulls. Primary training data for the behavioural
risk engine and the ATO classifier. The other 5 variants exist for fairness/bias
testing and are intentionally not used yet.

- **This dataset has no social-engineering labels.** A column-token scan confirms
  it: no payee, victim-authorised, or self-initiated-transfer fields. The
  social-engineering detector is therefore **NOT trained on it** — see Dataset 2
  and the rule-based engine. We never claim this dataset validates the SE detector.
- **No raw `device_id` / `account_id` columns.** Only aggregate device features
  (`device_os`, `device_distinct_emails_8w`, `device_fraud_count`). So:
  - the literal device-to-account **graph is a runtime structure**, populated by
    incoming scoring events (a bank computes this from its own session logs);
  - the **`mule_network` class label is a DERIVED HEURISTIC**, not ground truth:
    among fraud rows, `device_distinct_emails_8w >= 2` (a device tied to multiple
    email identities) is labelled mule. This is documented as a heuristic
    everywhere it appears — it is not presented as ground truth.
  - `device_fraud_count` is constant (0) across the Base variant and is dropped.

**Dataset 2 — Synthetic Financial Fraud.** 10,000 rows × 10 cols, fraud rate **5%**.
**Synthetic** — results on it are calibration, not real-world validation. Used to
calibrate the transaction-context signals that feed the SE risk-policy engine. The
data is near-deterministic (e.g. all night-time rows are fraud, `ip_risk>0.7` is
always fraud), so we take *direction and rough thresholds* from it but temper some
weights with domain judgement (real night-time transactions are mostly legitimate).

## How the engine is trained / calibrated

**Behavioural engine (supervised, on BAF).** Temporal split (train months 0–5,
test 6–7, no leakage). Headline result: **XGBoost P(fraud) ROC-AUC = 0.889**.
The 3-class type classifier uses tempered class weighting (`power=0.45`, chosen
via `scripts/tune_class_weights.py` to maximise macro-F1 and cut false positives
~9× vs naive inverse-frequency weighting). IsolationForest anomaly AUC is **0.545**
— weak on this dataset (BAF fraud doesn't sit in low-density feature regions), so
it contributes a minor signal only and XGBoost carries the discriminative load.

**Social-engineering engine (rule-weighted policy, NOT yet supervised).** Because
neither dataset has SE ground truth, this is an explicit, inspectable rule-weighted
engine. Each weight is documented as either *data-calibrated* (amount-vs-baseline,
high-risk-geo — from Dataset 2) or *expert prior* (first-transfer-to-payee,
payee-age, failed-attempts — no dataset analogue). The honest line for judges:

> *"We start with a rule-weighted risk-policy engine calibrated on transaction-
> context features, and the architecture is designed to transition to a supervised
> model as labelled fraud cases accumulate post-deployment."*

The code backs this: the policy engine and any future ML model share the same
canonical signal vector (`SE_SIGNAL_NAMES` + `signal_vector()`) behind one
`SocialEngineeringScorer` interface, so swapping it is a contained change.

## Tiered actions

`trust_score` thresholds (all configurable in `CombinerConfig`):

| Trust | Action |
|---|---|
| ≥ 70 | `allow` |
| 35–70 | `step_up_auth` (OTP by default; OTP / face / push configurable) |
| 15–35 | `cooling_off` (timed hold + analyst alert; 20 min default) |
| < 15 | `block_and_flag_analyst` |

## Known limitations / roadmap

- **No labelled social-engineering ground truth** exists in either dataset. The SE
  layer is a rule-weighted policy engine, not a trained classifier — by design,
  pending labelled cases post-deployment. (The architecture is ready for the swap.)
- **No real-time behavioural biometrics** (keystroke / mouse / gesture). This is a
  deliberate scope choice: every Aegis signal is derivable from data a bank
  *already logs*. Client-side biometric telemetry is explicitly out of scope.
- **Mule classifier is modest** (derived heuristic label, thin signal). The real
  mule defence is the runtime device-graph fan-out, not the BAF classifier.
- **IsolationForest is weak** on BAF (AUC 0.545); kept as a minor signal only.
- **Single-node**, in-memory device graph and user profiles (JSON-persistable).
  Not yet horizontally scaled — the natural next step is a shared store (Redis /
  graph DB) behind the same interfaces.
- **Explanation** is templated/deterministic by default. An LLM-backed generator
  is a clearly-marked optional extension point, not wired in.

## Repository layout

```
src/features/   feature engineering, device graph, user profile, SE signals
src/models/     train / inference / artifact registry (behavioural engine)
src/scoring/    SE policy engine, score combiner, orchestrator
src/explain/    pluggable explanation interface + templated default
api/            FastAPI app + Pydantic contract
scripts/        download_data, train, evaluate, calibrate, demo, examples
tests/          31 tests across all layers
```
