# Aegis
### Continuous Identity Trust Infrastructure for Digital Banking

**Bank of Baroda Hackathon 2026 — Identity Trust, Protection & Safety**
Team Aegis

---

## The problem, in one sentence

Banks verify identity once — at login — then trust everything that happens
in the session afterward. That single check misses two of the three ways
fraud actually happens.

## The question every existing system asks (and why it's not enough)

Device fingerprinting, geo-location checks, and login-anomaly detection all
ask the same question: **"does this login look suspicious?"**

That question has a blind spot. It assumes the fraud requires the legitimate
account holder to be absent — replaced by an attacker on a strange device,
in a strange location. It has no answer for the case where **the real
account holder is the one clicking "confirm,"** because they've been
talked into it.

## The question Aegis asks instead

**"Is this money movement legitimate — regardless of how the session
looks?"**

One trust engine. Three disguises the same underlying fraud wears to hide
from detection:

| # | Disguise | What it looks like | What catches it |
|---|---|---|---|
| 1 | **Stolen Identity** (Account Takeover) | New device, unfamiliar location, account-recovery abuse | Behavioral risk model |
| 2 | **Coordinated Network** (Mule accounts) | Money cashed out through accounts that share a device | Device-to-account graph |
| 3 | **Willing Victim** (Social Engineering) | Same device, same location, same user — no anomaly at all | Transaction-context risk signals |

Disguise 3 is the one every device/geo-based system structurally cannot
see. It's also responsible for a large and growing share of real fraud
losses in India — and it's the reason Aegis exists.

## What Aegis actually does, end to end

1. A banking session generates an event (login, device, transaction).
2. Three scoring layers run in parallel:
   - A **behavioral risk model** trained on real account/transaction data,
     scoring how anomalous this session is.
   - A **device-to-account graph** that tracks how many distinct accounts
     have used this device — the mule-network signal.
   - A **social-engineering risk-policy engine** that scores transaction
     context (new payee, payee age, amount vs. baseline, failed attempts,
     time of day) — signals every bank already logs, with no new
     instrumentation required.
3. The three scores combine into a single **trust score (0–100)**.
4. Instead of a binary block/allow decision, the trust score drives a
   **proportional response**:

| Trust score | Action |
|---|---|
| ≥ 70 | Allow — no friction |
| 35–69 | Step-up authentication (e.g. OTP) |
| 15–34 | Cooling-off hold (timed delay) + analyst alert |
| < 15 | Block transaction + flag to analyst |

5. Every decision comes with a plain-language explanation, generated from
   the actual contributing signals — not a black-box number.

This is the core business case: **reduce fraud losses without punishing
legitimate customers with constant friction.** Most fraud-detection systems
optimize for catching fraud. Aegis optimizes for catching fraud *and*
preserving a frictionless experience for the other 95%+ of sessions that
are completely legitimate.

## Proof it works — real output from the running system

The following are **real, unedited responses from the live API**, not
illustrative numbers. Every figure below was produced by calling the actual
running service.

### The three disguises, scored independently

**Legitimate session** (known device, known city, existing payee):
```
trust_score: 96.96  →  action: allow
"No risk signals fired across any layer."
```

**Disguise 1 — Account takeover** (new device, foreign IP, account-recovery
attempt):
```
trust_score: 23.03  →  action: cooling_off (20 min hold, analyst alerted)
predicted_type: account_takeover  (model confidence 0.81)
"Key drivers: behavioural risk 85/100 (model fraud probability 0.91)."
```

**Disguise 2 — Mule network** (a device used by four different accounts in
quick succession, simulated against the live device-graph endpoint):
```
trust_score: 8.78  →  action: block_and_flag_analyst
predicted_type: mule_network  (model confidence 0.68)
"Key drivers: device linked to 4 distinct accounts (mule fan-out)."
```

**Disguise 3 — Social engineering** (same device, same city as the genuine
owner; first-ever transfer to a brand-new payee, 7× the customer's usual
amount, two failed attempts beforehand):
```
trust_score: 32.38  →  action: cooling_off (20 min hold, analyst alerted)
predicted_type: social_engineering  (model confidence 0.95)
"Key drivers: amount far above baseline, first-ever transfer to this payee,
 payee added very recently, multiple failed attempts beforehand."
```

**The headline result**: Disguise 3 is caught with the customer's own
device, in their own city — zero device or location anomaly — purely from
transaction-context signals already present in a standard bank's logs. This
is the exact blind spot described above, closed.

### A live session, degrading step by step

The same customer, one continuous session, real consecutive API calls:

| Step | Event | Trust score | Action |
|---|---|---|---|
| 0 | Normal session, known device | **96.96** | Allow |
| 1 | New device + new beneficiary added | **60.89** | Step-up auth (OTP) |
| 2 | ₹150,000 transfer attempted at 2am | **47.51** | Step-up auth (OTP) |
| 3 | Login from a new country + 3 failed auth attempts | **21.92** | Cooling-off hold + analyst alert |

The score never resets to a hard binary state — it degrades continuously as
risk accumulates, and the *response* escalates proportionally rather than
jumping straight to a block.

---

## Technical detail (for judges who want to verify the claims above)

### What the model is actually trained on

- **Bank Account Fraud Dataset Suite (NeurIPS 2022)** — `Base` variant, the
  real dataset downloaded via the Kaggle API, **1,000,000 real rows**, not
  synthetic. Split temporally (months 0–5 train, months 6–7 test — no
  random shuffling, so the test set is genuinely unseen future data).
- **Synthetic Financial Fraud Dataset** — used to calibrate the
  transaction-context signal weights in the social-engineering risk-policy
  engine (amount deviation, new-payee, velocity, time-of-day thresholds).

### Model performance (real, from the test split — 205,011 rows)

| Metric | Value | What it means |
|---|---|---|
| XGBoost fraud-probability ROC-AUC | **0.889** | Strong — reliably ranks risky sessions above safe ones |
| Isolation Forest anomaly ROC-AUC | 0.545 | Weak by design — deliberately down-weighted in the combiner (20% weight) because it's a secondary signal, not the primary one |
| Macro F1 (3-class type attribution) | 0.516 | Modest — see honest limitation below |

**Per-class breakdown** (test set, real confusion matrix):

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Legitimate | 0.989 | 0.989 | 0.989 | 202,133 |
| Account takeover | 0.246 | 0.246 | 0.246 | 2,719 |
| Mule network | 0.259 | 0.390 | 0.312 | 159 |

**Why we lead with AUC 0.889, not the 97.9% overall accuracy figure**: with
fraud at roughly 1% of all sessions, a model that calls everything
"legitimate" already scores ~99% accuracy while being useless. AUC measures
whether the model correctly *ranks* risky sessions above safe ones across
every threshold — that's the number that matters for a system whose whole
design is "score everything continuously," not "classify once."

### Exact train/test split (Base variant, temporal split, 1,000,000 rows)

| Split | Legitimate | Account takeover | Mule network | **Total** |
|---|---|---|---|---|
| **Train** (months 0–5) | 786,838 | 7,275 | 876 | **794,989** |
| **Test** (months 6–7) | 202,133 | 2,719 | 159 | **205,011** |

`account_takeover` and `mule_network` are both derived from the dataset's
single binary `fraud_bool` (see limitation #1), so per split they sum to the
total fraud count (train 8,151; test 2,878).

### Honest limitations (we'd rather state these than have a judge find them)

1. **The account-takeover and mule-network per-class metrics above are
   weak** (F1 of 0.25 and 0.31). This is a direct consequence of severe
   class imbalance — the test set has only 159 real mule-network examples
   out of 205,011 rows. Both fraud-type labels are also **derived
   heuristics**, not native dataset columns: the NeurIPS BAF dataset only
   ships a single binary `fraud_bool` label, so `account_takeover` vs.
   `mule_network` is a label we constructed from velocity and device-sharing
   features. We're stating this plainly rather than presenting it as ground
   truth.
2. **When multiple risk signals fire simultaneously** (as in the
   degrading-session demo above), the system's type-attribution currently
   tends to label the result `social_engineering` even in cases that a
   human analyst might call account takeover. The underlying per-disguise
   detectors work correctly in isolation (see the three standalone results
   above); the combiner's logic for picking a single "winning" label when
   several signals overlap is the next thing we'd refine. **The trust
   score and the recommended action are correct and real in every case
   above — it's specifically the human-readable type label that can be
   ambiguous under combined signals.**
3. **The social-engineering detector is a rule-weighted risk-policy engine,
   not a supervised classifier**, because neither dataset used here
   contains real social-engineering ground-truth labels. This is by design,
   not an oversight: we start with calibrated rules and the architecture is
   built so it can transition to a supervised model as labelled
   fraud cases accumulate post-deployment — the feature format is already
   shared between the rule engine and the ML classifiers for exactly that
   reason.
4. **The device-to-account graph is an online structure, not a column in
   the training data** — the NeurIPS BAF dataset has no device-identifier
   field. The mule-network detection shown above is real and works against
   live API calls, but wasn't validated against a held-out labelled
   mule-fraud dataset, because no such dataset exists in what we used.

### Architecture

```
Session/Transaction Event
        |
        v
Feature Engineering  --  device-graph lookup, transaction-context features
        |
        v
Three parallel scoring layers:
  - Behavioral risk model (XGBoost + Isolation Forest, trained on NeurIPS BAF)
  - Device-to-account graph (mule-network signal)
  - Social-engineering risk-policy engine (rule-weighted, calibrated on
    synthetic financial fraud data)
        |
        v
Score Combiner  -->  trust_score (0-100) + predicted_type + confidence
        |
        v
Explanation Generator  --  plain-language, audit-ready (pluggable for a
        |                  private/on-prem LLM per deployment; default is
        |                  deterministic/templated, zero external calls)
        v
Tiered Action Decision  --  allow / step_up_auth / cooling_off / block
        |
        v
REST API  --  the only thing any frontend talks to
```

### Tech stack

Python · scikit-learn (Isolation Forest) · XGBoost · FastAPI · NetworkX
(device graph) · React (demo client)

### Roadmap (Phase 2, prototype due July 26)

- Improve combiner type-attribution when multiple signals overlap
  (limitation #2 above)
- Transition the social-engineering engine toward a supervised model as
  labelled cases accumulate
- 3D, interactive force-directed visualization of the device/mule graph
- Live analyst dashboard consuming the API over WebSocket

---

## Running this yourself (it's not deployed anywhere — you'll need to run it locally)

This is a hackathon prototype, not a hosted product — there is no live URL.
To see it, you need to clone the repo and run two things at once: the
backend (API + model) and the frontend (demo UI). Both must be running
simultaneously, in two separate terminals, for the demo to work.

### 1. Clone the repo

```bash
git clone https://github.com/KTZ0001/PSB-HACKATHON.git
cd PSB-HACKATHON
```

### 2. Backend — terminal 1

```bash
pip install -r requirements.txt

# The trained model ships with the repo (models_store/aegis_models.joblib),
# so you can start the API immediately — no dataset download or training needed:
uvicorn api.app:app --reload --port 8000
```

Leave this running. Confirm it's up by opening
**`http://localhost:8000/docs`** in a browser — you should see the
interactive API documentation (Swagger UI). If you see that page, the
backend is live.

<details>
<summary><b>Optional: reproduce training from scratch instead of using the bundled model</b></summary>

```bash
# Requires Kaggle CLI auth first (one-time):
#   kaggle auth login
# and you must accept both datasets' terms on kaggle.com first (visit each
# dataset page while logged in and click "I Understand and Accept"):
#   kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
#   kaggle.com/datasets/umitka/synthetic-financial-fraud-dataset

python scripts/download_data.py   # ~1.3 GB
python scripts/train.py           # rewrites models_store/aegis_models.joblib
```

</details>

### 3. Frontend — terminal 2 (separate terminal, while the backend keeps running)

```bash
cd frontend
npm install
npm run dev
```

This starts the demo UI, typically at **`http://localhost:5173`**
(the terminal output will print the exact URL — use that one if it
differs). Open that URL in a browser.

### 4. What you should see

The Aegis homepage with a single **"Launch Simulation"** button. Click it
to start the Rajesh Patel narrative described above, and use the
secondary buttons to view the Mule Network and Social Engineering cases
on demand. Every number on screen is a real, live response from the
backend you started in terminal 1 — not hardcoded.

If the frontend loads but shows connection errors, confirm the backend
(terminal 1) is still running on port 8000. The dev server automatically
proxies `/api` → `http://127.0.0.1:8000` (see `frontend/vite.config.js`);
if you run the backend on a different port, update that proxy. For a
production build served outside the dev server, set `VITE_API_BASE` to the
backend URL at build time.

### Don't want to set any of this up? Run the CLI demo instead

```bash
python scripts/demo.py
```

This runs the same narrative and scenario calls against the live backend
and prints the results directly to the terminal — it auto-launches the API,
drives it, and shuts it down, so no frontend/npm is required. Useful as a
fallback or for a quick sanity check that the backend works.
