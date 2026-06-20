# Aegis — Live Trust Score (demo client)

A single-screen, disposable demo client for the Aegis trust-scoring API. It tells
one continuous story — a single customer session degrading step by step into an
adaptive intervention — to make the trust-score collapse visible to judges. Not
the final product UI. Enterprise-calm aesthetic (Stripe/Linear), not sci-fi.

## Run

1. Start the backend API first (from the repo root) so it's live on port 8000:
   ```
   python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
   ```
2. Then start this demo:
   ```
   cd frontend
   npm install
   npm run dev
   ```
3. Open http://localhost:5173 — Vite proxies `/api` → `127.0.0.1:8000`.

> **Demo note:** the `/api` proxy only exists in the `npm run dev` server. A
> production build (`npm run build` + `npm run preview` or any static host) does
> **not** proxy — point it at a real backend by setting `VITE_API_BASE` at build
> time, e.g. `VITE_API_BASE=http://127.0.0.1:8000 npm run build`. For judging,
> just use `npm run dev` with the backend already running on :8000.

## The demo

**Launch Simulation → the Rajesh Patel narrative.** Three panels (Customer
Session · Trust Score · AI Explanation). "Next Event ▸" advances one customer
through four real `/api/v1/score` calls — every score, explanation and action is
the live API response, nothing hardcoded:

| Event | Real score | Action |
|---|---|---|
| Baseline | ~97 | allow |
| New device + new beneficiary | ~61 | step_up_auth |
| High-value transfer (₹1,50,000), 02:00 | ~47 | step_up_auth |
| Login from new location + failed auth | ~22 | **cooling_off** |

The climax replaces "Next Event" with the **Action Selected** block — the real
adaptive intervention (cooling period · analyst notified from `action_detail`),
*not* a binary block. Risk factors accumulate in the left panel; the score tweens
+ color-shifts between steps and idle-flickers when settled.

**Secondary cases** (small buttons below): "Mule Network" and "Social
Engineering" swap the view in place to that case's real response (badged, with
Back). The device fan-out graph appears only in the Mule case.

## Notes

- Every launch uses fresh per-run ids, so the stateful backend (per-user baseline
  + global device graph) starts clean and reproduces the numbers above.
- Payloads/scenarios are in `src/narrative.js`; verified against the live API.
- The score collapse is two real actions (step-up → cooling-off); the backend has
  no standalone "new device" signal, so step 2's drop is driven by the new
  beneficiary (device shown as an accompanying risk factor).
