// The Rajesh Patel narrative: one customer, one escalating session, four real
// /api/v1/score calls. Payloads are tuned to produce a monotonic trust collapse
// ending on a real adaptive intervention (cooling_off + analyst alert). The exact
// numbers depend on the loaded model bundle; nothing is hardcoded in the UI —
// scores/explanations/actions come from the live responses to these payloads.
//
// All ids are suffixed per run so the stateful backend (per-user baseline +
// global device graph) starts clean on every launch and reproduces the numbers.

// API base. Empty by default so the browser talks same-origin and Vite's dev
// proxy (vite.config.js) forwards /api -> :8000. For a built/previewed bundle
// pointed at a real backend, set VITE_API_BASE (e.g. "http://127.0.0.1:8000").
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function makeSuffix() {
  return `${Date.now()}_${Math.floor(Math.random() * 1000)}`;
}

export function buildNarrative(suffix) {
  const user = `rajesh_${suffix}`;
  const phone = `rajesh_phone_${suffix}`;
  const laptop = `rajesh_laptop_${suffix}`;
  const payee = `rajesh_landlord_${suffix}`;
  const scam = `new_payee_${suffix}`;

  // 3x to establish the customer's normal baseline before the first scored event.
  const prime = {
    user_id: user, device_id: phone, amount: 8000.0,
    payee_id: payee, country: "IN", hour: 10, timestamp: 0.0,
  };

  const steps = [
    {
      title: "Baseline session",
      customer: { device: "Known — iPhone", location: "Ahmedabad", beneficiary: "Existing" },
      riskFactors: [],
      payload: {
        user_id: user, device_id: phone, amount: 8200.0,
        payee_id: payee, country: "IN", hour: 10, timestamp: 600000.0,
      },
    },
    {
      title: "New device + new beneficiary",
      customer: { device: "Unrecognised laptop", location: "Ahmedabad", beneficiary: "New — unverified" },
      riskFactors: ["New device detected", "New beneficiary added"],
      payload: {
        user_id: user, device_id: laptop, amount: 8200.0,
        payee_id: scam, country: "IN", hour: 10, timestamp: 600100.0,
      },
    },
    {
      title: "High-value transfer",
      customer: { device: "Unrecognised laptop", location: "Ahmedabad", beneficiary: "New — unverified" },
      riskFactors: ["New device detected", "New beneficiary added", "High-value transfer · ₹1,50,000", "Unusual hour · 02:00"],
      payload: {
        user_id: user, device_id: laptop, amount: 150000.0,
        payee_id: scam, country: "IN", hour: 2, timestamp: 600200.0,
      },
    },
    {
      title: "Login from new location",
      customer: { device: "Unrecognised laptop", location: "Unknown — high-risk", beneficiary: "New — unverified" },
      riskFactors: ["New device detected", "New beneficiary added", "High-value transfer · ₹1,50,000", "Unusual hour · 02:00", "New location · high-risk geo", "Failed authentication · ×3"],
      payload: {
        user_id: user, device_id: laptop, amount: 150000.0,
        payee_id: scam, country: "RU", hour: 2, failed_attempts: 3,
        is_high_risk_geo: true, timestamp: 600300.0,
      },
    },
  ];

  return { prime, steps };
}

export async function postScore(payload) {
  const res = await fetch(`${API_BASE}/api/v1/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`score ${res.status}`);
  return res.json();
}

export async function fetchDeviceRisk(deviceId) {
  const res = await fetch(`${API_BASE}/api/v1/device/${encodeURIComponent(deviceId)}/risk`);
  if (!res.ok) throw new Error(`device risk ${res.status}`);
  return res.json();
}

// --- secondary "proof" cases (verified: mule -> block, social -> cooling_off) ---

export function buildMuleCase(suffix) {
  const deviceId = `mule_dev_${suffix}`;
  const calls = [0, 1, 2, 3].map((i) => ({
    user_id: `cashout_${i}_${suffix}`, device_id: deviceId, amount: 480.0,
    payee_id: `drop_${i}_${suffix}`, country: "IN", hour: 2, timestamp: 700000.0 + i,
  }));
  return { deviceId, calls };
}

export function buildSocialCase(suffix) {
  const user = `priya_${suffix}`;
  const device = `priya_phone_${suffix}`;
  const prime = {
    user_id: user, device_id: device, amount: 5000.0,
    payee_id: `priya_known_${suffix}`, country: "IN", hour: 14, timestamp: 0.0,
  };
  const score = {
    user_id: user, device_id: device, amount: 90000.0,
    payee_id: `scam_${suffix}`, country: "IN", hour: 15, failed_attempts: 2, timestamp: 1000.0,
  };
  return { prime, score };
}
