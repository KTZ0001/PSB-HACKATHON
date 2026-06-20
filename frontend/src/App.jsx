import { useState, useEffect, useRef } from "react";
import {
  buildNarrative, makeSuffix, postScore,
  buildMuleCase, buildSocialCase, fetchDeviceRisk,
} from "./narrative.js";

// --- trust tier helpers (thresholds match backend _trust_style) -------------
function tierLabel(score) {
  if (score >= 70) return "Trusted";
  if (score >= 35) return "Elevated risk";
  if (score >= 15) return "High risk";
  return "Critical";
}
// Idle flicker stays inside the current tier band so the color never jitters.
function tierBand(score) {
  if (score >= 70) return [70, 100];
  if (score >= 35) return [35, 70];
  if (score >= 15) return [15, 35];
  return [0, 15];
}
// Concrete hex per tier for interpolating the color during the tween.
function tierHex(score) {
  if (score >= 70) return "#02c39a";
  if (score >= 35) return "#f4a535";
  if (score >= 15) return "#e8743b";
  return "#c0504d";
}
function parseColor(c) {
  if (c[0] === "#") {
    const n = parseInt(c.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  return c.match(/\d+/g).map(Number);
}
function lerpColor(a, b, t) {
  const [r1, g1, b1] = parseColor(a);
  const [r2, g2, b2] = parseColor(b);
  return `rgb(${Math.round(r1 + (r2 - r1) * t)}, ${Math.round(g1 + (g2 - g1) * t)}, ${Math.round(b1 + (b2 - b1) * t)})`;
}
const easeInOutCubic = (p) =>
  p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;

// Turn the real recommended_action + action_detail into the adaptive
// intervention readout. Lines are derived only from what the API returns.
const ACTION_TITLES = {
  allow: "Allow transaction",
  step_up_auth: "Step-up authentication",
  cooling_off: "Cooling-off hold",
  block_and_flag_analyst: "Block & flag to analyst",
};
function actionReadout(result) {
  const a = result.recommended_action;
  const d = result.action_detail || {};
  const lines = [];
  if (a === "block_and_flag_analyst") lines.push("Transaction blocked");
  if (d.delay_minutes) lines.push(`Cooling period · ${d.delay_minutes} minutes`);
  if (a === "step_up_auth") lines.push(`Customer verification · ${(d.method || "otp").toUpperCase()}`);
  if (d.alert_analyst) lines.push("Analyst notified");
  return { title: ACTION_TITLES[a] || a.replace(/_/g, " "), lines };
}

export default function App() {
  const [screen, setScreen] = useState("home");

  if (screen === "home") {
    return (
      <div className="home">
        <h1 className="brand">AEGIS</h1>
        <p className="tagline">Continuous Identity Trust Infrastructure</p>
        <button className="launch-btn" onClick={() => setScreen("main")}>
          Launch Simulation
        </button>
      </div>
    );
  }

  return <MainScreen />;
}

// STEP 2: the live 4-step Rajesh narrative driven by Next Event.
function MainScreen() {
  // Build a hermetic, per-run narrative once on mount.
  const narrativeRef = useRef(null);
  if (!narrativeRef.current) narrativeRef.current = buildNarrative(makeSuffix());
  const { prime, steps } = narrativeRef.current;

  const [stepIndex, setStepIndex] = useState(0);
  const [result, setResult] = useState(null); // latest /score response
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [caseView, setCaseView] = useState(null); // secondary mule/social case
  const startedRef = useRef(false);

  // --- animated score display (tween between steps + idle flicker) ---
  const baseScore = result ? result.trust_score : 94;
  const [displayScore, setDisplayScore] = useState(94);
  const [displayColor, setDisplayColor] = useState(tierHex(94));
  const displayScoreRef = useRef(94);
  const displayColorRef = useRef(tierHex(94));
  const pausedUntil = useRef(0);
  const rafRef = useRef(0);
  const animToken = useRef(0);

  const setScore = (v) => { displayScoreRef.current = v; setDisplayScore(v); };
  const setColor = (c) => { displayColorRef.current = c; setDisplayColor(c); };

  // Idle "continuously monitoring" flicker within the current tier band.
  useEffect(() => {
    const id = setInterval(() => {
      if (Date.now() < pausedUntil.current) return;
      const [lo, hi] = tierBand(baseScore);
      const nudged = baseScore + (Math.random() * 6 - 3);
      setScore(Math.max(0, Math.min(100, Math.max(lo + 0.5, Math.min(hi - 0.5, nudged)))));
    }, 2500);
    return () => clearInterval(id);
  }, [baseScore]);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  // Tween the number + crossfade the tier color to each new real score.
  function animateTo(target) {
    cancelAnimationFrame(rafRef.current);
    const token = ++animToken.current;
    const startScore = displayScoreRef.current;
    const startColor = displayColorRef.current;
    const endColor = tierHex(target);
    const duration = 1300;
    const t0 = performance.now();
    const tick = (now) => {
      if (animToken.current !== token) return;
      const p = Math.min(1, (now - t0) / duration);
      const e = easeInOutCubic(p);
      setScore(startScore + (target - startScore) * e);
      setColor(lerpColor(startColor, endColor, e));
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    // Safety net for throttled rAF (hidden/background tabs).
    setTimeout(() => {
      if (animToken.current === token) { setScore(target); setColor(endColor); }
    }, duration + 250);
  }

  // Tween whenever a new real result lands.
  useEffect(() => {
    if (!result) return;
    pausedUntil.current = Date.now() + 3000;
    animateTo(result.trust_score);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  // Prime the baseline, then score event 1, once.
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    (async () => {
      setBusy(true);
      try {
        for (let i = 0; i < 3; i++) await postScore(prime);
        const r = await postScore(steps[0].payload);
        setResult(r);
      } catch {
        setError("Could not reach the Aegis API on :8000.");
      } finally {
        setBusy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function nextEvent() {
    if (busy || stepIndex >= steps.length - 1) return;
    const next = stepIndex + 1;
    setBusy(true);
    setError(null);
    try {
      const r = await postScore(steps[next].payload);
      setResult(r);
      setStepIndex(next);
    } catch {
      setError("Could not reach the Aegis API on :8000.");
    } finally {
      setBusy(false);
    }
  }

  async function runMuleCase() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const mc = buildMuleCase(makeSuffix());
      let last;
      for (const c of mc.calls) last = await postScore(c);
      const risk = await fetchDeviceRisk(mc.deviceId);
      setCaseView({
        kind: "mule",
        label: "Mule Network Cluster",
        result: last,
        accounts: risk.linked_accounts,
        meta: { Customer: "4 cash-out accounts", Device: "Shared device", Pattern: "Coordinated fan-out" },
      });
    } catch {
      setError("Could not reach the Aegis API on :8000.");
    } finally {
      setBusy(false);
    }
  }

  async function runSocialCase() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const sc = buildSocialCase(makeSuffix());
      for (let i = 0; i < 3; i++) await postScore(sc.prime);
      const last = await postScore(sc.score);
      setCaseView({
        kind: "social",
        label: "Social Engineering",
        result: last,
        meta: { Customer: "Priya (genuine owner)", Device: "Known — clean", Beneficiary: "New — coached transfer" },
      });
    } catch {
      setError("Could not reach the Aegis API on :8000.");
    } finally {
      setBusy(false);
    }
  }

  // A secondary case temporarily replaces the narrative view.
  if (caseView) {
    return (
      <CaseView
        view={caseView}
        onBack={() => setCaseView(null)}
      />
    );
  }

  const step = steps[stepIndex];
  const isFinal = stepIndex >= steps.length - 1;

  return (
    <div className="main">
      <header className="main-header">
        <span className="logo">AEGIS</span>
        <span className="sub">Live Session Monitor · Rajesh Patel</span>
      </header>

      <div className="panels">
        {/* ---- left: customer session ---- */}
        <section className="card">
          <div className="panel-label">Customer Session</div>
          <div className="cust-name">Rajesh Patel</div>
          <div className="cust-meta">
            <div className="meta-row"><span className="k">Device</span><span className="v">{step.customer.device}</span></div>
            <div className="meta-row"><span className="k">Location</span><span className="v">{step.customer.location}</span></div>
            <div className="meta-row"><span className="k">Beneficiary</span><span className="v">{step.customer.beneficiary}</span></div>
          </div>
          <div className="panel-label risk-title">Risk Factors</div>
          {step.riskFactors.length === 0 ? (
            <div className="risk-empty">None — session matches baseline.</div>
          ) : (
            <div className="risk-list">
              {step.riskFactors.map((rf) => (
                <div key={rf} className="risk-item"><span className="dot" />{rf}</div>
              ))}
            </div>
          )}
        </section>

        {/* ---- center: trust score ---- */}
        <section className="card score-panel">
          <div className="panel-label">Trust Score</div>
          <div className="score-value" style={{ color: displayColor }}>{Math.round(displayScore)}</div>
          <div className="score-tier" style={{ color: displayColor }}>{tierLabel(baseScore)}</div>
          <div className="score-cta">
            {isFinal && result ? (
              (() => {
                const { title, lines } = actionReadout(result);
                return (
                  <div className="action-block action-enter" key={stepIndex}>
                    <div className="ab-label">Action Selected</div>
                    <div className="action-headline">{title}</div>
                    {lines.map((l) => (
                      <div key={l} className="action-line"><span className="tick">✓</span>{l}</div>
                    ))}
                  </div>
                );
              })()
            ) : (
              <button className="next-btn" onClick={nextEvent} disabled={busy || !result}>
                {busy ? "Scoring…" : "Next Event ▸"}
              </button>
            )}
          </div>
        </section>

        {/* ---- right: AI explanation ---- */}
        <section className="card">
          <div className="panel-label">AI Explanation</div>
          <p className="explanation">
            {error ? error : result ? result.explanation : "Establishing baseline…"}
          </p>
        </section>
      </div>

      {/* ---- secondary case buttons ---- */}
      <div className="secondary-row">
        <span className="sec-hint">Other detected disguises:</span>
        <button className="sec-btn" onClick={runMuleCase} disabled={busy}>Show: Mule Network Case</button>
        <button className="sec-btn" onClick={runSocialCase} disabled={busy}>Show: Social Engineering Case</button>
      </div>
    </div>
  );
}

// Secondary "proof" view: one case's real response in the same 3-panel frame,
// clearly badged, with a way back. The device graph appears only for the mule
// case (it has nothing to show otherwise).
function CaseView({ view, onBack }) {
  const { result } = view;
  const color = tierHex(result.trust_score);
  return (
    <div className="main">
      <header className="main-header">
        <span className="logo">AEGIS</span>
        <span className="sub">Secondary case</span>
        <span className="case-badge">Case · {view.label}</span>
      </header>

      <div className="panels">
        <section className="card">
          <div className="panel-label">Case Context</div>
          <div className="cust-meta">
            {Object.entries(view.meta).map(([k, v]) => (
              <div key={k} className="meta-row"><span className="k">{k}</span><span className="v">{v}</span></div>
            ))}
          </div>
        </section>

        <section className="card score-panel">
          <div className="panel-label">Trust Score</div>
          <div className="score-value" style={{ color }}>{Math.round(result.trust_score)}</div>
          <div className="score-tier" style={{ color }}>{tierLabel(result.trust_score)}</div>
          {view.kind === "mule" && (
            <div className="graph-wrap">
              <DeviceGraph accounts={view.accounts} />
            </div>
          )}
        </section>

        <section className="card">
          <div className="panel-label">AI Explanation</div>
          <p className="explanation">{result.explanation}</p>
          <div className="expl-meta">
            Predicted: {result.predicted_type.replace(/_/g, " ")} · Action:{" "}
            {result.recommended_action.replace(/_/g, " ")}
          </div>
        </section>
      </div>

      <div className="secondary-row">
        <button className="back-btn" onClick={onBack}>← Back to main demo</button>
      </div>
    </div>
  );
}

// One device fanning out to its real linked accounts (mule case only).
const DEVICE = { x: 175, y: 75 };
function accountPositions(n) {
  if (n <= 1) return [{ x: 470, y: 75 }];
  const top = 24, bottom = 126;
  return Array.from({ length: n }, (_, i) => {
    const f = i / (n - 1);
    return { x: 470 + Math.abs(f - 0.5) * 60, y: top + (bottom - top) * f };
  });
}
// Strip the per-run suffix so labels read cleanly (cashout_0_1718_42 -> cashout_0).
const shortLabel = (id) => id.split("_").slice(0, 2).join("_");

function DeviceGraph({ accounts }) {
  const list = (accounts || []).slice(0, 4);
  const positions = accountPositions(list.length);
  const accent = "var(--orange)";
  return (
    <svg width="100%" height="150" viewBox="0 0 700 150">
      {positions.map((p, i) => (
        <line
          key={`e${i}`}
          x1={DEVICE.x} y1={DEVICE.y} x2={p.x} y2={p.y}
          stroke={accent} strokeWidth="2"
          className={i > 0 ? "edge-enter" : undefined}
          style={i > 0 ? { animationDelay: `${0.15 * i}s` } : undefined}
        />
      ))}
      <g>
        <circle className="device-node" cx={DEVICE.x} cy={DEVICE.y} r="34" fill="var(--red)" />
        <text x={DEVICE.x} y={DEVICE.y + 5} textAnchor="middle" fill="#fff" fontSize="13" fontWeight="700">DEVICE</text>
      </g>
      {positions.map((p, i) => (
        <g
          key={`n${i}`}
          className={i > 0 ? "node-enter" : undefined}
          style={i > 0 ? { animationDelay: `${0.15 * i}s` } : undefined}
        >
          <circle cx={p.x} cy={p.y} r="28" fill="#fff" stroke={accent} strokeWidth="2" />
          <text x={p.x} y={p.y + 4} textAnchor="middle" fill={accent} fontSize="10" fontWeight="700">
            {shortLabel(list[i])}
          </text>
        </g>
      ))}
    </svg>
  );
}
