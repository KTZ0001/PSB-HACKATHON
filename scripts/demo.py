"""Aegis live demo — the scriptable 'trust score collapse' moment.

Drives a sequence of REAL /api/v1/score calls that tell the Aegis story:
  1. Normal session, normal device           -> high trust, allow
  2. Same user, NEW device + geo + recovery   -> trust collapses, account_takeover
  3. One device across many accounts          -> mule_network + graph cluster
  4. Genuine owner, clean device/geo, but a
     coached first transfer to a new payee    -> social_engineering (the blind
                                                 spot device/geo scoring misses)

This is the safety net: if no UI is ready, this proves the whole thesis through
the real API. It talks HTTP (stdlib urllib only) and renders with `rich`.

Usage:
    python scripts/demo.py                      # auto-launches the API, runs, stops it
    python scripts/demo.py --base-url URL       # use an already-running API
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# --- tiny HTTP client (stdlib only) ------------------------------------------

def _post(base: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read().decode())


def _health_ok(base: str) -> bool:
    try:
        return _get(base, "/api/v1/health").get("status") == "ok"
    except Exception:
        return False


# --- server lifecycle --------------------------------------------------------

def ensure_server(base: str, port: int):
    """Return a Popen if we launched the server, else None (already running)."""
    if _health_ok(base):
        console.print(f"[dim]Using already-running API at {base}[/dim]")
        return None
    console.print(f"[dim]Launching API on port {port} ...[/dim]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "error"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    for _ in range(40):  # up to ~20s for model load
        if _health_ok(base):
            console.print("[green]API ready.[/green]\n")
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("API did not become healthy in time.")


# --- rendering ---------------------------------------------------------------

def _trust_style(t: float) -> str:
    if t >= 70:
        return "bold green"
    if t >= 35:
        return "bold yellow"
    if t >= 15:
        return "bold dark_orange"
    return "bold red"


def render(step_title: str, resp: dict) -> None:
    t = resp["trust_score"]
    b = resp["score_breakdown"]
    style = _trust_style(t)

    head = Text()
    head.append(f"TRUST {t:.1f}/100", style=style)
    head.append(f"   →  {resp['predicted_type'].upper()}", style="bold")
    head.append(f"   ·  action: {resp['recommended_action']}", style="cyan")

    body = Text()
    body.append("Risk breakdown   ", style="dim")
    body.append(f"behavioural={b['behavioral']:.0f}  ", style="white")
    body.append(f"mule_graph={b['mule_graph']:.0f}  ", style="white")
    body.append(f"social_eng={b['social_engineering']:.0f}\n\n", style="white")
    body.append(resp["explanation"], style="italic")

    console.print(Panel(body, title=f"[bold]{step_title}[/bold]", subtitle=head,
                        border_style=style.split()[-1]))


def render_device_cluster(base: str, device_id: str) -> None:
    info = _get(base, f"/api/v1/device/{device_id}/risk")
    table = Table(title=f"Mule device-graph: {device_id}", title_style="bold red")
    table.add_column("linked accounts", style="red")
    table.add_column("account count", justify="right")
    table.add_column("mule flag", justify="center")
    table.add_row(", ".join(info["linked_accounts"]),
                  str(info["linked_account_count"]),
                  "⚠ YES" if info["risk_flag"] else "no")
    console.print(table)


# --- the story ---------------------------------------------------------------

def run_story(base: str) -> None:
    console.rule("[bold cyan]AEGIS — continuous identity-trust scoring[/bold cyan]")
    console.print("One engine. Three disguises the same fraud wears.\n")

    # warm up Daniel's normal behaviour (baseline + known payee), silently.
    for _ in range(3):
        _post(base, "/api/v1/score", {
            "user_id": "daniel", "device_id": "daniel_iphone",
            "amount": 90.0, "payee_id": "daniel_landlord", "country": "UK",
            "hour": 9, "timestamp": 0.0,
        })

    console.rule("[1] Normal session")
    r1 = _post(base, "/api/v1/score", {
        "user_id": "daniel", "device_id": "daniel_iphone", "amount": 95.0,
        "payee_id": "daniel_landlord", "country": "UK", "hour": 9, "timestamp": 600000.0,
    })
    render("1 · Daniel pays rent from his usual phone", r1)

    console.rule("[2] Same user — new device, new geo, recovery abuse")
    # The STORY here is the hijacked session, not the money movement: the payment
    # goes to Daniel's known payee at a modest amount, so the social-engineering
    # layer stays quiet and the behavioural/session anomaly is what drives it.
    r2 = _post(base, "/api/v1/score", {
        "user_id": "daniel", "device_id": "BRAND_NEW_DEVICE", "amount": 120.0,
        "payee_id": "daniel_landlord", "country": "RU", "hour": 3, "timestamp": 600600.0,
        "is_high_risk_geo": True,
        "behavioral_features": {
            "income": 0.1, "name_email_similarity": 0.03, "email_is_free": 1,
            "credit_risk_score": 360, "bank_months_count": -1,
            "phone_home_valid": 0, "phone_mobile_valid": 0, "foreign_request": 1,
            "proposed_credit_limit": 2000, "customer_age": 55,
            "days_since_request": 0.001, "keep_alive_session": 0, "has_other_cards": 0,
        },
    })
    render("2 · Account takeover — the session itself is anomalous", r2)
    delta = r1["trust_score"] - r2["trust_score"]
    console.print(f"   [bold red]↓ trust collapsed {delta:.0f} points[/bold red] "
                  f"({r1['trust_score']:.0f} → {r2['trust_score']:.0f})\n")

    console.rule("[3] Coordinated network — one device, many accounts")
    last = None
    for i in range(4):
        last = _post(base, "/api/v1/score", {
            "user_id": f"cashout_{i}", "device_id": "MULE_DEVICE_9", "amount": 480.0,
            "payee_id": f"drop_{i}", "country": "UK", "hour": 2, "timestamp": 700000.0 + i,
        })
    render("3 · Mule network — caught by the device-to-account graph", last)
    render_device_cluster(base, "MULE_DEVICE_9")
    console.print()

    console.rule("[4] Willing victim — the blind spot every other system misses")
    # Build the genuine owner's normal baseline first.
    for _ in range(3):
        _post(base, "/api/v1/score", {
            "user_id": "grace", "device_id": "grace_phone", "amount": 100.0,
            "payee_id": "grace_groceries", "country": "UK", "hour": 18, "timestamp": 0.0,
        })
    r4 = _post(base, "/api/v1/score", {
        "user_id": "grace", "device_id": "grace_phone",   # HER OWN phone
        "amount": 700.0,                                   # 7x her baseline
        "payee_id": "scammer_account",                     # never seen
        "country": "UK", "hour": 16,                        # domestic, daytime
        "failed_attempts": 2, "timestamp": 1000.0,
    })
    render("4 · Social engineering — clean device, clean geo, still flagged", r4)
    console.print(
        "   [yellow]Device fingerprint, geo and IP all look perfectly normal — "
        "the genuine owner authorised it.[/yellow]\n"
        "   [bold]Only the shape of the money movement gives it away.[/bold]\n"
    )

    console.rule("[bold cyan]Demo complete[/bold cyan]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aegis live demo via the real API.")
    parser.add_argument("--base-url", default=None, help="Target an already-running API.")
    parser.add_argument("--port", type=int, default=8000, help="Port to auto-launch on.")
    args = parser.parse_args()

    base = args.base_url or f"http://127.0.0.1:{args.port}"
    proc = None
    try:
        if args.base_url:
            if not _health_ok(base):
                console.print(f"[red]No healthy API at {base}.[/red]")
                return 1
        else:
            proc = ensure_server(base, args.port)
        run_story(base)
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            console.print("[dim]API server stopped.[/dim]")


if __name__ == "__main__":
    raise SystemExit(main())
