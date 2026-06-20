"""Per-user behavioural profile for computing social-engineering signals.

Like the device graph, this is an ONLINE structure populated by scoring events.
For each user it tracks just enough state to derive the SE signal dict from a
raw transaction request:
  * a rolling amount baseline  -> amount_vs_user_baseline_ratio
  * the set of known payees + when each was first seen -> first_transfer_to_payee,
    payee_age_hours
  * recent transaction times   -> txn_velocity_short_window

All of this is derivable from data a bank already logs (transfer history, payee
registry, session timestamps). High-risk-geo, night-time and failed-attempt
counts come straight off the request (the bank computes them at the edge).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Countries treated as high-risk by default. In Dataset 2, NG was 100% fraud;
# real deployments would source this from a maintained risk list / sanctions feed.
DEFAULT_HIGH_RISK_COUNTRIES = {"NG"}
NIGHT_HOURS = set(range(0, 6))  # 00:00-05:59 local
VELOCITY_WINDOW_SECONDS = 600.0  # "short window" = 10 minutes


@dataclass
class _UserState:
    amount_sum: float = 0.0
    amount_count: int = 0
    payees_first_seen: dict = field(default_factory=dict)  # payee_id -> ts (seconds)
    recent_txn_times: list = field(default_factory=list)   # list of ts (seconds)

    @property
    def baseline(self) -> float:
        return self.amount_sum / self.amount_count if self.amount_count else 0.0


class UserProfileStore:
    """Tracks per-user baselines and payee history; derives SE signals."""

    def __init__(
        self,
        high_risk_countries: set[str] | None = None,
        velocity_window_seconds: float = VELOCITY_WINDOW_SECONDS,
    ):
        self.users: dict[str, _UserState] = {}
        self.high_risk_countries = set(high_risk_countries or DEFAULT_HIGH_RISK_COUNTRIES)
        self.velocity_window_seconds = velocity_window_seconds

    def compute_signals(
        self,
        user_id: str,
        amount: float,
        payee_id: str | None = None,
        timestamp: float = 0.0,
        country: str | None = None,
        hour: int | None = None,
        failed_attempts_before_success: int = 0,
        is_high_risk_geo: bool | None = None,
    ) -> dict:
        """Compute the SE signal dict for an incoming transaction.

        Reads the user's state as it was BEFORE this transaction (so a brand-new
        payee correctly reads as a first transfer), then `update()` should be
        called to fold the event in.
        """
        st = self.users.get(user_id, _UserState())

        # amount vs baseline (1.0 if no history yet -> neutral)
        baseline = st.baseline
        ratio = (amount / baseline) if baseline > 0 else 1.0

        # payee novelty / freshness
        if payee_id is None:
            first_transfer = False
            payee_age_hours = 1e9
        elif payee_id not in st.payees_first_seen:
            first_transfer = True
            payee_age_hours = 0.0
        else:
            first_transfer = False
            payee_age_hours = max(0.0, (timestamp - st.payees_first_seen[payee_id]) / 3600.0)

        # velocity in the short window (count of recent txns + this one)
        window_start = timestamp - self.velocity_window_seconds
        recent = [t for t in st.recent_txn_times if t >= window_start]
        velocity = len(recent) + 1

        # geo / night
        if is_high_risk_geo is None:
            is_high_risk_geo = bool(country) and country in self.high_risk_countries
        night = hour is not None and int(hour) in NIGHT_HOURS

        return {
            "first_transfer_to_payee": first_transfer,
            "payee_age_hours": payee_age_hours,
            "amount_vs_user_baseline_ratio": round(ratio, 3),
            "txn_velocity_short_window": velocity,
            "failed_attempts_before_success": int(failed_attempts_before_success),
            "is_new_high_risk_country_or_geo": bool(is_high_risk_geo),
            "night_time_transaction": bool(night),
        }

    def update(
        self,
        user_id: str,
        amount: float,
        payee_id: str | None = None,
        timestamp: float = 0.0,
    ) -> None:
        """Fold a transaction into the user's profile (call after compute_signals)."""
        st = self.users.setdefault(user_id, _UserState())
        st.amount_sum += float(amount)
        st.amount_count += 1
        if payee_id is not None and payee_id not in st.payees_first_seen:
            st.payees_first_seen[payee_id] = timestamp
        st.recent_txn_times.append(timestamp)
        # keep only the last 50 timestamps to bound memory
        st.recent_txn_times = st.recent_txn_times[-50:]

    def seed_baseline(self, user_id: str, amounts: list[float]) -> None:
        """Pre-load a user's normal spending so the first scored txn has a baseline."""
        st = self.users.setdefault(user_id, _UserState())
        for a in amounts:
            st.amount_sum += float(a)
            st.amount_count += 1

    # --- persistence ---
    def to_dict(self) -> dict:
        return {
            "high_risk_countries": sorted(self.high_risk_countries),
            "velocity_window_seconds": self.velocity_window_seconds,
            "users": {
                uid: {
                    "amount_sum": s.amount_sum,
                    "amount_count": s.amount_count,
                    "payees_first_seen": s.payees_first_seen,
                    "recent_txn_times": s.recent_txn_times,
                }
                for uid, s in self.users.items()
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UserProfileStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            high_risk_countries=set(data.get("high_risk_countries", [])),
            velocity_window_seconds=data.get("velocity_window_seconds", VELOCITY_WINDOW_SECONDS),
        )
        for uid, s in data.get("users", {}).items():
            obj.users[uid] = _UserState(
                amount_sum=s["amount_sum"],
                amount_count=s["amount_count"],
                payees_first_seen=s.get("payees_first_seen", {}),
                recent_txn_times=s.get("recent_txn_times", []),
            )
        return obj
