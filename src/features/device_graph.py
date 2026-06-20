"""Device-to-account graph for the mule-network layer.

Honest scope note
-----------------
The NeurIPS BAF dataset has NO raw `device_id` / `account_id` columns (only
aggregate device features like `device_distinct_emails_8w`). A literal
device->account graph therefore cannot be reconstructed from that dataset.

Instead this graph is an ONLINE structure: every `/api/v1/score` call carries a
`user_id` and a `device_id`, and we record that linkage. Over a session (or a
demo run) the graph accumulates, and a single device appearing across many
distinct user_ids in quick succession is the mule cash-out signature the
`GET /api/v1/device/{device_id}/risk` endpoint surfaces. This is exactly the
real-world signal a bank computes from its own session logs.

The graph is a bipartite networkx graph: account nodes and device nodes, edges
= "this account was seen on this device". It is persistable to JSON so the API
and the demo script can share state.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

# A device linked to >= this many distinct accounts is flagged as mule-risky.
# Rationale: a normal personal/household device serves 1-2 accounts; a value of
# 3+ distinct accounts on one device in a short window is anomalous for retail
# banking and is the cash-out-fan-in pattern of a mule network.
DEFAULT_RISK_ACCOUNT_THRESHOLD = 3


class DeviceAccountGraph:
    """Bipartite account<->device graph with mule-risk lookup."""

    ACCOUNT = "account"
    DEVICE = "device"

    def __init__(self, risk_threshold: int = DEFAULT_RISK_ACCOUNT_THRESHOLD):
        self.g = nx.Graph()
        self.risk_threshold = risk_threshold

    # --- mutation ---
    def add_event(self, account_id: str, device_id: str) -> None:
        """Record that `account_id` was seen on `device_id`."""
        acc = self._acc(account_id)
        dev = self._dev(device_id)
        if not self.g.has_node(acc):
            self.g.add_node(acc, kind=self.ACCOUNT, raw_id=str(account_id))
        if not self.g.has_node(dev):
            self.g.add_node(dev, kind=self.DEVICE, raw_id=str(device_id))
        if self.g.has_edge(acc, dev):
            self.g[acc][dev]["count"] += 1
        else:
            self.g.add_edge(acc, dev, count=1)

    # --- lookup ---
    def get_device_risk(self, device_id: str) -> dict:
        """Return mule-risk info for a device.

        Shape matches the build-prompt contract:
            {account_count, risk_flag, linked_accounts}
        """
        dev = self._dev(device_id)
        if not self.g.has_node(dev):
            return {
                "device_id": str(device_id),
                "account_count": 0,
                "risk_flag": False,
                "linked_accounts": [],
            }
        accounts = [self.g.nodes[n]["raw_id"] for n in self.g.neighbors(dev)]
        return {
            "device_id": str(device_id),
            "account_count": len(accounts),
            "risk_flag": len(accounts) >= self.risk_threshold,
            "linked_accounts": sorted(accounts),
        }

    def device_risk_score(self, device_id: str) -> float:
        """Normalised mule risk in [0,1] from the device's account fan-out.

        0 accounts -> 0.0; at/above threshold saturates toward 1.0.
        """
        info = self.get_device_risk(device_id)
        n = info["account_count"]
        if n <= 1:
            return 0.0
        # Linear ramp from 1 account (0.0) to threshold (1.0), then clamp.
        return float(min(1.0, (n - 1) / max(1, self.risk_threshold - 1)))

    # --- persistence ---
    def to_dict(self) -> dict:
        return {
            "risk_threshold": self.risk_threshold,
            "nodes": [
                {"id": n, **self.g.nodes[n]} for n in self.g.nodes
            ],
            "edges": [
                {"u": u, "v": v, "count": d.get("count", 1)}
                for u, v, d in self.g.edges(data=True)
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DeviceAccountGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(risk_threshold=data.get("risk_threshold", DEFAULT_RISK_ACCOUNT_THRESHOLD))
        for node in data["nodes"]:
            nid = node.pop("id")
            obj.g.add_node(nid, **node)
        for edge in data["edges"]:
            obj.g.add_edge(edge["u"], edge["v"], count=edge.get("count", 1))
        return obj

    # --- helpers ---
    def _acc(self, account_id: str) -> str:
        return f"acct::{account_id}"

    def _dev(self, device_id: str) -> str:
        return f"dev::{device_id}"
