"""Pluggable explanation-generator interface (Phase 3).

The default implementation (TemplatedExplanationGenerator) is rule-based,
deterministic, and makes ZERO network calls and needs ZERO API keys. A future
LLM-backed implementation can subclass this same interface without touching the
scoring engine or the API — that is the only extension point, and it is opt-in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ExplanationGenerator(ABC):
    @abstractmethod
    def explain(self, features: dict, score_breakdown: dict) -> str:
        """Return a human-readable, audit-log-style explanation of the decision.

        `features` is the full context dict (raw signals + combined result).
        `score_breakdown` is {behavioral, mule_graph, social_engineering} risk.
        """
        ...
