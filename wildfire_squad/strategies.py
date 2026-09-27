"""The three team strategies compared in the experiments.

* ``independent`` — no shared map, no Coordinator: each firefighter uses only what it sees.
* ``greedy``      — shared map; the Coordinator sends each zone its nearest free firefighter.
* ``auction``     — full system: shared map + threat-ordered auction + risk-aware A*.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    """Switches that distinguish the strategies."""

    name: str
    shared_map: bool  # firefighters read/write the scouts' shared belief map
    coordinator: bool  # a Coordinator clusters zones and assigns firefighters
    allocation: str | None  # "auction" | "greedy" | None


STRATEGIES: dict[str, Strategy] = {
    "independent": Strategy("independent", shared_map=False, coordinator=False, allocation=None),
    "greedy": Strategy("greedy", shared_map=True, coordinator=True, allocation="greedy"),
    "auction": Strategy("auction", shared_map=True, coordinator=True, allocation="auction"),
}


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
