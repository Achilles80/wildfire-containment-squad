"""The six Review 1 test scenarios, expressed as config overrides."""

from __future__ import annotations

from typing import Any

from settings import deep_merge

# name -> (scenario number, what it tests, config overrides)
SCENARIOS: dict[str, tuple[int, str, dict[str, Any]]] = {
    "calm_single": (1, "Calm wind, 1 ignition — baseline behaviour", {"fire": {"wind_k": 0.0, "ignitions": 1}}),
    "windy_single": (
        2,
        "Strong wind, 1 ignition — directional spread, firebreaks",
        {"fire": {"wind_k": 0.8, "ignitions": 1}},
    ),
    "multi_ignition": (
        3,
        "3 ignitions ≥ 15 cells apart — task allocation across zones",
        {"fire": {"wind_k": 0.4, "ignitions": 3, "ignition_min_separation": 15}},
    ),
    "no_scouts": (4, "Scouts disabled — value of shared perception", {"agents": {"n_scouts": 0}}),
    "scaling_2": (5, "2 firefighters — scalability", {"agents": {"n_firefighters": 2}}),
    "scaling_5": (5, "5 firefighters — scalability", {"agents": {"n_firefighters": 5}}),
    "scaling_10": (5, "10 firefighters — scalability", {"agents": {"n_firefighters": 10}}),
    "river": (6, "River with two narrow crossings — path conflicts, re-planning", {"simulation": {"preset": "river"}}),
}

GROUPS: dict[str, list[str]] = {"scaling": ["scaling_2", "scaling_5", "scaling_10"]}


def scenario_overrides(name: str, strategy: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Config overrides for a scenario, optionally with a strategy and further overrides."""
    if name == "default":
        overrides: dict[str, Any] = {}
    else:
        overrides = SCENARIOS[name][2]
    if strategy is not None:
        overrides = deep_merge(overrides, {"simulation": {"strategy": strategy}})
    return deep_merge(overrides, extra)


def expand(names: list[str]) -> list[str]:
    """Expand ``all`` and group names (``scaling``) into individual scenario names."""
    out: list[str] = []
    for name in names:
        if name == "all":
            out.extend(SCENARIOS)
        elif name in GROUPS:
            out.extend(GROUPS[name])
        elif name in SCENARIOS:
            out.append(name)
        else:
            raise ValueError(f"unknown scenario {name!r}; choose from all, {', '.join([*SCENARIOS, *GROUPS])}")
    return list(dict.fromkeys(out))
