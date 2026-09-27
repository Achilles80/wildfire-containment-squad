"""Stochastic, wind- and fuel-driven fire spread.

A Tree cell ``n`` next to a Burning cell ``b`` ignites with probability::

    P(ignite from b) = p_base × fuel[n] × (1 + k × cos θ)

where θ is the angle between the wind and the spread direction ``n − b``. Several burning
neighbours act as independent chances: ``P(ignite) = 1 − Π_b (1 − P(ignite from b))``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from environment.cells import NEIGHBOURS_4, NEIGHBOURS_8, Cell, CellState

Vector = tuple[float, float]


def wind_vector(direction_deg: float) -> Vector:
    """Unit vector of the wind; 0° blows toward +x (east), 90° toward +y."""
    rad = math.radians(direction_deg)
    return (math.cos(rad), math.sin(rad))


def spread_offsets(fire_cfg: dict[str, Any]) -> tuple[Cell, ...]:
    """Neighbour offsets fire can spread along (4- or 8-neighbourhood)."""
    return NEIGHBOURS_8 if fire_cfg["spread_neighbourhood"] == 8 else NEIGHBOURS_4


def _raw_factor(dx: int, dy: int, wind: Vector, fire_cfg: dict[str, Any]) -> float:
    """``p_base × (1 + k cos θ)`` for spread direction ``(dx, dy)``, before fuel and clipping."""
    norm = math.hypot(dx, dy)
    cos_theta = (dx * wind[0] + dy * wind[1]) / norm
    return fire_cfg["p_base"] * (1.0 + fire_cfg["wind_k"] * cos_theta)


def ignition_probability(b: Cell, n: Cell, fuel: float, wind: Vector, cfg: dict[str, Any]) -> float:
    """Probability that burning cell ``b`` ignites neighbour ``n`` (with fuel ``fuel``) this step.

    ``cfg`` is the ``fire`` section of the config. The result is clipped to [0, 1].
    """
    dx, dy = n[0] - b[0], n[1] - b[1]
    if dx == 0 and dy == 0:
        return 0.0
    p = _raw_factor(dx, dy, wind, cfg) * fuel
    return min(1.0, max(0.0, p))


def _shift(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Return ``out`` with ``out[x, y] = arr[x - dx, y - dy]`` (zero outside the grid)."""
    out = np.zeros_like(arr)
    w, h = arr.shape
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    out[xs_dst, ys_dst] = arr[xs_src, ys_src]
    return out


def ignition_probability_map(
    burning: np.ndarray, fuel: np.ndarray, wind: Vector, fire_cfg: dict[str, Any]
) -> np.ndarray:
    """Per-cell probability of igniting next step, combining all burning neighbours."""
    p_not = np.ones(burning.shape, dtype=float)
    for dx, dy in spread_offsets(fire_cfg):
        src = _shift(burning, dx, dy)
        if not src.any():
            continue
        p = np.clip(_raw_factor(dx, dy, wind, fire_cfg) * fuel, 0.0, 1.0)
        p_not *= np.where(src, 1.0 - p, 1.0)
    return 1.0 - p_not


def risk_map(burning: np.ndarray, fuel: np.ndarray, wind: Vector, fire_cfg: dict[str, Any]) -> np.ndarray:
    """``risk(n) = max over burning neighbours b of ignition_probability(b, n)`` (0 if none).

    Used for the firefighter's A* step cost and the Coordinator's threat score.
    """
    risk = np.zeros(burning.shape, dtype=float)
    for dx, dy in spread_offsets(fire_cfg):
        src = _shift(burning, dx, dy)
        if not src.any():
            continue
        p = np.clip(_raw_factor(dx, dy, wind, fire_cfg) * fuel, 0.0, 1.0)
        np.maximum(risk, np.where(src, p, 0.0), out=risk)
    return risk


def spread_step(
    state: np.ndarray,
    fuel: np.ndarray,
    burn_timer: np.ndarray,
    wind: Vector,
    fire_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    """Advance the fire one step in place. Returns the number of newly ignited cells.

    A full grid of random numbers is drawn every step, so the random stream depends only on
    the step count and runs with the same seed stay comparable across strategies.
    """
    burning = state == CellState.BURNING
    p_ignite = ignition_probability_map(burning, fuel, wind, fire_cfg)
    draws = rng.random(state.shape)
    ignite = (state == CellState.TREE) & (draws < p_ignite)

    burn_timer[burning] -= 1
    burnt_out = burning & (burn_timer <= 0)
    state[burnt_out] = CellState.BURNT
    burn_timer[burnt_out] = 0

    state[ignite] = CellState.BURNING
    burn_timer[ignite] = fire_cfg["burn_duration"]
    return int(ignite.sum())


def ignite(state: np.ndarray, burn_timer: np.ndarray, cells: list[Cell], duration: int) -> None:
    """Set the given cells burning with a full burn timer."""
    for x, y in cells:
        state[x, y] = CellState.BURNING
        burn_timer[x, y] = duration
