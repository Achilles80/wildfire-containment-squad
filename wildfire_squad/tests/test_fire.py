"""Phase 1 — fire-spread model."""

from __future__ import annotations

import numpy as np
import pytest

from environment.cells import CellState
from environment.fire import ignition_probability, ignition_probability_map, spread_step, wind_vector
from model import WildfireModel


def test_downwind_and_upwind_factors(cfg):
    fire = cfg["fire"]  # k = 0.8, wind toward +x
    wind = wind_vector(0)
    base = fire["p_base"]
    downwind = ignition_probability((10, 10), (11, 10), 1.0, wind, fire)
    upwind = ignition_probability((10, 10), (9, 10), 1.0, wind, fire)
    crosswind = ignition_probability((10, 10), (10, 11), 1.0, wind, fire)
    assert downwind == pytest.approx(1.8 * base)
    assert upwind == pytest.approx(0.2 * base)
    assert crosswind == pytest.approx(base)


def test_fuel_scales_probability(cfg):
    fire = cfg["fire"]
    wind = wind_vector(90)
    dense = ignition_probability((5, 5), (5, 6), fire["fuel_dense"], wind, fire)
    sparse = ignition_probability((5, 5), (5, 6), fire["fuel_sparse"], wind, fire)
    assert sparse == pytest.approx(dense * fire["fuel_sparse"] / fire["fuel_dense"])


def test_probability_is_clipped(cfg):
    fire = {**cfg["fire"], "p_base": 0.9, "wind_k": 1.0}
    assert ignition_probability((0, 0), (1, 0), 1.0, wind_vector(0), fire) == 1.0


def test_independent_neighbours_combine(cfg):
    fire = {**cfg["fire"], "wind_k": 0.0}
    burning = np.zeros((5, 5), dtype=bool)
    burning[1, 2] = burning[3, 2] = True
    fuel = np.ones((5, 5))
    p = ignition_probability_map(burning, fuel, wind_vector(0), fire)
    assert p[2, 2] == pytest.approx(1 - (1 - fire["p_base"]) ** 2)


def test_non_tree_cells_never_ignite(cfg):
    fire = {**cfg["fire"], "p_base": 1.0}
    rng = np.random.default_rng(0)
    for s in (CellState.EMPTY, CellState.WATER, CellState.FIREBREAK, CellState.BURNT):
        state = np.full((5, 5), s, dtype=np.int8)
        state[2, 2] = CellState.BURNING
        timer = np.zeros((5, 5), dtype=np.int16)
        timer[2, 2] = fire["burn_duration"]
        fuel = np.ones((5, 5))
        for _ in range(3):
            spread_step(state, fuel, timer, wind_vector(0), fire, rng)
        assert not ((state == CellState.BURNING).sum() > 1)
        assert (state[np.arange(5) != 2] == s).all()


def test_burning_becomes_burnt_after_burn_duration(cfg):
    fire = cfg["fire"]
    state = np.full((3, 3), CellState.EMPTY, dtype=np.int8)
    state[1, 1] = CellState.BURNING
    timer = np.zeros((3, 3), dtype=np.int16)
    timer[1, 1] = fire["burn_duration"]
    rng = np.random.default_rng(0)
    for _ in range(fire["burn_duration"] - 1):
        spread_step(state, np.zeros((3, 3)), timer, wind_vector(0), fire, rng)
        assert state[1, 1] == CellState.BURNING
    spread_step(state, np.zeros((3, 3)), timer, wind_vector(0), fire, rng)
    assert state[1, 1] == CellState.BURNT


def test_same_seed_gives_identical_fire():
    overrides = {"agents": {"n_scouts": 0, "n_firefighters": 0}, "simulation": {"strategy": "independent"}}
    a, b = WildfireModel(overrides, seed=7), WildfireModel(overrides, seed=7)
    for _ in range(40):
        a.step()
        b.step()
    assert np.array_equal(a.state, b.state)
    c = WildfireModel(overrides, seed=8)
    for _ in range(40):
        c.step()
    assert not np.array_equal(a.state, c.state)


def test_ignitions_respect_base_distance(cfg):
    m = WildfireModel({"fire": {"ignitions": 3}}, seed=3)
    base = m.base
    for x, y in m.ignitions:
        assert max(abs(x - base[0]), abs(y - base[1])) >= cfg["fire"]["ignition_min_base_distance"]


def test_river_preset_layout(cfg):
    m = WildfireModel({"simulation": {"preset": "river"}}, seed=1)
    river = cfg["presets"]["river"]
    col = m.state[river["x"]]
    for y0, y1 in river["crossings"]:
        assert (col[y0 : y1 + 1] != CellState.WATER).all()
    crossing_rows = {y for y0, y1 in river["crossings"] for y in range(y0, y1 + 1)}
    assert all(col[y] == CellState.WATER for y in range(m.height) if y not in crossing_rows)
    assert all(x > river["x"] for x, _ in m.ignitions)  # far side from the base at (2, 2)
