"""Phase 2 — risk-aware A* and the UCS baseline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from algorithms.astar import astar, goal_cells, plan_path, step_cost_grid, ucs
from environment.cells import CellState
from environment.fire import risk_map, wind_vector


def random_grid(seed: int, size: int = 20, block: float = 0.25):
    rng = np.random.default_rng(seed)
    passable = rng.random((size, size)) > block
    cost = 1.0 + rng.random((size, size)) * 3.0
    passable[0, 0] = passable[-1, -1] = True
    return passable, cost


@pytest.mark.parametrize("seed", range(25))
def test_astar_matches_ucs_cost_and_expands_no_more(seed):
    passable, cost = random_grid(seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(5):
        start = tuple(int(v) for v in rng.integers(0, 20, 2))
        goal = tuple(int(v) for v in rng.integers(0, 20, 2))
        passable[start] = passable[goal] = True
        pa, ca, na = astar(start, {goal}, passable, cost)
        pu, cu, nu = ucs(start, {goal}, passable, cost)
        if pu is None:
            assert pa is None
            continue
        assert ca == pytest.approx(cu)  # admissible heuristic ⇒ optimal
        assert na <= nu  # and never expands more
        assert sum(cost[c] for c in pa) == pytest.approx(ca)


def test_path_is_contiguous_and_avoids_impassable():
    passable = np.ones((6, 6), dtype=bool)
    passable[3, 0:5] = False  # wall with a gap at y = 5
    path, cost, _ = astar((0, 0), {(5, 0)}, passable, np.ones((6, 6)))
    assert path is not None and path[-1] == (5, 0)
    prev = (0, 0)
    for cell in path:
        assert abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) == 1
        assert passable[cell]
        prev = cell
    assert (3, 5) in path


def test_avoids_burning_cells():
    size = 7
    state = np.full((size, size), CellState.TREE)
    state[3, 1:6] = CellState.BURNING
    passable = state != CellState.BURNING
    path, _, _ = astar((0, 3), {(6, 3)}, passable, np.ones((size, size)))
    assert path is not None
    assert all(state[c] != CellState.BURNING for c in path)


def test_high_lambda_detours_around_fire(cfg):
    size = 15
    burning = np.zeros((size, size), dtype=bool)
    burning[7, 5:10] = True  # fire line between start and goal's straight route
    passable = ~burning
    fuel = np.ones((size, size))
    risk = risk_map(burning, fuel, wind_vector(0), cfg["fire"])
    start, goal = (7, 1), (7, 13)

    def fire_exposure(path):
        return sum(risk[c] for c in path)

    low, _, _ = astar(start, {goal}, passable, step_cost_grid(risk, 0.0))
    high, _, _ = astar(start, {goal}, passable, step_cost_grid(risk, 50.0))
    assert fire_exposure(high) < fire_exposure(low)
    assert len(high) >= len(low)


def test_no_path_when_blocked():
    passable = np.ones((5, 5), dtype=bool)
    passable[2, :] = False
    path, cost, expanded = astar((0, 0), {(4, 4)}, passable, np.ones((5, 5)))
    assert path is None and math.isinf(cost) and expanded > 0


def test_blocked_first_only_affects_first_move():
    passable = np.ones((3, 1), dtype=bool)
    path, _, _ = astar((0, 0), {(2, 0)}, passable, np.ones((3, 1)), blocked_first={(1, 0)})
    assert path is None  # the only first step is reserved
    path, _, _ = astar((0, 0), {(2, 0)}, passable, np.ones((3, 1)), blocked_first={(2, 0)})
    assert path == [(1, 0), (2, 0)]  # reservation of a later cell is ignored


def test_impassable_target_goes_adjacent():
    passable = np.ones((5, 5), dtype=bool)
    passable[2, 2] = False
    assert goal_cells((2, 2), passable) == {(1, 2), (3, 2), (2, 1), (2, 3)}
    path, _, _ = plan_path((0, 0), (2, 2), passable, np.ones((5, 5)))
    assert path[-1] in goal_cells((2, 2), passable)
