"""Phase 3/4 — conflict resolution: cell reservations, scout target exclusion, replanning."""

from __future__ import annotations

import pytest

from environment.cells import CellState, chebyshev
from model import WildfireModel


@pytest.mark.parametrize("strategy", ["independent", "greedy", "auction"])
def test_two_firefighters_never_share_a_cell(strategy):
    for seed in range(3):
        m = WildfireModel({"simulation": {"strategy": strategy}, "agents": {"n_firefighters": 10}}, seed=seed)
        while m.running and m.steps < 150:
            m.step()
            positions = [ff.pos for ff in m.firefighters]
            assert len(positions) == len(set(positions)), f"collision at step {m.steps}"


def test_reserved_cell_blocks_later_firefighter():
    """In a one-cell-wide corridor the second firefighter must wait behind the first."""
    m = WildfireModel(
        {
            "agents": {"n_scouts": 0, "n_firefighters": 2},
            "fire": {"ignitions": 0},
            "simulation": {"strategy": "independent"},
        },
        seed=0,
    )
    m.state[:] = CellState.WATER
    m.state[0:10, 0] = CellState.EMPTY
    m.water_mask = m.state == CellState.WATER
    for ff in m.firefighters:
        ff.belief.water_prior = m.water_mask.copy()
        ff.belief.version += 1
    a, b = sorted(m.firefighters, key=lambda f: f.unique_id)
    m.grid.move_agent(a, (1, 0))
    m.grid.move_agent(b, (0, 0))
    m.reserved = {a.pos, b.pos}
    # a is blocked by nothing; b wants a's cell, which a still holds until it moves
    b._go_to((5, 0))
    assert b.pos == (0, 0) and b._blocked_steps == 1
    a._go_to((6, 0))
    assert a.pos == (2, 0) and (1, 0) not in m.reserved
    b._go_to((5, 0))
    assert b.pos == (1, 0)


def test_scout_targets_respect_exclusion_radius(cfg):
    radius = cfg["scout"]["target_exclusion_radius"]
    m = WildfireModel({"agents": {"n_scouts": 4}, "fire": {"ignitions": 0}}, seed=5)
    for _ in range(120):
        m.step()
        targets = [t for t in m.scout_targets.values() if t is not None]
        for i, t1 in enumerate(targets):
            for t2 in targets[i + 1 :]:
                assert chebyshev(t1, t2) > radius


def test_firefighter_replans_when_path_blocked_by_fire():
    m = WildfireModel(
        {
            "agents": {"n_scouts": 0, "n_firefighters": 1},
            "fire": {"ignitions": 0},
            "simulation": {"strategy": "independent"},
        },
        seed=0,
    )
    ff = m.firefighters[0]
    m.state[:] = CellState.EMPTY
    goal = (ff.pos[0] + 8, ff.pos[1])
    ff._go_to(goal)
    assert ff.path
    blocker = ff.path[0]
    m.state[blocker] = CellState.BURNING  # fire appears on the planned route
    ff._go_to(goal)
    assert ff.pos != blocker  # never steps into fire
    assert ff.path is not None and blocker not in ff.path  # re-planned around it
    assert ff.belief.state[blocker] == CellState.BURNING  # and remembered what it saw


def test_boxed_in_firefighter_is_not_deadlocked():
    """Regression: the firefighter starting on the base cell used to be walled in by idle teammates."""
    m = WildfireModel({"agents": {"n_scouts": 0, "n_firefighters": 5}, "simulation": {"strategy": "auction"}}, seed=0)
    start = {ff.unique_id: ff.pos for ff in m.firefighters}
    for _ in range(40):
        m.step()
    moved = [ff for ff in m.firefighters if ff.pos != start[ff.unique_id]]
    assert len(moved) == len(m.firefighters), "every firefighter should have left the base area"
