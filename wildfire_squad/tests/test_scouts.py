"""Phase 4 — scouts, frontier exploration and the shared belief map."""

from __future__ import annotations

import numpy as np

from algorithms.frontier import frontier_mask, select_target
from model import WildfireModel


def test_frontier_is_unknown_next_to_known():
    seen = np.full((6, 6), -1)
    seen[0:2, 0:2] = 5
    f = frontier_mask(seen, step=6, stale_after=15)
    assert f[2, 0] and f[2, 2] and f[0, 2]
    assert not f[0, 0] and not f[4, 4]


def test_stale_cells_become_frontier():
    seen = np.full((6, 6), 25)  # fresh everywhere ...
    seen[3, 3] = 0  # ... except one cell not seen for 30 steps
    f = frontier_mask(seen, step=30, stale_after=15)
    assert f[3, 3] and f.sum() == 1


def test_select_target_prefers_near_unknown_and_excludes_others():
    seen = np.full((20, 20), -1)
    seen[0:15, 0:2] = 1  # a long known strip, so the frontier is the row y = 2
    t = select_target((1, 1), seen, 2, 15, 300)
    assert t[1] == 2 and t[0] <= 2  # nearest unknown frontier cell
    t2 = select_target((1, 1), seen, 2, 15, 300, other_targets=[t], exclusion_radius=5)
    assert max(abs(t2[0] - t[0]), abs(t2[1] - t[1])) > 5


def test_scouts_explore_most_of_grid():
    m = WildfireModel({"fire": {"ignitions": 0}, "agents": {"n_firefighters": 0}}, seed=0)
    for _ in range(m.t_max):
        m.step()
    explored = m.shared_belief.known_mask().mean()
    assert explored > 0.8, f"explored only {explored:.0%}"


def test_scouts_share_observations_with_firefighters():
    m = WildfireModel({"simulation": {"strategy": "auction"}}, seed=1)
    for _ in range(5):
        m.step()
    assert all(ff.belief is m.shared_belief for ff in m.firefighters)
    assert m.shared_belief.known_mask().sum() > 9 * len(m.firefighters)


def test_independent_firefighters_have_private_maps():
    m = WildfireModel({"simulation": {"strategy": "independent"}}, seed=1)
    for _ in range(5):
        m.step()
    maps = {id(ff.belief) for ff in m.firefighters}
    assert len(maps) == len(m.firefighters) and id(m.shared_belief) not in maps


def test_exclusion_falls_back_to_unknown_cells():
    """Regression: when exclusion rules out the whole frontier ring, pick another unknown cell."""
    seen = np.full((30, 30), -1)
    seen[0:3, 0:3] = 1  # tiny known area: the frontier is a small ring
    t = select_target((1, 1), seen, 2, 15, 300, other_targets=[(3, 3)], exclusion_radius=5)
    assert t is not None and max(abs(t[0] - 3), abs(t[1] - 3)) > 5
