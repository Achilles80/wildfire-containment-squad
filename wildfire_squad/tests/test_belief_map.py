"""Phase 2 — shared belief map fusion and staleness."""

from __future__ import annotations

import numpy as np

from belief_map import BeliefMap
from environment.cells import UNKNOWN, CellState


def test_starts_unknown():
    b = BeliefMap(5, 5)
    assert (b.state == UNKNOWN).all()
    assert (b.seen_step == -1).all()


def test_newest_observation_wins():
    b = BeliefMap(5, 5)
    b.observe({(1, 1): CellState.BURNING}, step=40)  # scout sees it burning at step 40
    b.observe({(1, 1): CellState.BURNT}, step=42)  # firefighter sees it burnt at step 42
    assert b.state[1, 1] == CellState.BURNT
    assert b.seen_step[1, 1] == 42


def test_older_observation_is_ignored():
    b = BeliefMap(5, 5)
    b.observe({(2, 3): CellState.BURNT}, step=42)
    b.observe({(2, 3): CellState.BURNING}, step=40)  # arrives late
    assert b.state[2, 3] == CellState.BURNT
    assert b.seen_step[2, 3] == 42


def test_staleness_and_age():
    b = BeliefMap(4, 4)
    b.observe({(0, 0): CellState.TREE, (1, 1): CellState.TREE}, step=10)
    b.observe({(1, 1): CellState.TREE}, step=20)
    stale = b.stale_mask(step=30, stale_after=15)
    assert stale[0, 0] and not stale[1, 1]
    assert not stale[3, 3]  # unknown cells are not "stale", they are unknown
    age = b.age(30)
    assert age[0, 0] == 20 and age[1, 1] == 10
    assert age[3, 3] > 10**6


def test_passability():
    water = np.zeros((3, 3), dtype=bool)
    water[2, 2] = True
    b = BeliefMap(3, 3, water_prior=water)
    b.observe({(0, 1): CellState.BURNING, (1, 0): CellState.BURNT}, step=1)
    assert b.is_passable_for_firefighter(0, 0)  # unknown → passable
    assert not b.is_passable_for_firefighter(0, 1)  # burning
    assert b.is_passable_for_firefighter(1, 0)  # burnt ground
    assert not b.is_passable_for_firefighter(2, 2)  # known terrain water
    assert b.burning_cells() == [(0, 1)]
