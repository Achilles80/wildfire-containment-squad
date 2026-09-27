"""Coordinator life-cycle, robustness fallback, firefighter safety and study-only switches."""

from __future__ import annotations

import numpy as np

from algorithms.zones import connected_components, live_zone_cells
from environment.cells import CellState
from model import WildfireModel


def run_until(model: WildfireModel, condition, limit: int = 300) -> None:
    """Step ``model`` until ``condition(model)`` holds (or the run ends)."""
    while model.running and model.steps < limit and not condition(model):
        model.step()


def assigned_model(seed: int = 0, **overrides) -> WildfireModel:
    """An auction model advanced until at least one firefighter holds a zone."""
    cfg = {"simulation": {"strategy": "auction", "measure_ucs": False}, **overrides}
    m = WildfireModel(cfg, seed=seed)
    run_until(m, lambda mm: bool(mm.coordinator.assignments))
    assert m.coordinator.assignments, "fixture expects a fire that gets found"
    return m


# ---------------------------------------------------------------------- Coordinator life-cycle
def test_absent_firefighter_is_reauctioned():
    m = assigned_model()
    ff_id, zone_id = next(iter(m.coordinator.assignments.items()))
    ff = next(f for f in m.firefighters if f.unique_id == ff_id)
    ff.water = 0
    ff.refilling_since = m.steps - m.cfg["coordinator"]["reauction_after_absent"] - 1
    m.coordinator.release_absent()
    assert ff_id not in m.coordinator.assignments
    assert ff.assigned_zone is None  # the REVOKE reached the firefighter
    assert any(e["kind"] == "revoke" and e["reason"] == "absent" and e["agent"] == ff_id for e in m.events)


def test_abandoned_zone_is_offered_again():
    m = assigned_model(coordinator={"assign_idle": False})  # only the design slots are filled
    ff_id, zone_id = next(iter(m.coordinator.assignments.items()))
    ff = next(f for f in m.firefighters if f.unique_id == ff_id)
    ff._send_done("abandoned")
    assert ff_id not in m.coordinator.assignments
    assert zone_id in {o.zone_id for o in m.coordinator._offers()}
    assert m.bus.counts["DONE"] >= 1


def test_freed_firefighter_is_reallocated_without_waiting_for_the_next_round():
    m = assigned_model()
    while (m.steps + 1) % m.cfg["coordinator"]["cluster_every"] == 0:  # make the next step a non-clustering step
        m.step()
    ff_id = next(iter(m.coordinator.assignments))
    ff = next(f for f in m.firefighters if f.unique_id == ff_id)
    ff._send_done("abandoned")
    m.step()
    if m.coordinator.zones and ff in m.firefighters:
        assert ff.unique_id in m.coordinator.assignments  # event-driven allocation


def test_greedy_awards_without_bids():
    m = WildfireModel({"simulation": {"strategy": "greedy", "measure_ucs": False}}, seed=0)
    run_until(m, lambda mm: mm.bus.counts["AWARD"] > 0)
    assert m.bus.counts["AWARD"] > 0
    assert m.bus.counts["BID"] == 0 and m.bus.counts["ANNOUNCE"] == 0


# ---------------------------------------------------------------------- robustness fallback
def test_coordinator_failure_triggers_fallback():
    m = WildfireModel({"coordinator": {"fail_at_step": 1}, "simulation": {"measure_ucs": False}}, seed=0)
    run_until(m, lambda mm: any(f.fallback for f in mm.firefighters))
    assert m.coordinator.failed
    assert m.bus.counts["AWARD"] == 0  # a failed Coordinator never allocates
    assert any(f.fallback for f in m.firefighters)
    assert any(e["kind"] == "fallback" for e in m.events)
    assert any(e["kind"] == "coordinator_failed" for e in m.events)


def test_no_fallback_when_timeout_disabled():
    m = WildfireModel(
        {
            "coordinator": {"fail_at_step": 1},
            "firefighter": {"award_timeout": None},
            "simulation": {"measure_ucs": False, "t_max": 150},
        },
        seed=0,
    )
    m.run()
    assert not any(e["kind"] == "fallback" for e in m.events)
    assert m.cells_extinguished < 20  # without an allocator or fallback, firefighters barely engage


# ---------------------------------------------------------------------- firefighter safety
def test_firefighter_secures_footing_before_anything_else():
    m = WildfireModel(
        {
            "agents": {"n_scouts": 0, "n_firefighters": 1},
            "fire": {"ignitions": 0},
            "simulation": {"strategy": "independent"},
        },
        seed=0,
    )
    ff = m.firefighters[0]
    x, y = 10, 10
    m.grid.move_agent(ff, (x, y))
    m.state[x - 2 : x + 3, y - 2 : y + 3] = CellState.TREE
    m.state[x + 1, y + 1] = CellState.BURNING  # diagonal only
    ff.water = 0  # even when it should be refilling
    ff.step()
    assert m.state[x, y] == CellState.FIREBREAK
    assert ff.pos == (x, y)


# ---------------------------------------------------------------------- zones
def test_live_zone_follows_the_fire():
    burning = np.zeros((10, 10), dtype=bool)
    burning[4:7, 5] = True  # fire has moved one cell east of the snapshot
    snapshot = {(4, 4), (5, 4)}  # no longer burning
    assert live_zone_cells(snapshot, burning) == frozenset({(4, 5), (5, 5), (6, 5)})
    assert live_zone_cells({(0, 0)}, burning) == frozenset()


def test_zone_link_distance_joins_nearby_fragments():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 2] = mask[2, 4] = True  # two burning cells with one burnt cell between them
    assert len(connected_components(mask, link_distance=1)) == 2
    assert len(connected_components(mask, link_distance=2)) == 1


# ---------------------------------------------------------------------- environment switches
def test_fire_updates_only_every_spread_every_steps():
    m = WildfireModel(
        {
            "agents": {"n_scouts": 0, "n_firefighters": 0},
            "fire": {"spread_every": 5},
            "simulation": {"strategy": "independent"},
        },
        seed=0,
    )
    timer0 = m.burn_timer.copy()
    for _ in range(4):
        m.step()
        assert np.array_equal(m.burn_timer, timer0)
    m.step()
    assert not np.array_equal(m.burn_timer, timer0)


def test_competing_preset_places_the_two_fires(cfg):
    m = WildfireModel({"simulation": {"preset": "competing"}}, seed=0)
    preset = cfg["presets"]["competing"]
    assert m.ignitions == [tuple(preset["near_ignition"]), tuple(preset["far_ignition"])]
    assert m.fuel[tuple(preset["near_ignition"])] == cfg["fire"]["fuel_sparse"]
    assert m.fuel[tuple(preset["far_ignition"])] == cfg["fire"]["fuel_dense"]


def test_oracle_knowledge_gives_true_fire_state():
    m = WildfireModel({"simulation": {"oracle_knowledge": True, "measure_ucs": False}}, seed=0)
    m.step()
    assert (m.shared_belief.state[m.shared_belief.state == CellState.BURNING].size) > 0
    assert m.shared_belief.known_mask().all()
