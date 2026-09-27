"""Phase 3/6 — whole-model smoke tests, metrics and scalability."""

from __future__ import annotations

import pytest

from environment.cells import CellState
from model import WildfireModel
from scenarios import SCENARIOS, scenario_overrides


@pytest.mark.parametrize("strategy", ["independent", "greedy", "auction"])
def test_strategy_runs_to_t_max_without_error(strategy):
    m = WildfireModel({"simulation": {"strategy": strategy}}, seed=0)
    metrics = m.run()
    assert metrics["steps"] <= m.t_max
    assert 0.0 <= metrics["pct_forest_saved"] <= 100.0
    assert not m.running


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_scenario_runs(scenario):
    m = WildfireModel(scenario_overrides(scenario, "auction", {"simulation": {"t_max": 60}}), seed=1)
    m.run()
    assert m.steps <= 60


@pytest.mark.parametrize("n_scouts,n_ff", [(0, 1), (0, 30), (10, 5), (1, 0)])
def test_scales_to_any_agent_count(n_scouts, n_ff):
    m = WildfireModel({"agents": {"n_scouts": n_scouts, "n_firefighters": n_ff}, "simulation": {"t_max": 80}}, seed=3)
    m.run()
    assert len(m.scouts) == n_scouts and m.n_firefighters == n_ff


def test_metrics_and_score_are_consistent():
    m = WildfireModel({"simulation": {"strategy": "auction"}}, seed=4)
    r = m.run()
    w = m.cfg["score"]
    expected = (
        w["w1"] * r["trees_saved"] / r["trees_initial"]
        - w["w2"] * r["t_contain"] / m.t_max
        - w["w3"] * r["agents_lost"] / m.n_firefighters
    )
    assert r["score"] == pytest.approx(expected)
    assert r["trees_saved"] == int((m.state == CellState.TREE).sum())
    assert r["astar_nodes_expanded"] <= r["ucs_nodes_expanded"]
    df = m.datacollector.get_model_vars_dataframe()
    assert len(df) == m.steps + 1 and df["pct_forest_saved"].is_monotonic_decreasing


def test_same_seed_same_result():
    a = WildfireModel({"simulation": {"strategy": "auction"}}, seed=11).run()
    b = WildfireModel({"simulation": {"strategy": "auction"}}, seed=11).run()
    assert a == b


def test_caught_firefighter_is_removed():
    m = WildfireModel({"agents": {"n_scouts": 0}, "simulation": {"strategy": "independent"}}, seed=0)
    ff = m.firefighters[0]
    m.state[ff.pos] = CellState.BURNING
    m._safety_check()
    assert ff not in m.firefighters and m.agents_lost == 1
    assert any(e["kind"] == "agent_lost" for e in m.events)
