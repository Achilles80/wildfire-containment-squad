"""The demo helpers, the experiment runner and the studies runner work end to end."""

from __future__ import annotations

import pandas as pd
import pytest

import app
import run_experiments
import run_studies
from scenarios import SCENARIOS, expand, scenario_overrides
from strategies import get_strategy


def test_scenario_expansion_and_overrides():
    assert expand(["all"]) == list(SCENARIOS)
    assert expand(["scaling"]) == ["scaling_2", "scaling_5", "scaling_10"]
    assert expand(["river", "river"]) == ["river"]
    with pytest.raises(ValueError):
        expand(["no_such_scenario"])
    assert scenario_overrides("default") == {}
    assert scenario_overrides("no_scouts", "greedy")["simulation"]["strategy"] == "greedy"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        get_strategy("telepathy")


def test_demo_model_builds_from_flat_parameters():
    m = app.DemoModel(strategy="greedy", preset="river", n_scouts=2, n_firefighters=4, coordinator_fails_at=50, seed=3)
    assert m.strategy.name == "greedy" and len(m.scouts) == 2 and m.n_firefighters == 4
    assert m.cfg["simulation"]["preset"] == "river" and m.cfg["coordinator"]["fail_at_step"] == 50
    summary = app.settings_summary(m)
    assert "greedy" in summary and "seed **3**" in summary and "fails at step 50" in summary


@pytest.mark.parametrize("belief", [False, True])
def test_demo_drawing(belief):
    m = app.DemoModel(strategy="independent", seed=1)
    for _ in range(12):
        m.step()
    fig = app.draw_grid(m, belief)
    assert fig.axes
    img = app.belief_image(m)
    assert img.shape == (m.width, m.height, 3)


def test_demo_comparison_runs_all_strategies():
    m = app.DemoModel(preset="windy_single", seed=0)
    table, images = app.compare_strategies(m.cfg)
    assert list(table["strategy"]) == ["independent", "greedy", "auction"]
    assert len(images) == 3 and table["% forest saved"].between(0, 100).all()


def test_run_experiments_writes_results_and_charts(tmp_path):
    run_experiments.main(["--scenarios", "river", "--runs", "2", "--workers", "1", "--out", str(tmp_path)])
    for name in ("results.csv", "summary.csv", "paired_gains.csv", "forest_saved_by_scenario.png", "paired_gains.png"):
        assert (tmp_path / name).exists(), name
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 2 * 3 and set(df["strategy"]) == {"independent", "greedy", "auction"}
    run_experiments.main(["--replot", "--out", str(tmp_path)])  # rebuilds from results.csv


def test_run_studies_writes_outputs(tmp_path):
    run_studies.main(["--studies", "robustness", "allocation", "--runs", "1", "--workers", "1", "--out", str(tmp_path)])
    for name in ("robustness_runs.csv", "robustness_summary.csv", "robustness.png", "allocation_paired.csv"):
        assert (tmp_path / name).exists(), name
