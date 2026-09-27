"""Batch experiments: every scenario × strategy × seed, headless, with CSV output and charts.

Usage::

    python run_experiments.py --scenarios all --runs 30
    python run_experiments.py --quick                 # 3 runs per cell, for a fast check
    python run_experiments.py --scenarios windy_single multi_ignition --runs 10
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from model import WildfireModel  # noqa: E402
from scenarios import SCENARIOS, expand, scenario_overrides  # noqa: E402
from settings import load_config  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

RESULTS_DIR = Path(__file__).with_name("results")

# Validated categorical slots (dataviz reference palette; the first three pass all-pairs CVD checks).
STRATEGY_COLORS = {"auction": "#2a78d6", "independent": "#eb6834", "greedy": "#1baf7a"}
STRATEGY_LABELS = {"independent": "Independent", "greedy": "Greedy nearest", "auction": "Auction (full system)"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["pct_forest_saved", "t_contain", "agents_lost", "mean_belief_staleness", "score"]


def run_one(job: tuple[str, str, int]) -> dict[str, Any]:
    """Run one scenario/strategy/seed combination and return its final metrics."""
    scenario, strategy, seed = job
    t0 = time.perf_counter()
    metrics = WildfireModel(scenario_overrides(scenario, strategy), seed=seed).run()
    return {
        "scenario": scenario,
        "scenario_no": SCENARIOS[scenario][0],
        "strategy": strategy,
        "seed": seed,
        **metrics,
        "runtime_s": time.perf_counter() - t0,
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each metric per scenario × strategy."""
    agg = df.groupby(["scenario_no", "scenario", "strategy"], sort=False)[
        METRICS + ["astar_nodes_expanded", "ucs_nodes_expanded", "contained"]
    ].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "runs", df.groupby(["scenario_no", "scenario", "strategy"], sort=False).size())
    return agg.reset_index().sort_values(["scenario_no", "scenario", "strategy"], key=_strategy_sort)


def _strategy_sort(col: pd.Series) -> pd.Series:
    """Sort key: scenarios and strategies in their definition order."""
    if col.name == "strategy":
        return col.map({name: i for i, name in enumerate(STRATEGIES)})
    if col.name == "scenario":
        return col.map({name: i for i, name in enumerate(SCENARIOS)})
    return col


def paired_gains(df: pd.DataFrame) -> pd.DataFrame:
    """Auction minus each baseline on the same seed (paired comparison of % forest saved).

    Every strategy faces the identical map and fire random stream for a given seed, so the
    per-seed difference removes most of the fire-to-fire variance. ``ci95`` is the normal
    approximation ``1.96 × std / sqrt(n)`` of the mean difference.
    """
    wide = df.pivot_table(index=["scenario", "seed"], columns="strategy", values="pct_forest_saved")
    rows = []
    for scenario in [s for s in SCENARIOS if s in wide.index.get_level_values("scenario")]:
        grp = wide.xs(scenario, level="scenario")
        for base in ("independent", "greedy"):
            if base in grp and "auction" in grp:
                diff = (grp["auction"] - grp[base]).dropna()
                rows.append(
                    {
                        "scenario": scenario,
                        "vs": base,
                        "mean_gain_pct_points": diff.mean(),
                        "ci95": 1.96 * diff.std() / math.sqrt(len(diff)) if len(diff) > 1 else float("nan"),
                        "std": diff.std(),
                        "auction_better_runs": int((diff > 0).sum()),
                        "ties": int((diff == 0).sum()),
                        "runs": int(diff.count()),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- charts
def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_saved_by_scenario(summary: pd.DataFrame, out: Path) -> None:
    """Grouped bars: mean % forest saved per scenario by strategy, with std error bars."""
    scenarios = [s for s in SCENARIOS if s in set(summary["scenario"])]
    strategies = [s for s in STRATEGIES if s in set(summary["strategy"])]
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    _style(ax)
    width = 0.8 / len(strategies)
    for i, strat in enumerate(strategies):
        sub = summary[summary["strategy"] == strat].set_index("scenario").reindex(scenarios)
        xs = [x + (i - (len(strategies) - 1) / 2) * width for x in range(len(scenarios))]
        ax.bar(
            xs,
            sub["pct_forest_saved_mean"],
            width * 0.92,
            color=STRATEGY_COLORS[strat],
            label=STRATEGY_LABELS[strat],
            yerr=sub["pct_forest_saved_std"],
            ecolor=INK_2,
            error_kw={"elinewidth": 1, "capsize": 2},
        )
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([f"S{SCENARIOS[s][0]}\n{s}" for s in scenarios], color=INK, fontsize=9)
    ax.set_ylabel("% forest saved (mean ± std)", color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_title("Forest saved by scenario and strategy", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_paired_gains(gains: pd.DataFrame, out: Path) -> None:
    """Dot plot: mean paired gain of the auction over each baseline, with ≈95% CI, per scenario."""
    if gains.empty:
        return
    scenarios = list(dict.fromkeys(gains["scenario"]))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(scenarios) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.axvline(0, color=INK_2, linewidth=1)
    for offset, base in ((-0.15, "independent"), (0.15, "greedy")):
        g = gains[gains["vs"] == base].set_index("scenario").reindex(scenarios)
        ys = [i + offset for i in range(len(scenarios))]
        ax.errorbar(
            g["mean_gain_pct_points"],
            ys,
            xerr=g["ci95"],
            fmt="o",
            color=STRATEGY_COLORS[base],
            markersize=7,
            elinewidth=2,
            capsize=3,
            label=f"Auction − {STRATEGY_LABELS[base].lower()}",
        )
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"S{SCENARIOS[s][0]} {s}" for s in scenarios], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Paired gain in % forest saved (percentage points, same seeds; bars ≈ 95% CI)", color=INK_2)
    ax.set_title("Where coordination helps: auction vs each baseline", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_scaling(summary: pd.DataFrame, out: Path) -> None:
    """Line chart for Scenario 5: % forest saved vs number of firefighters."""
    sub = summary[summary["scenario"].str.startswith("scaling_")].copy()
    if sub.empty:
        return
    sub["n_ff"] = sub["scenario"].str.split("_").str[1].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    _style(ax)
    for strat in STRATEGIES:
        s = sub[sub["strategy"] == strat].sort_values("n_ff")
        if s.empty:
            continue
        ax.errorbar(
            s["n_ff"],
            s["pct_forest_saved_mean"],
            yerr=s["pct_forest_saved_std"],
            color=STRATEGY_COLORS[strat],
            linewidth=2,
            marker="o",
            markersize=7,
            capsize=3,
            label=STRATEGY_LABELS[strat],
        )
    ax.set_xticks(sorted(sub["n_ff"].unique()))
    ax.set_xlabel("Firefighters", color=INK_2)
    ax.set_ylabel("% forest saved (mean ± std)", color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_title("Scenario 5 — scaling the firefighter team", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_astar_vs_ucs(df: pd.DataFrame, out: Path) -> None:
    """Bars: mean nodes expanded per search call, A* vs UCS, per scenario (same queries)."""
    d = df[df["astar_calls"] > 0].copy()
    if d.empty or d["ucs_nodes_expanded"].sum() == 0:
        return
    per = d.groupby("scenario", sort=False)[["astar_nodes_expanded", "ucs_nodes_expanded", "astar_calls"]].sum()
    per = per.reindex([s for s in SCENARIOS if s in per.index])
    a = per["astar_nodes_expanded"] / per["astar_calls"]
    u = per["ucs_nodes_expanded"] / per["astar_calls"]
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    _style(ax)
    xs = range(len(per))
    ax.bar([x - 0.2 for x in xs], u, 0.38, color="#eb6834", label="Uniform-cost search")
    ax.bar([x + 0.2 for x in xs], a, 0.38, color="#2a78d6", label="Risk-aware A* (Manhattan h)")
    for x, (av, uv) in enumerate(zip(a, u, strict=True)):
        ax.annotate(
            f"−{100 * (1 - av / uv):.0f}%",
            (x + 0.2, av),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK_2,
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"S{SCENARIOS[s][0]}\n{s}" for s in per.index], color=INK, fontsize=9)
    ax.set_ylabel("Nodes expanded per search", color=INK_2)
    ax.set_title("A* vs UCS on identical path queries (same optimal cost)", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, run the batch, write CSVs and charts."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=["all"], help="scenario names, 'scaling' or 'all'")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--runs", type=int, default=cfg["experiments"]["runs_per_scenario"])
    parser.add_argument("--quick", action="store_true", help="3 runs per scenario × strategy")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
        help="parallel worker processes (each needs ~150 MB)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--replot", action="store_true", help="rebuild summaries and charts from results.csv")
    args = parser.parse_args(argv)
    if args.replot:
        write_outputs(pd.read_csv(args.out / "results.csv"), args.out)
        return
    runs = 3 if args.quick else args.runs

    scenarios = expand(args.scenarios)
    jobs = [(s, strat, seed) for s in scenarios for strat in args.strategies for seed in range(runs)]
    print(
        f"{len(jobs)} runs: {len(scenarios)} scenarios × {len(args.strategies)} strategies × {runs} seeds "
        f"on {args.workers} workers"
    )
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_one, jobs, chunksize=2), 1):
            rows.append(row)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    write_outputs(df, args.out)
    print(
        f"\nWrote {args.out}/results.csv, summary.csv, paired_gains.csv and charts in {time.perf_counter() - t0:.0f}s"
    )


def write_outputs(df: pd.DataFrame, out: Path) -> None:
    """Write results/summary/paired-gain CSVs and all charts, and print the headline tables."""
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out / "summary.csv", index=False)
    gains = paired_gains(df)
    gains.to_csv(out / "paired_gains.csv", index=False)
    chart_saved_by_scenario(summary, out / "forest_saved_by_scenario.png")
    chart_paired_gains(gains, out / "paired_gains.png")
    chart_scaling(summary, out / "scaling.png")
    chart_astar_vs_ucs(df, out / "astar_vs_ucs.png")

    pd.set_option("display.width", 160)
    print("\nMean % forest saved (± std):")
    table = summary.pivot(index="scenario", columns="strategy", values="pct_forest_saved_mean")
    std = summary.pivot(index="scenario", columns="strategy", values="pct_forest_saved_std")
    order = [s for s in SCENARIOS if s in table.index]
    cols = [s for s in STRATEGIES if s in table.columns]
    print((table.loc[order, cols].round(1).astype(str) + " ± " + std.loc[order, cols].round(1).astype(str)).to_string())
    if not gains.empty:
        print("\nPaired gain of auction over each baseline (percentage points, same seeds):")
        print(gains.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
