"""Supporting studies behind the design decisions (run after ``run_experiments.py``).

Each study compares variants of the system on the same seeds (paired) and writes a CSV and a
chart to ``results/studies/``:

* ``timing``      — fire time scale (``fire.spread_every``) 1…8 for every strategy, plus an
                    upper bound where firefighters are told the true fire state ("oracle").
* ``tactics``     — what a firefighter targets inside its zone: the plan's downwind
                    firebreak edge vs nearest burning cell vs utility (spread stopped per step).
* ``allocation``  — targeted test of threat-aware allocation: a low-threat fire near the base
                    and a high-threat fire far away, 3 firefighters, zone linking 1 vs 3.
* ``robustness``  — the Coordinator fails at step 60, with and without the firefighters'
                    AWARD-timeout fallback.

Usage::

    python run_studies.py                      # all studies, 20–30 seeds each (~10 min on 4 workers)
    python run_studies.py --studies timing --runs 10
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
from run_experiments import GRID, INK, INK_2, STRATEGY_COLORS, STRATEGY_LABELS, SURFACE, _style  # noqa: E402
from scenarios import scenario_overrides  # noqa: E402
from settings import deep_merge  # noqa: E402

OUT_DIR = Path(__file__).with_name("results") / "studies"
NO_UCS = {"simulation": {"measure_ucs": False}}  # studies do not need the A*-vs-UCS measurement

Variant = tuple[str, str, dict[str, Any]]  # (label, strategy, config overrides)


def timing_variants() -> list[Variant]:
    """Fire time scale sweep on the windy single-ignition scenario, plus the oracle upper bound."""
    out: list[Variant] = []
    for every in (1, 2, 3, 5, 8):
        base = deep_merge(scenario_overrides("windy_single"), {"fire": {"spread_every": every}})
        for strategy in ("independent", "greedy", "auction"):
            out.append((f"{strategy} | every {every}", strategy, base))
        oracle = deep_merge(base, {"simulation": {"oracle_knowledge": True}})
        out.append((f"oracle | every {every}", "auction", oracle))
    return out


def tactics_variants() -> list[Variant]:
    """In-zone target choice for both coordinated strategies, plus the independent reference."""
    out: list[Variant] = []
    for scenario in ("windy_single", "scaling_10"):
        base = scenario_overrides(scenario)
        out.append((f"{scenario} | independent", "independent", base))
        for tactic in ("firebreak", "nearest", "utility"):
            for strategy in ("greedy", "auction"):
                cfg = deep_merge(base, {"firefighter": {"zone_tactic": tactic}})
                out.append((f"{scenario} | {strategy} | {tactic}", strategy, cfg))
    return out


def allocation_variants() -> list[Variant]:
    """Competing zones: near low-threat fire vs far high-threat fire, 3 firefighters."""
    base = {"simulation": {"preset": "competing"}, "agents": {"n_firefighters": 3}}
    out: list[Variant] = [("independent", "independent", base)]
    for link in (1, 3):
        for strategy in ("greedy", "auction"):
            cfg = deep_merge(base, {"coordinator": {"zone_link_distance": link}})
            out.append((f"{strategy} | link {link}", strategy, cfg))
    return out


def robustness_variants() -> list[Variant]:
    """Coordinator failure at step 60, with and without the AWARD-timeout fallback."""
    fail = {"coordinator": {"fail_at_step": 60}}
    return [
        ("independent", "independent", {}),
        ("auction, Coordinator works", "auction", {}),
        ("auction, Coordinator fails @60, fallback on", "auction", fail),
        (
            "auction, Coordinator fails @60, no fallback",
            "auction",
            deep_merge(fail, {"firefighter": {"award_timeout": None}}),
        ),
    ]


STUDIES = {
    "timing": (timing_variants, 20),
    "tactics": (tactics_variants, 30),
    "allocation": (allocation_variants, 30),
    "robustness": (robustness_variants, 30),
}


def run_variant(job: tuple[str, str, str, dict[str, Any], int]) -> dict[str, Any]:
    """Run one (study, variant, seed) and return its metrics."""
    study, label, strategy, overrides, seed = job
    cfg = deep_merge(deep_merge(overrides, NO_UCS), {"simulation": {"strategy": strategy}})
    m = WildfireModel(cfg, seed=seed)
    metrics = m.run()
    return {
        "study": study,
        "variant": label,
        "strategy": strategy,
        "seed": seed,
        "fallbacks": sum(e["kind"] == "fallback" for e in m.events),
        **metrics,
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and ≈95% CI of % forest saved per variant (in definition order)."""
    order = list(dict.fromkeys(df["variant"]))
    g = df.groupby("variant")["pct_forest_saved"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(), "runs": g.size()}).reindex(order)
    out["ci95"] = 1.96 * out["std"] / out["runs"].map(math.sqrt)
    extra = df.groupby("variant")[["agents_lost", "cells_extinguished", "firebreaks_cut", "fallbacks"]].mean()
    return out.join(extra).reset_index()


def paired(df: pd.DataFrame, a: str, b: str) -> tuple[float, float, int, int]:
    """Mean and ≈95% CI of (a − b) over shared seeds, plus how often a > b."""
    wa = df[df["variant"] == a].set_index("seed")["pct_forest_saved"]
    wb = df[df["variant"] == b].set_index("seed")["pct_forest_saved"]
    d = (wa - wb).dropna()
    ci = 1.96 * d.std() / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return float(d.mean()), float(ci), int((d > 0).sum()), int(len(d))


# ---------------------------------------------------------------------------- charts
def chart_timing(df: pd.DataFrame, out: Path) -> None:
    """% forest saved vs fire time scale, per strategy, with the oracle upper bound."""
    d = df.copy()
    d["every"] = d["variant"].str.extract(r"every (\d+)").astype(int)
    d["who"] = d["variant"].str.split(" | ", regex=False).str[0]
    agg = d.groupby(["who", "every"])["pct_forest_saved"].agg(["mean", "std", "size"]).reset_index()
    agg["ci"] = 1.96 * agg["std"] / agg["size"].map(math.sqrt)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    _style(ax)
    styles = {
        **{s: (STRATEGY_COLORS[s], "-", STRATEGY_LABELS[s]) for s in STRATEGY_COLORS},
        "oracle": (INK_2, "--", "Auction with perfect fire knowledge (upper bound)"),
    }
    for who in ("independent", "greedy", "auction", "oracle"):
        s = agg[agg["who"] == who].sort_values("every")
        color, ls, label = styles[who]
        ax.errorbar(
            s["every"],
            s["mean"],
            yerr=s["ci"],
            color=color,
            linestyle=ls,
            linewidth=2,
            marker="o",
            markersize=6,
            capsize=3,
            label=label,
        )
    ax.set_xticks(sorted(agg["every"].unique()))
    ax.set_xlabel("Agent steps per fire update (1 = a fire update every step)", color=INK_2)
    ax.set_ylabel("% forest saved (mean, ≈95% CI)", color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_title("Fire time scale: when can agents make a difference?", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_bars(summary: pd.DataFrame, title: str, out: Path) -> None:
    """Horizontal bars of mean % forest saved per variant with ≈95% CI."""
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(summary) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    colors = []
    for label in summary["variant"]:
        key = next((s for s in ("independent", "greedy", "auction") if s in label), "auction")
        colors.append(STRATEGY_COLORS[key])
    ys = range(len(summary))
    ax.barh(
        list(ys),
        summary["mean"],
        xerr=summary["ci95"],
        color=colors,
        height=0.7,
        ecolor=INK_2,
        error_kw={"elinewidth": 1, "capsize": 2},
    )
    for y, v, ci in zip(ys, summary["mean"], summary["ci95"], strict=True):
        ax.text(v + ci + 1.5, y, f"{v:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(summary["variant"], color=INK, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% forest saved (mean, ≈95% CI)", color=INK_2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


TITLES = {
    "tactics": "What a firefighter targets inside its zone",
    "allocation": "Targeted test: near low-threat fire vs far high-threat fire (3 firefighters)",
    "robustness": "Coordinator failure at step 60 and the AWARD-timeout fallback",
}


def report(study: str, df: pd.DataFrame, out: Path) -> None:
    """Write the study's CSVs and chart and print its key comparisons."""
    df.to_csv(out / f"{study}_runs.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out / f"{study}_summary.csv", index=False)
    if study == "timing":
        chart_timing(df, out / "timing.png")
    else:
        chart_bars(summary, TITLES[study], out / f"{study}.png")
    print(f"\n=== {study}")
    print(summary.round(2).to_string(index=False))
    comparisons: list[tuple[str, str]] = []
    if study == "tactics":
        for sc in ("windy_single", "scaling_10"):
            for strat in ("greedy", "auction"):
                comparisons.append((f"{sc} | {strat} | utility", f"{sc} | {strat} | firebreak"))
            comparisons.append((f"{sc} | auction | utility", f"{sc} | independent"))
            comparisons.append((f"{sc} | auction | utility", f"{sc} | greedy | utility"))
    elif study == "allocation":
        comparisons = [
            ("auction | link 1", "greedy | link 1"),
            ("auction | link 3", "greedy | link 3"),
            ("auction | link 1", "independent"),
        ]
    elif study == "robustness":
        comparisons = [
            ("auction, Coordinator fails @60, fallback on", "auction, Coordinator fails @60, no fallback"),
            ("auction, Coordinator fails @60, fallback on", "auction, Coordinator works"),
        ]
    elif study == "timing":
        for every in (1, 5):
            comparisons.append((f"auction | every {every}", f"independent | every {every}"))
    rows = []
    for a, b in comparisons:
        mean, ci, wins, n = paired(df, a, b)
        rows.append({"a": a, "b": b, "mean_diff": mean, "ci95": ci, "a_better": wins, "runs": n})
        print(f"  {a}  −  {b}:  {mean:+.2f} ± {ci:.2f}  (better on {wins}/{n})")
    pd.DataFrame(rows).to_csv(out / f"{study}_paired.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    """Run the selected studies in parallel and write their outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", nargs="+", default=list(STUDIES), choices=list(STUDIES))
    parser.add_argument("--runs", type=int, default=None, help="seeds per variant (default: per study)")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--replot", action="store_true", help="rebuild summaries and charts from *_runs.csv")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        for study in args.studies:
            report(study, pd.read_csv(args.out / f"{study}_runs.csv"), args.out)
        return

    jobs = []
    for study in args.studies:
        make, default_runs = STUDIES[study]
        runs = args.runs or default_runs
        jobs += [(study, label, strat, cfg, seed) for label, strat, cfg in make() for seed in range(runs)]
    print(f"{len(jobs)} runs across {len(args.studies)} studies on {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_variant, jobs, chunksize=2), 1):
            rows.append(row)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    for study in args.studies:
        report(study, df[df["study"] == study], args.out)
    print(f"\nWrote {args.out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
