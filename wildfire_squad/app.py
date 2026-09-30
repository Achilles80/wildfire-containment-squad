"""Live visual demo (Mesa + Solara).

Run from this folder with::

    solara run app.py

Left: the forest (or the team's belief map), agents and wind. Right: live metrics, zone
assignments, firefighter status, the latest auction rounds and message counts. Below: forest
saved and burning cells over time. Change parameters in the sidebar, then press Reset.

The sidebar uses Mesa's own SolaraViz controllers (``ModelController``, ``ModelCreator``). The
page layout is our own because Mesa 3.5's draggable grid layout does not render with the Vue 3
front end of current Solara releases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import solara
from matplotlib import patheffects
from matplotlib.figure import Figure
from mesa.visualization.solara_viz import ModelController, ModelCreator
from mesa.visualization.utils import update_counter

from environment.cells import UNKNOWN, CellState
from model import WildfireModel
from scenarios import SCENARIOS, scenario_overrides
from settings import deep_merge, load_config
from strategies import STRATEGIES

_DEFAULTS = load_config()

# ----------------------------------------------------------------------------- colours
COLORS = {
    "tree_dense": (0.10, 0.36, 0.17),
    "tree_sparse": (0.49, 0.72, 0.40),
    "burning": (0.93, 0.33, 0.10),
    "burnt": (0.25, 0.25, 0.25),
    "firebreak": (0.55, 0.36, 0.20),
    "water": (0.20, 0.47, 0.82),
    "empty": (0.89, 0.84, 0.72),
    "unknown": (0.0, 0.0, 0.0),
}
SCOUT_COLOR = "#5fe3f0"
FIREFIGHTER_COLOR = "#ffd21f"


class DemoModel(WildfireModel):
    """WildfireModel with flat keyword parameters, so SolaraViz can rebuild it from the sidebar.

    A scenario preset (other than ``custom``) is applied on top of the sliders.
    """

    def __init__(
        self,
        strategy: str = _DEFAULTS["simulation"]["strategy"],
        preset: str = "custom",
        n_scouts: int = _DEFAULTS["agents"]["n_scouts"],
        n_firefighters: int = _DEFAULTS["agents"]["n_firefighters"],
        wind_direction_deg: int = _DEFAULTS["fire"]["wind_direction_deg"],
        wind_k: float = _DEFAULTS["fire"]["wind_k"],
        ignitions: int = _DEFAULTS["fire"]["ignitions"],
        spread_every: int = _DEFAULTS["fire"]["spread_every"],
        coordinator_fails_at: int = 0,
        seed: int = _DEFAULTS["seed"],
    ) -> None:
        overrides: dict[str, Any] = {
            "simulation": {"strategy": strategy},
            "agents": {"n_scouts": n_scouts, "n_firefighters": n_firefighters},
            "fire": {
                "wind_direction_deg": wind_direction_deg,
                "wind_k": wind_k,
                "ignitions": ignitions,
                "spread_every": spread_every,
            },
            "coordinator": {"fail_at_step": coordinator_fails_at or None},  # 0 = never fails
        }
        if preset != "custom":
            overrides = deep_merge(overrides, scenario_overrides(preset))
        super().__init__(overrides, seed=int(seed))
        self.preset = preset


# ----------------------------------------------------------------------------- drawing
show_belief = solara.reactive(False)


def state_image(state: np.ndarray, fuel: np.ndarray, dense_fuel: float) -> np.ndarray:
    """RGB image (``[x, y, rgb]``) of a cell-state array using the demo colours."""
    img = np.zeros((*state.shape, 3))
    img[state == CellState.EMPTY] = COLORS["empty"]
    img[(state == CellState.TREE) & (fuel >= dense_fuel)] = COLORS["tree_dense"]
    img[(state == CellState.TREE) & (fuel < dense_fuel)] = COLORS["tree_sparse"]
    img[state == CellState.BURNING] = COLORS["burning"]
    img[state == CellState.BURNT] = COLORS["burnt"]
    img[state == CellState.FIREBREAK] = COLORS["firebreak"]
    img[state == CellState.WATER] = COLORS["water"]
    img[state == UNKNOWN] = COLORS["unknown"]
    return img


def belief_image(model: WildfireModel) -> np.ndarray:
    """What the team believes: unknown cells black, stale cells faded toward grey."""
    belief = model.shared_belief
    if not model.strategy.shared_map and model.firefighters:
        belief = model.firefighters[0].belief  # independent: show one firefighter's private view
    img = state_image(belief.state, model.fuel, model.cfg["fire"]["fuel_dense"])
    img[belief.water_prior & (belief.state == UNKNOWN)] = COLORS["water"]  # terrain is known a priori
    stale = belief.stale_mask(model.steps, model.cfg["scout"]["stale_after"])
    img[stale] = 0.45 * img[stale] + 0.55 * np.array([0.75, 0.75, 0.75])
    return img


def draw_grid(model: WildfireModel, belief: bool) -> Figure:
    """Forest (or belief) raster with agents, zone targets and a wind arrow."""
    fig = Figure(figsize=(6.2, 6.2), facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    img = belief_image(model) if belief else state_image(model.state, model.fuel, model.cfg["fire"]["fuel_dense"])
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = "Belief map (black = unknown, faded = stale)" if belief else "Ground truth"
    ax.set_title(f"{title} — step {model.steps}", fontsize=10, loc="left")

    if model.coordinator is not None:
        for zone in model.coordinator.zones.values():
            if zone.target_cell is not None:
                ax.plot(*zone.target_cell, marker="x", color="white", markersize=6, mew=1.5)
    for scout in model.scouts:
        ax.plot(*scout.pos, marker="o", markersize=7, color=SCOUT_COLOR, mec="black", mew=0.8)
    for ff in model.firefighters:
        ax.plot(*ff.pos, marker="s", markersize=9, color=FIREFIGHTER_COLOR, mec="black", mew=0.8)
        ax.annotate(str(ff.unique_id), ff.pos, color="black", fontsize=6, ha="center", va="center", weight="bold")

    wx, wy = model.wind
    k = model.cfg["fire"]["wind_k"]
    cx, cy = model.width - 6, model.height - 6
    outline = [patheffects.withStroke(linewidth=3, foreground="black")]
    if k > 0:
        arrow = ax.annotate(
            "",
            xy=(cx + 4 * wx, cy + 4 * wy),
            xytext=(cx - 4 * wx, cy - 4 * wy),
            arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 2.5},
        )
        arrow.arrow_patch.set_path_effects([patheffects.withStroke(linewidth=4.5, foreground="black")])
    ax.text(cx, cy - 5.5, f"wind k={k:g}", color="white", ha="center", fontsize=8, weight="bold", path_effects=outline)
    return fig


def settings_summary(model: WildfireModel) -> str:
    """One line describing the run currently on screen (the sliders do not show their values)."""
    cfg = model.cfg
    fire, agents = cfg["fire"], cfg["agents"]
    fail = cfg["coordinator"]["fail_at_step"]
    coordinator = "none" if model.coordinator is None else (f"fails at step {fail}" if fail else "working")
    return (
        f"Strategy **{model.strategy.name}** · preset **{getattr(model, 'preset', 'custom')}** · "
        f"seed **{model.seed_value}** · scouts {agents['n_scouts']} · firefighters {agents['n_firefighters']} · "
        f"wind {fire['wind_direction_deg']}° k={fire['wind_k']:g} · ignitions {fire['ignitions']} · "
        f"fire updates every {fire['spread_every']} steps · Coordinator {coordinator}"
    )


def compare_strategies(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray]]]:
    """Run independent, greedy and auction headless on exactly this configuration and seed.

    Returns a results table and the final ground-truth image of each run.
    """
    rows, images = [], []
    for name in STRATEGIES:
        run_cfg = deep_merge(cfg, {"simulation": {"strategy": name, "measure_ucs": False}})
        m = WildfireModel(run_cfg)
        r = m.run()
        rows.append(
            {
                "strategy": name,
                "% forest saved": round(r["pct_forest_saved"], 1),
                "steps": r["steps"],
                "contained": "yes" if r["contained"] else "no",
                "firefighters lost": r["agents_lost"],
                "cells extinguished": r["cells_extinguished"],
                "messages": r["messages"],
            }
        )
        images.append((name, state_image(m.state, m.fuel, cfg["fire"]["fuel_dense"])))
    return pd.DataFrame(rows), images


@solara.lab.task
def comparison_task(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray]]]:
    """Background task behind the Compare button (keeps the page responsive)."""
    return compare_strategies(cfg)


# ----------------------------------------------------------------------------- components
@solara.component
def GridView(model: WildfireModel) -> None:
    """Main map with the belief-map toggle."""
    update_counter.get()
    with solara.Column():
        solara.Checkbox(label="Show belief map instead of ground truth", value=show_belief)
        solara.FigureMatplotlib(draw_grid(model, show_belief.value), format="png", dpi=110)
        solara.Markdown(
            "**Legend:** dark green = dense forest · light green = sparse · orange-red = burning · "
            "grey = burnt · brown = firebreak · blue = water · beige = empty · "
            "cyan ● scout · yellow ■ firefighter (id) · white × zone target"
        )


def _fmt_bids(bids: dict[int, float]) -> str:
    return ", ".join(f"FF{a}: {u:.3f}" for a, u in sorted(bids.items(), key=lambda kv: -kv[1])) or "no bids"


@solara.component
def SidePanel(model: WildfireModel) -> None:
    """Live metrics, assignments, firefighter status, auction log and messages."""
    update_counter.get()
    burning = int((model.state == CellState.BURNING).sum())
    status = "contained" if model.t_contain is not None else ("running" if model.running else "time limit")
    lines = [
        settings_summary(model),
        "",
        f"### {model.pct_forest_saved:.1f}% forest saved",
        f"Step **{model.steps}** / {model.t_max} · strategy **{model.strategy.name}** · "
        f"burning cells **{burning}** · firefighters lost **{model.agents_lost}** · {status}",
        "",
    ]
    if model.coordinator is not None:
        lines += ["**Zones and assignments**", "", "| zone | cells | threat | firefighters |", "|---|---|---|---|"]
        crew: dict[int, list[int]] = {}
        for ff_id, zid in model.coordinator.assignments.items():
            crew.setdefault(zid, []).append(ff_id)
        for zid, zone in sorted(model.coordinator.zones.items(), key=lambda kv: -kv[1].threat)[:8]:
            names = ", ".join(f"FF{i}" for i in sorted(crew.get(zid, []))) or "—"
            lines.append(f"| {zid} | {len(zone.cells)} | {zone.threat:.2f} | {names} |")
        if not model.coordinator.zones:
            lines.append("| — | no known fire | | |")
        lines.append("")
    lines += ["**Firefighters**", "", "| id | position | water | doing |", "|---|---|---|---|"]
    for ff in model.firefighters:
        lines.append(f"| FF{ff.unique_id} | {ff.pos} | {ff.water} | {ff.status} |")
    if model.coordinator is not None:
        lines += ["", "**Last auction rounds**", ""]
        for r in model.auction_log[-5:][::-1]:
            win = f"FF{r['winner']}" if r["winner"] is not None else "unassigned"
            head = f"- step {r['step']}, zone {r['zone_id']} (threat {r['threat']:.2f})"
            lines.append(f"{head}: {_fmt_bids(r['bids'])} → **{win}**")
        if not model.auction_log:
            lines.append("- none yet (waiting for scouts to find fire)")
    counts = " · ".join(f"{k} {v}" for k, v in sorted(model.bus.counts.items()))
    lines += ["", f"**Messages:** {counts or 'none'}", "", "**Recent events**", ""]
    for e in model.events[-6:][::-1]:
        detail = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("step", "kind"))
        lines.append(f"- step {e['step']}: {e['kind']} ({detail})")
    solara.Markdown("\n".join(lines))


@solara.component
def TimeSeries(model: WildfireModel) -> None:
    """Forest saved and burning cells over time (two charts, one axis each)."""
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    fig = Figure(figsize=(7.5, 3.2), facecolor="white")
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(df.index, df["pct_forest_saved"], color="#2a78d6", linewidth=2)
    ax1.set_ylim(0, 101)
    ax1.set_title("% forest saved", fontsize=10, loc="left")
    ax2.plot(df.index, df["burning_cells"], color="#eb6834", linewidth=2)
    ax2.set_title("Burning cells", fontsize=10, loc="left")
    for ax in (ax1, ax2):
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, color="#e4e3df")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png", dpi=100)


@solara.component
def ComparePanel(model: WildfireModel) -> None:
    """Run all three strategies on the current setup and show them side by side."""
    with solara.Card("Why multi-agent? Same fire, three strategies"):
        solara.Button(
            "Compare all 3 strategies on this exact setup",
            color="primary",
            on_click=lambda: comparison_task(model.cfg),
            disabled=comparison_task.pending,
        )
        if comparison_task.pending:
            solara.ProgressLinear(True)
            solara.Text("Running independent, greedy and auction to the end (a few seconds)…")
        elif comparison_task.finished and comparison_task.value is not None:
            table, images = comparison_task.value
            solara.Markdown(table.to_markdown(index=False))
            fig = Figure(figsize=(9, 3.2), facecolor="white")
            for ax, (name, img) in zip(fig.subplots(1, len(images)), images, strict=True):
                ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest")
                ax.set_title(name, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            solara.FigureMatplotlib(fig, format="png", dpi=100)
        elif comparison_task.error:
            solara.Error(f"Comparison failed: {comparison_task.exception}")


# ----------------------------------------------------------------------------- page
model_params = {
    "strategy": {
        "type": "Select",
        "value": _DEFAULTS["simulation"]["strategy"],
        "values": list(STRATEGIES),
        "label": "Strategy",
    },
    "preset": {"type": "Select", "value": "custom", "values": ["custom", *SCENARIOS], "label": "Scenario preset"},
    "n_scouts": {
        "type": "SliderInt",
        "value": _DEFAULTS["agents"]["n_scouts"],
        "min": 0,
        "max": 10,
        "step": 1,
        "label": "Scouts",
    },
    "n_firefighters": {
        "type": "SliderInt",
        "value": _DEFAULTS["agents"]["n_firefighters"],
        "min": 1,
        "max": 30,
        "step": 1,
        "label": "Firefighters",
    },
    "wind_direction_deg": {
        "type": "SliderInt",
        "value": _DEFAULTS["fire"]["wind_direction_deg"],
        "min": 0,
        "max": 315,
        "step": 45,
        "label": "Wind direction (°, 0 = east)",
    },
    "wind_k": {
        "type": "SliderFloat",
        "value": _DEFAULTS["fire"]["wind_k"],
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "label": "Wind strength k",
    },
    "ignitions": {
        "type": "SliderInt",
        "value": _DEFAULTS["fire"]["ignitions"],
        "min": 1,
        "max": 5,
        "step": 1,
        "label": "Ignitions",
    },
    "spread_every": {
        "type": "SliderInt",
        "value": _DEFAULTS["fire"]["spread_every"],
        "min": 1,
        "max": 8,
        "step": 1,
        "label": "Fire updates every N steps",
    },
    "coordinator_fails_at": {
        "type": "SliderInt",
        "value": 0,
        "min": 0,
        "max": 200,
        "step": 10,
        "label": "Coordinator failure step",
    },
    "seed": {"type": "SliderInt", "value": _DEFAULTS["seed"], "min": 0, "max": 99, "step": 1, "label": "Seed"},
}


HEADER_HTML = (
    '<div style="background:#14213D; color:#F7F5EF; padding:18px 28px; border-radius:12px; margin:8px 0 16px; '
    'font-family:Segoe UI, Roboto, Arial, sans-serif">'
    '<div style="font-size:26px; font-weight:600; letter-spacing:0.2px">Wildfire Containment Squad</div>'
    '<div style="font-size:15px; color:#C9D3E3; margin-top:4px">'
    "Scout drones find the fire &middot; the Coordinator auctions fire zones &middot; "
    "firefighters bid, travel with risk-aware A* and fight it</div></div>"
)
PARAMS_HINT_HTML = (
    '<div style="font-size:13px; color:#5F6B7A; margin:-4px 0 12px; line-height:1.4">'
    "Changes apply when you press Reset.<br>Coordinator failure step 0 = never fails.</div>"
)


@solara.component
def Page() -> None:
    """Sidebar with Mesa's run controls and parameters; map + panel side by side; charts below."""
    initial = solara.use_memo(DemoModel, [])
    model = solara.use_reactive(initial)
    parameters = solara.use_reactive({})
    play_interval = solara.use_reactive(150)
    render_interval = solara.use_reactive(1)
    use_threads = solara.use_reactive(False)

    solara.Title("Wildfire Containment Squad")
    solara.HTML(tag="div", unsafe_innerHTML=HEADER_HTML)
    with solara.Columns([1, 2, 1.6]):
        with solara.Column():
            with solara.Card("Run"):
                ModelController(
                    model,
                    model_parameters=parameters,
                    play_interval=play_interval,
                    render_interval=render_interval,
                    use_threads=use_threads,
                )
                solara.SliderInt("Play interval (ms)", value=play_interval, min=10, max=500, step=10)
                solara.SliderInt("Steps per frame", value=render_interval, min=1, max=20)
            with solara.Card("Parameters"):
                solara.HTML(tag="div", unsafe_innerHTML=PARAMS_HINT_HTML)
                ModelCreator(model, model_params, model_parameters=parameters)
        with solara.Column():
            GridView(model.value)
            TimeSeries(model.value)
        SidePanel(model.value)
    ComparePanel(model.value)
