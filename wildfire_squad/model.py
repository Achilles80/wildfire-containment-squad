"""WildfireModel: forest grid, agents, the fixed step order, and metrics."""

from __future__ import annotations

from typing import Any

import mesa
import numpy as np
from mesa.space import MultiGrid

from agents.coordinator import Coordinator
from agents.firefighter import Firefighter
from agents.scout import ScoutDrone
from algorithms.astar import goal_cells, plan_path, step_cost_grid, ucs
from belief_map import BeliefMap
from environment.cells import UNKNOWN, Cell, CellState
from environment.fire import ignite, risk_map, spread_step, wind_vector
from environment.forest import build_forest, refill_cells, start_cells
from messages import Message, MessageBus, Observe, Target
from settings import load_config
from strategies import get_strategy


class WildfireModel(mesa.Model):
    """Scouts, firefighters and a Coordinator containing a stochastic wildfire.

    Each step runs in a fixed order:

    1. scouts sense, update the shared map, choose targets and move;
    2. the Coordinator (every ``cluster_every`` steps) clusters zones and runs auctions;
    3. firefighters act in id order using per-step cell reservations;
    4. the fire spreads and burn timers advance;
    5. firefighters standing on a burning cell are caught and removed;
    6. metrics are collected and termination is checked.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int | None = None) -> None:
        """
        Args:
            config: overrides deep-merged onto ``config.yaml`` (e.g. a scenario).
            seed: overrides ``config['seed']``.
        """
        cfg = load_config(config)
        if seed is not None:
            cfg["seed"] = seed
        super().__init__(rng=int(cfg["seed"]))
        self.cfg = cfg
        self.seed_value = int(cfg["seed"])
        self.strategy = get_strategy(cfg["simulation"]["strategy"])
        self.width = cfg["grid"]["width"]
        self.height = cfg["grid"]["height"]
        self.t_max = cfg["simulation"]["t_max"]

        # separate RNG streams: the map and the fire never depend on what the agents do
        map_rng = np.random.default_rng([self.seed_value, 0])
        self.fire_rng = np.random.default_rng([self.seed_value, 1])

        # ---- environment (ground truth)
        self.state, self.fuel, self.ignitions = build_forest(cfg, map_rng)
        self.burn_timer = np.zeros(self.state.shape, dtype=np.int16)
        ignite(self.state, self.burn_timer, self.ignitions, cfg["fire"]["burn_duration"])
        self.wind = wind_vector(cfg["fire"]["wind_direction_deg"])
        self.water_mask = self.state == CellState.WATER
        self.refill_cells = refill_cells(self.state)
        self.base = tuple(cfg["grid"]["base_station"])
        self.grid = MultiGrid(self.width, self.height, torus=False)

        # ---- communication and shared knowledge
        self.bus = MessageBus(cfg["simulation"]["message_log_size"])
        self.shared_belief = BeliefMap(self.width, self.height, self.water_mask)
        self.scout_targets: dict[int, Cell | None] = {}
        self.explore_targets: dict[int, Cell | None] = {}
        self.bus.subscribe("belief_map", self._on_observe)
        self.bus.subscribe("scouts", self._on_target(self.scout_targets))
        self.bus.subscribe("explorers", self._on_target(self.explore_targets))
        self.reserved: set[Cell] = set()
        self.yield_requests: dict[Cell, tuple[int, int]] = {}  # blocker cell -> (requester id, step asked)
        self.events: list[dict[str, Any]] = []
        self.auction_log: list[dict[str, Any]] = []
        self._plan_cache: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
        self.astar_nodes_expanded = 0
        self.ucs_nodes_expanded = 0
        self.astar_calls = 0

        # ---- agents
        agents_cfg = cfg["agents"]
        self.scouts = [ScoutDrone(self, self.base) for _ in range(agents_cfg["n_scouts"])]
        self.coordinator = Coordinator(self) if self.strategy.coordinator else None
        n_ff = agents_cfg["n_firefighters"]
        self.firefighters: list[Firefighter] = [
            Firefighter(self, cell, self._firefighter_belief()) for cell in start_cells(self.state, self.base, n_ff)
        ]
        self.n_firefighters = len(self.firefighters)

        # ---- metrics
        self.trees_initial = int((self.state == CellState.TREE).sum()) + len(self.ignitions)
        self.agents_lost = 0
        self.cells_extinguished = 0
        self.t_contain: int | None = None
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "pct_forest_saved": lambda m: m.pct_forest_saved,
                "burning_cells": lambda m: int((m.state == CellState.BURNING).sum()),
                "agents_lost": "agents_lost",
                "mean_belief_staleness": lambda m: m.mean_belief_staleness,
                "zones": lambda m: len(m.coordinator.zones) if m.coordinator else 0,
            }
        )
        self.running = True
        self.datacollector.collect(self)

    # ------------------------------------------------------------------ plumbing
    def _firefighter_belief(self) -> BeliefMap:
        if self.strategy.shared_map:
            return self.shared_belief
        return BeliefMap(self.width, self.height, self.water_mask)

    def _on_observe(self, msg: Message) -> None:
        assert isinstance(msg, Observe)
        self.shared_belief.observe(msg.cells, msg.step)

    @staticmethod
    def _on_target(table: dict[int, Cell | None]):
        def handler(msg: Message) -> None:
            assert isinstance(msg, Target)
            table[msg.sender] = msg.cell

        return handler

    def request_yield(self, cell: Cell, requester: int) -> None:
        """Ask whoever stands on ``cell`` to step aside if it is idle (handled on its next turn)."""
        self.yield_requests[cell] = (requester, self.steps)

    def log_event(self, kind: str, **data: Any) -> None:
        """Append a key event (award, abandon, agent lost, ...) to ``self.events``."""
        self.events.append({"step": self.steps, "kind": kind, **data})

    def _planning_grids(self, belief: BeliefMap) -> tuple[np.ndarray, np.ndarray]:
        """Passable mask and step-cost grid for a belief map (cached per belief version)."""
        key = id(belief)
        cached = self._plan_cache.get(key)
        if cached is not None and cached[0] == belief.version:
            return cached[1], cached[2]
        burning = belief.burning_mask()
        fuel_believed = np.where((belief.state == CellState.TREE) | (belief.state == UNKNOWN), self.fuel, 0.0)
        risk = risk_map(burning, fuel_believed, self.wind, self.cfg["fire"])
        passable = belief.passable_mask()
        cost = step_cost_grid(risk, self.cfg["firefighter"]["risk_lambda"])
        self._plan_cache[key] = (belief.version, passable, cost)
        return passable, cost

    def plan(
        self, start: Cell, target: Cell, belief: BeliefMap, blocked_first: set[Cell] | frozenset = frozenset()
    ) -> tuple[list[Cell] | None, float]:
        """Risk-aware A* on ``belief``; also runs UCS on the same query when measuring."""
        passable, cost = self._planning_grids(belief)
        path, total, expanded = plan_path(start, target, passable, cost, blocked_first)
        self.astar_calls += 1
        self.astar_nodes_expanded += expanded
        if self.cfg["simulation"]["measure_ucs"]:
            _, _, ucs_expanded = ucs(start, goal_cells(target, passable), passable, cost, blocked_first)
            self.ucs_nodes_expanded += ucs_expanded
        return path, total

    @property
    def fire_spreads_this_step(self) -> bool:
        """Fire updates once every ``fire.spread_every`` agent steps (the rules are known to agents)."""
        return self.steps % self.cfg["fire"]["spread_every"] == 0

    # ------------------------------------------------------------------ metrics
    @property
    def trees_saved(self) -> int:
        """Tree cells still standing (firebreaks do not count)."""
        return int((self.state == CellState.TREE).sum())

    @property
    def pct_forest_saved(self) -> float:
        """Primary metric: ``trees_saved / trees_initial × 100``."""
        return 100.0 * self.trees_saved / self.trees_initial if self.trees_initial else 100.0

    @property
    def mean_belief_staleness(self) -> float:
        """Mean age of known cells in the map(s) the firefighters plan on."""
        if self.strategy.shared_map or not self.firefighters:
            return self.shared_belief.mean_staleness(self.steps)
        return float(np.mean([ff.belief.mean_staleness(self.steps) for ff in self.firefighters]))

    @property
    def score(self) -> float:
        """``w1 × T_saved/T_initial − w2 × t_contain/t_max − w3 × A_lost/A_total``."""
        w = self.cfg["score"]
        t_contain = self.t_contain if self.t_contain is not None else self.t_max
        saved = self.trees_saved / self.trees_initial if self.trees_initial else 1.0
        lost = self.agents_lost / self.n_firefighters if self.n_firefighters else 0.0
        return w["w1"] * saved - w["w2"] * t_contain / self.t_max - w["w3"] * lost

    def final_metrics(self) -> dict[str, Any]:
        """Metrics recorded per run in the experiments."""
        return {
            "trees_initial": self.trees_initial,
            "trees_saved": self.trees_saved,
            "pct_forest_saved": self.pct_forest_saved,
            "t_contain": self.t_contain if self.t_contain is not None else self.t_max,
            "contained": self.t_contain is not None,
            "agents_lost": self.agents_lost,
            "mean_belief_staleness": self.mean_belief_staleness,
            "astar_nodes_expanded": self.astar_nodes_expanded,
            "ucs_nodes_expanded": self.ucs_nodes_expanded,
            "astar_calls": self.astar_calls,
            "cells_extinguished": self.cells_extinguished,
            "firebreaks_cut": int((self.state == CellState.FIREBREAK).sum()),
            "messages": sum(self.bus.counts.values()),
            "score": self.score,
            "steps": self.steps,
        }

    # ------------------------------------------------------------------ step
    def _safety_check(self) -> None:
        """Firefighters on a cell that is now burning are caught and removed."""
        for ff in list(self.firefighters):
            if self.state[ff.pos] == CellState.BURNING:
                self.log_event("agent_lost", agent=ff.unique_id, pos=ff.pos)
                self.firefighters.remove(ff)
                self.grid.remove_agent(ff)
                ff.remove()
                self.agents_lost += 1

    def _grant_oracle_knowledge(self) -> None:
        """Studies only: overwrite every belief map with the ground truth (perfect information)."""
        maps = {id(self.shared_belief): self.shared_belief}
        maps.update({id(ff.belief): ff.belief for ff in self.firefighters})
        for belief in maps.values():
            belief.state[:] = self.state
            belief.seen_step[:] = self.steps
            belief.version += 1

    def step(self) -> None:
        """Advance the simulation by one step (see class docstring for the order)."""
        if self.cfg["simulation"]["oracle_knowledge"]:
            self._grant_oracle_knowledge()
        for scout in self.scouts:
            scout.step()
        if self.coordinator is not None:
            self.coordinator.step()
        self.reserved = {ff.pos for ff in self.firefighters}
        for ff in sorted(self.firefighters, key=lambda f: f.unique_id):
            ff.step()
        if self.fire_spreads_this_step:
            spread_step(self.state, self.fuel, self.burn_timer, self.wind, self.cfg["fire"], self.fire_rng)
        self._safety_check()

        self.datacollector.collect(self)
        if not (self.state == CellState.BURNING).any():
            self.t_contain = self.steps
            self.running = False
        elif self.steps >= self.t_max:
            self.running = False

    def run(self) -> dict[str, Any]:
        """Run headless until termination and return :meth:`final_metrics`."""
        while self.running:
            self.step()
        return self.final_metrics()


def run_once(config: dict[str, Any] | None = None, seed: int | None = None) -> dict[str, Any]:
    """Convenience: build a model, run it to the end, return the final metrics."""
    return WildfireModel(config, seed).run()
