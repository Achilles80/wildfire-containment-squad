"""Firefighter — model-based, utility-based agent that bids for zones and fights the fire."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa
import numpy as np

from agents.scout import sense_window
from algorithms.auction import utility
from algorithms.frontier import select_target
from algorithms.zones import Zone, firebreak_candidates, live_zone_cells
from belief_map import BeliefMap
from environment.cells import NEIGHBOURS_4, NEIGHBOURS_8, UNKNOWN, Cell, CellState, manhattan, neighbours
from environment.fire import ignition_probability
from messages import Announce, Award, Bid, Done, Message, Observe, Revoke, Target

if TYPE_CHECKING:
    from model import WildfireModel


class Firefighter(mesa.Agent):
    """Moves one cell in 4 directions; extinguishes, cuts firebreaks, refills at water.

    Action priority each step (exactly one action):

    1. out of water → go to the nearest water-side cell and refill;
    2. burning 4-neighbour → extinguish the one with the highest downwind spread probability;
    3. standing on the firebreak target and it is Tree → cut a firebreak;
    4. otherwise → next step of the risk-aware A* path to the target.
    """

    def __init__(self, model: WildfireModel, pos: Cell, belief: BeliefMap) -> None:
        super().__init__(model)
        cfg = model.cfg
        self.belief = belief
        self.water: int = cfg["agents"]["water_max"]
        self.assigned_zone: int | None = None
        self.target_cell: Cell | None = None
        self.path: list[Cell] | None = None
        self.steps_since_plan = 0
        self.refilling_since: int | None = None
        # navigation bookkeeping
        self._nav_goal: Cell | None = None
        self._blocked_steps = 0
        self._explore_target: Cell | None = None
        self._explore_selected_step = -1
        self._unreachable: set[Cell] = set()
        self._bid_cache: dict[tuple[int, int], float] = {}
        self._free_since: int | None = None  # step since which we know of fire but hold no award
        self._live_zone: Zone | None = None
        self._live_zone_key: tuple[int, int, int] | None = None
        self.fallback = False  # acting alone because the Coordinator has gone silent
        # statistics
        self.cells_extinguished = 0
        self.firebreaks_cut = 0
        self.refills = 0
        model.grid.place_agent(self, pos)
        model.bus.subscribe(self.unique_id, self.receive)

    # ------------------------------------------------------------------ helpers
    @property
    def coordinated(self) -> bool:
        """True when a Coordinator allocates zones (greedy / auction strategies)."""
        return self.model.strategy.coordinator

    @property
    def status(self) -> str:
        """Short label for the demo panel."""
        if self.water <= 0:
            return "refilling"
        if self.assigned_zone is not None:
            return f"zone {self.assigned_zone}"
        if self.fallback:
            return "fallback (no award)"
        if self.target_cell is not None:
            return "to fire"
        if self._explore_target is not None:
            return "exploring"
        return "idle"

    def _truth(self, cell: Cell) -> int:
        return int(self.model.state[cell])

    def _zone(self) -> Zone | None:
        """Our assigned zone, tracked live on the belief map between clustering rounds."""
        coordinator = self.model.coordinator
        if coordinator is None or self.assigned_zone is None:
            return None
        snapshot = coordinator.zones.get(self.assigned_zone)
        if snapshot is None:
            return None
        key = (snapshot.zone_id, self.model.steps, self.belief.version)
        if self._live_zone_key != key:
            cells = live_zone_cells(snapshot.cells, self.belief.burning_mask())
            self._live_zone = Zone(snapshot.zone_id, cells, snapshot.threat) if cells else None
            self._live_zone_key = key
        return self._live_zone

    def _become_free(self) -> None:
        self.assigned_zone = None
        self.target_cell = None
        self.path = None
        self._nav_goal = None

    # ------------------------------------------------------------------ perception
    def sense(self) -> None:
        """Observe radius 1 and write it to the (shared or private) belief map."""
        step = self.model.steps
        cells = sense_window(self.model, self.pos, self.model.cfg["agents"]["firefighter_sense_radius"])
        if self.model.strategy.shared_map:
            self.model.bus.send(Observe(self.unique_id, "belief_map", step, cells))
        else:
            self.belief.observe(cells, step)

    def _record(self, cell: Cell) -> None:
        """Write the current true state of a cell we just changed into our belief map."""
        self.belief.observe({cell: self._truth(cell)}, self.model.steps)

    # ------------------------------------------------------------------ messages
    def receive(self, msg: Message) -> None:
        """Handle an instantly delivered message (ANNOUNCE, AWARD, REVOKE)."""
        if isinstance(msg, Announce):
            u = self.bid_utility(msg)
            if u > 0:
                self.model.bus.send(Bid(self.unique_id, "coordinator", self.model.steps, msg.zone_id, u))
        elif isinstance(msg, Award):
            self._become_free()
            self.assigned_zone = msg.zone_id
            # the award's target cell is the zone's reference point used to price bids;
            # with a tactic other than "firebreak" we choose our own cell inside the zone
            tactic = self.model.cfg["firefighter"]["zone_tactic"]
            self.target_cell = msg.target_cell if tactic == "firebreak" else None
            self._explore_target = None
            self._free_since = None
            self.fallback = False
        elif isinstance(msg, Revoke):
            if msg.zone_id == self.assigned_zone:
                self._become_free()

    def bid_utility(self, announce: Announce) -> float:
        """``U(f, z) = threat / (1 + pathcost) × water / water_max`` (0 if unreachable)."""
        if announce.target_cell is None or self.water <= 0:
            return 0.0
        key = (announce.zone_id, self.model.steps)
        if key not in self._bid_cache:
            _, cost = self.model.plan(self.pos, announce.target_cell, self.belief)
            self._bid_cache = {key: cost}  # only the current round is ever reused
        return utility(announce.threat, self._bid_cache[key], self.water, self.model.cfg["agents"]["water_max"])

    def _send_done(self, status: str) -> None:
        zone_id = self.assigned_zone
        self._become_free()
        self.model.bus.send(Done(self.unique_id, "coordinator", self.model.steps, zone_id, status))
        self.model.log_event(f"done_{status}", agent=self.unique_id, zone=zone_id)

    # ------------------------------------------------------------------ actions
    def _adjacent_burning(self) -> list[Cell]:
        m = self.model
        return [c for c in neighbours(self.pos, m.width, m.height, NEIGHBOURS_4) if self._truth(c) == CellState.BURNING]

    def _spread_score(self, b: Cell) -> float:
        """Highest probability that burning cell ``b`` ignites one of its Tree neighbours."""
        m = self.model
        return max(
            (
                ignition_probability(b, n, float(m.fuel[n]), m.wind, m.cfg["fire"])
                for n in neighbours(b, m.width, m.height, NEIGHBOURS_8)
                if self._truth(n) == CellState.TREE
            ),
            default=0.0,
        )

    def extinguish(self, burning: list[Cell]) -> None:
        """Put out the adjacent burning cell that threatens the most downwind spread."""
        cell = max(sorted(burning), key=self._spread_score)
        self.model.state[cell] = CellState.BURNT
        self.model.burn_timer[cell] = 0
        self.water -= self.model.cfg["agents"]["extinguish_water_cost"]
        self.cells_extinguished += 1
        self.model.cells_extinguished += 1
        self._record(cell)

    def _standing_at_risk(self) -> bool:
        """Our own cell is Tree and next to fire, so the fire could spread onto us."""
        m = self.model
        if self._truth(self.pos) != CellState.TREE:
            return False
        return any(self._truth(b) == CellState.BURNING for b in neighbours(self.pos, m.width, m.height, NEIGHBOURS_8))

    def cut_firebreak(self, retarget: bool = True) -> None:
        """Turn the Tree under us into a Firebreak (no water cost), then pick the next target."""
        self.model.state[self.pos] = CellState.FIREBREAK
        self.firebreaks_cut += 1
        self._record(self.pos)
        if retarget:
            self.target_cell = None
            self._choose_zone_target()

    def _refill(self) -> None:
        """Go to the nearest water-side cell; when next to water, refill (uses the step)."""
        m = self.model
        if self.refilling_since is None:
            self.refilling_since = m.steps
        if any(self._truth(c) == CellState.WATER for c in neighbours(self.pos, m.width, m.height, NEIGHBOURS_4)):
            self.water = m.cfg["agents"]["water_max"]
            self.refilling_since = None
            self.refills += 1
            self.path = None
            self._nav_goal = None
            return
        spots = sorted(m.refill_cells, key=lambda c: (manhattan(c, self.pos), c))
        for spot in spots[: m.cfg["firefighter"]["target_candidates"]]:
            if spot in self._unreachable:
                continue
            if self._go_to(spot):
                return
            self._unreachable.add(spot)

    # ------------------------------------------------------------------ targets
    def _is_fuel(self, cell: Cell) -> bool:
        s = self.belief.state[cell]
        return s == CellState.TREE or (s == UNKNOWN and self.model.fuel[cell] > 0)

    def _spread_value(self, b: Cell) -> float:
        """Expected ignitions per fire update that burning cell ``b`` threatens (believed fuel only)."""
        m = self.model
        return sum(
            ignition_probability(b, n, float(m.fuel[n]), m.wind, m.cfg["fire"])
            for n in neighbours(b, m.width, m.height, NEIGHBOURS_8)
            if self._is_fuel(n)
        )

    def _utility_candidates(self, zone: Zone) -> list[Cell]:
        """Burning cells of the zone ranked by spread stopped per step of travel.

        ``value(b) = spread threatened by b / (1 + distance)``; cells a teammate already targets
        are skipped so two firefighters do not queue for the same cell.
        """
        m = self.model
        taken = {ff.target_cell for ff in m.firefighters if ff is not self and ff.target_cell is not None}
        burning = [c for c in zone.cells if self.belief.state[c] == CellState.BURNING and c not in taken]
        scored = [(self._spread_value(b) / (1 + manhattan(self.pos, b)), b) for b in burning]
        return [b for v, b in sorted(scored, key=lambda vb: (-vb[0], vb[1])) if v > 0]

    def _choose_zone_target(self) -> None:
        """Pick our target inside the assigned zone (see ``firefighter.zone_tactic``)."""
        zone = self._zone()
        if zone is None:
            return
        m = self.model
        tactic = m.cfg["firefighter"]["zone_tactic"]
        cands: list[Cell] = []
        if tactic == "utility":
            cands = self._utility_candidates(zone)
        elif tactic == "nearest":
            cands = sorted(
                (c for c in zone.cells if self.belief.state[c] == CellState.BURNING),
                key=lambda c: (manhattan(self.pos, c), c),
            )
        if not cands:  # "firebreak" tactic, or nothing left to put out: downwind firebreak edge
            cands = firebreak_candidates(zone, self.belief.state, m.fuel, m.wind, from_pos=self.pos)
        for cand in cands[: m.cfg["firefighter"]["target_candidates"]]:
            path, _ = m.plan(self.pos, cand, self.belief, self._blocked_first())
            if path is not None:
                self.target_cell = cand
                self.path = path
                self._nav_goal = cand
                self.steps_since_plan = 0
                return
        self._send_done("abandoned")

    def _nearest_known_fire(self) -> Cell | None:
        burning = self.belief.burning_mask()
        if not burning.any():
            return None
        cells = [tuple(int(v) for v in c) for c in np.argwhere(burning)]
        cells = [c for c in cells if c not in self._unreachable]
        return min(cells, key=lambda c: (manhattan(c, self.pos), c)) if cells else None

    def _choose_explore_target(self) -> None:
        """Frontier exploration when there is nothing else to do."""
        m = self.model
        t = self._explore_target
        if t is not None and t != self.pos and self.belief.seen_step[t] < self._explore_selected_step:
            return
        others: list[Cell] = []
        if m.strategy.shared_map:
            others = [c for aid, c in m.explore_targets.items() if aid != self.unique_id and c]
            others += [c for c in m.scout_targets.values() if c]
        excluded = np.zeros(self.belief.state.shape, dtype=bool)
        for c in self._unreachable:
            excluded[c] = True
        excluded |= ~self.belief.passable_mask()
        self._explore_target = select_target(
            self.pos,
            self.belief.seen_step,
            m.steps,
            m.cfg["scout"]["stale_after"],
            m.cfg["simulation"]["t_max"],
            others,
            m.cfg["scout"]["target_exclusion_radius"],
            excluded,
        )
        self._explore_selected_step = m.steps
        if m.strategy.shared_map:
            m.bus.send(Target(self.unique_id, "explorers", m.steps, self._explore_target))

    def _task_target(self) -> Cell | None:
        """Where to go this step when not refilling/extinguishing/cutting."""
        m = self.model
        if self.coordinated:
            if self.assigned_zone is not None:
                t = self.target_cell
                stale = self.steps_since_plan >= m.cfg["firefighter"]["replan_every"]
                valid = t is not None and (self._is_fuel(t) or self.belief.state[t] == CellState.BURNING)
                if not valid or stale:
                    self._choose_zone_target()
                return self.target_cell
            if not m.shared_belief.burning_mask().any():
                self._free_since = None
                self.fallback = False
                self.target_cell = None
                if m.cfg["firefighter"]["explore_when_idle"]:
                    self._choose_explore_target()
                    return self._explore_target
                return None
            self._explore_target = None
            return self._await_award_or_fallback()
        # independent: nearest fire this firefighter knows of, else explore on its own
        fire = self._nearest_known_fire()
        self.target_cell = fire
        if fire is not None:
            self._explore_target = None
            return fire
        if m.cfg["firefighter"]["explore_when_idle"]:
            self._choose_explore_target()
            return self._explore_target
        return None

    def _await_award_or_fallback(self) -> Cell | None:
        """Wait for an AWARD; after ``award_timeout`` silent steps, fight the nearest known fire alone.

        This is the Coordinator-failure fallback from Review 1: if the Coordinator stops
        allocating, firefighters degrade to greedy nearest-fire behaviour instead of idling.
        """
        m = self.model
        timeout = m.cfg["firefighter"]["award_timeout"]
        if self._free_since is None:
            self._free_since = m.steps
        if timeout is None or m.steps - self._free_since < timeout:
            return None
        if not self.fallback:
            self.fallback = True
            m.log_event("fallback", agent=self.unique_id)
        self.target_cell = self._nearest_known_fire()
        return self.target_cell

    # ------------------------------------------------------------------ navigation
    def _unsafe_to_enter(self, cell: Cell) -> bool:
        """Entering ``cell`` now would put us on fuel near known fire just as the fire spreads.

        The buffer is 2 cells because the map may be a little out of date near the fire front.
        """
        m = self.model
        if not m.cfg["firefighter"]["secure_footing"] or not m.fire_spreads_this_step:
            return False
        if not self._is_fuel(cell):
            return False
        x, y = cell
        near = self.belief.state[max(0, x - 2) : x + 3, max(0, y - 2) : y + 3]
        return bool((near == CellState.BURNING).any())

    def _blocked_first(self) -> set[Cell]:
        return self.model.reserved - {self.pos}

    def _replan(self, goal: Cell) -> bool:
        blocked = self._blocked_first()
        path, _ = self.model.plan(self.pos, goal, self.belief, blocked)
        if path is None and blocked:
            # the only way on is through a reserved cell: keep that route and wait for it to clear
            path, _ = self.model.plan(self.pos, goal, self.belief)
        self.path = path
        self._nav_goal = goal
        self.steps_since_plan = 0
        self._blocked_steps = 0
        return path is not None

    def _go_to(self, goal: Cell) -> bool:
        """Take one A* step toward ``goal``. Returns False if no path exists."""
        m = self.model
        need = (
            self.path is None
            or goal != self._nav_goal
            or self.steps_since_plan >= m.cfg["firefighter"]["replan_every"]
            or (self.path and not self.belief.is_passable_for_firefighter(*self.path[0]))
        )
        if need and not self._replan(goal):
            return False
        if not self.path:
            return True  # already at (or next to) the goal
        nxt = self.path[0]
        if self._truth(nxt) in (CellState.BURNING, CellState.WATER):
            self._record(nxt)
            if not self._replan(goal):
                return False
            if not self.path:
                return True
            nxt = self.path[0]
        if self._unsafe_to_enter(nxt):
            return True  # hold one step rather than stand on threatened fuel as the fire moves
        if nxt in m.reserved:
            m.request_yield(nxt, self.unique_id)  # an idle teammate standing there will step aside
            self._blocked_steps += 1
            if self._blocked_steps < m.cfg["firefighter"]["blocked_replan_after"]:
                return True  # wait
            if not self._replan(goal):
                return False
            if not self.path or self.path[0] in m.reserved:
                return True
            nxt = self.path[0]
        m.reserved.discard(self.pos)
        m.reserved.add(nxt)
        m.grid.move_agent(self, nxt)
        self.path.pop(0)
        self._blocked_steps = 0
        return True

    def _yield_if_asked(self) -> None:
        """Idle and blocking a teammate: step onto a free neighbouring cell to let it pass."""
        m = self.model
        request = m.yield_requests.pop(self.pos, None)
        if request is None or request[1] < m.steps - 1:  # requests expire after one step
            return
        requester = request[0]
        for cell in neighbours(self.pos, m.width, m.height, NEIGHBOURS_4):
            if (
                cell not in m.reserved
                and self.belief.is_passable_for_firefighter(*cell)
                and not self._unsafe_to_enter(cell)
            ):
                if self._truth(cell) in (CellState.BURNING, CellState.WATER):
                    continue
                m.reserved.discard(self.pos)
                m.reserved.add(cell)
                m.grid.move_agent(self, cell)
                self.path = None
                m.log_event("yield", agent=self.unique_id, to=requester)
                return

    # ------------------------------------------------------------------ main loop
    def _check_zone(self) -> None:
        """Report DONE(contained) once our zone has no believed-burning cells left."""
        if not self.coordinated or self.assigned_zone is None:
            return
        if self._zone() is None:  # nothing of our zone still burns
            self._send_done("contained")

    def step(self) -> None:
        """Sense, then exactly one action by priority."""
        self.steps_since_plan += 1
        self.sense()
        self._check_zone()

        if self.model.cfg["firefighter"]["secure_footing"] and self._standing_at_risk():
            self.cut_firebreak(retarget=False)  # safety first: never stand on fuel next to fire
            return
        if self.water <= 0:
            self._refill()
            return
        burning = self._adjacent_burning()
        if burning:
            self.extinguish(burning)
            return
        if self.assigned_zone is not None and self.target_cell == self.pos and self._truth(self.pos) == CellState.TREE:
            self.cut_firebreak()
            return

        goal = self._task_target()
        if goal is None:
            self._yield_if_asked()
            return
        if not self._go_to(goal):
            if self.assigned_zone is not None:
                self._send_done("abandoned")
            else:
                self._unreachable.add(goal)
                self.target_cell = None
                self._explore_target = None
