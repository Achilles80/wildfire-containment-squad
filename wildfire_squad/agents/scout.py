"""Scout Drone — model-based, goal-based explorer that feeds the shared belief map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa

from algorithms.frontier import select_target
from environment.cells import Cell
from messages import Observe, Target

if TYPE_CHECKING:
    from model import WildfireModel


def sense_window(model: WildfireModel, pos: Cell, radius: int) -> dict[Cell, int]:
    """Ground-truth states of all cells within Chebyshev ``radius`` of ``pos``."""
    x, y = pos
    x0, x1 = max(0, x - radius), min(model.width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(model.height, y + radius + 1)
    window = model.state[x0:x1, y0:y1]
    return {(x0 + i, y0 + j): int(window[i, j]) for i in range(x1 - x0) for j in range(y1 - y0)}


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


class ScoutDrone(mesa.Agent):
    """Flies one cell in 8 directions per step (or hovers), cannot be harmed by fire.

    Cycle each step: sense (radius 3) → write belief map → choose frontier target → move.
    """

    def __init__(self, model: WildfireModel, pos: Cell) -> None:
        super().__init__(model)
        self.target: Cell | None = None
        self.target_selected_step = -1
        self.wind = model.wind  # wind sensor
        model.grid.place_agent(self, pos)

    def sense(self) -> None:
        """Observe the ground truth around the drone and broadcast it to the belief map."""
        step = self.model.steps
        cells = sense_window(self.model, self.pos, self.model.cfg["agents"]["scout_sense_radius"])
        self.wind = self.model.wind
        self.model.bus.send(Observe(self.unique_id, "belief_map", step, cells))

    def _needs_new_target(self) -> bool:
        if self.target is None or self.pos == self.target:
            return True
        # re-select once anyone has observed the target since we picked it
        return self.model.shared_belief.seen_step[self.target] >= self.target_selected_step

    def choose_target(self) -> None:
        """Pick the best frontier cell away from other scouts' published targets."""
        if not self._needs_new_target():
            return
        model = self.model
        scout_cfg = model.cfg["scout"]
        others = [t for sid, t in model.scout_targets.items() if sid != self.unique_id and t]
        self.target = select_target(
            self.pos,
            model.shared_belief.seen_step,
            model.steps,
            scout_cfg["stale_after"],
            model.cfg["simulation"]["t_max"],
            others,
            scout_cfg["target_exclusion_radius"],
        )
        self.target_selected_step = model.steps
        model.bus.send(Target(self.unique_id, "scouts", model.steps, self.target))

    def move(self) -> None:
        """One Chebyshev step straight toward the target (scouts fly, so no pathfinding)."""
        if self.target is None:
            return
        x, y = self.pos
        nxt = (x + _sign(self.target[0] - x), y + _sign(self.target[1] - y))
        self.model.grid.move_agent(self, nxt)

    def step(self) -> None:
        """Sense → update belief map → choose goal → act."""
        self.sense()
        self.choose_target()
        self.move()
