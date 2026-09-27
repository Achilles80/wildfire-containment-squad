"""Shared belief map: the team's knowledge of every cell and when it was last seen."""

from __future__ import annotations

import numpy as np

from environment.cells import UNKNOWN, Cell, CellState

UNKNOWN_AGE = np.iinfo(np.int32).max // 2  # "very large" age for never-seen cells


class BeliefMap:
    """Last observed state and observation step for every cell.

    Conflicting observations are fused with *newest timestamp wins*. Static terrain (where the
    lakes and rivers are) is treated as known a priori through ``water_prior``; the dynamic fire
    state is not, so every cell starts ``UNKNOWN``.
    """

    def __init__(self, width: int, height: int, water_prior: np.ndarray | None = None) -> None:
        self.width = width
        self.height = height
        self.state = np.full((width, height), UNKNOWN, dtype=np.int8)
        self.seen_step = np.full((width, height), -1, dtype=np.int32)
        self.water_prior = np.zeros((width, height), dtype=bool) if water_prior is None else water_prior.copy()
        self.version = 0  # bumped whenever a cell's believed state changes

    def observe(self, cells: dict[Cell, int], step: int) -> None:
        """Record observations; a cell is written only if ``step >= seen_step`` for it."""
        for (x, y), cell_state in cells.items():
            if step >= self.seen_step[x, y]:
                if self.state[x, y] != cell_state:
                    self.state[x, y] = cell_state
                    self.version += 1
                self.seen_step[x, y] = step

    def age(self, step: int) -> np.ndarray:
        """``step − seen_step`` per cell; never-seen cells get a very large age."""
        return np.where(self.seen_step < 0, UNKNOWN_AGE, step - self.seen_step)

    def known_mask(self) -> np.ndarray:
        """Cells observed at least once."""
        return self.seen_step >= 0

    def stale_mask(self, step: int, stale_after: int) -> np.ndarray:
        """Known cells whose information is older than ``stale_after`` steps."""
        return self.known_mask() & (step - self.seen_step > stale_after)

    def burning_mask(self) -> np.ndarray:
        """Cells believed to be burning."""
        return self.state == CellState.BURNING

    def burning_cells(self) -> list[Cell]:
        """Cells believed to be burning, as ``(x, y)`` tuples."""
        return [(int(x), int(y)) for x, y in np.argwhere(self.burning_mask())]

    def passable_mask(self) -> np.ndarray:
        """Cells a firefighter may plan through: not Burning, not Water (Unknown is passable)."""
        return ~(self.burning_mask() | (self.state == CellState.WATER) | self.water_prior)

    def is_passable_for_firefighter(self, x: int, y: int) -> bool:
        """Not Burning, not Water; Unknown treated as passable."""
        s = self.state[x, y]
        return s != CellState.BURNING and s != CellState.WATER and not self.water_prior[x, y]

    def mean_staleness(self, step: int) -> float:
        """Mean age of known cells (0 if nothing is known yet)."""
        known = self.known_mask()
        if not known.any():
            return 0.0
        return float((step - self.seen_step[known]).mean())
