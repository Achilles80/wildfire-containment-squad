"""Risk-aware A* (and UCS for comparison) on a 4-connected grid.

Step cost entering cell ``n`` is ``c(n) = 1 + λ × risk(n)``. The heuristic is the Manhattan
distance to the nearest goal: every step costs at least 1 and moves exactly one cell, so it
never overestimates (admissible) and satisfies the triangle inequality (consistent).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Iterable

import numpy as np

from environment.cells import NEIGHBOURS_4, Cell

SearchResult = tuple[list[Cell] | None, float, int]


def step_cost_grid(risk: np.ndarray, risk_lambda: float) -> np.ndarray:
    """``c(n) = 1 + λ × risk(n)`` for every cell."""
    return 1.0 + risk_lambda * risk


def goal_cells(target: Cell, passable: np.ndarray) -> set[Cell]:
    """The target itself if passable, else its passable 4-neighbours."""
    if passable[target]:
        return {target}
    width, height = passable.shape
    x, y = target
    return {
        (x + dx, y + dy)
        for dx, dy in NEIGHBOURS_4
        if 0 <= x + dx < width and 0 <= y + dy < height and passable[x + dx, y + dy]
    }


def astar(
    start: Cell,
    goals: Iterable[Cell],
    passable: np.ndarray,
    step_cost: np.ndarray,
    blocked_first: Iterable[Cell] = (),
    use_heuristic: bool = True,
) -> SearchResult:
    """Cheapest 4-connected path from ``start`` to any cell in ``goals``.

    Args:
        start: current cell (always allowed, even if it is itself impassable).
        goals: acceptable end cells (impassable goals are ignored).
        passable: boolean ``[x, y]`` array of cells that may be entered.
        step_cost: ``[x, y]`` array, cost of entering each cell (all ≥ 1).
        blocked_first: cells that may not be entered on the *first* move only
            (other agents' reservations for this step).
        use_heuristic: ``False`` turns A* into uniform-cost search (h = 0).

    Returns:
        ``(path, cost, nodes_expanded)``; ``path`` excludes ``start`` and is ``None`` (with
        ``cost = inf``) when no goal is reachable.
    """
    width, height = passable.shape
    pas = passable.ravel().tolist()
    cost = step_cost.ravel().tolist()
    s = start[0] * height + start[1]
    goal_idx = {gx * height + gy for gx, gy in goals if pas[gx * height + gy] or (gx, gy) == start}
    if not goal_idx:
        return None, math.inf, 0
    if s in goal_idx:
        return [], 0.0, 1
    gxs = [g // height for g in goal_idx]
    gys = [g % height for g in goal_idx]
    blocked = {bx * height + by for bx, by in blocked_first}

    def h(i: int) -> int:
        if not use_heuristic:
            return 0
        x, y = divmod(i, height)
        return min(abs(x - gx) + abs(y - gy) for gx, gy in zip(gxs, gys, strict=True))

    counter = itertools.count()
    h0 = h(s)
    frontier: list[tuple[float, int, int, int]] = [(h0, h0, next(counter), s)]
    g: dict[int, float] = {s: 0.0}
    parent: dict[int, int] = {}
    closed: set[int] = set()
    expanded = 0
    while frontier:
        _, _, _, i = heapq.heappop(frontier)
        if i in closed:
            continue
        closed.add(i)
        expanded += 1
        if i in goal_idx:
            total = g[i]
            path: list[Cell] = []
            while i != s:
                x, y = divmod(i, height)
                path.append((x, y))
                i = parent[i]
            path.reverse()
            return path, total, expanded
        x, y = divmod(i, height)
        gi = g[i]
        for dx, dy in NEIGHBOURS_4:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            j = nx * height + ny
            if not pas[j] or j in closed or (i == s and j in blocked):
                continue
            ng = gi + cost[j]
            if ng < g.get(j, math.inf):
                g[j] = ng
                parent[j] = i
                hj = h(j)
                heapq.heappush(frontier, (ng + hj, hj, next(counter), j))
    return None, math.inf, expanded


def ucs(
    start: Cell,
    goals: Iterable[Cell],
    passable: np.ndarray,
    step_cost: np.ndarray,
    blocked_first: Iterable[Cell] = (),
) -> SearchResult:
    """Uniform-cost search: A* with ``h = 0``. Used only as the comparison baseline."""
    return astar(start, goals, passable, step_cost, blocked_first, use_heuristic=False)


def plan_path(
    start: Cell,
    target: Cell,
    passable: np.ndarray,
    step_cost: np.ndarray,
    blocked_first: Iterable[Cell] = (),
    use_heuristic: bool = True,
) -> SearchResult:
    """Path to ``target``, or to a cell next to it if the target itself is impassable."""
    return astar(start, goal_cells(target, passable), passable, step_cost, blocked_first, use_heuristic)
