"""Frontier-based exploration target selection for scouts (and idle firefighters).

Frontier cells are unknown or stale cells next to a known, fresh cell. A scout picks the
frontier cell maximising ``age / (1 + distance)`` and ignores cells within the exclusion radius
of targets other scouts have already published.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from environment.cells import Cell


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Cells that are, or are 8-adjacent to, a True cell."""
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    w, h = mask.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out |= padded[1 + dx : 1 + dx + w, 1 + dy : 1 + dy + h]
    return out


def candidate_mask(seen_step: np.ndarray, step: int, stale_after: int) -> np.ndarray:
    """Unknown cells (never seen) or stale cells (seen more than ``stale_after`` steps ago)."""
    return (seen_step < 0) | (step - seen_step > stale_after)


def frontier_mask(seen_step: np.ndarray, step: int, stale_after: int) -> np.ndarray:
    """Unknown/stale cells adjacent to a known fresh cell.

    Falls back to all unknown/stale cells if no such boundary exists (e.g. nothing known yet).
    """
    candidates = candidate_mask(seen_step, step, stale_after)
    frontier = candidates & _dilate8(~candidates)
    return frontier if frontier.any() else candidates


def chebyshev_grid(shape: tuple[int, int], origin: Cell) -> np.ndarray:
    """Chebyshev distance from ``origin`` to every cell."""
    xs, ys = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return np.maximum(np.abs(xs - origin[0]), np.abs(ys - origin[1]))


def exploration_values(
    pos: Cell, seen_step: np.ndarray, step: int, stale_after: int, t_max: int, frontier_only: bool = True
) -> np.ndarray:
    """``age / (1 + chebyshev distance)`` on frontier cells, ``-inf`` elsewhere.

    Never-seen cells count as the oldest possible information (age ``t_max + 1``). With
    ``frontier_only=False`` every unknown/stale cell is a candidate.
    """
    age = np.where(seen_step < 0, t_max + 1, step - seen_step).astype(float)
    values = age / (1.0 + chebyshev_grid(seen_step.shape, pos))
    mask = (
        frontier_mask(seen_step, step, stale_after) if frontier_only else candidate_mask(seen_step, step, stale_after)
    )
    return np.where(mask, values, -np.inf)


def select_target(
    pos: Cell,
    seen_step: np.ndarray,
    step: int,
    stale_after: int,
    t_max: int,
    other_targets: Iterable[Cell] = (),
    exclusion_radius: int = 0,
    excluded: np.ndarray | None = None,
) -> Cell | None:
    """Best frontier cell for an explorer at ``pos``, or ``None`` if every candidate is excluded.

    Cells within ``exclusion_radius`` (Chebyshev) of any cell in ``other_targets`` are skipped,
    as are cells where ``excluded`` is True. If that rules out the whole frontier (e.g. early on,
    when the frontier is a small ring around the base), every unknown/stale cell is considered.
    """
    blocked = np.zeros(seen_step.shape, dtype=bool) if excluded is None else excluded.copy()
    for target in other_targets:
        blocked |= chebyshev_grid(seen_step.shape, target) <= exclusion_radius
    for frontier_only in (True, False):
        values = exploration_values(pos, seen_step, step, stale_after, t_max, frontier_only)
        values[blocked] = -np.inf
        best = int(np.argmax(values))
        if np.isfinite(values.flat[best]):
            x, y = divmod(best, seen_step.shape[1])
            return (x, y)
    return None
