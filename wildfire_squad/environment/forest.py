"""Forest map generation: random lake/tree/sparse maps, the river preset, ignitions, start cells."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from environment.cells import NEIGHBOURS_4, NEIGHBOURS_8, Cell, CellState, chebyshev, neighbours


def _chebyshev_grid(width: int, height: int, origin: Cell) -> np.ndarray:
    """Array of Chebyshev distances from ``origin`` to every cell."""
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    return np.maximum(np.abs(xs - origin[0]), np.abs(ys - origin[1]))


def _add_lakes(state: np.ndarray, grid_cfg: dict[str, Any], rng: np.random.Generator) -> None:
    """Carve ``grid_cfg['lakes']`` rough elliptical water blobs away from the base station."""
    width, height = state.shape
    base = tuple(grid_cfg["base_station"])
    r_min, r_max = grid_cfg["lake_radius"]
    noise = grid_cfg["lake_edge_noise"]
    keep_clear = grid_cfg["base_clear_radius"] + r_max
    base_dist = _chebyshev_grid(width, height, base)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    centres = np.argwhere(base_dist > keep_clear)
    for _ in range(grid_cfg["lakes"]):
        cx, cy = centres[rng.integers(len(centres))]
        rx, ry = rng.uniform(r_min, r_max, size=2)
        d = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
        edge = 1.0 + noise * rng.uniform(-1.0, 1.0, size=state.shape)
        lake = (d <= edge) & (base_dist > grid_cfg["base_clear_radius"])
        state[lake] = CellState.WATER


def _plant_trees(
    state: np.ndarray, grid_cfg: dict[str, Any], fire_cfg: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Turn land into Tree/Empty and mark rectangular sparse patches. Returns the fuel array."""
    land = state != CellState.WATER
    trees = land & (rng.random(state.shape) < grid_cfg["tree_density"])
    state[trees] = CellState.TREE
    fuel = np.where(trees, fire_cfg["fuel_dense"], 0.0)

    n_trees = int(trees.sum())
    target_sparse = grid_cfg["sparse_region_fraction"] * n_trees
    width, height = state.shape
    s_min, s_max = grid_cfg["sparse_patch_size"]
    sparse = np.zeros(state.shape, dtype=bool)
    max_patches = width * height  # generous guard; normally a handful of patches suffice
    for _ in range(max_patches):
        if (sparse & trees).sum() >= target_sparse:
            break
        pw, ph = rng.integers(s_min, s_max + 1, size=2)
        x0 = rng.integers(0, max(1, width - pw))
        y0 = rng.integers(0, max(1, height - ph))
        sparse[x0 : x0 + pw, y0 : y0 + ph] = True
    fuel[sparse & trees] = fire_cfg["fuel_sparse"]
    return fuel


def generate_forest(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random forest map from the seed: lakes, trees, sparse patches. Returns ``(state, fuel)``."""
    grid_cfg = cfg["grid"]
    state = np.full((grid_cfg["width"], grid_cfg["height"]), CellState.EMPTY, dtype=np.int8)
    _add_lakes(state, grid_cfg, rng)
    fuel = _plant_trees(state, grid_cfg, cfg["fire"], rng)
    return state, fuel


def generate_river_forest(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Scenario 6 map: a vertical river at ``presets.river.x`` with two 3-cell-wide crossings."""
    grid_cfg = cfg["grid"]
    river = cfg["presets"]["river"]
    state = np.full((grid_cfg["width"], grid_cfg["height"]), CellState.EMPTY, dtype=np.int8)
    state[river["x"], :] = CellState.WATER
    fuel = _plant_trees(state, grid_cfg, cfg["fire"], rng)
    for y0, y1 in river["crossings"]:
        state[river["x"], y0 : y1 + 1] = CellState.EMPTY
        fuel[river["x"], y0 : y1 + 1] = 0.0
    return state, fuel


def place_ignitions(state: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> list[Cell]:
    """Pick ignition cells: Tree cells far enough from base and from each other.

    On the river preset, ignitions are placed on the far side of the river from the base.
    """
    fire_cfg = cfg["fire"]
    base = tuple(cfg["grid"]["base_station"])
    width, height = state.shape
    far_enough = _chebyshev_grid(width, height, base) >= max(
        fire_cfg["ignition_min_base_distance"], cfg["grid"]["base_clear_radius"] + 1
    )
    allowed = (state == CellState.TREE) & far_enough
    if cfg["simulation"].get("preset") == "river":
        river_x = cfg["presets"]["river"]["x"]
        side = np.arange(width)[:, None] > river_x if base[0] < river_x else np.arange(width)[:, None] < river_x
        allowed &= side

    chosen: list[Cell] = []
    for _ in range(fire_cfg["ignitions"]):
        candidates = [
            (int(x), int(y))
            for x, y in np.argwhere(allowed)
            if all(chebyshev((x, y), c) >= fire_cfg["ignition_min_separation"] for c in chosen)
        ]
        if not candidates:  # separation impossible: fall back to any allowed cell
            candidates = [(int(x), int(y)) for x, y in np.argwhere(allowed) if (x, y) not in chosen]
        if not candidates:
            break
        chosen.append(candidates[rng.integers(len(candidates))])
    return chosen


def generate_competing_forest(
    cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[Cell]]:
    """Targeted allocation test: a low-threat fire near the base and a high-threat fire far away.

    The near ignition sits in a patch of sparse forest (slow spread) and is usually found first;
    the far ignition sits in dense forest (fast spread). A threat-blind allocator sends its crews
    to the near fire; a threat-aware one should prioritise the far fire.
    """
    preset = cfg["presets"]["competing"]
    state, fuel = generate_forest(cfg, rng)
    width, height = state.shape
    r = preset["patch_radius"]
    for (cx, cy), patch_fuel in ((preset["near_ignition"], "fuel_sparse"), (preset["far_ignition"], "fuel_dense")):
        xs = slice(max(0, cx - r), min(width, cx + r + 1))
        ys = slice(max(0, cy - r), min(height, cy + r + 1))
        patch = state[xs, ys]
        patch[patch == CellState.WATER] = CellState.TREE  # keep both fires on land
        fuel[xs, ys] = np.where(patch == CellState.TREE, cfg["fire"][patch_fuel], 0.0)
        state[cx, cy] = CellState.TREE
        fuel[cx, cy] = cfg["fire"][patch_fuel]
    ignitions = [tuple(preset["near_ignition"]), tuple(preset["far_ignition"])]
    return state, fuel, [(int(x), int(y)) for x, y in ignitions]


def build_forest(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[Cell]]:
    """Generate the map selected by ``simulation.preset`` plus its ignition cells."""
    preset = cfg["simulation"].get("preset")
    if preset == "competing":
        return generate_competing_forest(cfg, rng)
    if preset == "river":
        state, fuel = generate_river_forest(cfg, rng)
    else:
        state, fuel = generate_forest(cfg, rng)
    return state, fuel, place_ignitions(state, cfg, rng)


def start_cells(state: np.ndarray, base: Cell, count: int) -> list[Cell]:
    """The ``count`` non-water cells closest to ``base`` (breadth-first), base first.

    Firefighters start on distinct cells of the base-station area so no two share a cell.
    """
    width, height = state.shape
    seen = {base}
    queue = deque([base])
    cells: list[Cell] = []
    while queue and len(cells) < count:
        cell = queue.popleft()
        if state[cell] != CellState.WATER:
            cells.append(cell)
        for nb in neighbours(cell, width, height, NEIGHBOURS_8):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return cells


def refill_cells(state: np.ndarray) -> list[Cell]:
    """Land cells 4-adjacent to water, where a firefighter can refill."""
    width, height = state.shape
    water = state == CellState.WATER
    return [
        (int(x), int(y))
        for x, y in np.argwhere(~water)
        if any(water[nx, ny] for nx, ny in neighbours((x, y), width, height, NEIGHBOURS_4))
    ]
