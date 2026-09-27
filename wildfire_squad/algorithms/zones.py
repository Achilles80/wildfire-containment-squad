"""Fire-zone clustering (connected components), stable zone ids, threat and firebreak targets."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from environment.cells import NEIGHBOURS_8, UNKNOWN, Cell, CellState, manhattan, neighbours
from environment.fire import Vector, ignition_probability


@dataclass
class Zone:
    """A connected group of burning cells that the Coordinator auctions as one task."""

    zone_id: int
    cells: frozenset[Cell]
    threat: float = 0.0
    target_cell: Cell | None = None
    changed: bool = False  # created by a merge or split this round

    @property
    def centroid(self) -> tuple[float, float]:
        """Mean position of the zone's cells."""
        xs, ys = zip(*self.cells, strict=True)
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class ClusterResult:
    """Output of one clustering round."""

    zones: dict[int, Zone]
    new_ids: set[int] = field(default_factory=set)
    changed_ids: set[int] = field(default_factory=set)
    removed_ids: set[int] = field(default_factory=set)
    next_id: int = 0


def connected_components(mask: np.ndarray, link_distance: int = 1) -> list[frozenset[Cell]]:
    """Connected components of True cells, in scan order.

    Two True cells are linked when their Chebyshev distance is at most ``link_distance``;
    ``link_distance = 1`` is ordinary 8-connectivity. Larger values keep a ragged fire front
    that is broken by a few burnt or extinguished cells together as one zone.
    """
    width, height = mask.shape
    d = link_distance
    offsets = tuple((dx, dy) for dx in range(-d, d + 1) for dy in range(-d, d + 1) if (dx, dy) != (0, 0))
    seen = np.zeros_like(mask, dtype=bool)
    components: list[frozenset[Cell]] = []
    for sx, sy in np.argwhere(mask):
        start = (int(sx), int(sy))
        if seen[start]:
            continue
        seen[start] = True
        comp = [start]
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for nb in neighbours(cell, width, height, offsets):
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    comp.append(nb)
                    queue.append(nb)
        components.append(frozenset(comp))
    return components


def match_zones(components: list[frozenset[Cell]], previous: dict[int, Zone], next_id: int) -> ClusterResult:
    """Give components stable ids by maximum cell overlap with the previous zones.

    A component that overlaps exactly one old zone (and is that zone's best match) keeps its id.
    Merges keep the id of the largest-overlap old zone; in a split the largest-overlap piece keeps
    the id. Both are flagged ``changed``. Components with no overlap get new ids.
    """
    overlaps = [{zid: len(comp & z.cells) for zid, z in previous.items() if comp & z.cells} for comp in components]
    # each old id goes to the component with the largest overlap (ties: earlier component)
    owner: dict[int, int] = {}
    for ci, ov in enumerate(overlaps):
        for zid, n in ov.items():
            if zid not in owner or n > overlaps[owner[zid]][zid]:
                owner[zid] = ci
    claimed: dict[int, int] = {}  # component index -> id it keeps
    for ci, ov in enumerate(overlaps):
        mine = [zid for zid in ov if owner[zid] == ci]
        if mine:
            claimed[ci] = max(mine, key=lambda zid: (ov[zid], -zid))

    result = ClusterResult(zones={}, next_id=next_id)
    split_ids = {zid for zid in owner if sum(1 for ov in overlaps if zid in ov) > 1}
    for ci, comp in enumerate(components):
        if ci in claimed:
            zid = claimed[ci]
            merged = len(overlaps[ci]) > 1
            changed = merged or zid in split_ids
            result.zones[zid] = Zone(zid, comp, changed=changed)
            if changed:
                result.changed_ids.add(zid)
        else:
            zid = result.next_id
            result.next_id += 1
            result.zones[zid] = Zone(zid, comp, changed=bool(overlaps[ci]))
            result.new_ids.add(zid)
    result.removed_ids = set(previous) - set(result.zones)
    return result


def live_zone_cells(snapshot: Iterable[Cell], burning: np.ndarray) -> frozenset[Cell]:
    """The zone as it is *now*: burning cells connected to the Coordinator's last snapshot.

    Seeds are snapshot cells that still burn plus burning cells next to the snapshot (the fire
    has moved on since the last clustering round); the zone is everything 8-connected to them.
    """
    width, height = burning.shape
    snapshot = set(snapshot)
    seeds = {c for c in snapshot if burning[c]}
    for c in snapshot:
        seeds.update(nb for nb in neighbours(c, width, height, NEIGHBOURS_8) if burning[nb])
    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        cell = queue.popleft()
        for nb in neighbours(cell, width, height, NEIGHBOURS_8):
            if burning[nb] and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return frozenset(seen)


def _fuel_believed(belief_state: np.ndarray, fuel: np.ndarray, cell: Cell) -> bool:
    """Cell is believed to be (or may be) unburnt forest."""
    s = belief_state[cell]
    return s == CellState.TREE or (s == UNKNOWN and fuel[cell] > 0)


def neighbouring_trees(cells: Iterable[Cell], belief_state: np.ndarray, fuel: np.ndarray) -> set[Cell]:
    """Believed-Tree cells 8-adjacent to the given burning cells (outside them)."""
    cells = set(cells)
    width, height = belief_state.shape
    out: set[Cell] = set()
    for c in cells:
        for nb in neighbours(c, width, height, NEIGHBOURS_8):
            if nb not in cells and _fuel_believed(belief_state, fuel, nb):
                out.add(nb)
    return out


def downwind_probability(
    n: Cell, zone_cells: frozenset[Cell], fuel: np.ndarray, wind: Vector, fire_cfg: dict[str, Any]
) -> float:
    """Highest ignition probability of ``n`` from any adjacent burning cell of the zone."""
    width, height = fuel.shape
    return max(
        (
            ignition_probability(b, n, float(fuel[n]), wind, fire_cfg)
            for b in neighbours(n, width, height, NEIGHBOURS_8)
            if b in zone_cells
        ),
        default=0.0,
    )


def zone_threat(
    zone_cells: frozenset[Cell],
    belief_state: np.ndarray,
    fuel: np.ndarray,
    wind: Vector,
    fire_cfg: dict[str, Any],
) -> float:
    """``threat(z) = |z| × mean ignition probability of the zone's neighbouring Tree cells``."""
    trees = neighbouring_trees(zone_cells, belief_state, fuel)
    if not trees:
        return 0.0
    mean_p = sum(downwind_probability(n, zone_cells, fuel, wind, fire_cfg) for n in trees) / len(trees)
    return len(zone_cells) * mean_p


def firebreak_candidates(
    zone: Zone,
    belief_state: np.ndarray,
    fuel: np.ndarray,
    wind: Vector,
    from_pos: Cell | None = None,
    exclude: Iterable[Cell] = (),
) -> list[Cell]:
    """Firebreak targets for a zone, best first.

    Candidates are Tree cells next to the zone on its downwind side (projection of
    ``cell − centroid`` onto the wind > 0), ordered by largest projection, ties by distance to
    ``from_pos`` (or the centroid). If there are none, every cell next to the zone is returned,
    nearest first.
    """
    exclude = set(exclude)
    cx, cy = zone.centroid
    origin = from_pos if from_pos is not None else (round(cx), round(cy))
    trees = [c for c in neighbouring_trees(zone.cells, belief_state, fuel) if c not in exclude]

    def projection(c: Cell) -> float:
        return (c[0] - cx) * wind[0] + (c[1] - cy) * wind[1]

    downwind = [c for c in trees if projection(c) > 0]
    if downwind:
        return sorted(downwind, key=lambda c: (-round(projection(c), 6), manhattan(c, origin), c))

    width, height = belief_state.shape
    ring = {
        nb
        for c in zone.cells
        for nb in neighbours(c, width, height, NEIGHBOURS_8)
        if nb not in zone.cells and nb not in exclude
    }
    return sorted(ring, key=lambda c: (manhattan(c, origin), c))


def cluster_zones(
    belief_state: np.ndarray,
    fuel: np.ndarray,
    wind: Vector,
    fire_cfg: dict[str, Any],
    previous: dict[int, Zone],
    next_id: int,
    link_distance: int = 1,
) -> ClusterResult:
    """Cluster believed-burning cells into zones with stable ids, threat and target cell."""
    comps = connected_components(belief_state == CellState.BURNING, link_distance)
    result = match_zones(comps, previous, next_id)
    for zone in result.zones.values():
        zone.threat = zone_threat(zone.cells, belief_state, fuel, wind, fire_cfg)
        cands = firebreak_candidates(zone, belief_state, fuel, wind)
        zone.target_cell = cands[0] if cands else None
    return result
