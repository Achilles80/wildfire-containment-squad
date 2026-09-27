"""Phase 5 — zone clustering, stable ids, threat and firebreak targets."""

from __future__ import annotations

import numpy as np

from algorithms.zones import Zone, cluster_zones, connected_components, firebreak_candidates, match_zones, zone_threat
from environment.cells import CellState
from environment.fire import wind_vector


def mask_of(cells, size=12):
    m = np.zeros((size, size), dtype=bool)
    for c in cells:
        m[c] = True
    return m


def test_components_use_8_connectivity():
    comps = connected_components(mask_of([(1, 1), (2, 2), (3, 3), (8, 8), (8, 9)]))
    assert sorted(len(c) for c in comps) == [2, 3]  # the diagonal chain is one zone


def test_ids_stable_when_zone_grows():
    first = match_zones(connected_components(mask_of([(5, 5)])), {}, 0)
    assert first.new_ids == {0}
    grown = match_zones(connected_components(mask_of([(5, 5), (5, 6), (6, 6)])), first.zones, first.next_id)
    assert set(grown.zones) == {0}
    assert not grown.new_ids and not grown.changed_ids


def test_merge_keeps_largest_overlap_id_and_is_changed():
    a = Zone(0, frozenset({(1, 1), (1, 2), (1, 3)}))
    b = Zone(1, frozenset({(1, 5)}))
    merged = match_zones([frozenset({(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)})], {0: a, 1: b}, 2)
    assert set(merged.zones) == {0}
    assert merged.changed_ids == {0}
    assert merged.removed_ids == {1}


def test_split_gives_new_id_to_smaller_piece():
    old = Zone(0, frozenset({(1, 1), (1, 2), (1, 3), (1, 7)}))
    split = match_zones([frozenset({(1, 1), (1, 2), (1, 3)}), frozenset({(1, 7)})], {0: old}, 1)
    assert split.zones[0].cells == frozenset({(1, 1), (1, 2), (1, 3)})
    assert 0 in split.changed_ids and split.new_ids == {1}


def test_disappeared_zone_is_removed():
    old = Zone(3, frozenset({(4, 4)}))
    result = match_zones([], {3: old}, 4)
    assert result.removed_ids == {3} and not result.zones


def test_threat_grows_with_size_and_wind(cfg):
    size = 20
    fuel = np.ones((size, size))
    belief = np.full((size, size), CellState.TREE, dtype=np.int8)
    small = frozenset({(10, 10)})
    big = frozenset({(10, 10), (10, 11), (11, 10), (11, 11)})
    for c in big:
        belief[c] = CellState.BURNING
    wind = wind_vector(0)
    assert zone_threat(big, belief, fuel, wind, cfg["fire"]) > zone_threat(small, belief, fuel, wind, cfg["fire"])
    calm = {**cfg["fire"], "wind_k": 0.0}
    assert zone_threat(small, belief, fuel, wind, calm) > 0


def test_firebreak_target_is_downwind(cfg):
    size = 20
    belief = np.full((size, size), CellState.TREE, dtype=np.int8)
    zone = Zone(0, frozenset({(10, 10), (10, 11)}))
    for c in zone.cells:
        belief[c] = CellState.BURNING
    cands = firebreak_candidates(zone, belief, np.ones((size, size)), wind_vector(0))
    assert cands and cands[0][0] == 11  # wind blows toward +x
    assert all(c[0] > 10 for c in cands)


def test_cluster_zones_assigns_target_and_threat(cfg):
    size = 20
    belief = np.full((size, size), CellState.TREE, dtype=np.int8)
    belief[3, 3] = belief[15, 15] = CellState.BURNING
    result = cluster_zones(belief, np.ones((size, size)), wind_vector(0), cfg["fire"], {}, 0)
    assert len(result.zones) == 2
    assert all(z.threat > 0 and z.target_cell is not None for z in result.zones.values())
