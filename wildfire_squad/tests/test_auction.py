"""Phase 5 — utility, sequential single-item auction and the Coordinator's message flow."""

from __future__ import annotations

import math

import pytest

from algorithms.auction import ZoneOffer, greedy_assignment, pick_winner, sequential_auction, slots_for, utility
from model import WildfireModel


def fixed_bids(table):
    """collect_bids callback returning preset utilities {(agent, zone): u}."""
    return lambda offer, free: {a: table.get((a, offer.zone_id), 0.0) for a in free}


def test_utility_formula():
    assert utility(10.0, 4.0, 5, 10) == pytest.approx(10.0 / 5.0 * 0.5)
    assert utility(10.0, math.inf, 10, 10) == 0.0  # unreachable → no bid
    assert utility(10.0, 3.0, 0, 10) == 0.0  # no water → no bid


def test_highest_utility_wins():
    offers = [ZoneOffer(0, 5.0, 3, 1)]
    awards, rounds = sequential_auction(offers, [1, 2, 3], fixed_bids({(1, 0): 0.2, (2, 0): 0.9, (3, 0): 0.5}))
    assert awards == {2: 0}
    assert rounds[0].winner == 2 and set(rounds[0].bids) == {1, 2, 3}


def test_tie_break_by_lower_id():
    assert pick_winner({7: 0.5, 3: 0.5, 9: 0.1}) == 3


def test_zones_auctioned_by_threat_and_loser_bids_on_next():
    offers = [ZoneOffer(0, 1.0, 3, 1), ZoneOffer(1, 9.0, 3, 1)]
    table = {(1, 0): 0.1, (1, 1): 0.8, (2, 0): 0.3, (2, 1): 0.7}
    awards, rounds = sequential_auction(offers, [1, 2], fixed_bids(table))
    assert [r.zone_id for r in rounds] == [1, 0]  # highest threat first
    assert awards == {1: 1, 2: 0}  # agent 2 lost zone 1, won zone 0


def test_large_zone_gets_two_firefighters(cfg):
    large = cfg["coordinator"]["large_zone_cells"]
    assert slots_for(large, large) == 1 and slots_for(large + 1, large) == 2
    offers = [ZoneOffer(0, 5.0, large + 1, slots_for(large + 1, large))]
    awards, _ = sequential_auction(offers, [1, 2, 3], fixed_bids({(1, 0): 0.4, (2, 0): 0.6, (3, 0): 0.5}))
    assert awards == {2: 0, 3: 0}


def test_unreachable_firefighter_does_not_bid():
    offers = [ZoneOffer(0, 5.0, 3, 1)]
    bids = lambda offer, free: {1: utility(5.0, math.inf, 10, 10), 2: utility(5.0, 8.0, 10, 10)}  # noqa: E731
    awards, rounds = sequential_auction(offers, [1, 2], bids)
    assert awards == {2: 0} and 1 not in rounds[0].bids


def test_no_bids_leaves_zone_open():
    awards, rounds = sequential_auction([ZoneOffer(0, 5.0, 3, 1)], [1], fixed_bids({}))
    assert awards == {} and rounds[0].winner is None


def test_greedy_ignores_threat_and_picks_nearest():
    offers = [ZoneOffer(0, 1.0, 3, 1), ZoneOffer(1, 9.0, 3, 1)]
    dist = {(1, 0): 2, (2, 0): 5, (1, 1): 1, (2, 1): 9}
    awards, rounds = greedy_assignment(offers, [1, 2], lambda a, o: dist[(a, o.zone_id)])
    assert [r.zone_id for r in rounds] == [0, 1]  # discovery order, not threat
    assert awards == {1: 0, 2: 1}


def test_model_auction_messages_flow():
    """End to end: announcements, bids and awards travel over the message bus."""
    m = WildfireModel({"simulation": {"strategy": "auction"}}, seed=0)  # a fire that survives to be found
    while m.running and not m.auction_log:
        m.step()
    assert m.auction_log, "an auction should happen once scouts find the fire"
    assert m.bus.counts["ANNOUNCE"] > 0 and m.bus.counts["BID"] > 0 and m.bus.counts["AWARD"] > 0
    first = m.auction_log[0]
    assert first["winner"] is None or first["winner"] == pick_winner(first["bids"])
    assert any(e["kind"] == "award" for e in m.events)
    assigned = [ff for ff in m.firefighters if ff.assigned_zone is not None]
    assert assigned and all(ff.assigned_zone in m.coordinator.zones for ff in assigned)
    assert all(m.coordinator.assignments[ff.unique_id] == ff.assigned_zone for ff in assigned)
