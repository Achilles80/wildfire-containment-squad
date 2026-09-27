"""Utility function and sequential single-item auction for zone allocation.

Firefighter ``f`` values zone ``z`` at::

    U(f, z) = threat(z) / (1 + pathcost(f, z)) × (water_f / water_max)

Zones are auctioned in decreasing order of threat; each slot goes to the highest bidder
(ties to the lower agent id) and the winner leaves the free pool.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass
class ZoneOffer:
    """What is being auctioned: a zone and how many more firefighters it needs."""

    zone_id: int
    threat: float
    n_cells: int
    vacancies: int


@dataclass
class AuctionRound:
    """Record of one auctioned slot, kept for the demo panel and debugging."""

    zone_id: int
    slot: int
    threat: float
    bids: dict[int, float] = field(default_factory=dict)
    winner: int | None = None


def utility(threat: float, pathcost: float, water: float, water_max: float) -> float:
    """``U(f, z)``; 0 when the zone is unreachable or the firefighter has no water."""
    if not math.isfinite(pathcost) or water <= 0 or water_max <= 0:
        return 0.0
    return threat / (1.0 + pathcost) * (water / water_max)


def slots_for(n_cells: int, large_zone_cells: int) -> int:
    """2 firefighters for zones larger than ``large_zone_cells``, else 1."""
    return 2 if n_cells > large_zone_cells else 1


def pick_winner(bids: dict[int, float]) -> int:
    """Highest bid wins; ties go to the lower agent id."""
    return min(bids, key=lambda agent_id: (-bids[agent_id], agent_id))


def sequential_auction(
    offers: Sequence[ZoneOffer],
    free_ids: Sequence[int],
    collect_bids: Callable[[ZoneOffer, list[int]], dict[int, float]],
) -> tuple[dict[int, int], list[AuctionRound]]:
    """Run a sequential single-item auction.

    Args:
        offers: zones with open slots.
        free_ids: ids of firefighters that may bid.
        collect_bids: announces a zone to the given free firefighters and returns their bids
            ``{agent_id: utility}`` (only positive bids count).

    Returns:
        ``({agent_id: zone_id}, rounds)``.
    """
    free = list(free_ids)
    awards: dict[int, int] = {}
    rounds: list[AuctionRound] = []
    for offer in sorted(offers, key=lambda o: (-o.threat, o.zone_id)):
        for slot in range(offer.vacancies):
            if not free:
                return awards, rounds
            bids = {a: u for a, u in collect_bids(offer, list(free)).items() if u > 0}
            record = AuctionRound(offer.zone_id, slot, offer.threat, bids)
            rounds.append(record)
            if not bids:
                break
            winner = pick_winner(bids)
            record.winner = winner
            awards[winner] = offer.zone_id
            free.remove(winner)
    return awards, rounds


def greedy_assignment(
    offers: Sequence[ZoneOffer],
    free_ids: Sequence[int],
    distance: Callable[[int, ZoneOffer], float],
) -> tuple[dict[int, int], list[AuctionRound]]:
    """Baseline: zones in discovery (id) order, each slot to the nearest free firefighter.

    Ignores threat entirely. Unreachable firefighters (infinite distance) are skipped.
    """
    free = list(free_ids)
    awards: dict[int, int] = {}
    rounds: list[AuctionRound] = []
    for offer in sorted(offers, key=lambda o: o.zone_id):
        for slot in range(offer.vacancies):
            dists = {a: distance(a, offer) for a in free}
            dists = {a: d for a, d in dists.items() if math.isfinite(d)}
            record = AuctionRound(offer.zone_id, slot, offer.threat, {a: -d for a, d in dists.items()})
            rounds.append(record)
            if not dists:
                break
            winner = min(dists, key=lambda a: (dists[a], a))
            record.winner = winner
            awards[winner] = offer.zone_id
            free.remove(winner)
    return awards, rounds
