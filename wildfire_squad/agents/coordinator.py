"""Coordinator — utility-based task allocator with no body on the grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa

from algorithms.auction import AuctionRound, ZoneOffer, greedy_assignment, sequential_auction, slots_for
from algorithms.zones import Zone, cluster_zones
from environment.cells import manhattan
from messages import Announce, Award, Bid, Done, Message, Revoke

if TYPE_CHECKING:
    from model import WildfireModel


class Coordinator(mesa.Agent):
    """Clusters known fire into zones and allocates firefighters to them.

    Every ``cluster_every`` steps: cluster burning cells (connected components) with stable ids,
    score each zone's threat, take zones back from firefighters that are gone or absent too long,
    then announce the open slots of new, changed and under-staffed zones and fill them by auction
    (or greedily, for the baseline).
    """

    def __init__(self, model: WildfireModel) -> None:
        super().__init__(model)
        self.zones: dict[int, Zone] = {}
        self.assignments: dict[int, int] = {}  # firefighter id -> zone id
        self.next_zone_id = 0
        self._bids: list[Bid] = []
        model.bus.subscribe("coordinator", self.receive)

    # ------------------------------------------------------------------ messages
    def receive(self, msg: Message) -> None:
        """Collect BIDs; process DONE reports immediately."""
        if isinstance(msg, Bid):
            self._bids.append(msg)
        elif isinstance(msg, Done):
            self.assignments.pop(msg.sender, None)

    def _revoke(self, ff_id: int, reason: str) -> None:
        zone_id = self.assignments.pop(ff_id)
        self.model.bus.send(Revoke(self.unique_id, ff_id, self.model.steps, zone_id, reason))
        self.model.log_event("revoke", agent=ff_id, zone=zone_id, reason=reason)

    # ------------------------------------------------------------------ zones
    def recluster(self) -> None:
        """Rebuild the zone table from the shared belief map and release stale assignments."""
        m = self.model
        result = cluster_zones(m.shared_belief.state, m.fuel, m.wind, m.cfg["fire"], self.zones, self.next_zone_id)
        self.zones = result.zones
        self.next_zone_id = result.next_id
        for zid in sorted(result.removed_ids):
            m.log_event("zone_contained", zone=zid)
        # a zone that merged into another or burnt out is gone; its firefighters are re-auctioned.
        # Zones that merely changed keep their firefighters and advertise any new open slots.
        for ff_id, zid in sorted(self.assignments.items()):
            if zid in result.removed_ids:
                self._revoke(ff_id, "zone_gone")

    def release_absent(self) -> None:
        """Take zones back from lost firefighters and ones away refilling for too long."""
        m = self.model
        limit = m.cfg["coordinator"]["reauction_after_absent"]
        alive = {ff.unique_id: ff for ff in m.firefighters}
        for ff_id in sorted(self.assignments):
            ff = alive.get(ff_id)
            if ff is None:
                self.assignments.pop(ff_id)
            elif ff.refilling_since is not None and m.steps - ff.refilling_since > limit:
                self._revoke(ff_id, "absent")

    def _offers(self) -> list[ZoneOffer]:
        large = self.model.cfg["coordinator"]["large_zone_cells"]
        taken: dict[int, int] = {}
        for zid in self.assignments.values():
            taken[zid] = taken.get(zid, 0) + 1
        offers = []
        for zid, zone in sorted(self.zones.items()):
            vacancies = slots_for(len(zone.cells), large) - taken.get(zid, 0)
            if vacancies > 0 and zone.target_cell is not None:
                offers.append(ZoneOffer(zid, zone.threat, len(zone.cells), vacancies))
        return offers

    # ------------------------------------------------------------------ allocation
    def _collect_bids(self, offer: ZoneOffer, free_ids: list[int]) -> dict[int, float]:
        """Send ANNOUNCE to each free firefighter and gather the BIDs that come back."""
        m = self.model
        zone = self.zones[offer.zone_id]
        self._bids = []
        for ff_id in free_ids:
            m.bus.send(
                Announce(self.unique_id, ff_id, m.steps, zone.zone_id, zone.cells, zone.threat, zone.target_cell)
            )
        return {b.sender: b.utility for b in self._bids if b.zone_id == offer.zone_id}

    def _support_offers(self) -> list[ZoneOffer]:
        """One extra slot per zone, used to put otherwise idle firefighters to work."""
        return [
            ZoneOffer(zid, z.threat, len(z.cells), 1)
            for zid, z in sorted(self.zones.items())
            if z.target_cell is not None
        ]

    def _run_allocation(self, offers: list[ZoneOffer], free: list[int]) -> dict[int, int]:
        m = self.model
        if m.strategy.allocation == "auction":
            awards, rounds = sequential_auction(offers, free, self._collect_bids)
        else:
            by_id = {ff.unique_id: ff for ff in m.firefighters}

            def distance(ff_id: int, offer: ZoneOffer) -> float:
                return float(manhattan(by_id[ff_id].pos, self.zones[offer.zone_id].target_cell))

            awards, rounds = greedy_assignment(offers, free, distance)
        self._log_rounds(rounds)
        for ff_id, zid in sorted(awards.items()):
            self.assignments[ff_id] = zid
            m.bus.send(Award(self.unique_id, ff_id, m.steps, zid, self.zones[zid].target_cell))
            m.log_event("award", agent=ff_id, zone=zid)
        return awards

    def allocate(self) -> None:
        """Fill open zone slots and send AWARDs to the winners.

        First the design slots (1 per zone, 2 for large zones). Then, if ``assign_idle`` is on,
        repeated support rounds give each still-free firefighter a zone, so none stays idle
        while fire is known.
        """
        m = self.model

        def free_ids() -> list[int]:
            return [ff.unique_id for ff in m.firefighters if ff.unique_id not in self.assignments]

        offers = self._offers()
        if offers and free_ids():
            self._run_allocation(offers, free_ids())
        if not m.cfg["coordinator"]["assign_idle"]:
            return
        while free_ids() and self.zones:
            if not self._run_allocation(self._support_offers(), free_ids()):
                break

    def _log_rounds(self, rounds: list[AuctionRound]) -> None:
        for r in rounds:
            self.model.auction_log.append({"step": self.model.steps, **r.__dict__})

    # ------------------------------------------------------------------ main loop
    def step(self) -> None:
        """Every ``cluster_every`` steps: cluster, release, allocate."""
        m = self.model
        if m.steps % m.cfg["coordinator"]["cluster_every"] != 0:
            return
        self.recluster()
        self.release_absent()
        self.allocate()
