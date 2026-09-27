"""Message types and the instant-delivery message bus that carries all agent interaction.

Every interaction between agents goes through :class:`MessageBus`, so it can be counted and
shown in the demo: OBSERVE, TARGET, ANNOUNCE, BID, AWARD, DONE (Review 1, Section 8.1), plus
REVOKE for the Coordinator taking a zone back from an absent or re-zoned firefighter.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from environment.cells import Cell

Recipient = int | str  # agent id, or a named service such as "belief_map" / "coordinator"


@dataclass
class Message:
    """Base class: who sent it, who it is for, and the step it was sent."""

    sender: int
    recipient: Recipient
    step: int

    @property
    def kind(self) -> str:
        """Upper-case message name, e.g. ``"BID"``."""
        return type(self).__name__.upper()


@dataclass
class Observe(Message):
    """Scout/firefighter → belief map: cell states seen this step."""

    cells: dict[Cell, int] = field(default_factory=dict)


@dataclass
class Target(Message):
    """Scout → other scouts: the frontier cell this scout is heading for (``None`` = hovering)."""

    cell: Cell | None = None


@dataclass
class Announce(Message):
    """Coordinator → firefighter: a zone open for bids."""

    zone_id: int = -1
    cells: frozenset[Cell] = frozenset()
    threat: float = 0.0
    target_cell: Cell | None = None


@dataclass
class Bid(Message):
    """Firefighter → Coordinator: utility for an announced zone."""

    zone_id: int = -1
    utility: float = 0.0


@dataclass
class Award(Message):
    """Coordinator → firefighter: you won this zone; start at this target cell."""

    zone_id: int = -1
    target_cell: Cell | None = None


@dataclass
class Done(Message):
    """Firefighter → Coordinator: zone finished. ``status`` is ``contained`` or ``abandoned``."""

    zone_id: int = -1
    status: str = "contained"


@dataclass
class Revoke(Message):
    """Coordinator → firefighter: assignment withdrawn (``reason``: absent / zone_gone)."""

    zone_id: int = -1
    reason: str = ""


class MessageBus:
    """Instant-delivery message queue.

    A recipient either registers a handler (called immediately on ``send``) or reads its
    inbox with :meth:`receive`. Every message is counted by type and the most recent ones are
    kept for the demo panel.
    """

    def __init__(self, log_size: int) -> None:
        self._inboxes: dict[Recipient, list[Message]] = defaultdict(list)
        self._handlers: dict[Recipient, Callable[[Message], None]] = {}
        self.counts: Counter[str] = Counter()
        self.recent: deque[Message] = deque(maxlen=log_size)

    def subscribe(self, recipient: Recipient, handler: Callable[[Message], None]) -> None:
        """Deliver messages for ``recipient`` straight to ``handler``."""
        self._handlers[recipient] = handler

    def send(self, msg: Message) -> None:
        """Send one message (delivered instantly)."""
        self.counts[msg.kind] += 1
        if not isinstance(msg, Observe):  # observations are too frequent for the panel
            self.recent.append(msg)
        handler = self._handlers.get(msg.recipient)
        if handler is not None:
            handler(msg)
        else:
            self._inboxes[msg.recipient].append(msg)

    def receive(self, recipient: Recipient) -> list[Message]:
        """Remove and return all queued messages for ``recipient``."""
        return self._inboxes.pop(recipient, [])
