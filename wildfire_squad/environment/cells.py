"""Cell states and small grid-geometry helpers shared by every module."""

from __future__ import annotations

from enum import IntEnum

Cell = tuple[int, int]


class CellState(IntEnum):
    """Ground-truth state of one forest cell."""

    EMPTY = 0
    TREE = 1
    BURNING = 2
    BURNT = 3
    FIREBREAK = 4
    WATER = 5


UNKNOWN = -1  # used only in the belief map

NEIGHBOURS_4: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBOURS_8: tuple[Cell, ...] = NEIGHBOURS_4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    """True if ``(x, y)`` lies on a ``width`` × ``height`` grid."""
    return 0 <= x < width and 0 <= y < height


def chebyshev(a: Cell, b: Cell) -> int:
    """Chebyshev (king-move) distance between two cells."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def manhattan(a: Cell, b: Cell) -> int:
    """Manhattan (4-neighbour) distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbours(cell: Cell, width: int, height: int, offsets: tuple[Cell, ...]) -> list[Cell]:
    """In-bounds neighbours of ``cell`` for the given offsets."""
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in offsets if in_bounds(x + dx, y + dy, width, height)]
