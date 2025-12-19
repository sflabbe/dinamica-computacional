from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Node:
    x: float
    y: float
    dof_u: Tuple[int, int]
    dof_th: int


class DofManager:
    def __init__(self) -> None:
        self._next = 0

    def new_trans(self) -> Tuple[int, int]:
        ux = self._next
        uy = self._next + 1
        self._next += 2
        return ux, uy

    def new_rot(self) -> int:
        th = self._next
        self._next += 1
        return th

    @property
    def ndof(self) -> int:
        return self._next
