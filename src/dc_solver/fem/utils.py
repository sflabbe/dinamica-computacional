"""Helpers for FE model construction."""

from __future__ import annotations

from typing import List

import numpy as np

from .nodes import Node, DofManager


def discretize_member(n1: int, n2: int, nseg: int, nodes: List[Node], dm: DofManager) -> List[int]:
    """Return node indices along member (including ends), inserting internal nodes as needed."""
    if nseg < 1:
        raise ValueError("nseg must be >= 1")
    if nseg == 1:
        return [n1, n2]
    ni = nodes[n1]
    nj = nodes[n2]
    xs = np.linspace(ni.x, nj.x, nseg + 1)
    ys = np.linspace(ni.y, nj.y, nseg + 1)
    node_ids = [n1]
    for k in range(1, nseg):
        nk = Node(float(xs[k]), float(ys[k]), dm.new_trans(), dm.new_rot())
        nodes.append(nk)
        node_ids.append(len(nodes) - 1)
    node_ids.append(n2)
    return node_ids
