from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class NMSurfacePolygon:
    vertices: List[Tuple[float, float]]


def moment_capacity_from_polygon(surface: NMSurfacePolygon, N: float) -> float:
    V = np.asarray(surface.vertices, float)
    Nmin, Nmax = float(np.min(V[:, 0])), float(np.max(V[:, 0]))
    Nc = float(np.clip(N, Nmin, Nmax))

    Ms = []
    for i in range(V.shape[0]):
        a = V[i]
        b = V[(i + 1) % V.shape[0]]
        Na, Ma = float(a[0]), float(a[1])
        Nb, Mb = float(b[0]), float(b[1])
        if (Na - Nc) == 0.0 and (Nb - Nc) == 0.0:
            Ms.extend([Ma, Mb])
            continue
        if (Na - Nc) * (Nb - Nc) > 0:
            continue
        if abs(Nb - Na) < 1e-18:
            continue
        t = (Nc - Na) / (Nb - Na)
        if -1e-12 <= t <= 1.0 + 1e-12:
            Mi = Ma + t * (Mb - Ma)
            Ms.append(float(Mi))

    if len(Ms) == 0:
        j = int(np.argmin(np.abs(V[:, 0] - Nc)))
        return float(abs(V[j, 1]))
    return float(max(abs(min(Ms)), abs(max(Ms))))
