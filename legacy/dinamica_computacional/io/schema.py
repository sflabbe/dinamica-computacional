from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InputSchema:
    nodes: Dict[int, tuple]
    elements: List[dict]
