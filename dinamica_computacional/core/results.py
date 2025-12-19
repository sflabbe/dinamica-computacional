from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from dinamica_computacional.core.model import Model
from dinamica_computacional.utils.plotting import plot_results


@dataclass
class Results:
    model: Model
    static_steps: Dict[str, dict] = field(default_factory=dict)
    dynamic_steps: Dict[str, dict] = field(default_factory=dict)

    def save_plots(self, outfile_prefix: str = "problem4") -> None:
        plot_results(self.model, self, outfile_prefix=outfile_prefix)
