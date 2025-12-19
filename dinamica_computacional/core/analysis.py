from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dinamica_computacional.core.model import Model
from dinamica_computacional.integrators.hht_alpha import hht_alpha_newton
from dinamica_computacional.integrators.static_newton import solve_static_newton
from dinamica_computacional.core.results import Results


@dataclass
class StaticStep:
    name: str
    load_const: np.ndarray
    geometry: str = "linear"
    max_iter: int = 60
    tol: float = 1e-10


@dataclass
class HHTStep:
    name: str
    load_const: np.ndarray
    geometry: str = "linear"
    t_end: float
    dt: float
    alpha: float = -0.05
    drift_limit: float = 0.10
    drift_snapshot: float = 0.04
    base_accel_expr: str = "0.0"


@dataclass
class AnalysisPlan:
    steps: List[object] = field(default_factory=list)


def _make_time(t_end: float, dt: float) -> np.ndarray:
    return np.arange(0.0, t_end + 1e-12, dt)


def _eval_base_accel(expr: str, t: np.ndarray) -> np.ndarray:
    from dinamica_computacional.utils.misc import safe_eval_expr

    return safe_eval_expr(expr, {"t": t})


def run_analysis(model: Model, plan: AnalysisPlan) -> Results:
    results = Results(model=model)
    for step in plan.steps:
        if isinstance(step, StaticStep):
            model.load_const = step.load_const.copy()
            model.options.geometry = step.geometry
            for e in model.elements:
                e.geometry = step.geometry
            u = solve_static_newton(model, max_iter=step.max_iter, tol=step.tol)
            results.static_steps[step.name] = {"u": u.copy()}
        elif isinstance(step, HHTStep):
            model.load_const = step.load_const.copy()
            model.options.geometry = step.geometry
            for e in model.elements:
                e.geometry = step.geometry
            t = _make_time(step.t_end, step.dt)
            ag = _eval_base_accel(step.base_accel_expr, t)
            out = hht_alpha_newton(
                model,
                t,
                ag,
                drift_height=max(nd.y for nd in model.nodes),
                drift_limit=step.drift_limit,
                drift_snapshot=step.drift_snapshot,
                alpha=step.alpha,
                max_iter=50,
                tol=1e-6,
                verbose=False,
            )
            results.dynamic_steps[step.name] = out
        else:
            raise ValueError(f"Unknown step type: {step}")
    return results
