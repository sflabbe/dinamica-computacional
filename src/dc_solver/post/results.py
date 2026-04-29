from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrameGeometryData:
    node_xy: np.ndarray
    element_node_pairs: list[tuple[int, int]]


@dataclass
class FrameStateResult:
    label: str
    u: np.ndarray
    scale: float
    node_xy_ref: np.ndarray
    node_xy_def: np.ndarray
    node_umag: np.ndarray
    member_sigma_max: np.ndarray
    drift: float


@dataclass
class HingeTimeHistory:
    hinge_idx: int
    kind: str
    name: str
    t: np.ndarray
    M: np.ndarray
    dtheta: np.ndarray
    a: np.ndarray


@dataclass
class DynamicResult:
    t: np.ndarray
    ag: np.ndarray
    drift: np.ndarray
    Vb: np.ndarray
    u: np.ndarray
    v: np.ndarray | None = None
    a: np.ndarray | None = None
    iters: np.ndarray | None = None
    hinges: list[HingeTimeHistory] = field(default_factory=list)
    energy: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _node_xy(model, u: np.ndarray | None = None) -> np.ndarray:
    xy = np.array([[nd.x, nd.y] for nd in model.nodes], dtype=float)
    if u is None:
        return xy
    out = xy.copy()
    for i, nd in enumerate(model.nodes):
        ux, uy = nd.dof_u
        out[i, 0] += float(u[ux])
        out[i, 1] += float(u[uy])
    return out


def frame_geometry_from_model(model) -> FrameGeometryData:
    return FrameGeometryData(
        node_xy=_node_xy(model, None),
        element_node_pairs=[(int(eb.ni), int(eb.nj)) for eb in model.beams],
    )


def _member_sigma(model, u: np.ndarray) -> np.ndarray:
    vals = []
    for eb in model.beams:
        f = eb.end_forces_local(u)
        n, mi, mj = float(f["N"]), float(f["Mi"]), float(f["Mj"])
        A, I = float(eb.A), float(eb.I)
        if A <= 0.0 or I <= 0.0:
            vals.append(0.0)
            continue
        c = 0.5 * np.sqrt(12.0 * I / A)
        sigma_n = n / A
        sbi = mi * c / I
        sbj = mj * c / I
        vals.append(float(max(abs(sigma_n + sbi), abs(sigma_n - sbi), abs(sigma_n + sbj), abs(sigma_n - sbj))))
    return np.asarray(vals, dtype=float)


def _drift_from_state(model, u: np.ndarray) -> float:
    ys = np.array([nd.y for nd in model.nodes], dtype=float)
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    h = y1 - y0
    if h <= 0.0:
        return 0.0
    i_top = np.where(np.isclose(ys, y1))[0]
    i_bot = np.where(np.isclose(ys, y0))[0]
    ux_top = np.mean([u[model.nodes[i].dof_u[0]] for i in i_top])
    ux_bot = np.mean([u[model.nodes[i].dof_u[0]] for i in i_bot])
    return float((ux_top - ux_bot) / h)


def frame_state_from_model(model, u, *, label="state", scale=1.0) -> FrameStateResult:
    u_arr = np.asarray(u, dtype=float)
    xy_ref = _node_xy(model, None)
    xy_def = _node_xy(model, u_arr * float(scale))
    node_umag = np.array([np.hypot(u_arr[nd.dof_u[0]], u_arr[nd.dof_u[1]]) for nd in model.nodes], dtype=float)
    return FrameStateResult(
        label=label,
        u=u_arr,
        scale=float(scale),
        node_xy_ref=xy_ref,
        node_xy_def=xy_def,
        node_umag=node_umag,
        member_sigma_max=_member_sigma(model, u_arr),
        drift=_drift_from_state(model, u_arr),
    )


def dynamic_result_from_dict(raw: dict, *, name: str = "dynamic") -> DynamicResult:
    meta = {"name": name, "warnings": []}
    t = np.asarray(raw.get("t", []), dtype=float)
    ag = np.asarray(raw.get("ag", np.zeros_like(t)), dtype=float)
    drift = np.asarray(raw.get("drift", np.zeros_like(t)), dtype=float)
    vb = np.asarray(raw.get("Vb", np.zeros_like(t)), dtype=float)
    u = np.asarray(raw.get("u", []), dtype=float)
    v = None if raw.get("v", None) is None else np.asarray(raw.get("v"), dtype=float)
    a = None if raw.get("a", None) is None else np.asarray(raw.get("a"), dtype=float)
    iters = None if raw.get("iters", None) is None else np.asarray(raw.get("iters"), dtype=float)
    energy = {}
    if isinstance(raw.get("energy", None), dict):
        energy = {str(k): np.asarray(vv, dtype=float) for k, vv in raw["energy"].items()}

    hinges: list[HingeTimeHistory] = []
    hinge_hist = raw.get("hinges", raw.get("hinge_hist", None))
    if hinge_hist is None:
        meta["warnings"].append("hinge history missing")
    elif not isinstance(hinge_hist, list) or len(hinge_hist) == 0:
        meta["warnings"].append("hinge history empty or invalid")
    elif t.size == 0:
        meta["warnings"].append("time vector missing; hinges skipped")
    else:
        nstep = len(hinge_hist)
        if nstep == max(t.size - 1, 0):
            t_h = t[1:]
        elif nstep == t.size:
            t_h = t
        else:
            nmin = min(nstep, t.size)
            t_h = t[:nmin]
            meta["warnings"].append(f"hinge_hist length mismatch: {nstep} vs t {t.size}; truncated to {nmin}")
        n_use = min(len(t_h), len(hinge_hist))
        if n_use == 0:
            meta["warnings"].append("insufficient hinge data")
        else:
            first = hinge_hist[0]
            if not isinstance(first, list) or len(first) == 0:
                meta["warnings"].append("insufficient hinge data")
            else:
                n_hinges = min(len(step) for step in hinge_hist[:n_use] if isinstance(step, list))
                for ih in range(n_hinges):
                    seq = [step[ih] if isinstance(step, list) and len(step) > ih and isinstance(step[ih], dict) else {} for step in hinge_hist[:n_use]]
                    M = np.array([item.get("M", item.get("moment", item.get("M_trial", np.nan))) for item in seq], dtype=float)
                    dtheta = np.array([item.get("dtheta", item.get("theta", item.get("rot", np.nan))) for item in seq], dtype=float)
                    aval = np.array([item.get("a", item.get("damage", item.get("alpha", np.nan))) for item in seq], dtype=float)
                    kind = str(seq[0].get("kind", seq[0].get("type", "unknown")))
                    name_h = str(seq[0].get("name", f"hinge_{ih}"))
                    hinges.append(HingeTimeHistory(ih, kind, name_h, t_h[:n_use], M, dtheta, aval))
                if n_hinges == 0:
                    meta["warnings"].append("insufficient hinge data")

    if "A_input_g" in raw:
        meta["A_input_g"] = float(raw["A_input_g"])
    else:
        meta["warnings"].append("A_input_g missing")

    return DynamicResult(t=t, ag=ag, drift=drift, Vb=vb, u=u, v=v, a=a, iters=iters, hinges=hinges, energy=energy, meta=meta)
