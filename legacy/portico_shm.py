#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portico_shm.py — Pórtico 1 piso (SDOF) no lineal con:
- Resorte columnas (EPP o Bouc–Wen, opcional 2 columnas en paralelo)
- Resorte viga (Bouc–Wen con degradación por energía)
- P-Delta como rigidez geométrica negativa: Kgeo = P_total / H
- Integradores:
    * Velocity-Verlet (explícito, con fixed-point para damping + histeresis)
    * HHT-α (implícito, amortiguamiento numérico alta frecuencia)
- Excitación:
    * seno tipo tarea
    * choque (Ricker / impulso)
    * combo (seno + choque)
    * archivo CSV (p.ej. señales de tren: Time_s + Acceleration_g)
- Estudio paramétrico y plots (u(t), V-u, gradiente temporal, ringing HF)
Uso: ver al final ("COMANDOS ÚTILES").
"""

from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl


# -------------------------
# Constantes
# -------------------------
g = 9.81


# -------------------------
# Helpers ENV (CLI sin argparse)
# -------------------------
def _envs(key: str, default: str) -> str:
    return str(os.environ.get(key, default)).strip()


def _envi(key: str, default: int) -> int:
    try:
        return int(float(os.environ.get(key, str(default))))
    except Exception:
        return int(default)


def _envf(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return float(default)


# -------------------------
# Señal: métricas de "ringing" / serrucho
# -------------------------
def jerk_rms(a: np.ndarray, dt: float) -> float:
    """RMS del jerk (derivada de a) en unidades de g/s."""
    if len(a) < 3:
        return 0.0
    j = np.diff(a) / dt
    return float(np.sqrt(np.mean(j * j)) / g)


def roughness_second_diff(x: np.ndarray) -> float:
    """
    Proxy serrucho / ringing: norma L1 de la segunda diferencia (sin normalizar).
    Crece fuerte cuando la señal se vuelve "diente de sierra".
    """
    if len(x) < 3:
        return 0.0
    d2 = np.diff(x, n=2)
    return float(np.sum(np.abs(d2)))


def hf_ratio(x: np.ndarray, dt: float, f_cut: float) -> float:
    """
    Fracción de energía espectral por arriba de f_cut.
    (0 = nada HF, 1 = todo HF). Usa rFFT y potencia.
    """
    n = len(x)
    if n < 8:
        return 0.0
    x0 = x - float(np.mean(x))
    X = np.fft.rfft(x0)
    f = np.fft.rfftfreq(n, d=dt)
    p = (X.real * X.real + X.imag * X.imag)
    tot = float(np.sum(p)) + 1e-30
    hf = float(np.sum(p[f >= f_cut]))
    return hf / tot


# -------------------------
# Excitación ag(t)
# -------------------------
@dataclass
class Excitation:
    type: str = "sine"  # sine|shock|combo|file
    # Sine (default de la tarea)
    sine_w1_hz: float = 2.0     # sin(2π*w1*t) factor
    sine_w2_hz: float = 2.0     # cos(2π*w2*t) factor
    sine_scale: float = 1.0

    # Shock: Ricker wavelet (pico en t0)
    shock_f0_hz: float = 20.0
    shock_t0_s: float = 1.0
    shock_scale: float = 1.0

    # Combo
    combo_sine_scale: float = 1.0
    combo_shock_scale: float = 0.5

    # File
    file_path: str = ""
    file_time_col: str = "Time_s"
    file_acc_col: str = "Acceleration_g"  # o Building_a_g
    file_units: str = "auto"  # auto|g|m/s2
    file_scale: float = 1.0

    # (cache interno)
    _t: Optional[np.ndarray] = None
    _a_ms2: Optional[np.ndarray] = None

    def _ricker(self, t: float) -> float:
        # Ricker / Mexican hat: (1 - 2π^2 f0^2 τ^2) e^{-π^2 f0^2 τ^2}
        f0 = self.shock_f0_hz
        tau = t - self.shock_t0_s
        a = (math.pi * f0 * tau)
        return (1.0 - 2.0 * a * a) * math.exp(-a * a)

    def _sine(self, t: float) -> float:
        # señal tipo tarea (modulada)
        w1 = 2.0 * math.pi * self.sine_w1_hz
        w2 = 2.0 * math.pi * self.sine_w2_hz
        return math.cos(w2 * t) * math.sin(w1 * t)

    def _load_file(self) -> None:
        if self._t is not None and self._a_ms2 is not None:
            return
        if not self.file_path:
            raise ValueError("EXC_TYPE=file pero EXC_FILE está vacío.")

        path = self.file_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe EXC_FILE='{path}'")

        # lectura CSV robusta: intenta pandas, si no, numpy.genfromtxt
        t = None
        a = None

        try:
            import pandas as pd  # type: ignore
            df = pd.read_csv(path)
            if self.file_time_col not in df.columns:
                raise KeyError(f"No encuentro columna tiempo '{self.file_time_col}' en {list(df.columns)[:20]}...")
            if self.file_acc_col not in df.columns:
                raise KeyError(f"No encuentro columna accel '{self.file_acc_col}' en {list(df.columns)[:20]}...")
            t = df[self.file_time_col].to_numpy(dtype=float)
            a = df[self.file_acc_col].to_numpy(dtype=float)
        except Exception:
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if self.file_time_col not in data.dtype.names:
                raise KeyError(f"No encuentro columna tiempo '{self.file_time_col}' en {data.dtype.names}")
            if self.file_acc_col not in data.dtype.names:
                raise KeyError(f"No encuentro columna accel '{self.file_acc_col}' en {data.dtype.names}")
            t = np.asarray(data[self.file_time_col], dtype=float)
            a = np.asarray(data[self.file_acc_col], dtype=float)

        # normaliza: t ascendente
        idx = np.argsort(t)
        t = t[idx]
        a = a[idx]

        units = self.file_units.lower().strip()
        if units == "auto":
            # heurística: si el nombre dice _g o el rango es ~ decenas -> g
            if ("_g" in self.file_acc_col.lower()) or (np.nanmax(np.abs(a)) < 80.0):
                units = "g"
            else:
                units = "m/s2"

        if units in ("g", "gal", "gee"):
            a_ms2 = a * g
        elif units in ("m/s2", "m/s^2", "ms2"):
            a_ms2 = a
        else:
            raise ValueError(f"EXC_FILE_UNITS inválido: {self.file_units} (usa auto|g|m/s2)")

        # aplica escala extra
        a_ms2 = self.file_scale * a_ms2

        self._t = t
        self._a_ms2 = a_ms2

    def ag(self, t: float, A_ms2: float) -> float:
        """
        Devuelve aceleración [m/s²]. A_ms2 suele ser (A_factor_g*g).
        Convención:
          - sine/shock/combo: A_ms2 escala la señal base.
          - file: la señal del archivo se escala por (A_ms2/g) para que A_factor_g actúe como factor.
        """
        typ = self.type.lower().strip()
        if typ == "sine":
            return A_ms2 * self.sine_scale * self._sine(t)
        if typ == "shock":
            return A_ms2 * self.shock_scale * self._ricker(t)
        if typ == "combo":
            return (A_ms2 * self.combo_sine_scale * self._sine(t)
                    + A_ms2 * self.combo_shock_scale * self._ricker(t))
        if typ == "file":
            self._load_file()
            assert self._t is not None and self._a_ms2 is not None
            # interp lineal
            a0 = float(np.interp(t, self._t, self._a_ms2))
            scale = A_ms2 / g  # => A_factor_g
            return scale * a0
        raise ValueError(f"EXC_TYPE inválido: {self.type} (sine|shock|combo|file)")


def plot_excitation(exc: Excitation, A_factor_g: float, dt: float, tmax: float) -> None:
    t = np.arange(0.0, tmax + 0.5 * dt, dt)
    a = np.array([exc.ag(float(ti), A_factor_g * g) for ti in t], dtype=float)
    plt.figure()
    plt.plot(t, a / g)
    plt.xlabel("t [s]")
    plt.ylabel("a_g(t) [g]")
    plt.title(f"Excitación: {exc.type} | A={A_factor_g:.2f}g")
    plt.grid(True)


# -------------------------
# Modelo: Elementos no lineales (interfaz con snapshot/restore)
# -------------------------
class Element:
    def snapshot(self) -> Dict[str, Any]:
        return {}

    def restore(self, state: Dict[str, Any]) -> None:
        _ = state

    def update_and_force(self, u_old: float, u_new: float, v_mid: float, dt: float) -> float:
        raise NotImplementedError


class Elastic(Element):
    def __init__(self, k: float):
        self.k = float(k)

    def update_and_force(self, u_old: float, u_new: float, v_mid: float, dt: float) -> float:
        _ = (u_old, v_mid, dt)
        return self.k * float(u_new)


class EPP(Element):
    """
    Elastoplástico perfecto en fuerza (cortante): f = k(u - up), |f|<=Fy, up evoluciona.
    """
    def __init__(self, k: float, fy: float):
        self.k = float(k)
        self.fy = float(abs(fy))
        self.up = 0.0  # offset plástico

    def snapshot(self) -> Dict[str, Any]:
        return {"up": float(self.up)}

    def restore(self, state: Dict[str, Any]) -> None:
        self.up = float(state.get("up", 0.0))

    def update_and_force(self, u_old: float, u_new: float, v_mid: float, dt: float) -> float:
        _ = (u_old, v_mid, dt)
        u = float(u_new)
        f_trial = self.k * (u - self.up)
        if abs(f_trial) <= self.fy:
            return float(f_trial)
        f = self.fy * math.copysign(1.0, f_trial)
        # actualiza up para que f = k(u-up)
        self.up = u - f / self.k
        return float(f)


class BoucWenDegrading(Element):
    """
    Bouc–Wen (histerético) + degradación simple por energía disipada.
    Fuerza:
        f = α K u + (1-α) Fy z
    Evolución (forma clásica):
        z_dot = x_dot - β|x_dot||z|^{n-1}z - γ x_dot |z|^n
    donde x_dot ~ u_dot / uy
    Degradación:
        damage = 1 - exp(-E_diss / E_ref)
        K = K0*(1 - c_stiff*damage)
        Fy = Fy0*(1 - c_strength*damage)
    """
    def __init__(
        self,
        k0: float,
        fy0: float,
        alpha: float = 0.05,
        beta: float = 0.5,
        gamma: float = 0.5,
        n: float = 2.0,
        c_strength: float = 0.4,
        c_stiff: float = 0.2,
        e_ref: Optional[float] = None,
    ):
        self.k0 = float(k0)
        self.fy0 = float(abs(fy0))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n = float(n)
        self.c_strength = float(c_strength)
        self.c_stiff = float(c_stiff)

        uy0 = self.fy0 / max(1e-30, self.k0)
        self.e_ref = float(e_ref) if e_ref is not None else float(self.fy0 * uy0)

        self.z = 0.0
        self.E_diss = 0.0
        self.K = self.k0
        self.Fy = self.fy0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "z": float(self.z),
            "E_diss": float(self.E_diss),
            "K": float(self.K),
            "Fy": float(self.Fy),
        }

    def restore(self, state: Dict[str, Any]) -> None:
        self.z = float(state.get("z", 0.0))
        self.E_diss = float(state.get("E_diss", 0.0))
        self.K = float(state.get("K", self.k0))
        self.Fy = float(state.get("Fy", self.fy0))

    def _update_damage(self) -> None:
        dmg = 1.0 - math.exp(-self.E_diss / max(1e-30, self.e_ref))
        self.K = self.k0 * max(0.02, 1.0 - self.c_stiff * dmg)
        self.Fy = self.fy0 * max(0.05, 1.0 - self.c_strength * dmg)

    def update_and_force(self, u_old: float, u_new: float, v_mid: float, dt: float) -> float:
        # normaliza velocidad a x_dot ~ u_dot/uy
        uy = self.Fy / max(1e-30, self.K)
        xdot = float(v_mid) / max(1e-12, uy)

        # RK2 para z
        def z_dot(z: float, xdot_: float) -> float:
            az = abs(z)
            return xdot_ - self.beta * abs(xdot_) * (az ** (self.n - 1.0)) * z - self.gamma * xdot_ * (az ** self.n)

        z0 = self.z
        k1 = z_dot(z0, xdot)
        z_half = z0 + 0.5 * dt * k1
        k2 = z_dot(z_half, xdot)
        z1 = z0 + dt * k2

        # clipping suave para estabilidad (evitar z explote si dt es grande)
        z1 = float(np.clip(z1, -2.0, 2.0))
        self.z = z1

        # fuerza
        u = float(u_new)
        f_hys = (1.0 - self.alpha) * self.Fy * self.z
        f = self.alpha * self.K * u + f_hys

        # energía disipada aprox (trabajo de la parte histerética)
        du = float(u_new - u_old)
        self.E_diss += abs(f_hys * du)

        # degradación
        self._update_damage()
        return float(f)


class Parallel(Element):
    def __init__(self, springs: List[Element]):
        self.springs = springs

    def snapshot(self) -> Dict[str, Any]:
        return {"spr": [s.snapshot() for s in self.springs]}

    def restore(self, state: Dict[str, Any]) -> None:
        arr = state.get("spr", [])
        for s, st in zip(self.springs, arr):
            s.restore(st)

    def update_and_force(self, u_old: float, u_new: float, v_mid: float, dt: float) -> float:
        return float(sum(s.update_and_force(u_old, u_new, v_mid, dt) for s in self.springs))


# -------------------------
# Pórtico lineal: K0 y momentos por unidad de cortante (mini-FE 2D)
# -------------------------
def _frame_k_local(E: float, A: float, I: float, L: float) -> np.ndarray:
    # elemento marco 2D (u,v,θ) por nodo: 6x6 local
    EA_L = E * A / L
    EI = E * I
    k = np.zeros((6, 6), dtype=float)

    k[0, 0] = EA_L
    k[0, 3] = -EA_L
    k[3, 0] = -EA_L
    k[3, 3] = EA_L

    k[1, 1] = 12 * EI / (L ** 3)
    k[1, 2] = 6 * EI / (L ** 2)
    k[1, 4] = -12 * EI / (L ** 3)
    k[1, 5] = 6 * EI / (L ** 2)

    k[2, 1] = 6 * EI / (L ** 2)
    k[2, 2] = 4 * EI / L
    k[2, 4] = -6 * EI / (L ** 2)
    k[2, 5] = 2 * EI / L

    k[4, 1] = -12 * EI / (L ** 3)
    k[4, 2] = -6 * EI / (L ** 2)
    k[4, 4] = 12 * EI / (L ** 3)
    k[4, 5] = -6 * EI / (L ** 2)

    k[5, 1] = 6 * EI / (L ** 2)
    k[5, 2] = 2 * EI / L
    k[5, 4] = -6 * EI / (L ** 2)
    k[5, 5] = 4 * EI / L

    return k


def _frame_T(c: float, s: float) -> np.ndarray:
    T = np.zeros((6, 6), dtype=float)
    # node 1
    T[0, 0] = c
    T[0, 1] = s
    T[1, 0] = -s
    T[1, 1] = c
    T[2, 2] = 1.0
    # node 2
    T[3, 3] = c
    T[3, 4] = s
    T[4, 3] = -s
    T[4, 4] = c
    T[5, 5] = 1.0
    return T


def portal_linear_response(
    H: float,
    L: float,
    Ec: float,
    Ac: float,
    Ic: float,
    Eb: float,
    Ab: float,
    Ib: float,
) -> Dict[str, float]:
    """
    Portal 2D con 4 nodos:
      1=(0,0), 2=(L,0), 3=(0,H), 4=(L,H)
    Bases fijas (1 y 2). Top con v=0 y diafragma rígido: u3=u4=u.
    Aplica cortante total V=1N (0.5 en u3 + 0.5 en u4).
    Retorna:
      K0 = 1/u
      m_col_base = max |M_base| por N
      m_beam_end = max |M_end| por N
      r_col = U_col / (U_total)
    """
    # dof mapping: node i in [1..4], dof 0=u,1=v,2=th
    def dof(node: int, comp: int) -> int:
        return (node - 1) * 3 + comp

    nd = 12
    K = np.zeros((nd, nd), dtype=float)
    F = np.zeros(nd, dtype=float)

    # elementos: col izq (1-3), col der (2-4), viga (3-4)
    elems = [
        (1, 3, Ec, Ac, Ic),
        (2, 4, Ec, Ac, Ic),
        (3, 4, Eb, Ab, Ib),
    ]

    elem_info = []  # para post: (n1,n2,k_local,T)
    coords = {
        1: (0.0, 0.0),
        2: (L, 0.0),
        3: (0.0, H),
        4: (L, H),
    }

    for (n1, n2, E, A, I) in elems:
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        dx = x2 - x1
        dy = y2 - y1
        Le = float(math.hypot(dx, dy))
        c = dx / Le
        s = dy / Le

        k_loc = _frame_k_local(E, A, I, Le)
        T = _frame_T(c, s)
        k_g = T.T @ k_loc @ T

        dofs = [
            dof(n1, 0), dof(n1, 1), dof(n1, 2),
            dof(n2, 0), dof(n2, 1), dof(n2, 2),
        ]
        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += k_g[i, j]

        elem_info.append((n1, n2, k_loc, T))

    # cargas: V=1N repartido
    F[dof(3, 0)] += 0.5
    F[dof(4, 0)] += 0.5

    # BC + constraint u3=u4 con mapeo B
    fixed = set()
    for n in (1, 2):
        fixed.update([dof(n, 0), dof(n, 1), dof(n, 2)])
    # top vertical fijo
    fixed.update([dof(3, 1), dof(4, 1)])

    # B: d_full = B q, q = [u_top, th3, th4]
    B = np.zeros((nd, 3), dtype=float)
    B[dof(3, 0), 0] = 1.0
    B[dof(4, 0), 0] = 1.0
    B[dof(3, 2), 1] = 1.0
    B[dof(4, 2), 2] = 1.0

    # anula filas fixed en B
    for r in fixed:
        B[r, :] = 0.0

    K_red = B.T @ K @ B
    F_red = B.T @ F

    q = np.linalg.solve(K_red, F_red)
    u_top = float(q[0])
    K0 = 1.0 / max(1e-30, u_top)

    # Post: momentos y energías
    d_full = B @ q

    U_col = 0.0
    U_tot = 0.0
    M_col_bases = []
    M_beam_ends = []

    for (n1, n2, k_loc, T) in elem_info:
        dofs = [
            dof(n1, 0), dof(n1, 1), dof(n1, 2),
            dof(n2, 0), dof(n2, 1), dof(n2, 2),
        ]
        d_g = d_full[dofs]
        d_l = T @ d_g
        f_l = k_loc @ d_l  # [N, N, Nm, N, N, Nm]
        M1 = float(f_l[2])
        M2 = float(f_l[5])

        Ue = 0.5 * float(d_l @ (k_loc @ d_l))
        U_tot += Ue

        # columnas: n1 base, beam: n1,n2 son top
        if (n1, n2) in [(1, 3), (2, 4)]:
            U_col += Ue
            M_col_bases.append(M1)
        if (n1, n2) == (3, 4):
            M_beam_ends.extend([M1, M2])

    r_col = float(U_col / max(1e-30, U_tot))
    m_col_base = float(max(abs(m) for m in M_col_bases))  # [Nm] por N
    m_beam_end = float(max(abs(m) for m in M_beam_ends))  # [Nm] por N

    return {
        "K0": K0,
        "m_col_base": m_col_base,
        "m_beam_end": m_beam_end,
        "r_col": r_col,
    }


# -------------------------
# Restoring + snapshot/restore (agnóstico al tipo de elemento)
# -------------------------
def _snapshot(col: Element, beam: Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return col.snapshot(), beam.snapshot()


def _restore(col: Element, beam: Element, snap: Tuple[Dict[str, Any], Dict[str, Any]]) -> None:
    col.restore(snap[0])
    beam.restore(snap[1])


def _eval_restoring(u_old: float, u_new: float, v_mid: float, dt: float, Kgeo: float, col: Element, beam: Element) -> float:
    fcol = col.update_and_force(u_old, u_new, v_mid, dt)
    fbeam = beam.update_and_force(u_old, u_new, v_mid, dt)
    return float(fcol + fbeam - Kgeo * u_new)


# -------------------------
# Integradores
# -------------------------
def run_time_history_verlet(
    M: float,
    c: float,
    H: float,
    col: Element,
    beam: Element,
    exc: Excitation,
    A_factor_g: float,
    dt: float,
    tmax: float,
    drift_collapse: float = 0.10,
    fp_iters: int = 4,
    fp_tol: float = 1e-10,
) -> Dict[str, Any]:
    n = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    V = np.zeros(n)

    P_total = M * g
    Kgeo = P_total / H

    # init
    snap0 = _snapshot(col, beam)
    R0 = _eval_restoring(0.0, 0.0, 0.0, dt, Kgeo, col, beam)
    V[0] = R0
    P0 = -M * exc.ag(float(t[0]), A_factor_g * g)
    a[0] = (P0 - c * v[0] - R0) / M

    collapsed = False
    collapse_idx = None

    for i in range(n - 1):
        # predictor
        v_half = v[i] + 0.5 * dt * a[i]
        u_new = u[i] + dt * v_half

        Pnp1 = -M * exc.ag(float(t[i + 1]), A_factor_g * g)

        # fixed-point: v_mid depende de a_new, pero a_new depende de R(u_new, v_mid)
        snap_n = _snapshot(col, beam)
        v_mid = v_half  # start

        a_new = 0.0
        v_new = 0.0
        R_new = 0.0

        for _ in range(fp_iters):
            _restore(col, beam, snap_n)
            R_try = _eval_restoring(u[i], u_new, v_mid, dt, Kgeo, col, beam)
            # damping semi-implícito (v_{n+1} usa v_half)
            a_try = (Pnp1 - c * v_half - R_try) / max(1e-30, (M + 0.5 * c * dt))
            v_try = v_half + 0.5 * dt * a_try
            v_mid_new = 0.5 * (v[i] + v_try)

            if abs(v_mid_new - v_mid) <= fp_tol * max(1.0, abs(v_mid)):
                a_new, v_new, R_new = float(a_try), float(v_try), float(R_try)
                v_mid = float(v_mid_new)
                break

            a_new, v_new, R_new = float(a_try), float(v_try), float(R_try)
            v_mid = float(v_mid_new)

        # commit final con estado consistente a n
        _restore(col, beam, snap_n)
        R_new = _eval_restoring(u[i], u_new, v_mid, dt, Kgeo, col, beam)
        a_new = (Pnp1 - c * v_half - R_new) / max(1e-30, (M + 0.5 * c * dt))
        v_new = v_half + 0.5 * dt * a_new

        u[i + 1] = u_new
        v[i + 1] = v_new
        a[i + 1] = a_new
        V[i + 1] = R_new

        if abs(u_new) / H >= drift_collapse:
            collapsed = True
            collapse_idx = i + 1
            break

    if collapsed and collapse_idx is not None:
        t = t[: collapse_idx + 1]
        u = u[: collapse_idx + 1]
        v = v[: collapse_idx + 1]
        a = a[: collapse_idx + 1]
        V = V[: collapse_idx + 1]

    return {
        "t": t, "u": u, "v": v, "a": a, "V": V,
        "collapsed": collapsed,
        "P_total": P_total,
        "Kgeo": Kgeo,
    }


def run_time_history_hht(
    M: float,
    c: float,
    H: float,
    col: Element,
    beam: Element,
    exc: Excitation,
    A_factor_g: float,
    dt: float,
    tmax: float,
    drift_collapse: float = 0.10,
    alpha_hht: float = -0.10,
    newton_tol: float = 1e-8,
    newton_max: int = 25,
    fd_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    HHT-α (Hilber–Hughes–Taylor) para amortiguamiento numérico HF.
    α ∈ [-1/3, 0]. Newmark:
      γ = 0.5 - α
      β = (1-α)^2 / 4
    """
    if not (-1.0 / 3.0 <= alpha_hht <= 0.0):
        raise ValueError("alpha_hht debe estar en [-1/3, 0].")

    alpha = float(alpha_hht)
    gamma = 0.5 - alpha
    beta = (1.0 - alpha) ** 2 / 4.0

    n = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    V = np.zeros(n)

    P_total = M * g
    Kgeo = P_total / H

    # init
    snap0 = _snapshot(col, beam)
    R0 = _eval_restoring(0.0, 0.0, 0.0, dt, Kgeo, col, beam)
    V[0] = R0
    P0 = -M * exc.ag(float(t[0]), A_factor_g * g)
    a[0] = (P0 - c * v[0] - R0) / M

    collapsed = False
    collapse_idx = None

    for i in range(n - 1):
        tn = float(t[i])
        tnp1 = float(t[i + 1])

        Pn = -M * exc.ag(tn, A_factor_g * g)
        Pnp1 = -M * exc.ag(tnp1, A_factor_g * g)
        P_eff = (1.0 + alpha) * Pnp1 - alpha * Pn

        Rn = float(V[i])

        # predictores Newmark
        u_pred = u[i] + dt * v[i] + dt ** 2 * (0.5 - beta) * a[i]
        v_pred = v[i] + dt * (1.0 - gamma) * a[i]

        u_guess = float(u_pred)

        snap_n = _snapshot(col, beam)

        for _it in range(newton_max):
            a_guess = (u_guess - u_pred) / (beta * dt ** 2)
            v_guess = v_pred + gamma * dt * a_guess
            v_mid = 0.5 * (v[i] + v_guess)

            _restore(col, beam, snap_n)
            R_guess = _eval_restoring(u[i], u_guess, v_mid, dt, Kgeo, col, beam)

            res = M * a_guess + c * v_guess + (1.0 + alpha) * R_guess - alpha * Rn - P_eff

            scale = max(1.0, abs(P_eff) + abs((1.0 + alpha) * R_guess) + abs(M * a_guess))
            if abs(res) <= newton_tol * scale:
                break

            du = fd_eps * max(1.0, abs(u_guess))
            u2 = u_guess + du

            a2 = (u2 - u_pred) / (beta * dt ** 2)
            v2 = v_pred + gamma * dt * a2
            v_mid2 = 0.5 * (v[i] + v2)

            _restore(col, beam, snap_n)
            R2 = _eval_restoring(u[i], u2, v_mid2, dt, Kgeo, col, beam)

            res2 = M * a2 + c * v2 + (1.0 + alpha) * R2 - alpha * Rn - P_eff
            dres = (res2 - res) / du

            if abs(dres) < 1e-14:
                # fallback estable
                dres = (M / (beta * dt ** 2)) + (c * gamma / (beta * dt)) + (1.0 + alpha) * 1.0

            step = res / dres

            # limitador de paso
            max_step = 0.25 * max(1e-6, abs(u_pred) + 1e-6)
            step = float(np.clip(step, -max_step, max_step))
            u_guess = u_guess - step

        # commit n+1
        _restore(col, beam, snap_n)
        u_new = float(u_guess)
        a_new = float((u_new - u_pred) / (beta * dt ** 2))
        v_new = float(v_pred + gamma * dt * a_new)
        v_mid = 0.5 * (v[i] + v_new)
        R_new = _eval_restoring(u[i], u_new, v_mid, dt, Kgeo, col, beam)

        u[i + 1] = u_new
        v[i + 1] = v_new
        a[i + 1] = a_new
        V[i + 1] = R_new

        if abs(u_new) / H >= drift_collapse:
            collapsed = True
            collapse_idx = i + 1
            break

    if collapsed and collapse_idx is not None:
        t = t[: collapse_idx + 1]
        u = u[: collapse_idx + 1]
        v = v[: collapse_idx + 1]
        a = a[: collapse_idx + 1]
        V = V[: collapse_idx + 1]

    return {
        "t": t, "u": u, "v": v, "a": a, "V": V,
        "collapsed": collapsed,
        "P_total": P_total,
        "Kgeo": Kgeo,
    }


# -------------------------
# Plots: histéresis con gradiente temporal
# -------------------------
def plot_hysteresis_time_gradient(x: np.ndarray, y: np.ndarray, t: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = mpl.colors.Normalize(vmin=float(np.min(t)), vmax=float(np.max(t)))
    lc = LineCollection(segs, cmap="plasma", norm=norm)
    lc.set_array(t[:-1])
    lc.set_linewidth(2.5)
    ax.add_collection(lc)
    ax.autoscale_view()
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("t [s]")
    ax.set_title(title)
    ax.grid(True)
    return


# -------------------------
# Main (config + estudio)
# -------------------------
def build_excitation_from_env() -> Excitation:
    exc = Excitation()
    exc.type = _envs("EXC_TYPE", exc.type)
    exc.sine_w1_hz = _envf("EXC_SINE_W1_HZ", exc.sine_w1_hz)
    exc.sine_w2_hz = _envf("EXC_SINE_W2_HZ", exc.sine_w2_hz)
    exc.sine_scale = _envf("EXC_SINE_SCALE", exc.sine_scale)

    exc.shock_f0_hz = _envf("EXC_SHOCK_F0", exc.shock_f0_hz)
    exc.shock_t0_s = _envf("EXC_SHOCK_T0", exc.shock_t0_s)
    exc.shock_scale = _envf("EXC_SHOCK_SCALE", exc.shock_scale)

    exc.combo_sine_scale = _envf("EXC_COMBO_SINE", exc.combo_sine_scale)
    exc.combo_shock_scale = _envf("EXC_COMBO_SHOCK", exc.combo_shock_scale)

    exc.file_path = _envs("EXC_FILE", exc.file_path)
    exc.file_time_col = _envs("EXC_FILE_TIME", exc.file_time_col)
    exc.file_acc_col = _envs("EXC_FILE_COL", exc.file_acc_col)
    exc.file_units = _envs("EXC_FILE_UNITS", exc.file_units)
    exc.file_scale = _envf("EXC_FILE_SCALE", exc.file_scale)
    return exc


def main() -> None:
    # -------------------------
    # Parámetros (puedes tunear por ENV)
    # -------------------------
    # Geometría pórtico
    H = _envf("H", 5.0)     # altura piso [m]
    L = _envf("L", 6.0)     # luz [m]

    # Concreto/Acero (solo para prints; la capacidad My la dejamos por env)
    Ec = _envf("EC_GPA", 23.5) * 1e9
    fc_MPa = _envf("FC_MPA", 25.0)
    fy_MPa = _envf("FY_MPA", 411.9)

    # Secciones (para K0 lineal via FE)
    # (valores default razonables; si ya los tienes en el script viejo, ajusta por ENV)
    b_col = _envf("BCOL_M", 0.40)
    h_col = _envf("HCOL_M", 0.40)
    b_beam = _envf("BBEAM_M", 0.30)
    h_beam = _envf("HBEAM_M", 0.50)

    Ac = b_col * h_col
    Ic = b_col * (h_col ** 3) / 12.0
    Ab = b_beam * h_beam
    Ib = b_beam * (h_beam ** 3) / 12.0
    Eb = Ec  # mismo material

    # Periodo target para fijar masa
    T0 = _envf("T0", 0.5)  # [s]
    zeta = _envf("ZETA", 0.05)

    # Excitación
    exc = build_excitation_from_env()

    # Integración / estudio
    dt = _envf("DT", 0.001)
    tmax = _envf("TMAX", 10.0)
    drift_collapse = _envf("DRIFT_COL", 0.10)

    DO_PLOTS = bool(_envi("DO_PLOTS", 1))
    DO_STUDY = bool(_envi("DO_STUDY", 1))
    DO_DT_SWEEP = bool(_envi("DO_DT_SWEEP", 1))
    PLOT_EXC = bool(_envi("PLOT_EXC", 1))

    # Modelo columnas: epp|bw
    COL_MODEL = _envs("COL_MODEL", "epp").lower()
    # Beam siempre BW (por ahora)
    BEAM_MODEL = _envs("BEAM_MODEL", "bw").lower()

    # My (en MNm) — deja defaults como los de tu output previo
    My_col = _envf("MY_COL_MNM", 0.615) * 1e6
    My_beam = _envf("MY_BEAM_MNM", 0.108) * 1e6

    # Plástico efectivo (momento por V): Lp y brazos eficaces
    Lp_col = _envf("LP_COL_M", 0.55)
    Lp_beam = _envf("LP_BEAM_M", 0.65)

    # Bouc–Wen params (comunes)
    bw_alpha = _envf("BW_ALPHA", 0.05)
    bw_beta = _envf("BW_BETA", 0.5)
    bw_gamma = _envf("BW_GAMMA", 0.5)
    bw_n = _envf("BW_N", 2.0)
    bw_c_strength = _envf("BW_C_STRENGTH", 0.40)
    bw_c_stiff = _envf("BW_C_STIFF", 0.20)
    bw_eref_factor = _envf("BW_EREF_FACTOR", 1.0)  # multiplica Fy*uy

    # HHT alpha values
    HHT_ALPHAS = [None, -0.05, -0.10, -0.15]
    # A barrido (g)
    A_max = _envf("A_MAX_G", 0.8 if COL_MODEL == "bw" else 0.5)
    A_step = _envf("A_STEP_G", 0.1)
    A_list = np.arange(A_step, A_max + 1e-12, A_step)
    A_ref = _envf("A_REF_G", 0.4)

    # --------------------------------
    # 1) Lineal FE para K0 y brazos
    # --------------------------------
    lin = portal_linear_response(H=H, L=L, Ec=Ec, Ac=Ac, Ic=Ic, Eb=Eb, Ab=Ab, Ib=Ib)
    K0 = lin["K0"]
    m_col_base = lin["m_col_base"]
    m_beam_end = lin["m_beam_end"]
    r_col = lin["r_col"]

    # Masa para T0
    w0 = 2.0 * math.pi / T0
    M = K0 / (w0 ** 2)

    # Damping físico
    c_scale = _envf("C_SCALE", 1.0)
    c = 2.0 * c_scale * zeta * w0 * M

    # Axial por columna (solo print)
    Peso = M * g
    N0_col = 0.5 * Peso  # 2 columnas

    # brazos efectivos
    m_col_eff = m_col_base - 0.5 * Lp_col
    m_beam_eff = m_beam_end - 0.5 * Lp_beam

    # cortantes de fluencia (a nivel SDOF)
    Vy_col = My_col / max(1e-30, m_col_eff)
    Vy_beam = My_beam / max(1e-30, m_beam_eff)
    Vy_first = min(Vy_col, Vy_beam)
    uy_first = Vy_first / max(1e-30, K0)

    Kc = r_col * K0
    Kb = (1.0 - r_col) * K0

    print("\n=== Calibración elástica / capacidades ===")
    print(f"Ec = {Ec/1e9:.2f} GPa (aprox), fc'={fc_MPa:.1f} MPa, fy={fy_MPa:.1f} MPa")
    print(f"K0 = {K0/1e6:.2f} MN/m")
    print(f"M (para T0={T0:.2f}s) = {M:.0f} kg  -> Peso = {Peso/1e3:.1f} kN")
    print(f"N0 por columna = {N0_col/1e3:.1f} kN")
    print(f"My_col(N0) = {My_col/1e6:.3f} MN·m | My_beam(N≈0) = {My_beam/1e6:.3f} MN·m")
    print(f"m_col_base = {m_col_base:.3f} m  | m_beam_end = {m_beam_end:.3f} m  (momento por 1N de V)")
    print(f"Lp_col={Lp_col:.3f} m, Lp_beam={Lp_beam:.3f} m | m_col_eff={m_col_eff:.3f} m, m_beam_eff={m_beam_eff:.3f} m")
    print(f"Vy_col = {Vy_col/1e3:.1f} kN | Vy_beam = {Vy_beam/1e3:.1f} kN")
    print(f"Vy_first = {Vy_first/1e3:.1f} kN -> uy_first={uy_first*1e3:.2f} mm")
    print(f"Partición (energía FEM): r_col={r_col:.2f} -> Kc={Kc/K0:.2f} K0, Kb={Kb/K0:.2f} K0")
    print(f"Damping c = {c:.2e} N·s/m (ζ={zeta*100:.0f}% en T0)")

    if PLOT_EXC and DO_PLOTS:
        plot_excitation(exc, A_ref, dt, tmax)

    # f_cut para ringing: por defecto 4*f1
    f1 = 1.0 / T0
    f_cut = _envf("F_CUT_HZ", 4.0 * f1)

    # --------------------------------
    # 2) Fabricar elementos (cols + beam)
    # --------------------------------
    def make_columns() -> Element:
        if COL_MODEL == "bw":
            # 2 columnas en paralelo: cada una con Kc/2 y Fy/2
            k_each = 0.5 * Kc
            fy_each = 0.5 * Vy_col
            uy0 = fy_each / max(1e-30, k_each)
            e_ref = bw_eref_factor * fy_each * uy0
            s1 = BoucWenDegrading(k_each, fy_each, alpha=bw_alpha, beta=bw_beta, gamma=bw_gamma, n=bw_n,
                                 c_strength=bw_c_strength, c_stiff=bw_c_stiff, e_ref=e_ref)
            s2 = BoucWenDegrading(k_each, fy_each, alpha=bw_alpha, beta=bw_beta, gamma=bw_gamma, n=bw_n,
                                 c_strength=bw_c_strength, c_stiff=bw_c_stiff, e_ref=e_ref)
            return Parallel([s1, s2])
        # default: EPP
        return EPP(Kc, Vy_col)

    def make_beam() -> Element:
        if BEAM_MODEL == "epp":
            return EPP(Kb, Vy_beam)
        uy0 = Vy_beam / max(1e-30, Kb)
        e_ref = bw_eref_factor * Vy_beam * uy0
        return BoucWenDegrading(Kb, Vy_beam, alpha=bw_alpha, beta=bw_beta, gamma=bw_gamma, n=bw_n,
                                c_strength=bw_c_strength, c_stiff=bw_c_stiff, e_ref=e_ref)

    # --------------------------------
    # 3) Study: comparar Verlet vs HHT-α
    # --------------------------------
    if DO_STUDY:
        print("\n=== Estudio paramétrico (Verlet vs HHT-α) ===")

        results = []  # list of dicts

        for a_hht in HHT_ALPHAS:
            tag = "Verlet" if a_hht is None else f"HHT a={a_hht:.2f}"
            for A in A_list:
                col = make_columns()
                beam = make_beam()

                t0 = time.perf_counter()
                if a_hht is None:
                    out = run_time_history_verlet(M=M, c=c, H=H, col=col, beam=beam, exc=exc,
                                                  A_factor_g=float(A), dt=dt, tmax=tmax,
                                                  drift_collapse=drift_collapse)
                else:
                    out = run_time_history_hht(M=M, c=c, H=H, col=col, beam=beam, exc=exc,
                                               A_factor_g=float(A), dt=dt, tmax=tmax,
                                               drift_collapse=drift_collapse, alpha_hht=float(a_hht))
                cpu = time.perf_counter() - t0

                drift = float(np.max(np.abs(out["u"])) / H)
                hf_u = hf_ratio(out["u"], dt, f_cut)
                hf_a = hf_ratio(out["a"], dt, f_cut)
                hf_V = hf_ratio(out["V"], dt, f_cut)
                roughV = roughness_second_diff(out["V"])
                jerk = jerk_rms(out["a"], dt)

                rec = {
                    "method": tag,
                    "alpha": a_hht,
                    "A_g": float(A),
                    "drift": drift,
                    "hf_u": hf_u,
                    "hf_a": hf_a,
                    "hf_V": hf_V,
                    "roughV": roughV,
                    "jerk": jerk,
                    "cpu_s": cpu,
                    "collapsed": bool(out["collapsed"]),
                    "out": out,
                }
                results.append(rec)

                coltxt = "  (COLAPSO)" if out["collapsed"] else ""
                print(f"[{tag}] A={A:.1f}g  drift={drift*100:5.2f}%"
                      f"  hf_u={hf_u:9.3e}  hf_a={hf_a:9.3e}  hf_V={hf_V:9.3e}"
                      f"  roughV={roughV:8.2e}  jerk={jerk:8.2e}  cpu={cpu:5.2f}s{coltxt}")

        # Resumen colapso
        print("\n=== Resumen colapso (criterio deriva %.1f%%) ===" % (drift_collapse * 100))
        for a_hht in HHT_ALPHAS:
            tag = "Verlet" if a_hht is None else f"HHT a={a_hht:.2f}"
            Acol = None
            for rec in results:
                if rec["method"] == tag and rec["collapsed"]:
                    Acol = rec["A_g"]
                    break
            if Acol is None:
                print(f"{tag:>12}: no colapsa en barrido hasta A={A_max:.1f}g")
            else:
                print(f"{tag:>12}: colapsa en A≈{Acol:.1f}g")

        # --------------------------------
        # Plots comparativos en A_ref
        # --------------------------------
        if DO_PLOTS:
            # extrae outputs en A_ref
            outs_Aref = {}
            for a_hht in HHT_ALPHAS:
                tag = "Verlet" if a_hht is None else f"HHT a={a_hht:.2f}"
                # busca el registro más cercano a A_ref
                cand = [r for r in results if r["method"] == tag]
                idx = int(np.argmin([abs(r["A_g"] - A_ref) for r in cand]))
                outs_Aref[tag] = cand[idx]["out"]

            # u(t)
            plt.figure()
            for tag, out in outs_Aref.items():
                plt.plot(out["t"], out["u"], label=tag)
            plt.grid(True)
            plt.xlabel("t [s]")
            plt.ylabel("u [m]")
            plt.title(f"Comparación u(t) en A_ref={A_ref:.1f}g")
            plt.legend()

            # V-u overlay
            plt.figure()
            for tag, out in outs_Aref.items():
                plt.plot(out["u"], out["V"] / 1e3, label=tag)  # kN
            plt.grid(True)
            plt.xlabel("u [m]")
            plt.ylabel("V [kN]")
            plt.title(f"Comparación V-u en A_ref={A_ref:.1f}g (overlay)")
            plt.legend()

            # time gradient (solo uno)
            outg = outs_Aref.get("HHT a=-0.10", list(outs_Aref.values())[0])
            plot_hysteresis_time_gradient(outg["u"], outg["V"] / 1e3, outg["t"],
                                          title=f"Histeresis V-u (gradiente) — {('HHT a=-0.10' if 'HHT a=-0.10' in outs_Aref else 'case')} — A_ref={A_ref:.1f}g")
            plt.xlabel("u [m]")
            plt.ylabel("V_restaurador [kN]")

        # --------------------------------
        # 4) Barrido dt (para que HHT muestre ventaja en HF cuando dt crece)
        # --------------------------------
        if DO_DT_SWEEP:
            dt_list = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
            print(f"\n=== Barrido dt | f_cut={f_cut:.2f} Hz (≈4·f1) ===")

            for c_test, c_tag, A_dt in [(c, "c=phys", A_ref), (0.0, "c=0", 0.25)]:
                print(f"\n--- {c_tag} (A={A_dt:.2f}g) ---")
                print("\nVerlet:")
                print("  dt [s] | drift[%] | hf_a[e] | hf_V[e] | roughV | jerk")
                print("  -------+----------+---------+---------+--------+------")

                lines = {}

                for a_hht in HHT_ALPHAS:
                    tag = "Verlet" if a_hht is None else f"HHT a={a_hht:.2f}"
                    lines[tag] = {"dt": [], "hf_a": [], "hf_V": []}

                for dt2 in dt_list:
                    # Verlet
                    col = make_columns()
                    beam = make_beam()
                    out = run_time_history_verlet(M=M, c=c_test, H=H, col=col, beam=beam, exc=exc,
                                                  A_factor_g=float(A_dt), dt=float(dt2), tmax=tmax,
                                                  drift_collapse=drift_collapse)
                    drift = float(np.max(np.abs(out["u"])) / H)
                    hf_a_ = hf_ratio(out["a"], float(dt2), f_cut)
                    hf_V_ = hf_ratio(out["V"], float(dt2), f_cut)
                    roughV = roughness_second_diff(out["V"])
                    jerk = jerk_rms(out["a"], float(dt2))
                    coltxt = "  *COL*" if out["collapsed"] else ""
                    print(f"  {dt2:7.4f} | {drift*100:8.2f} | {hf_a_:7.3e} | {hf_V_:7.3e} | {roughV:6.1e} | {jerk:4.1e}{coltxt}")
                    lines["Verlet"]["dt"].append(dt2)
                    lines["Verlet"]["hf_a"].append(hf_a_)
                    lines["Verlet"]["hf_V"].append(hf_V_)

                    # HHTs
                    for a_hht in [-0.05, -0.10, -0.15]:
                        col = make_columns()
                        beam = make_beam()
                        out = run_time_history_hht(M=M, c=c_test, H=H, col=col, beam=beam, exc=exc,
                                                   A_factor_g=float(A_dt), dt=float(dt2), tmax=tmax,
                                                   drift_collapse=drift_collapse, alpha_hht=float(a_hht))
                        hf_a_ = hf_ratio(out["a"], float(dt2), f_cut)
                        hf_V_ = hf_ratio(out["V"], float(dt2), f_cut)
                        tag = f"HHT a={a_hht:.2f}"
                        lines[tag]["dt"].append(dt2)
                        lines[tag]["hf_a"].append(hf_a_)
                        lines[tag]["hf_V"].append(hf_V_)

                if DO_PLOTS:
                    # HF en a(t)
                    plt.figure()
                    for tag, d in lines.items():
                        plt.plot(d["dt"], d["hf_a"], marker="o", label=tag)
                    plt.grid(True)
                    plt.xlabel("dt [s]")
                    plt.ylabel(f"HF_ratio a(t), f>{f_cut:.1f} Hz")
                    plt.title(f"Sensibilidad dt (A={A_dt:.1f}g) — {c_tag} — ringing en a(t)")
                    plt.legend()

                    # HF en V(t)
                    plt.figure()
                    for tag, d in lines.items():
                        plt.plot(d["dt"], d["hf_V"], marker="o", label=tag)
                    plt.grid(True)
                    plt.xlabel("dt [s]")
                    plt.ylabel(f"HF_ratio V(t), f>{f_cut:.1f} Hz")
                    plt.title(f"Sensibilidad dt (A={A_dt:.1f}g) — {c_tag} — ringing en V(t)")
                    plt.legend()

    if DO_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()


"""
COMANDOS ÚTILES (terminal)

# 1) Caso base (seno)
python3 portico_shm.py

# 2) Shock (choque) — ajusta frecuencia y tiempo del pico
EXC_TYPE=shock EXC_SHOCK_F0=25 EXC_SHOCK_T0=0.2 A_REF_G=0.4 python3 portico_shm.py

# 3) Combo seno + choque
EXC_TYPE=combo EXC_COMBO_SHOCK=0.7 EXC_SHOCK_T0=0.4 python3 portico_shm.py

# 4) Importar aceleración de CSV (p.ej. resultados tren)
#   - Usa Time_s y Acceleration_g (en g) desde tu results (1).csv
EXC_TYPE=file EXC_FILE="results (1).csv" EXC_FILE_TIME=Time_s EXC_FILE_COL=Acceleration_g EXC_FILE_UNITS=g A_REF_G=0.4 python3 portico_shm.py

# 5) Cambiar modelo columnas a Bouc–Wen (2 columnas en paralelo)
COL_MODEL=bw python3 portico_shm.py

# 6) Desactivar plots / estudios
DO_PLOTS=0 DO_STUDY=0 python3 portico_shm.py

# 7) Ajustar paso de tiempo / dt sweep
DT=0.002 DO_DT_SWEEP=1 python3 portico_shm.py
"""
