#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 20:26:21 2025

@author: sebastian
"""

"""
demo_portico_debug.py

Versión depurada y autocontenida del Problema 4.
Se han incluido las clases 'plastic_hinge' (Mock) dentro del script
para eliminar dependencias externas y asegurar la ejecución.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ==============================================================================
# 1. MOCK DE DEPENDENCIAS (plastic_hinge)
#    Implementación simplificada para que el código sea autocontenido.
# ==============================================================================

@dataclass
class RebarLayer:
    As: float
    y: float

@dataclass
class RCSectionRect:
    b: float
    h: float
    fc: float
    fy: float
    Es: float
    layers: List[RebarLayer]
    n_fibers: int = 50
    eps_c0: float = 0.002
    eps_cu: float = 0.0035

    def sample_interaction_curve(self, n: int = 20) -> np.ndarray:
        """
        Genera una curva de interacción simplificada (N, M) para el mock.
        Devuelve puntos en el primer cuadrante (Compresión, M positivo).
        """
        # Capacidad a compresión pura approx
        Nu = 0.85 * self.fc * (self.b * self.h) + sum(l.As for l in self.layers) * self.fy
        # Capacidad a tracción pura approx
        Tu = sum(l.As for l in self.layers) * self.fy
        # Momento puro approx
        d = self.h - 0.05 # recubrimiento asumido
        As_tot = sum(l.As for l in self.layers)
        Mu = 0.9 * (As_tot/2) * self.fy * 0.8 * self.h # estimación burda
        
        # Generar elipse deformada como curva de interacción
        t = np.linspace(0, np.pi, n)
        # Mapear t a N: de -Tu a +Nu
        N_range = np.linspace(-Tu, Nu, n)
        points = []
        for val_N in N_range:
            # Parabola simple para M
            if val_N > Nu or val_N < -Tu:
                val_M = 0.0
            else:
                # Perfil parabólico simple entre tracción y compresión
                # M es maximo cerca del "punto de balance" (aprox 0.3 Nu)
                N_norm = (val_N - (-Tu)) / (Nu - (-Tu)) # 0 a 1
                val_M = 4 * Mu * N_norm * (1 - N_norm)
            points.append([val_N, val_M])
        
        return np.array(points)

@dataclass
class NMSurfacePolygon:
    vertices: np.ndarray # Array Nx2

    @classmethod
    def from_points(cls, pts: np.ndarray) -> NMSurfacePolygon:
        # Envolvente convexa simplificada: solo ordenamos por ángulo polar o asumimos orden
        # Aquí asumimos que los puntos ya vienen "razonablemente" ordenados o son nube.
        # Para robustez, usamos ConvexHull de scipy si fuera necesario, 
        # pero aquí simplemente guardamos los puntos.
        return cls(vertices=pts)

# ==============================================================================
# 2. AYUDAS GRÁFICAS Y GEOMÉTRICAS
# ==============================================================================

def colored_line(ax, x, y, t, linewidth=2.0, cmap="viridis", label=None):
    x = np.asarray(x); y = np.asarray(y); t = np.asarray(t)
    if len(x) < 2: return # Evitar error con arrays vacíos
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t[:-1], cmap=cmap, linewidth=linewidth)
    ax.add_collection(lc)
    if label:
        ax.plot([], [], label=label) # Dummy plot para leyenda
    ax.autoscale()
    return lc

def mirror_section_about_middepth(sec: RCSectionRect) -> RCSectionRect:
    h = sec.h
    layers = [RebarLayer(As=l.As, y=h - l.y) for l in sec.layers]
    return RCSectionRect(
        b=sec.b, h=sec.h, fc=sec.fc, fy=sec.fy, Es=sec.Es,
        eps_c0=sec.eps_c0, eps_cu=sec.eps_cu,
        layers=layers, n_fibers=sec.n_fibers,
    )

def build_nm_surface(sec: RCSectionRect, npts: int = 90, tension_positive: bool = True) -> NMSurfacePolygon:
    # Generamos la curva base
    pts1 = sec.sample_interaction_curve(n=npts)
    # Reflejamos para tener M negativo (simetría de sección)
    pts = np.vstack([pts1, pts1 * np.array([1.0, -1.0])])
    
    # IMPORTANTE: El código original asume sample_interaction_curve da Compresión (+).
    # Si queremos Tensión (+), invertimos el eje N (índice 0).
    if tension_positive:
        pts[:, 0] *= -1.0 
    
    # Asegurar cierre del polígono
    pts = np.vstack([pts, pts[0]])
    return NMSurfacePolygon.from_points(pts)

def moment_capacity_from_polygon(surface: NMSurfacePolygon, N: float) -> float:
    """Intersección de línea vertical N=const con el polígono (N, M)."""
    V = np.asarray(surface.vertices, float)
    if V.shape[0] < 3: return 1e-6 # Protección contra polígonos vacíos

    Nmin, Nmax = float(np.min(V[:, 0])), float(np.max(V[:, 0]))
    Nc = float(np.clip(N, Nmin, Nmax))

    Ms = []
    n_v = V.shape[0]
    for i in range(n_v):
        a = V[i]
        b = V[(i + 1) % n_v] # Cierre circular
        Na, Ma = float(a[0]), float(a[1])
        Nb, Mb = float(b[0]), float(b[1])

        # Chequeo robusto de intersección
        if abs(Nb - Na) < 1e-9:
            # Segmento vertical: si coincide con Nc, tomamos los M
            if abs(Na - Nc) < 1e-9:
                Ms.extend([Ma, Mb])
            continue
        
        # Interpolar M para N=Nc
        if (Nc - Na) * (Nc - Nb) <= 0: # Nc está entre Na y Nb
            t = (Nc - Na) / (Nb - Na)
            Mi = Ma + t * (Mb - Ma)
            Ms.append(float(Mi))

    if len(Ms) == 0:
        # Fallback: punto más cercano
        j = int(np.argmin(np.abs(V[:, 0] - Nc)))
        return float(abs(V[j, 1]))
    
    # Retorna el máximo valor absoluto de momento (capacidad envolvente)
    return float(max(abs(min(Ms)), abs(max(Ms))))


# ==============================================================================
# 3. CORE DE ELEMENTOS FINITOS
# ==============================================================================

@dataclass
class Node:
    x: float
    y: float
    dof_u: Tuple[int, int]   # (ux, uy)
    dof_th: int              # theta

class DofManager:
    def __init__(self):
        self._next = 0

    def new_trans(self) -> Tuple[int, int]:
        ux = self._next; uy = self._next + 1
        self._next += 2
        return ux, uy

    def new_rot(self) -> int:
        th = self._next
        self._next += 1
        return th

    @property
    def ndof(self) -> int:
        return self._next

def rot2d(c, s):
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    T[:3, :3] = R
    T[3:, 3:] = R
    return T

@dataclass
class FrameElementLinear2D:
    ni: int
    nj: int
    E: float
    A: float
    I: float
    nodes: List[Node]

    def _geom(self):
        xi, yi = self.nodes[self.ni].x, self.nodes[self.ni].y
        xj, yj = self.nodes[self.nj].x, self.nodes[self.nj].y
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        if L < 1e-12: raise ValueError(f"Elemento de longitud cero detectado entre nodos {self.ni}-{self.nj}")
        c, s = dx / L, dy / L
        return L, c, s

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def k_local(self):
        L, c, s = self._geom()
        E, A, I = self.E, self.A, self.I
        k = np.zeros((6, 6))
        k_ax = E * A / L
        k[0, 0] = k_ax; k[0, 3] = -k_ax
        k[3, 0] = -k_ax; k[3, 3] = k_ax

        k11 = 12 * E * I / (L ** 3)
        k12 = 6 * E * I / (L ** 2)
        k22 = 4 * E * I / L
        k22b = 2 * E * I / L

        k[1, 1] = k11;   k[1, 2] = k12;   k[1, 4] = -k11;  k[1, 5] = k12
        k[2, 1] = k12;   k[2, 2] = k22;   k[2, 4] = -k12;  k[2, 5] = k22b
        k[4, 1] = -k11;  k[4, 2] = -k12;  k[4, 4] = k11;   k[4, 5] = -k12
        k[5, 1] = k12;   k[5, 2] = k22b;  k[5, 4] = -k12;  k[5, 5] = k22
        return k

    def stiffness_and_force_global(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        L, c, s = self._geom()
        T = rot2d(c, s)
        dofs = self.dofs()
        u_g = u[dofs]
        u_l = T @ u_g
        k_l = self.k_local()
        f_l = k_l @ u_l
        k_g = T.T @ k_l @ T
        f_g = T.T @ f_l

        # axial force in tension-positive convention
        k_ax = self.E * self.A / L
        du_ax = u_l[3] - u_l[0]
        N_tension = k_ax * du_ax

        return dofs, k_g, f_g, {"N": float(N_tension)}


@dataclass
class ColumnHingeNMRot:
    """Rótula M-Theta con My dependiente de N."""
    surface: NMSurfacePolygon
    k0: float
    alpha_post: float = 1e-4

    th_p_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0
    My_comm: float = 1.0

    def set_yield_from_N(self, N_ref: float):
        self.My_comm = max(1e-6, moment_capacity_from_polygon(self.surface, N_ref))

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float]:
        K0 = float(self.k0)
        Kp = float(self.alpha_post) * K0
        H = (K0 * Kp) / max(K0 - Kp, 1e-18)

        M_trial = float(self.M_comm) + K0 * float(dth)
        f = abs(M_trial) - (float(self.My_comm) + H * float(self.a_comm))
        
        if f <= 0.0:
            return M_trial, K0, self.th_p_comm, self.a_comm, M_trial

        dg = f / (K0 + H)
        sgn = 1.0 if M_trial >= 0.0 else -1.0
        th_p_new = float(self.th_p_comm) + dg * sgn
        a_new = float(self.a_comm) + dg
        M_new = M_trial - K0 * dg * sgn
        k_tan = (K0 * H) / (K0 + H)
        return M_new, k_tan, th_p_new, a_new, M_new


@dataclass
class SHMBeamHinge1D:
    K0_0: float
    My_0: float
    alpha_post: float = 0.02
    cK: float = 2.0
    cMy: float = 1.0

    th_p_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float]:
        K0 = self.K0_0 * math.exp(-self.cK * self.a_comm)
        My = self.My_0 * math.exp(-self.cMy * self.a_comm)
        Kp = self.alpha_post * K0
        H = (K0 * Kp) / max(K0 - Kp, 1e-18)

        M_trial = self.M_comm + K0 * dth
        f = abs(M_trial) - (My + H * self.a_comm)
        
        if f <= 0.0:
            return M_trial, K0, self.th_p_comm, self.a_comm, M_trial

        dg = f / (K0 + H)
        sgn = 1.0 if M_trial >= 0 else -1.0
        th_p_new = self.th_p_comm + dg * sgn
        a_new = self.a_comm + dg
        M_new = M_trial - K0 * dg * sgn
        k_tan = (K0 * H) / (K0 + H)
        return M_new, k_tan, th_p_new, a_new, M_new


@dataclass
class RotSpringElement:
    ni: int
    nj: int
    kind: str  # "col_nm" or "beam_shm"
    col_hinge: Optional[ColumnHingeNMRot]
    beam_hinge: Optional[SHMBeamHinge1D]
    nodes: List[Node]
    _trial: Dict = None

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        # Devuelve los DOFs: [u_i, v_i, th_i, u_j, v_j, th_j]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        dofs = self.dofs()
        th_i = float(u_trial[dofs[2]])
        th_j = float(u_trial[dofs[5]])
        th_i_c = float(u_comm[dofs[2]])
        th_j_c = float(u_comm[dofs[5]])

        # Incremento de rotación relativa respecto al paso "comm"
        dth_inc = (th_j - th_i) - (th_j_c - th_i_c)

        if self.kind == "col_nm":
            M, kM, th_p_new, a_new, M_new = self.col_hinge.eval_increment(dth_inc)
            trial_state = {"th_p_new": th_p_new, "a_new": a_new, "M_new": M_new, "M": float(M)}
        elif self.kind == "beam_shm":
            M, kM, th_p_new, a_new, M_new = self.beam_hinge.eval_increment(dth_inc)
            trial_state = {"th_p_new": th_p_new, "a_new": a_new, "M_new": M_new, "M": float(M)}
        else:
            raise ValueError("Unknown hinge kind")

        # Vector Bm para [ui, vi, thi, uj, vj, thj] -> dtheta = thj - thi
        # Bm = [0, 0, -1, 0, 0, 1]
        # Stiffness matrix (6x6)
        # f_l (6)
        
        # Optimizacion: Solo llenar indices 2 y 5
        k_l = np.zeros((6,6))
        k_l[2,2] = kM;  k_l[2,5] = -kM
        k_l[5,2] = -kM; k_l[5,5] = kM
        
        f_l = np.zeros(6)
        f_l[2] = -M # Momento en nodo i (reacción opuesta)
        f_l[5] = M  # Momento en nodo j

        self._trial = trial_state
        info = {"dtheta": float(dth_inc), "M": float(M)}
        if self.kind == "col_nm":
            info["My"] = float(self.col_hinge.My_comm)
        return k_l, f_l, info

    def commit(self):
        if self._trial is None: return
        if self.kind == "col_nm":
            self.col_hinge.th_p_comm = self._trial["th_p_new"]
            self.col_hinge.a_comm = self._trial["a_new"]
            self.col_hinge.M_comm = self._trial["M_new"]
        elif self.kind == "beam_shm":
            self.beam_hinge.th_p_comm = self._trial["th_p_new"]
            self.beam_hinge.a_comm = self._trial["a_new"]
            self.beam_hinge.M_comm = self._trial["M_new"]


@dataclass
class Model:
    nodes: List[Node]
    beams: List[FrameElementLinear2D]
    hinges: List[RotSpringElement]
    fixed_dofs: np.ndarray
    mass_diag: np.ndarray
    C_diag: np.ndarray
    load_const: np.ndarray
    col_hinge_groups: List[Tuple[int, int, int]]

    def ndof(self) -> int: return int(self.mass_diag.size)

    def free_dofs(self) -> np.ndarray:
        all_dofs = np.arange(self.ndof(), dtype=int)
        mask = np.ones(self.ndof(), dtype=bool)
        mask[self.fixed_dofs] = False
        return all_dofs[mask]

    def reset_state(self):
        for h in self.hinges:
            h._trial = None
            if h.kind == "col_nm":
                h.col_hinge.th_p_comm = 0.0
                h.col_hinge.a_comm = 0.0
                h.col_hinge.M_comm = 0.0
                h.col_hinge.My_comm = 1.0 # Reset seguro
            else:
                h.beam_hinge.th_p_comm = 0.0
                h.beam_hinge.a_comm = 0.0
                h.beam_hinge.M_comm = 0.0

    def update_column_yields(self, u_comm: np.ndarray):
        """Actualiza My basado en la carga axial actual."""
        N_beam = []
        for b in self.beams:
            _, _, _, meta = b.stiffness_and_force_global(u_comm)
            N_beam.append(meta["N"]) # tension +
        
        for hinge_idx, beam_idx, sign in self.col_hinge_groups:
            h = self.hinges[hinge_idx]
            if beam_idx < len(N_beam):
                Nref = float(sign) * float(N_beam[beam_idx])
                h.col_hinge.set_yield_from_N(Nref)

    def assemble(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        nd = self.ndof()
        K = np.zeros((nd, nd))
        R = np.zeros(nd)
        info = {"hinges": []}

        for e in self.beams:
            dofs, k_g, f_g, _ = e.stiffness_and_force_global(u_trial)
            # Ensamblaje vectorizado para velocidad
            ix_grid = np.ix_(dofs, dofs)
            K[ix_grid] += k_g
            R[dofs] += f_g

        for h in self.hinges:
            k_l, f_l, inf = h.eval_trial(u_trial, u_comm)
            dofs = h.dofs()
            ix_grid = np.ix_(dofs, dofs)
            K[ix_grid] += k_l
            R[dofs] += f_l
            info["hinges"].append(inf)

        fd = self.free_dofs()
        return K[np.ix_(fd, fd)], R[fd], info

    def commit(self):
        for h in self.hinges: h.commit()

    def base_shear(self, u: np.ndarray) -> float:
        base_ux = [self.nodes[0].dof_u[0], self.nodes[1].dof_u[0]]
        nd = self.ndof()
        R = np.zeros(nd)
        for e in self.beams:
            dofs, _, f_g, _ = e.stiffness_and_force_global(u)
            R[dofs] += f_g
        for h in self.hinges:
            # Usar u como comm para evaluar fuerza interna estática
            _, f_l, _ = h.eval_trial(u, u) 
            dofs = h.dofs()
            R[dofs] += f_l
        return float(np.sum(-R[base_ux])) # Reacción es opuesta a fuerza interna

# ==============================================================================
# 4. SOLVER HHT-ALPHA
# ==============================================================================

def hht_alpha_newton(model: Model, t: np.ndarray, ag: np.ndarray, drift_height: float,
                     drift_limit: float = 0.10, alpha: float = -0.05,
                     max_iter: int = 40, tol: float = 1e-6, verbose: bool = False) -> Dict[str, np.ndarray]:
    
    if not (-1.0/3.0 - 1e-12 <= alpha <= 1e-12):
        raise ValueError("HHT-alpha requiere alpha en [-1/3, 0].")

    model.reset_state()
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size

    # --- Gravedad Estática ---
    u = np.zeros(nd)
    u_free = u[fd].copy()
    
    # Pre-actualizar yields con N=0 para iniciar
    model.update_column_yields(u)

    for it in range(60):
        u_trial = u.copy(); u_trial[fd] = u_free
        
        # En estática, u_comm se asume igual a u_trial (iterativo) o anterior.
        # Aquí usamos u_previo (u) como base.
        K, Rint, _ = model.assemble(u_trial, u)
        
        res = model.load_const[fd] - Rint
        err = np.linalg.norm(res)
        ref = max(1.0, np.linalg.norm(model.load_const[fd]))
        
        if err < 1e-10 * ref:
            u = u_trial.copy()
            # Commit estado inicial de rotulas (probablemente elástico)
            model.commit()
            break
        
        try:
            du = np.linalg.solve(K + 1e-14*np.eye(nf), res)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(K + 1e-14*np.eye(nf), res, rcond=None)[0]
            
        u_free += du
        # Actualizar yields en cada iteración de gravedad porque N cambia
        model.update_column_yields(u_trial)
    else:
        raise RuntimeError("No converge el paso estático de gravedad.")

    # Inicializar estado dinámico
    u_n = u.copy()
    v_n = np.zeros(nd)
    a_n = np.zeros(nd) # Asumimos a=0 al inicio (carga estática balanceada)

    M = model.mass_diag
    C = model.C_diag

    dt = float(t[1] - t[0])
    gamma = 0.5 - alpha
    beta = 0.25 * (1.0 - alpha)**2
    
    # Vectores de influencia
    r = np.zeros(nd)
    r[np.where(M > 1e-9)[0]] = 1.0 # Solo DOFs con masa responden

    u_hist = np.zeros((t.size, nd))
    drift = np.zeros(t.size)
    Vb = np.zeros(t.size)
    iters = np.zeros(t.size-1, dtype=int)
    hinge_hist = []

    ux2 = model.nodes[2].dof_u[0]
    ux3 = model.nodes[3].dof_u[0]

    u_hist[0] = u_n
    drift[0] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
    Vb[0] = model.base_shear(u_n)

    p_const = model.load_const.copy()

    # --- Bucle de Tiempo ---
    for n in range(t.size - 1):
        # 1. Actualizar superficie de fluencia con el N del paso anterior (Explicit N-M coupling)
        model.update_column_yields(u_n)

        # 2. Cargas dinámicas
        p_n = p_const - M * r * ag[n]
        p_np1 = p_const - M * r * ag[n+1]
        p_alpha = (1.0 + alpha) * p_np1 - alpha * p_n

        # 3. Fuerza interna al inicio del paso (equilibrada)
        _, Rint_n, _ = model.assemble(u_n, u_n)

        # 4. Predictores
        dt = t[n+1] - t[n]
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        
        u_pred = u_n + dt * v_n + dt * dt * (0.5 - beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n
        
        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy() # Base para calcular incrementos

        # 5. Iteraciones Newton-Raphson
        converged = False
        info_converged = {}
        
        for it in range(max_iter):
            u_trial = u_comm_step.copy(); u_trial[fd] = u_free
            
            # Ensamblar K tangente y R interno
            K_tan, Rint, inf = model.assemble(u_trial, u_comm_step)

            # Cinemática HHT
            a_trial = a0 * (u_trial - u_pred)
            v_trial = v_pred + (gamma * dt) * a_trial # corrección v

            # Residual HHT
            force_inert = M[fd] * a_trial[fd]
            force_damp  = C[fd] * v_trial[fd]
            force_int   = (1.0 + alpha) * Rint - alpha * Rint_n
            
            res = p_alpha[fd] - (force_int + force_damp + force_inert)

            norm_res = np.linalg.norm(res)
            scale = 1.0 + np.linalg.norm(p_alpha[fd])
            
            if norm_res <= tol * scale:
                u_n = u_trial
                v_n = v_trial
                a_n = a_trial
                model.commit()
                iters[n] = it + 1
                hinge_hist.append(inf["hinges"])
                converged = True
                break

            # Matriz Efectiva
            K_eff = (1.0 + alpha) * K_tan + np.diag(M[fd] * a0) + np.diag(C[fd] * a1)
            
            # Solve
            try:
                du = np.linalg.solve(K_eff + 1e-14*np.eye(nf), res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K_eff + 1e-14*np.eye(nf), res, rcond=None)[0]
                
            u_free += du

        if not converged:
            raise RuntimeError(f"No converge en paso {n+1} / t={t[n+1]:.3f}s")

        u_hist[n+1] = u_n
        drift[n+1] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
        Vb[n+1] = model.base_shear(u_n)

        if abs(drift[n+1]) >= drift_limit:
            if verbose: print(f"COLLAPSE by drift >= {100*drift_limit:.1f}% at t={t[n+1]:.3f}s")
            u_hist = u_hist[:n+2]
            drift = drift[:n+2]
            Vb = Vb[:n+2]
            # ag se corta afuera
            iters = iters[:n+1]
            break

    # Retornar diccionarios cortados al tamaño real
    return {"t": t[:len(drift)], "ag": ag[:len(drift)], "u": u_hist, 
            "drift": drift, "Vb": Vb, "iters": iters, "hinges": hinge_hist}

# ==============================================================================
# 5. SETUP Y EJECUCIÓN
# ==============================================================================

def build_portal_beam_hinge(H=3.0, L=5.0, T0=0.5, zeta=0.02, P_gravity_total=1500e3):
    dm = DofManager()
    
    # Nodos físicos
    n0 = Node(0.0, 0.0, dm.new_trans(), dm.new_rot())
    n1 = Node(L,   0.0, dm.new_trans(), dm.new_rot())
    n2 = Node(0.0, H,   dm.new_trans(), dm.new_rot())
    n3 = Node(L,   H,   dm.new_trans(), dm.new_rot())
    nodes = [n0, n1, n2, n3]

    def aux_at(j):
        nj = nodes[j]
        # Nodo auxiliar comparte UX, UY con el nodo principal, pero tiene su propia ROTACION
        na = Node(nj.x, nj.y, nj.dof_u, dm.new_rot())
        nodes.append(na)
        return len(nodes) - 1

    # Nodos de rótulas
    i0L = aux_at(0)
    i2L = aux_at(2)
    i1R = aux_at(1)
    i3R = aux_at(3)
    i2B = aux_at(2)
    i3B = aux_at(3)

    # Materiales Mock
    fc, fy, Es = 30e6, 420e6, 200e9
    
    # Sección Columna
    b_col, h_col = 0.30, 0.40
    # Mock Layers
    layers = [RebarLayer(4e-4, 0.05), RebarLayer(4e-4, h_col-0.05)]
    sec_col = RCSectionRect(b_col, h_col, fc, fy, Es, layers)
    surf_col = build_nm_surface(sec_col, npts=40, tension_positive=True)

    # Propiedades viga (elástica)
    b_beam, h_beam = 0.30, 0.50
    A_beam = b_beam * h_beam
    I_beam = b_beam * h_beam**3 / 12.0
    A_col = b_col * h_col
    I_col = b_col * h_col**3 / 12.0
    E = 30e9

    # Elementos Barra (Elasticos)
    beams = [
        FrameElementLinear2D(i0L, i2L, E, A_col, I_col, nodes), # 0: Col Izq
        FrameElementLinear2D(i1R, i3R, E, A_col, I_col, nodes), # 1: Col Der
        FrameElementLinear2D(i2B, i3B, E, A_beam, I_beam, nodes), # 2: Viga
    ]

    # Rigidez inicial para las rótulas (muy alta para simular continuidad inicial)
    # k = n * EI/L. Usamos un valor alto relativo.
    k_col0 = 100.0 * E * I_col / H
    k_beam0 = 100.0 * E * I_beam / L

    hinges = []
    # Rótulas Columnas N-M (4)
    hinges.append(RotSpringElement(0, i0L, "col_nm", ColumnHingeNMRot(surf_col, k_col0), None, nodes))
    hinges.append(RotSpringElement(2, i2L, "col_nm", ColumnHingeNMRot(surf_col, k_col0), None, nodes))
    hinges.append(RotSpringElement(1, i1R, "col_nm", ColumnHingeNMRot(surf_col, k_col0), None, nodes))
    hinges.append(RotSpringElement(3, i3R, "col_nm", ColumnHingeNMRot(surf_col, k_col0), None, nodes))
    
    # Rótulas Viga SHM (2)
    # Capacidad aprox 300 kNm
    shm = SHMBeamHinge1D(K0_0=k_beam0, My_0=300e3)
    hinges.append(RotSpringElement(2, i2B, "beam_shm", None, shm, nodes))
    hinges.append(RotSpringElement(3, i3B, "beam_shm", None, shm, nodes))

    fixed = np.array([
        nodes[0].dof_u[0], nodes[0].dof_u[1], nodes[0].dof_th,
        nodes[1].dof_u[0], nodes[1].dof_u[1], nodes[1].dof_th,
    ], dtype=int)

    nd = dm.ndof
    mass = np.zeros(nd)
    C = np.zeros(nd)
    p0 = np.zeros(nd)

    # Carga gravedad (Vertical Y, negativo)
    p0[nodes[2].dof_u[1]] = -0.5 * P_gravity_total
    p0[nodes[3].dof_u[1]] = -0.5 * P_gravity_total

    model = Model(nodes, beams, hinges, fixed, mass, C, p0, 
                  col_hinge_groups=[(0,0,1), (1,0,1), (2,1,1), (3,1,1)])

    # Calibrar Masa para T0
    # Stiffness lateral simple
    K_story = story_stiffness_linear(model)
    omega0 = 2.0 * math.pi / T0
    M_total = K_story / (omega0**2)
    
    # Asignar masa horizontal
    mass[nodes[2].dof_u[0]] = 0.5 * M_total
    mass[nodes[3].dof_u[0]] = 0.5 * M_total
    
    # Damping Rayleigh simplificado (prop. a masa)
    C[:] = 2.0 * zeta * omega0 * mass

    return model, {"K_story": K_story, "M_total": M_total, "surface": surf_col}

def story_stiffness_linear(model: Model) -> float:
    nd = model.ndof()
    fd = model.free_dofs()
    
    K, _, _ = model.assemble(np.zeros(nd), np.zeros(nd))
    f = np.zeros(nd)
    f[model.nodes[2].dof_u[0]] = 0.5
    f[model.nodes[3].dof_u[0]] = 0.5
    
    uf = np.linalg.solve(K[np.ix_(fd, fd)] + 1e-14*np.eye(fd.size), f[fd])
    u = np.zeros(nd); u[fd] = uf
    disp = 0.5*(u[model.nodes[2].dof_u[0]] + u[model.nodes[3].dof_u[0]])
    return 1.0 / disp

def run_ida():
    # Parámetros simulación
    print("Iniciando IDA (Versión Debug)...")
    amps = np.arange(0.1, 1.5, 0.2) # Pocos pasos para demo rápida
    drifts = []
    
    for ag_mult in amps:
        model, meta = build_portal_beam_hinge(T0=0.5)
        dt = 0.005
        t = np.arange(0, 5.0, dt)
        acc = ag_mult * 9.81 * np.cos(2*np.pi*t) * np.sin(4*np.pi*t) # Artificial
        
        try:
            res = hht_alpha_newton(model, t, acc, drift_height=3.0, verbose=False)
            max_d = np.max(np.abs(res["drift"]))
            drifts.append(max_d)
            print(f"Amp {ag_mult:.1f}g -> Drift {100*max_d:.2f}%")
            
            # Plot último resultado exitoso para verificar
            if ag_mult == amps[-1] or max_d > 0.02:
                plot_last_run(res, meta)
                
        except RuntimeError as e:
            print(f"Amp {ag_mult:.1f}g -> Fallo convergencia ({e})")
            break

    plt.figure()
    plt.plot(amps[:len(drifts)], np.array(drifts)*100, '-o')
    plt.xlabel("PGA [g]"); plt.ylabel("Max Drift [%]")
    plt.grid(True)
    plt.title("Curva IDA Generada")
    plt.savefig("debug_ida_curve.png")
    print("\nProceso terminado. Revisar 'debug_ida_curve.png'.")

def plot_last_run(res, meta):
    t = res["t"]
    drift = res["drift"]
    Vb = res["Vb"]
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    
    # Historia Drift
    ax[0].plot(t, drift*100)
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("Drift [%]")
    ax[0].grid(True)
    
    # Histeresis Global
    colored_line(ax[1], drift*100, Vb/1000, t)
    ax[1].set_xlabel("Drift [%]"); ax[1].set_ylabel("Vb [kN]")
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("debug_last_response.png")
    plt.close()

if __name__ == "__main__":
    run_ida()