#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(pórtico 1 piso) — análisis dinámico no lineal tipo SDOF calibrado con el pórtico elástico.

MODELO (pragmático para tarea):
- 1 GDL: desplazamiento lateral del “piso” u(t) (diafragma rígido).
- Rigidez elástica K0 y coeficientes de momentos por cortante (M/V) obtenidos con FE lineal del pórtico 2D.
- Resistencia no lineal como suma en PARALELO:
    (i) "Columnas" : resorte elasto-plástico ideal (EPP) en cortante (sin degradación),
        con Vy_col derivado de My_col(N) / (Mbase/V).
        My_col(N) se obtiene de la curva interacción N–M de la sección S1 (Problema 2).
        Axial N se toma constante = (M*g)/2 por columna (peso del piso).
    (ii) "Viga" : resorte Bouc–Wen “SHM” con degradación de rigidez y resistencia (editable),
        con Vy_beam derivado de My_beam(N≈0) / (Mendbeam/V) usando sección S2.
- Efecto P–Δ incluido como rigidez geométrica negativa: Kgeo = P_total/H.
- Amortiguamiento viscoso equivalente: ζ=5% en el período elástico T0=0.5s.
- Excitación: a_g(t)=A cos(0.2π t) sin(4π t), 0<=t<=10s.
- Barrido de amplitudes A = 0.1g, 0.2g, ... hasta colapso (criterio por deriva).

DEPENDENCIAS: numpy, matplotlib
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.collections import LineCollection
import matplotlib as mpl

def plot_hysteresis_time_gradient(x, y, t, ax=None, cmap="plasma", lw=2.5, cbar_label="t [s]"):
    """
    Dibuja y(x) coloreado por tiempo t usando LineCollection (gradiente temporal).
    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x); y = np.asarray(y); t = np.asarray(t)
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = mpl.colors.Normalize(vmin=float(t.min()), vmax=float(t.max()))
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(t[:-1])
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.autoscale_view()
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label(cbar_label)
    return ax

# -------------------------
# Constantes / unidades
# -------------------------
g = 9.80665  # m/s^2
PI = np.pi


# -------------------------
# Secciones RC (para interacción N–M)
# -------------------------
def area_bar(phi_m: float) -> float:
    return PI * (phi_m**2) / 4.0


@dataclass(frozen=True)
class RebarLayerSI:
    y_m_from_top: float
    n_bars: int
    phi_m: float

    @property
    def area_m2(self) -> float:
        return self.n_bars * area_bar(self.phi_m)


@dataclass(frozen=True)
class RCSectionSI:
    name: str
    b_m: float
    h_m: float
    layers: tuple  # tuple[RebarLayerSI, ...]

    @property
    def y_ref(self) -> float:
        return 0.5 * self.h_m


def NM_from_neutral_axis_SI(
    sec: RCSectionSI,
    c_m: float,
    fc_Pa: float,
    fy_Pa: float,
    alpha: float = 0.85,
    compression_at: str = "top",  # "top" o "bottom"
) -> tuple[float, float]:
    """
    Devuelve (N, M) en (N, N*m).
    N>0 compresión.
    M respecto al centroide (y_ref), y positivo con compresión "arriba" (convención simple).
    """
    b, h = sec.b_m, sec.h_m
    y_ref = sec.y_ref

    # espejo para "compresión abajo"
    def y_eff(y):
        return y if compression_at == "top" else (h - y)

    # hormigón: bloque rectangular en [0, a]
    a = float(np.clip(c_m, 0.0, h))
    sig_c = alpha * fc_Pa
    Nc = sig_c * b * a  # N
    yc = 0.5 * a
    yC_global = yc if compression_at == "top" else (h - yc)
    Mc = Nc * (yC_global - y_ref)

    # acero: perfectamente plástico ±fy
    Ns = 0.0
    Ms = 0.0
    for L in sec.layers:
        yL_eff = y_eff(L.y_m_from_top)
        sig_s = +fy_Pa if (yL_eff < c_m) else -fy_Pa
        Fs = sig_s * L.area_m2
        Ns += Fs
        y_global = L.y_m_from_top
        Ms += Fs * (y_global - y_ref)

    return float(Nc + Ns), float(Mc + Ms)


def sample_interaction_cloud_SI(
    sec: RCSectionSI,
    fc_Pa: float,
    fy_Pa: float,
    alpha: float = 0.85,
    n_c: int = 600,
) -> np.ndarray:
    """Nube de puntos (N,M) muestreando c en [0,h] para compresión arriba y abajo."""
    c_vals = np.linspace(0.0, sec.h_m, n_c)
    pts = []
    for comp in ("top", "bottom"):
        for c in c_vals:
            N, M = NM_from_neutral_axis_SI(sec, c, fc_Pa, fy_Pa, alpha=alpha, compression_at=comp)
            pts.append((N, M))
    return np.array(pts, dtype=float)


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Andrew monotone chain. Devuelve vértices CCW sin repetir.
    points: (N,2)
    """
    pts = np.unique(points.round(6), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def M_capacity_at_N(poly_NM: np.ndarray, N0: float) -> tuple[float, float]:
    """
    Dado un polígono (casco convexo) en (N,M), encuentra intersecciones con la recta N=N0.
    Retorna (Mmax_pos, Mmin_neg) en N*m.
    Si no intersecta, retorna (nan, nan).
    """
    Ms = []
    n = len(poly_NM)
    for i in range(n):
        N1, M1 = poly_NM[i]
        N2, M2 = poly_NM[(i+1) % n]
        if (N0 - N1) * (N0 - N2) <= 0 and abs(N2 - N1) > 1e-12:
            t = (N0 - N1) / (N2 - N1)
            if 0.0 <= t <= 1.0:
                Mx = M1 + t * (M2 - M1)
                Ms.append(Mx)

    if len(Ms) < 2:
        return (np.nan, np.nan)

    Ms = np.array(Ms)
    return float(np.max(Ms)), float(np.min(Ms))


# -------------------------
# FE lineal del pórtico para K0 y coeficientes de momentos por V
# -------------------------
def frame_element_k_global(E, A, I, x1, y1, x2, y2):
    L = float(np.hypot(x2 - x1, y2 - y1))
    c = (x2 - x1) / L
    s = (y2 - y1) / L

    EA_L = E * A / L
    EI = E * I

    kL = np.array([
        [ EA_L,        0,           0, -EA_L,        0,           0],
        [    0,  12*EI/L**3,  6*EI/L**2,    0, -12*EI/L**3,  6*EI/L**2],
        [    0,   6*EI/L**2,   4*EI/L,     0,  -6*EI/L**2,   2*EI/L],
        [-EA_L,       0,           0,  EA_L,        0,           0],
        [    0, -12*EI/L**3, -6*EI/L**2,   0,  12*EI/L**3, -6*EI/L**2],
        [    0,   6*EI/L**2,   2*EI/L,     0,  -6*EI/L**2,   4*EI/L],
    ], dtype=float)

    T = np.array([
        [ c,  s, 0, 0, 0, 0],
        [-s,  c, 0, 0, 0, 0],
        [ 0,  0, 1, 0, 0, 0],
        [ 0,  0, 0, c,  s, 0],
        [ 0,  0, 0,-s,  c, 0],
        [ 0,  0, 0, 0,  0, 1],
    ], dtype=float)

    kG = T.T @ kL @ T
    return kG, kL, T, L


def portal_elastic_K0_and_moment_coeffs(Ec, sec_col, sec_beam, H=3.0, L=5.0):
    """
    Pórtico con 4 nodos:
      1:(0,0)  2:(L,0)  3:(0,H)  4:(L,H)
    Bases fijas (u,v,θ=0). En techo: v3=v4=0. Diafragma: u3=u4 (=u).

    Devuelve:
      K0 [N/m]
      m_col_base [N*m per N]   (máximo abs entre 2 bases)
      m_beam_end [N*m per N]   (máximo abs entre 2 extremos)
    """
    coords = {
        1: (0.0, 0.0),
        2: (L,   0.0),
        3: (0.0, H),
        4: (L,   H),
    }

    # DOFs globales: [u,v,θ] por nodo
    def dof(node, comp):  # comp: 0=u,1=v,2=θ
        return 3*(node-1) + comp

    ndof = 12
    K = np.zeros((ndof, ndof))

    # propiedades geométricas elásticas (usamos concreto "bruto")
    Acol = sec_col.b_m * sec_col.h_m
    Icol = sec_col.b_m * sec_col.h_m**3 / 12.0
    Abeam = sec_beam.b_m * sec_beam.h_m
    Ibeam = sec_beam.b_m * sec_beam.h_m**3 / 12.0

    elements = [
        ("colL", 1, 3, Acol, Icol),
        ("colR", 2, 4, Acol, Icol),
        ("beam", 3, 4, Abeam, Ibeam),
    ]

    # ensamblaje
    elem_cache = {}
    for name, n1, n2, A, I in elements:
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        kG, kL, T, L_e = frame_element_k_global(Ec, A, I, x1, y1, x2, y2)
        elem_cache[name] = (n1, n2, A, I, kG, kL, T, L_e)

        edofs = [
            dof(n1,0), dof(n1,1), dof(n1,2),
            dof(n2,0), dof(n2,1), dof(n2,2),
        ]
        for i in range(6):
            for j in range(6):
                K[edofs[i], edofs[j]] += kG[i,j]

    # Restricciones:
    fixed = set()
    # bases totalmente fijas
    for n in (1,2):
        fixed.update([dof(n,0), dof(n,1), dof(n,2)])
    # techo v=0
    fixed.update([dof(3,1), dof(4,1)])

    # libres (antes de amarrar diafragma)
    free = [i for i in range(ndof) if i not in fixed]
    # free debería ser [u3,θ3,u4,θ4]
    # Construimos K_free
    Kf = K[np.ix_(free, free)]

    # Mapear q=[u,θ3,θ4] a d_free=[u3,θ3,u4,θ4]
    # d_free = [u, θ3, u, θ4]
    C = np.array([
        [1,0,0],  # u3
        [0,1,0],  # θ3
        [1,0,0],  # u4
        [0,0,1],  # θ4
    ], dtype=float)

    K_red = C.T @ Kf @ C

    # Fuerza lateral total 1 N en el diafragma: repartir 0.5 y 0.5 en u3 y u4
    F_free = np.array([0.5, 0.0, 0.5, 0.0], dtype=float)
    F_red = C.T @ F_free  # = [1,0,0]

    q = np.linalg.solve(K_red, F_red)
    u = q[0]
    K0 = 1.0 / u

    # reconstruir d_global para calcular fuerzas internas
    d_full = np.zeros(ndof)
    # set top
    d_full[dof(3,0)] = u
    d_full[dof(4,0)] = u
    d_full[dof(3,2)] = q[1]
    d_full[dof(4,2)] = q[2]
    # v3=v4=0 ya

    # extraer momentos de extremo por V=1
    M_col_bases = []
    M_beam_ends = []


    U_col = 0.0
    U_beam = 0.0
    for name, (n1, n2, A, I, kG, kL, T, L_e) in elem_cache.items():
        edofs = [
            dof(n1,0), dof(n1,1), dof(n1,2),
            dof(n2,0), dof(n2,1), dof(n2,2),
        ]
        d_e_global = d_full[edofs]
        d_e_local = T @ d_e_global
        f_local = kL @ d_e_local

        Ue = 0.5 * float(d_e_local.T @ (kL @ d_e_local))
        if name.startswith("col"):
            U_col += Ue
        elif name == "beam":
            U_beam += Ue
        # f_local: [N1, V1, M1, N2, V2, M2] (convención local)
        M1 = f_local[2]
        M2 = f_local[5]

        if name.startswith("col"):
            # base es el nodo de abajo (1 o 2)
            if n1 in (1,2):
                M_col_bases.append(M1)
            else:
                M_col_bases.append(M2)
        if name == "beam":
            M_beam_ends.append(M1)
            M_beam_ends.append(M2)

    m_col_base = float(np.max(np.abs(M_col_bases)))  # N*m por N
    m_beam_end = float(np.max(np.abs(M_beam_ends)))
    r_col = float(U_col / (U_col + U_beam + 1e-30))
    return float(K0), m_col_base, m_beam_end, r_col
# -------------------------
# Histeresis: columnas EPP + viga Bouc–Wen con degradación
# -------------------------
@dataclass
class ColEPP:
    K: float      # N/m
    Fy: float     # N
    up: float = 0.0  # desplazamiento plástico (m)

    def force_update(self, u: float) -> float:
        f_tr = self.K * (u - self.up)
        if abs(f_tr) <= self.Fy:
            return f_tr
        f = np.sign(f_tr) * self.Fy
        self.up = u - f / self.K
        return f


@dataclass
class BeamBoucWenDegrading:
    K0: float
    Fy0: float
    alpha: float = 0.05
    beta: float = 0.5
    gamma: float = 0.5
    n: float = 2.0
    # degradación (ajusta libremente)
    c_strength: float = 0.40
    c_stiff: float = 0.20

    z: float = 0.0      # variable histérica (dimensionless)
    E_diss: float = 0.0 # energía histérica acumulada (J aproximada = N*m)
    K: float = None
    Fy: float = None

    def __post_init__(self):
        self.K = float(self.K0)
        self.Fy = float(self.Fy0)

    def degrade(self):
        # Energía de referencia: Fy0 * uy0
        uy0 = self.Fy0 / (self.K0 + 1e-30)
        E0 = self.Fy0 * uy0 + 1e-30
        rF = 1.0 / (1.0 + self.c_strength * self.E_diss / E0)
        rK = 1.0 / (1.0 + self.c_stiff   * self.E_diss / E0)
        self.Fy = self.Fy0 * rF
        self.K  = self.K0  * rK

    def update_and_force(self, u_old: float, u_new: float, v_mid: float, dt: float) -> float:
        # actualizar degradación
        self.degrade()

        uy = self.Fy / (self.K + 1e-30)
        x_dot = v_mid / (uy + 1e-30)

        # RK2 para z
        def z_dot(z):
            return (x_dot
                    - self.beta * abs(x_dot) * (abs(z) ** (self.n - 1.0)) * z
                    - self.gamma * x_dot * (abs(z) ** self.n))

        z1 = self.z
        k1 = z_dot(z1)
        z_mid = z1 + 0.5 * dt * k1
        k2 = z_dot(z_mid)
        z2 = z1 + dt * k2
        self.z = float(np.clip(z2, -1.5, 1.5))  # clip suave

        # fuerza
        # parte elástica + parte histérica
        f = self.alpha * self.K * u_new + (1.0 - self.alpha) * self.Fy * self.z

        # energía histérica incremental (aprox)
        du = u_new - u_old
        f_h = (1.0 - self.alpha) * self.Fy * self.z
        self.E_diss += abs(f_h * du)

        return float(f)


# -------------------------
# Ground motion
# -------------------------
def ag(t: float, A_g: float) -> float:
    """
    A_g en [m/s^2] (o sea A*g).
    a_g(t)=A cos(0.2π t) sin(4π t)
    """
    return A_g * np.cos(0.2 * PI * t) * np.sin(4.0 * PI * t)


# -------------------------
# Integración dinámica (Velocity Verlet + actualización histérica)
# -------------------------
def run_time_history(
    K0: float,
    M: float,
    c: float,
    H: float,
    col: ColEPP,
    beam: BeamBoucWenDegrading,
    A_factor_g: float,
    dt: float = 0.001,
    tmax: float = 10.0,
    drift_collapse: float = 0.10
):
    n = int(tmax / dt) + 1
    t = np.linspace(0.0, tmax, n)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    Vres = np.zeros(n)  # fuerza restauradora total (incluye P–Δ)
    Fy_beam_hist = np.zeros(n)
    K_beam_hist = np.zeros(n)

    P_total = M * g
    Kgeo = P_total / H  # rigidez geométrica (N/m) -> desestabiliza

    # condición inicial
    fcol = col.force_update(0.0)
    fbeam = beam.update_and_force(0.0, 0.0, 0.0, dt)
    fres = fcol + fbeam - Kgeo * 0.0
    a[0] = (-c * v[0] - fres - M * ag(t[0], A_factor_g * g)) / M
    Vres[0] = fres
    Fy_beam_hist[0] = beam.Fy
    K_beam_hist[0] = beam.K

    collapsed = False
    collapse_idx = None

    for i in range(n - 1):
        # half-step velocity
        v_half = v[i] + 0.5 * dt * a[i]
        # update displacement
        u_new = u[i] + dt * v_half

        # update hysteresis using (u[i] -> u_new, v_half)
        fcol = col.force_update(u_new)
        fbeam = beam.update_and_force(u[i], u_new, v_half, dt)
        fres = fcol + fbeam - Kgeo * u_new

        # compute new acceleration
        a_new = (-c * v_half - fres - M * ag(t[i+1], A_factor_g * g)) / M
        # complete velocity
        v_new = v_half + 0.5 * dt * a_new

        u[i+1] = u_new
        v[i+1] = v_new
        a[i+1] = a_new
        Vres[i+1] = fres
        Fy_beam_hist[i+1] = beam.Fy
        K_beam_hist[i+1] = beam.K

        if abs(u_new) / H >= drift_collapse and not collapsed:
            collapsed = True
            collapse_idx = i + 1
            break

    if collapsed:
        t = t[:collapse_idx+1]
        u = u[:collapse_idx+1]
        v = v[:collapse_idx+1]
        a = a[:collapse_idx+1]
        Vres = Vres[:collapse_idx+1]
        Fy_beam_hist = Fy_beam_hist[:collapse_idx+1]
        K_beam_hist = K_beam_hist[:collapse_idx+1]

    return {
        "t": t, "u": u, "v": v, "a": a,
        "V": Vres,
        "Fy_beam": Fy_beam_hist,
        "K_beam": K_beam_hist,
        "collapsed": collapsed,
        "P_total": P_total,
        "Kgeo": Kgeo,
    }


# -------------------------
# MAIN
# -------------------------
def main():
    # -------------------------
    # Geometría pórtico
    # -------------------------
    H = 3.0   # m
    L = 5.0   # m
    T0 = 0.5  # s (objetivo)

    # -------------------------
    # Materiales (editables)
    # -------------------------
    # fy dado: 4.2 tonf/cm^2 -> 4.2 * 98.0665 MPa
    fy_MPa = 4.2 * 98.0665
    fy = fy_MPa * 1e6  # Pa

    # f'c no viene: pon tu valor
    fc_MPa = 25.0
    fc = fc_MPa * 1e6  # Pa
    alpha = 0.85

    # módulo Ec (puedes fijarlo o usar fórmula)
    Ec_MPa = 4700.0 * np.sqrt(fc_MPa)  # típica ACI, aprox.
    Ec = Ec_MPa * 1e6

    # -------------------------
    # Secciones Problema 2 (suposición cover efectivo)
    # -------------------------
    cover = 0.06  # m (ajusta si tu pauta usa otro)
    phi = 0.02    # m (Ø20)

    # S1: 40x60, 4Ø20 arriba + 4Ø20 abajo
    S1 = RCSectionSI(
        name="S1_col_40x60",
        b_m=0.40, h_m=0.60,
        layers=(
            RebarLayerSI(y_m_from_top=cover,        n_bars=4, phi_m=phi),
            RebarLayerSI(y_m_from_top=0.60-cover,   n_bars=4, phi_m=phi),
        ),
    )

    # S2: 25x50, 3Ø20 arriba + 2Ø20 abajo
    S2 = RCSectionSI(
        name="S2_beam_25x50",
        b_m=0.25, h_m=0.50,
        layers=(
            RebarLayerSI(y_m_from_top=cover,        n_bars=3, phi_m=phi),
            RebarLayerSI(y_m_from_top=0.50-cover,   n_bars=2, phi_m=phi),
        ),
    )

    # -------------------------
    # 1) Interacción N–M (polígonos)
    # -------------------------
    pts_col = sample_interaction_cloud_SI(S1, fc, fy, alpha=alpha, n_c=700)
    hull_col = convex_hull_2d(pts_col)

    pts_beam = sample_interaction_cloud_SI(S2, fc, fy, alpha=alpha, n_c=700)
    hull_beam = convex_hull_2d(pts_beam)

    # Plot interacción
    plt.figure()
    plt.plot(pts_col[:, 1]/1e6, pts_col[:, 0]/1e3, ".", ms=1.5, alpha=0.25, label="nube col")
    plt.plot(np.r_[hull_col[:,1]/1e6, hull_col[0,1]/1e6], np.r_[hull_col[:,0]/1e3, hull_col[0,0]/1e3], "-", lw=2, label="poligonal col")
    plt.xlabel("M [MN·m]")
    plt.ylabel("N [kN] (compresión +)")
    plt.title("Interacción N–M — columnas (S1)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.figure()
    plt.plot(pts_beam[:, 1]/1e6, pts_beam[:, 0]/1e3, ".", ms=1.5, alpha=0.25, label="nube viga")
    plt.plot(np.r_[hull_beam[:,1]/1e6, hull_beam[0,1]/1e6], np.r_[hull_beam[:,0]/1e3, hull_beam[0,0]/1e3], "-", lw=2, label="poligonal viga")
    plt.xlabel("M [MN·m]")
    plt.ylabel("N [kN]")
    plt.title("Interacción N–M — viga (S2)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # -------------------------
    # 2) Rigidez elástica K0 del pórtico + coeficientes (M/V)
    # -------------------------
    K0, m_col_base, m_beam_end, r_col = portal_elastic_K0_and_moment_coeffs(Ec, S1, S2, H=H, L=L)

    # Masa para T0
    w0 = 2.0 * PI / T0
    M = K0 / (w0**2)

    # Axial en columnas (peso del piso repartido)
    N0_col = 0.5 * M * g  # N

    # Capacidades a momento desde interacción (en esa N0)
    Mpos_col, Mneg_col = M_capacity_at_N(hull_col, N0_col)
    if np.isnan(Mpos_col):
        raise RuntimeError("N0_col queda fuera del polígono de la columna. Ajusta fc/Ec/cover o revisa la masa.")

    My_col = min(abs(Mpos_col), abs(Mneg_col))  # (simétrica, pero usamos min por robustez)

    # Para viga: N≈0
    Mpos_beam, Mneg_beam = M_capacity_at_N(hull_beam, 0.0)
    if np.isnan(Mpos_beam):
        raise RuntimeError("N=0 queda fuera del polígono de la viga (raro). Revisa nube/hull.")
    # viga asimétrica -> usar capacidad conservadora (menor |M|)
    My_beam = min(abs(Mpos_beam), abs(Mneg_beam))

    # Convertir a cortantes equivalentes usando momentos por unit shear
    Vy_col = My_col / (m_col_base + 1e-30)
    Vy_beam = My_beam / (m_beam_end + 1e-30)

    # Primer yield global (aprox)
    Vy_first = min(Vy_col, Vy_beam)
    uy_first = Vy_first / K0
    # Partición de rigidez en paralelo (más física): por energía elástica FEM (caso V=1 N)
    # r_col viene de portal_elastic_K0_and_moment_coeffs()
    Kc = max(r_col * K0, 0.05 * K0)
    Kb = max((1.0 - r_col) * K0, 0.05 * K0)
    # Renormaliza para asegurar Kc + Kb = K0 exactamente
    s = Kc + Kb
    Kc *= K0 / (s + 1e-30)
    Kb *= K0 / (s + 1e-30)

    # Amortiguamiento 5% en T0
    zeta = 0.05
    c = 2.0 * zeta * w0 * M

    print("\n=== Calibración elástica / capacidades ===")
    print(f"Ec = {Ec/1e9:.2f} GPa (aprox), fc'={fc_MPa:.1f} MPa, fy={fy_MPa:.1f} MPa")
    print(f"K0 = {K0/1e6:.2f} MN/m")
    print(f"M (para T0=0.5s) = {M:.0f} kg  -> Peso = {M*g/1e3:.1f} kN")
    print(f"N0 por columna = {N0_col/1e3:.1f} kN")
    print(f"My_col(N0) = {My_col/1e6:.3f} MN·m | My_beam(N≈0) = {My_beam/1e6:.3f} MN·m")
    print(f"m_col_base = {m_col_base:.3f} m  | m_beam_end = {m_beam_end:.3f} m  (momento por 1N de V)")
    print(f"Vy_col = {Vy_col/1e3:.1f} kN | Vy_beam = {Vy_beam/1e3:.1f} kN")
    print(f"Vy_first = {Vy_first/1e3:.1f} kN -> uy_first={uy_first*1000:.2f} mm")
    print(f"Partición (energía FEM): r_col={r_col:.2f} -> Kc={Kc/K0:.2f} K0, Kb={Kb/K0:.2f} K0")
    print(f"Damping c = {c:.2e} N·s/m (ζ=5% en T0)")

    # -------------------------
    # 3) Barrido de amplitudes A = 0.1g, 0.2g, ...
    # -------------------------
    dt = 0.001
    tmax = 10.0
    drift_collapse = 0.10

    A_list = np.arange(0.1, 2.1, 0.1)  # hasta 2.0g o colapso
    results = []
    collapsed_at = None

    for A in A_list:
        col = ColEPP(K=Kc, Fy=Vy_col)
        beam = BeamBoucWenDegrading(
            K0=Kb, Fy0=Vy_beam,
            alpha=0.05, beta=0.5, gamma=0.5, n=2.0,
            c_strength=0.40, c_stiff=0.20
        )

        out = run_time_history(
            K0=K0, M=M, c=c, H=H,
            col=col, beam=beam,
            A_factor_g=A,
            dt=dt, tmax=tmax,
            drift_collapse=drift_collapse
        )

        drift_max = np.max(np.abs(out["u"])) / H
        results.append((A, drift_max, out["collapsed"]))

        print(f"A={A:.1f}g -> drift_max={100*drift_max:.2f}%  {'(COLAPSO)' if out['collapsed'] else ''}")

        if out["collapsed"]:
            collapsed_at = A
            last_out = out
            break
        last_out = out

    results = np.array(results, dtype=float)

    # -------------------------
    # Plots resumen
    # -------------------------
    plt.figure()
    plt.plot(results[:,0], 100*results[:,1], "-o")
    plt.xlabel("Amplitud A [g]")
    plt.ylabel("Deriva máxima [%]")
    plt.title("Curva incremental (IDA) simplificada — pórtico 1 piso")
    plt.grid(True, alpha=0.3)

    # Plot del último run (colapso o el mayor A sin colapso)
    t = last_out["t"]
    u = last_out["u"]
    V = last_out["V"]
    Fy_b = last_out["Fy_beam"]
    K_b = last_out["K_beam"]

    plt.figure()
    plt.plot(t, u*1000)
    plt.xlabel("t [s]")
    plt.ylabel("u [mm]")
    plt.title(f"Historia u(t) — A={results[-1,0]:.1f}g" + (" (colapso)" if last_out["collapsed"] else ""))
    plt.grid(True, alpha=0.3)

    fig, ax = plt.subplots()
    plot_hysteresis_time_gradient(u, V/1e3, t, ax=ax, cmap="plasma", lw=2.5, cbar_label="t [s]")
    ax.set_xlabel("u [m]")
    ax.set_ylabel("V_restaurador [kN]")
    ax.set_title(f"Histeresis V–u (gradiente temporal) — A={results[-1,0]:.1f}g")
    ax.grid(True, alpha=0.3)
    plt.figure()
    plt.plot(t, Fy_b/1e3, label="Fy_beam")
    plt.plot(t, K_b/1e6, label="K_beam")
    plt.xlabel("t [s]")
    plt.ylabel("Fy [kN] / K [MN/m]")
    plt.title("Degradación viga (SHM Bouc–Wen) en el último run")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Ground motion del último run (en g)
    plt.figure()
    a_g = np.array([ag(tt, results[-1,0]*g) for tt in t])
    plt.plot(t, a_g/g)
    plt.xlabel("t [s]")
    plt.ylabel("a_g [g]")
    plt.title("Entrada sísmica seno-modulada (último run)")
    plt.grid(True, alpha=0.3)

    if collapsed_at is not None:
        print(f"\n>>> Colapso (criterio deriva {100*drift_collapse:.1f}%) alcanzado en A ≈ {collapsed_at:.1f}g")
    else:
        print(f"\n>>> No colapsó hasta A={results[-1,0]:.1f}g con el criterio deriva {100*drift_collapse:.1f}%")

    plt.show()


if __name__ == "__main__":
    main()
