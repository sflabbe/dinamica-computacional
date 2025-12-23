#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(N–M–V): Secciones RC con acero perfectamente plástico y hormigón solo a compresión.
- Calcula curvas de interacción N–M (y opcionalmente prisma N–M–V).
- Construye superficie de fluencia como casco convexo (polígono) en (N,M).
- Implementa modelo elasto-plástico en resultantes (N,M) con flujo asociado (return mapping a polígono).
- Simula historias cíclicas combinadas (ε0, κ) y grafica resultados.

UNIDADES:
- Longitudes: cm
- Tensiones: tonf/cm^2
- Fuerzas: tonf
- Momentos: tonf*cm (para plot se convierte a tonf*m)

NOTA: f'c no viene en el enunciado -> se deja como parámetro editable.
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
# Utilidades de unidades
# -------------------------
MPA_PER_TONF_PER_CM2 = 98.0665  # 1 tonf/cm^2 = 98.0665 MPa
def mpaa_to_tonfcm2(x_mpa: float) -> float:
    return x_mpa / MPA_PER_TONF_PER_CM2


# -------------------------
# Geometría / materiales
# -------------------------
def area_bar_cm2(phi_cm: float) -> float:
    return np.pi * (phi_cm ** 2) / 4.0

@dataclass(frozen=True)
class RebarLayer:
    y_cm_from_top: float  # posición del centro de la capa medida desde la cara superior (cm)
    n_bars: int
    phi_cm: float

    @property
    def area_cm2(self) -> float:
        return self.n_bars * area_bar_cm2(self.phi_cm)

@dataclass(frozen=True)
class RCSection:
    name: str
    b_cm: float
    h_cm: float
    layers: tuple  # tuple[RebarLayer, ...]

    @property
    def y_ref_cm(self) -> float:
        return 0.5 * self.h_cm

    @property
    def As_total_cm2(self) -> float:
        return float(sum(L.area_cm2 for L in self.layers))

    @property
    def Ag_cm2(self) -> float:
        return self.b_cm * self.h_cm


# -------------------------
# Resultantes plásticas (N,M) para una profundidad de NA = c
# Modelo: hormigón bloque rectangular (alpha*fc) en [0,c] y acero ±sy según lado del NA
# -------------------------
def NM_from_neutral_axis(
    sec: RCSection,
    c_cm: float,
    fc_tonf_cm2: float,
    sy_tonf_cm2: float,
    alpha: float = 0.85,
    compression_at: str = "top",  # "top" o "bottom"
) -> tuple[float, float]:
    """
    Devuelve (N, M) en (tonf, tonf*cm).
    Convención: y positivo hacia abajo desde cara superior, y_ref = h/2.
    N>0 compresión.
    M = sum(F_i * (y_i - y_ref)).
    """
    b, h = sec.b_cm, sec.h_cm
    y_ref = sec.y_ref_cm

    # espejo para "compresión abajo": intercambiamos coordenada y -> h - y
    def y_eff(y):
        return y if compression_at == "top" else (h - y)

    # Hormigón: compresión uniforme en zona 0..a
    a = float(np.clip(c_cm, 0.0, h))
    sig_c = alpha * fc_tonf_cm2  # tonf/cm^2
    Nc = sig_c * b * a  # tonf
    yc = 0.5 * a  # cm desde cara comprimida "efectiva" (si top, desde arriba; si bottom, ya espejamos con y_eff)
    # OJO: para compresión abajo, el "origen" efectivo es abajo; pero usamos y_eff en momentos:
    # representamos el bloque en coordenadas globales como un punto a distancia yc desde la cara comprimida.
    y_face_comp = 0.0 if compression_at == "top" else h
    yC_global = yc if compression_at == "top" else (h - yc)
    Mc = Nc * (yC_global - y_ref)  # tonf*cm

    # Acero: perfectamente plástico
    Ns = 0.0
    Ms = 0.0
    for L in sec.layers:
        yL = y_eff(L.y_cm_from_top)
        # Si yL < c -> compresión; si yL > c -> tracción (en el modelo ideal)
        sig_s = +sy_tonf_cm2 if (yL < c_cm) else -sy_tonf_cm2
        Fs = sig_s * L.area_cm2
        Ns += Fs
        # Momento respecto a y_ref en coordenada global: ojo con compresión abajo (usamos y_global real)
        y_global = L.y_cm_from_top
        Ms += Fs * (y_global - y_ref)

    N = Nc + Ns
    M = Mc + Ms
    return float(N), float(M)


def sample_interaction_cloud(
    sec: RCSection,
    fc_tonf_cm2: float,
    sy_tonf_cm2: float,
    alpha: float = 0.85,
    n_c: int = 400,
) -> np.ndarray:
    """
    Genera nube de puntos (N,M) muestreando c en [0,h] para compresión arriba y abajo.
    Devuelve array shape (Npts,2).
    """
    h = sec.h_cm
    c_vals = np.linspace(0.0, h, n_c)
    pts = []
    for comp in ("top", "bottom"):
        for c in c_vals:
            N, M = NM_from_neutral_axis(sec, c, fc_tonf_cm2, sy_tonf_cm2, alpha=alpha, compression_at=comp)
            pts.append((N, M))
    return np.array(pts, dtype=float)


# -------------------------
# Casco convexo 2D (Andrew monotone chain)
# -------------------------
def convex_hull(points: np.ndarray) -> np.ndarray:
    """
    points: (N,2) -> devuelve vértices del casco convexo en CCW, sin repetir último.
    """
    pts = np.unique(points.round(12), axis=0)  # dedup suave
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x then y

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


# -------------------------
# Point-in-polygon (ray casting) y proyección al borde (en espacio escalado)
# -------------------------
def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cond = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0)
        if cond:
            inside = not inside
    return inside

def closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-30)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab

def project_to_polygon_boundary_scaled(u_tr: np.ndarray, poly_u: np.ndarray) -> np.ndarray:
    """
    Proyecta u_tr al borde del polígono poly_u (2D) con distancia Euclídea.
    Si está dentro, devuelve u_tr.
    """
    if point_in_polygon(u_tr, poly_u):
        return u_tr.copy()

    best = None
    best_d2 = np.inf
    n = len(poly_u)
    for i in range(n):
        a = poly_u[i]
        b = poly_u[(i + 1) % n]
        q = closest_point_on_segment(u_tr, a, b)
        d2 = float(np.dot(u_tr - q, u_tr - q))
        if d2 < best_d2:
            best_d2 = d2
            best = q
    return best


# -------------------------
# Rigidez elástica equivalente (muy simple) para (N,M)
# -------------------------
def elastic_stiffness_NM(sec: RCSection, fc_tonf_cm2: float,
                         Es_tonf_cm2: float = mpaa_to_tonfcm2(200000.0),
                         Ec_tonf_cm2: float | None = None) -> tuple[float, float]:
    """
    EA, EI aproximados para usar en el modelo elástico inicial.
    Ec: si None, usa Ec = 25000 MPa aprox (editable).
    """
    if Ec_tonf_cm2 is None:
        Ec_tonf_cm2 = mpaa_to_tonfcm2(25000.0)

    b, h = sec.b_cm, sec.h_cm
    As = sec.As_total_cm2
    Ag = b * h
    Ac = Ag - As

    EA = Ec_tonf_cm2 * Ac + Es_tonf_cm2 * As

    # Inercia del rectángulo (concreto) respecto a eje centroidal:
    Ig = b * h**3 / 12.0
    # Aporte diferencial del acero (transformado): (Es - Ec)*Σ As_i*(y_i - yref)^2
    yref = sec.y_ref_cm
    Is = 0.0
    for L in sec.layers:
        dy = L.y_cm_from_top - yref
        Is += L.area_cm2 * dy**2
    EI = Ec_tonf_cm2 * Ig + (Es_tonf_cm2 - Ec_tonf_cm2) * Is

    return float(EA), float(EI)


# -------------------------
# Modelo elasto-plástico (N,M) con return mapping a polígono (flujo asociado en norma energética)
# -------------------------
def simulate_cyclic_NM(
    hull_NM: np.ndarray,
    EA: float,
    EI: float,
    eps0_hist: np.ndarray,
    kappa_hist: np.ndarray,
) -> dict:
    """
    Integra con deformaciones prescritas (ε0, κ). Estado: e_p = (εp0, κp).
    return mapping a la superficie (casco convexo) en norma definida por C = diag(1/EA, 1/EI).
    """
    n = len(eps0_hist)
    N_out = np.zeros(n)
    M_out = np.zeros(n)
    ep0 = 0.0
    kp = 0.0
    Wp = np.zeros(n)  # trabajo plástico acumulado

    # Escalamiento para proyección en norma energética:
    sN = np.sqrt(EA)
    sM = np.sqrt(EI)
    poly_u = np.column_stack([hull_NM[:, 0] / sN, hull_NM[:, 1] / sM])

    prev_N = 0.0
    prev_M = 0.0
    prev_ep0 = 0.0
    prev_kp = 0.0

    for i in range(n):
        e0 = float(eps0_hist[i])
        k  = float(kappa_hist[i])

        # predictor elástico
        N_tr = EA * (e0 - ep0)
        M_tr = EI * (k  - kp)

        # check / proyección
        u_tr = np.array([N_tr / sN, M_tr / sM], dtype=float)
        u_pr = project_to_polygon_boundary_scaled(u_tr, poly_u)
        N = float(u_pr[0] * sN)
        M = float(u_pr[1] * sM)

        # update plastic strains to satisfy N=EA(e0-ep0), M=EI(k-kp)
        ep0_new = e0 - N / EA
        kp_new  = k  - M / EI

        # trabajo plástico incremental (aprox trapezoidal): dWp = s_avg · d(ep)
        d_ep0 = ep0_new - ep0
        d_kp  = kp_new  - kp
        Wp[i] = (Wp[i-1] if i > 0 else 0.0) + 0.5 * (prev_N + N) * d_ep0 + 0.5 * (prev_M + M) * d_kp

        # guardar
        N_out[i] = N
        M_out[i] = M
        prev_N, prev_M = N, M
        prev_ep0, prev_kp = ep0, kp
        ep0, kp = ep0_new, kp_new

    return {
        "N": N_out,
        "M": M_out,
        "Wp": Wp,
    }


# -------------------------
# Plot helpers
# -------------------------
def plot_interaction(sec: RCSection, pts: np.ndarray, hull: np.ndarray):
    # Convertimos M a tonf*m
    M_pts_m = pts[:, 1] / 100.0
    M_h_m   = hull[:, 1] / 100.0

    plt.figure()
    plt.plot(M_pts_m, pts[:, 0], ".", markersize=2, alpha=0.35, label="nube (muestreo c)")
    plt.plot(np.r_[M_h_m, M_h_m[0]], np.r_[hull[:, 0], hull[0, 0]], "-", linewidth=2, label="poligonal (casco convexo)")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("M [tonf·m]")
    plt.ylabel("N [tonf] (compresión +)")
    plt.title(f"Interacción N–M: {sec.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_cyclic(title: str, eps0, kappa, N, M, Wp, t_hist=None):
    fig, ax = plt.subplots()
    if t_hist is None:
        ax.plot(M / 100.0, N, "-", linewidth=1.5)
    else:
        plot_hysteresis_time_gradient(M / 100.0, N, t_hist, ax=ax, cmap="plasma", lw=2.5, cbar_label="t")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("M [tonf·m]")
    plt.ylabel("N [tonf]")
    plt.title(f"{title} — trayectoria en N–M")
    plt.grid(True, alpha=0.3)

    fig, ax = plt.subplots()
    plot_hysteresis_time_gradient(kappa, M / 100.0, t_hist, ax=ax, cmap="viridis", lw=2.5, cbar_label="t")
    plt.xlabel("κ [1/cm]")
    plt.ylabel("M [tonf·m]")
    plt.title(f"{title} — histéresis M–κ")
    plt.grid(True, alpha=0.3)

    fig, ax = plt.subplots()
    plot_hysteresis_time_gradient(eps0, N, t_hist, ax=ax, cmap="cividis", lw=2.5, cbar_label="t")
    plt.xlabel("ε0 [-]")
    plt.ylabel("N [tonf]")
    plt.title(f"{title} — histéresis N–ε0")
    plt.grid(True, alpha=0.3)

    plt.figure()
    plt.plot(Wp, "-", linewidth=1.5)
    plt.xlabel("paso")
    plt.ylabel("Trabajo plástico acumulado Wp [tonf]")
    plt.title(f"{title} — disipación plástica (Wp ≥ 0 esperado)")
    plt.grid(True, alpha=0.3)


# -------------------------
# MAIN
# -------------------------
def main():
    # -------------------------
    # Parámetros del enunciado
    # -------------------------
    sy = 4.2  # tonf/cm^2 (dado)
    alpha = 0.85

    # f'c NO viene: pon tu valor aquí (en tonf/cm^2). Por ejemplo 25 MPa ~ 0.255 tonf/cm^2
    fc = 0.25  # tonf/cm^2  (≈ 24.5 MPa)

    # Recubrimiento efectivo asumido (centro de barras a 6 cm de la cara)
    cover = 6.0  # cm
    phi = 2.0    # cm (Ø20mm)

    # -------------------------
    # Secciones
    # -------------------------
    S1 = RCSection(
        name="S1 (40x60) — 4Ø20 arriba + 4Ø20 abajo",
        b_cm=40.0,
        h_cm=60.0,
        layers=(
            RebarLayer(y_cm_from_top=cover, n_bars=4, phi_cm=phi),
            RebarLayer(y_cm_from_top=60.0 - cover, n_bars=4, phi_cm=phi),
        ),
    )

    S2 = RCSection(
        name="S2 (25x50) — 3Ø20 arriba + 2Ø20 abajo",
        b_cm=25.0,
        h_cm=50.0,
        layers=(
            RebarLayer(y_cm_from_top=cover, n_bars=3, phi_cm=phi),
            RebarLayer(y_cm_from_top=50.0 - cover, n_bars=2, phi_cm=phi),
        ),
    )

    sections = [S1, S2]

    # -------------------------
    # 1) Curvas de interacción y poligonal (casco convexo)
    # -------------------------
    hulls = {}
    clouds = {}

    for sec in sections:
        pts = sample_interaction_cloud(sec, fc_tonf_cm2=fc, sy_tonf_cm2=sy, alpha=alpha, n_c=600)
        hull = convex_hull(pts)
        clouds[sec.name] = pts
        hulls[sec.name] = hull
        plot_interaction(sec, pts, hull)

    # -------------------------
    # 2–3) Modelo EP + verificación con historias cíclicas (ε0, κ)
    # -------------------------
    # Definimos 3 historias:
    n = 2000
    t = np.linspace(0, 4 * 2 * np.pi, n)  # 4 ciclos

    for sec in sections:
        hull = hulls[sec.name]
        EA, EI = elastic_stiffness_NM(sec, fc_tonf_cm2=fc)

        # amplitudes "automáticas" para que entre a plástico:
        Nmax = np.max(np.abs(hull[:, 0]))
        Mmax = np.max(np.abs(hull[:, 1]))
        eps_amp   = 1.2 * (Nmax / EA)
        kappa_amp = 1.2 * (Mmax / EI)

        # Historia A: proporcional (misma fase)
        eps0_A = eps_amp * np.sin(t)
        kappa_A = kappa_amp * np.sin(t)

        # Historia B: no proporcional (desfase 90°)
        eps0_B = eps_amp * np.sin(t)
        kappa_B = kappa_amp * np.cos(t)

        # Historia C: con sesgo axial (mean) + flexión cíclica
        eps0_C = 0.35 * eps_amp + 0.8 * eps_amp * np.sin(t)
        kappa_C = 1.0 * kappa_amp * np.sin(t)

        for tag, eps0_hist, kappa_hist in [
            ("A (proporcional)", eps0_A, kappa_A),
            ("B (no proporcional)", eps0_B, kappa_B),
            ("C (sesgo axial + flexión)", eps0_C, kappa_C),
        ]:
            out = simulate_cyclic_NM(hull, EA, EI, eps0_hist, kappa_hist)
            plot_cyclic(f"{sec.name} — {tag}", eps0_hist, kappa_hist, out["N"], out["M"], out["Wp"], t_hist=t)
    plt.show()


if __name__ == "__main__":
    main()
