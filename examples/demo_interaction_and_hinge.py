
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from plastic_hinge import RCSectionRect, RebarLayer, NMSurfacePolygon, PlasticHingeNM

def colored_line(ax, x, y, t, linewidth=2.0, cmap="viridis"):
    x = np.asarray(x); y = np.asarray(y); t = np.asarray(t)
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t[:-1], cmap=cmap, linewidth=linewidth)
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def main():
    # Example section (change to yours)
    b = 0.40
    h = 0.60
    fc = 25e6
    fy = 420e6
    Es = 200e9

    # 4Ø20 top + 4Ø20 bottom (rough)
    phi = 0.020
    As_bar = np.pi * (phi**2) / 4.0
    layers = [
        RebarLayer(As=4*As_bar, y=0.05),
        RebarLayer(As=4*As_bar, y=h-0.05),
    ]
    sec = RCSectionRect(b=b, h=h, fc=fc, fy=fy, Es=Es, layers=layers)

    # Build interaction points (N compression +, tension -)
    pts = sec.sample_interaction_curve(n=120)
    # Add mirrored moments to show +/- M explicitly
    pts = np.vstack([pts, pts*np.array([1.0, -1.0])])

    # Convert to tension-positive if you want (common in FE codes)
    pts_tension_pos = pts.copy()
    pts_tension_pos[:, 0] *= -1.0

    surf = NMSurfacePolygon.from_points(pts_tension_pos)

    # A prototype hinge with diagonal elastic stiffness
    Lp = 0.06  # [m] plastic hinge length (toy)
    Ec = 25e9
    EA = Ec*(b*h)
    EI = Ec*(b*h**3/12)
    KN = EA/max(Lp, 1e-9)
    KM = EI/max(Lp, 1e-9)
    hinge = PlasticHingeNM(surface=surf, K=np.diag([KN, KM]))

    # Drive with a cyclic rotation history at (almost) constant axial load
    nsteps = 600
    t = np.linspace(0, 6.0, nsteps)
    theta = 0.02*np.sin(2*np.pi*0.8*t)  # [rad]
    N0 = -0.3*np.max(np.abs(pts_tension_pos[:,0]))  # compressive (negative, since tension +)
    # initialize stress
    hinge.s = np.array([N0, 0.0])
    hinge.q_p = np.zeros(2)

    M = np.zeros_like(theta)
    N = np.zeros_like(theta)

    n_plastic = 0
    max_active = 0

    th_prev = 0.0
    for i in range(1, nsteps):
        dth = theta[i] - theta[i-1]
        # no axial increment (keep N roughly constant)
        out = hinge.update(np.array([0.0, dth]))
        act = out.get('active', np.zeros((0,), int))
        if act.size > 0:
            n_plastic += 1
            max_active = max(max_active, int(act.size))
        N[i] = hinge.s[0]
        M[i] = hinge.s[1]

    # Plot N-M polygon + history (and mirrored M)
    fig, ax = plt.subplots()
    v = surf.vertices
    ax.plot(v[:,0]/1e3, v[:,1]/1e3, "-k", lw=2, label="Yield polygon")
    ax.plot([v[-1,0]/1e3, v[0,0]/1e3], [v[-1,1]/1e3, v[0,1]/1e3], "-k", lw=2)

    lc = colored_line(ax, N/1e3, M/1e3, t, linewidth=2.0)
    colored_line(ax, N/1e3, (-M)/1e3, t, linewidth=1.5)
    cb = fig.colorbar(lc, ax=ax)
    cb.set_label("t [s]")
    ax.set_xlabel("N [kN] (tensión +)")
    ax.set_ylabel("M [kN·m]")
    ax.set_title("N–M (se muestra M y -M)")
    ax.grid(True)
    fig.tight_layout()

    # Plot M-theta with time gradient
    fig, ax = plt.subplots()
    lc2 = colored_line(ax, theta, M/1e3, t, linewidth=2.0)
    cb = fig.colorbar(lc2, ax=ax)
    cb.set_label("t [s]")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("M [kN·m]")
    ax.set_title("Moment–Rotation (color=time)")
    ax.grid(True)
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
