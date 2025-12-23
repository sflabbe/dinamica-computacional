"""FE model container and assembly routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

from dc_solver.fem.nodes import Node
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.hinges.models import RotSpringElement, HingeNM2DElement, FiberRotSpringElement

# Import JIT assembly kernels (Option 2 optimization)
from dc_solver.kernels.assemble_jit import (
    aggregate_element_stiffness,
    aggregate_hinge_stiffness,
    is_jit_enabled,
)


@dataclass
class Model:
    nodes: List[Node]
    beams: List[FrameElementLinear2D]
    hinges: List[RotSpringElement | FiberRotSpringElement | HingeNM2DElement]
    fixed_dofs: np.ndarray
    mass_diag: np.ndarray
    C_diag: np.ndarray
    load_const: np.ndarray
    col_hinge_groups: List[Tuple[int, int, int]]
    nlgeom: bool = False

    def ndof(self) -> int:
        return int(self.mass_diag.size)

    def free_dofs(self) -> np.ndarray:
        all_dofs = np.arange(self.ndof(), dtype=int)
        mask = np.ones(self.ndof(), dtype=bool)
        mask[self.fixed_dofs] = False
        return all_dofs[mask]

    def reset_state(self) -> None:
        for h in self.hinges:
            if hasattr(h, "reset_state"):
                h.reset_state()

    def update_column_yields(self, u_comm: np.ndarray) -> None:
        """Update My(N) for each column hinge based on the committed axial forces."""
        N_beam = []
        for b in self.beams:
            _, _, _, meta = b.stiffness_and_force_global(u_comm, include_geo=False)
            N_beam.append(meta["N"])
        for hinge_idx, beam_idx, sign in self.col_hinge_groups:
            h = self.hinges[hinge_idx]
            if isinstance(h, RotSpringElement) and h.col_hinge is not None:
                Nref = float(sign) * float(N_beam[beam_idx])
                h.col_hinge.set_yield_from_N(Nref)

    def assemble(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        nd = self.ndof()
        K = np.zeros((nd, nd))
        R = np.zeros(nd)
        info = {"hinges": []}

        # Collect trial axial forces (tension-positive convention) for each frame element.
        # These are used to provide physically meaningful N_target values to fiber beam hinges.
        N_beam_trial: List[float] = []

        for e in self.beams:
            dofs, k_g, f_g, meta = e.stiffness_and_force_global(u_trial, include_geo=self.nlgeom)
            N_beam_trial.append(float(meta.get("N", 0.0)))

            # Use JIT kernel for aggregation if available (Option 2 optimization)
            aggregate_element_stiffness(K, R, k_g, f_g, dofs)


        # Update hinge axial coupling from associated frame element axial force (tension-positive convention).
        # - Fiber hinges want N_target in compression-positive convention -> beam_sign=-1 by default.
        # - SHM beam hinges optionally reduce My under axial compression via N_comp_current.
        for h in self.hinges:
            # Dedicated fiber spring element (preferred)
            if isinstance(h, FiberRotSpringElement) and h.beam_idx is not None:
                bi = int(h.beam_idx)
                if 0 <= bi < len(N_beam_trial):
                    N_tension = float(N_beam_trial[bi])
                    h._beam_N_tension = N_tension
                    h.hinge.N_target = float(h.beam_sign) * N_tension

            # Compat mode: RotSpringElement can also host beam_shm / beam_fiber with beam_idx
            if isinstance(h, RotSpringElement) and h.beam_idx is not None:
                bi = int(h.beam_idx)
                if 0 <= bi < len(N_beam_trial):
                    N_tension = float(N_beam_trial[bi])
                    h._beam_N_tension = N_tension
                    kind = str(getattr(h, "kind", "")).lower().strip()

                    # beam_shm: update compression-positive axial force for My(N)
                    if kind == "beam_shm" and getattr(h, "beam_hinge", None) is not None:
                        N_comp = max(0.0, float(getattr(h, "beam_sign", -1.0)) * float(N_tension))
                        setattr(h.beam_hinge, "N_comp_current", float(N_comp))

                    # beam_fiber compat: update fiber N_target if present
                    if kind in ("beam_fiber", "fiber"):
                        fh = getattr(h, "fiber_hinge", None)
                        if fh is None:
                            fh = getattr(h, "beam_hinge", None)
                        if fh is not None and hasattr(fh, "N_target"):
                            setattr(fh, "N_target", float(getattr(h, "beam_sign", -1.0)) * float(N_tension))

        for h in self.hinges:
            k_l, f_l, inf = h.eval_trial(u_trial, u_comm)
            dofs = h.dofs()

            # Use JIT kernel for aggregation if available (Option 2 optimization)
            aggregate_hinge_stiffness(K, R, k_l, f_l, dofs)

            info["hinges"].append(inf)

        fd = self.free_dofs()
        return K[np.ix_(fd, fd)], R[fd], info

    def commit(self) -> None:
        for h in self.hinges:
            h.commit()

    def base_shear(self, u: np.ndarray, base_nodes: Tuple[int, int]) -> float:
        base_ux = [self.nodes[base_nodes[0]].dof_u[0], self.nodes[base_nodes[1]].dof_u[0]]
        nd = self.ndof()
        R = np.zeros(nd)
        for e in self.beams:
            # Keep base reactions consistent with the model's NLGEOM setting.
            dofs, _, f_g, _ = e.stiffness_and_force_global(u, include_geo=bool(self.nlgeom))
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
        for h in self.hinges:
            _, f_l, _ = h.eval_trial(u, u)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
        Vb = 0.0
        for d in base_ux:
            Vb += -R[d]
        return float(Vb)

    def internal_force(self, u: np.ndarray) -> np.ndarray:
        nd = self.ndof()
        R = np.zeros(nd)
        for e in self.beams:
            dofs, _, f_g, _ = e.stiffness_and_force_global(u, include_geo=bool(self.nlgeom))
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
        for h in self.hinges:
            _, f_l, _ = h.eval_trial(u, u)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
        return R
