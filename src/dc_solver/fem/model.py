"""FE model container and assembly routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

from dc_solver.fem.nodes import Node
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.hinges.models import RotSpringElement, HingeNM2DElement, FiberRotSpringElement


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

        for e in self.beams:
            dofs, k_g, f_g, _ = e.stiffness_and_force_global(u_trial, include_geo=self.nlgeom)
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_g[a, b]

        for h in self.hinges:
            k_l, f_l, inf = h.eval_trial(u_trial, u_comm)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_l[a, b]
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
            dofs, _, f_g, _ = e.stiffness_and_force_global(u, include_geo=False)
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
            dofs, _, f_g, _ = e.stiffness_and_force_global(u, include_geo=False)
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
        for h in self.hinges:
            _, f_l, _ = h.eval_trial(u, u)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
        return R
