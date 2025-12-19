from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from dinamica_computacional.core.dof import Node


@dataclass
class ModelOptions:
    geometry: str = "linear"


@dataclass
class Model:
    nodes: List[Node]
    elements: List
    hinges: List
    fixed_dofs: np.ndarray
    mass_diag: np.ndarray
    C_diag: np.ndarray
    load_const: np.ndarray
    col_hinge_groups: List[Tuple[int, int, int]] = field(default_factory=list)
    elements_meta: List[Dict[str, object]] = field(default_factory=list)
    options: ModelOptions = field(default_factory=ModelOptions)

    def ndof(self) -> int:
        return int(self.mass_diag.size)

    def free_dofs(self) -> np.ndarray:
        all_dofs = np.arange(self.ndof(), dtype=int)
        mask = np.ones(self.ndof(), dtype=bool)
        mask[self.fixed_dofs] = False
        return all_dofs[mask]

    def reset_state(self) -> None:
        for h in self.hinges:
            h.reset()

    def update_column_yields(self, u_comm: np.ndarray) -> None:
        N_beam = []
        for e in self.elements:
            _, _, _, meta = e.stiffness_and_force_global(u_comm)
            N_beam.append(meta.get("N", 0.0))
        for hinge_idx, beam_idx, sign in self.col_hinge_groups:
            h = self.hinges[hinge_idx]
            Nref = float(sign) * float(N_beam[beam_idx])
            h.set_yield_from_N(Nref)

    def assemble(self, u_trial: np.ndarray, u_comm: np.ndarray):
        nd = self.ndof()
        K = np.zeros((nd, nd))
        R = np.zeros(nd)
        info = {"hinges": []}

        for e in self.elements:
            dofs, k_g, f_g, _ = e.stiffness_and_force_global(u_trial)
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

    def base_shear(self, u: np.ndarray) -> float:
        base_ux = [self.nodes[0].dof_u[0], self.nodes[1].dof_u[0]]
        nd = self.ndof()
        R = np.zeros(nd)
        for e in self.elements:
            dofs, _, f_g, _ = e.stiffness_and_force_global(u)
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
