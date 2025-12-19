from __future__ import annotations

import numpy as np

from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model
from dc_solver.fem.nodes import DofManager, Node


def build_nodes(coords: list[tuple[float, float]]) -> tuple[list[Node], DofManager]:
    dm = DofManager()
    nodes = [Node(x, y, dm.new_trans(), dm.new_rot()) for x, y in coords]
    return nodes, dm


def assemble_full(model: Model, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nd = model.ndof()
    K = np.zeros((nd, nd))
    R = np.zeros(nd)
    for e in model.beams:
        dofs, k_g, f_g, _ = e.stiffness_and_force_global(u, include_geo=model.nlgeom)
        for a, ia in enumerate(dofs):
            R[ia] += f_g[a]
            for b, ib in enumerate(dofs):
                K[ia, ib] += k_g[a, b]
    for h in model.hinges:
        k_l, f_l, _ = h.eval_trial(u, u)
        dofs = h.dofs()
        for a, ia in enumerate(dofs):
            R[ia] += f_l[a]
            for b, ib in enumerate(dofs):
                K[ia, ib] += k_l[a, b]
    return K, R


def build_two_node_beam(L: float = 2.0,
                        E: float = 210e9,
                        A: float = 0.01,
                        I: float = 1e-4) -> tuple[Model, FrameElementLinear2D]:
    nodes, dm = build_nodes([(0.0, 0.0), (L, 0.0)])
    element = FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)
    ndof = dm.ndof
    model = Model(
        nodes=nodes,
        beams=[element],
        hinges=[],
        fixed_dofs=np.array([], dtype=int),
        mass_diag=np.zeros(ndof),
        C_diag=np.zeros(ndof),
        load_const=np.zeros(ndof),
        col_hinge_groups=[],
        nlgeom=False,
    )
    return model, element


def build_cantilever_model(L: float = 3.0,
                           E: float = 210e9,
                           A: float = 0.01,
                           I: float = 1e-4) -> Model:
    nodes, dm = build_nodes([(0.0, 0.0), (0.0, L)])
    element = FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)
    fixed = np.array([*nodes[0].dof_u, nodes[0].dof_th], dtype=int)
    ndof = dm.ndof
    return Model(
        nodes=nodes,
        beams=[element],
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=np.zeros(ndof),
        C_diag=np.zeros(ndof),
        load_const=np.zeros(ndof),
        col_hinge_groups=[],
        nlgeom=False,
    )


def build_portal_frame_model(L: float = 4.0,
                             H: float = 3.0,
                             E: float = 210e9,
                             A: float = 0.02,
                             I: float = 4e-4) -> Model:
    nodes, dm = build_nodes([(0.0, 0.0), (L, 0.0), (0.0, H), (L, H)])
    elements = [
        FrameElementLinear2D(0, 2, E=E, A=A, I=I, nodes=nodes),
        FrameElementLinear2D(1, 3, E=E, A=A, I=I, nodes=nodes),
        FrameElementLinear2D(2, 3, E=E, A=A, I=I, nodes=nodes),
    ]
    fixed = np.array([
        *nodes[0].dof_u, nodes[0].dof_th,
        *nodes[1].dof_u, nodes[1].dof_th,
    ], dtype=int)
    ndof = dm.ndof
    return Model(
        nodes=nodes,
        beams=elements,
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=np.zeros(ndof),
        C_diag=np.zeros(ndof),
        load_const=np.zeros(ndof),
        col_hinge_groups=[],
        nlgeom=False,
    )


def build_sdof_column_model(L: float = 3.0,
                            E: float = 210e9,
                            A: float = 0.02,
                            I: float = 4e-4,
                            mass: float = 1.0,
                            damping_ratio: float = 0.0) -> Model:
    nodes, dm = build_nodes([(0.0, 0.0), (0.0, L)])
    elements = [FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)]
    fixed = np.array([
        *nodes[0].dof_u, nodes[0].dof_th,
        nodes[1].dof_u[1], nodes[1].dof_th,
    ], dtype=int)
    ndof = dm.ndof
    mass_diag = np.zeros(ndof)
    mass_diag[nodes[1].dof_u[0]] = mass
    C_diag = np.zeros(ndof)
    if damping_ratio > 0.0:
        omega = np.sqrt(E * A / L / mass)
        C_diag[nodes[1].dof_u[0]] = 2.0 * damping_ratio * omega * mass
    return Model(
        nodes=nodes,
        beams=elements,
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=mass_diag,
        C_diag=C_diag,
        load_const=np.zeros(ndof),
        col_hinge_groups=[],
        nlgeom=False,
    )
