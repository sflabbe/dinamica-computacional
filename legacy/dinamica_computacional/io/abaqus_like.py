from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from dinamica_computacional.core.dof import DofManager, Node
from dinamica_computacional.core.model import Model, ModelOptions
from dinamica_computacional.core.analysis import AnalysisPlan, StaticStep, HHTStep
from dinamica_computacional.elements.beam2d import Beam2D
from dinamica_computacional.elements.hinge_nm import ColumnHingeNMRot, RotSpringElementNM
from dinamica_computacional.elements.hinge_mtheta import SHMBeamHinge1D, RotSpringElementMTheta
from dinamica_computacional.materials.nm_surface import NMSurfacePolygon


@dataclass
class SectionDef:
    elset: str
    material: str
    A: float
    I: float


def _parse_header(line: str) -> Tuple[str, Dict[str, str]]:
    parts = [p.strip() for p in line.strip().split(",")]
    key = parts[0].upper()
    opts = {}
    for p in parts[1:]:
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().upper()] = v.strip()
        else:
            opts[p.strip().upper()] = ""
    return key, opts


def _resolve_nodeset(target: str, nsets: Dict[str, List[int]]) -> List[int]:
    if target in nsets:
        return nsets[target]
    try:
        return [int(target)]
    except ValueError as exc:
        available = ", ".join(sorted(nsets)) or "none"
        raise ValueError(f"Unknown NSET '{target}'. Available: {available}.") from exc


def read_inp(path: str) -> Tuple[Model, AnalysisPlan]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("**")]

    nodes_raw: Dict[int, Tuple[float, float]] = {}
    elements_raw: List[Dict[str, object]] = []
    nsets: Dict[str, List[int]] = {}
    elsets: Dict[str, List[int]] = {}
    materials: Dict[str, Dict[str, float]] = {}
    sections: Dict[str, SectionDef] = {}
    hinge_nm_params: Dict[str, Dict[str, float]] = {}
    hinge_mtheta_params: Dict[str, Dict[str, float]] = {}
    nm_surfaces: Dict[str, NMSurfacePolygon] = {}
    mass_defs: List[Tuple[str, float]] = []
    damping_def: Dict[str, float] = {}

    steps: List[Dict] = []
    current_step: Dict | None = None

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("*"):
            key, opts = _parse_header(line)
            if key == "*NODE":
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("*"):
                    nid, x, y = [p.strip() for p in lines[idx].split(",")[:3]]
                    nodes_raw[int(nid)] = (float(x), float(y))
                    idx += 1
                continue
            if key == "*ELEMENT":
                etype = opts.get("TYPE", "").upper()
                elset = opts.get("ELSET", "").upper()
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",")]
                    eid = int(parts[0])
                    ni = int(parts[1]); nj = int(parts[2])
                    elements_raw.append({"eid": eid, "type": etype, "ni": ni, "nj": nj, "elset": elset})
                    elsets.setdefault(elset, []).append(eid)
                    idx += 1
                continue
            if key == "*NSET":
                name = opts.get("NSET", "").upper()
                idx += 1
                nodes = []
                while idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    nodes.extend([int(p) for p in parts])
                    idx += 1
                nsets[name] = nodes
                continue
            if key == "*ELSET":
                name = opts.get("ELSET", "").upper()
                idx += 1
                elems = []
                while idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    elems.extend([int(p) for p in parts])
                    idx += 1
                elsets[name] = elems
                continue
            if key == "*MATERIAL":
                name = opts.get("NAME", "").upper()
                materials[name] = {}
                idx += 1
                continue
            if key == "*ELASTIC":
                idx += 1
                if idx < len(lines):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    if parts:
                        E = float(parts[0])
                        nu = float(parts[1]) if len(parts) > 1 else 0.3
                        materials[list(materials.keys())[-1]]["E"] = E
                        materials[list(materials.keys())[-1]]["nu"] = nu
                    idx += 1
                continue
            if key == "*SECTION":
                elset = opts.get("ELSET", "").upper()
                mat = opts.get("MATERIAL", "").upper()
                A = float(opts.get("A", "0"))
                I = float(opts.get("I", "0"))
                sections[elset] = SectionDef(elset=elset, material=mat, A=A, I=I)
                idx += 1
                continue
            if key == "*NMSURFACE":
                name = opts.get("NAME", "").upper()
                idx += 1
                verts = []
                while idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    if len(parts) >= 2:
                        verts.append((float(parts[0]), float(parts[1])))
                    idx += 1
                nm_surfaces[name] = NMSurfacePolygon(vertices=verts)
                continue
            if key == "*HINGE_NM_PARAMS":
                name = opts.get("ELSET", "").upper()
                idx += 1
                if idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    params = {
                        "K0": float(parts[0]),
                        "ALPHA_POST": float(parts[1]) if len(parts) > 1 else 1e-4,
                        "SURFACE": opts.get("SURFACE", "").upper(),
                    }
                    hinge_nm_params[name] = params
                    idx += 1
                continue
            if key == "*HINGE_MTHETA_SHM_PARAMS":
                name = opts.get("ELSET", "").upper()
                idx += 1
                if idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    params = {
                        "K0": float(parts[0]),
                        "MY": float(parts[1]),
                        "ALPHA_POST": float(parts[2]) if len(parts) > 2 else 0.02,
                        "CK": float(parts[3]) if len(parts) > 3 else 2.0,
                        "CMY": float(parts[4]) if len(parts) > 4 else 1.0,
                    }
                    hinge_mtheta_params[name] = params
                    idx += 1
                continue
            if key == "*STEP":
                current_step = {"name": opts.get("NAME", "STEP"), "nlgeom": opts.get("NLGEOM", "NO").upper()}
                steps.append(current_step)
                idx += 1
                continue
            if key == "*STATIC":
                idx += 1
                if idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    current_step["static"] = {
                        "tol": float(parts[2]) if len(parts) > 2 else 1e-10,
                        "max_iter": int(parts[3]) if len(parts) > 3 else 60,
                    }
                    idx += 1
                continue
            if key == "*HHT":
                idx += 1
                if idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    current_step["hht"] = {
                        "alpha": float(opts.get("ALPHA", "-0.05")),
                        "t_start": float(parts[0]),
                        "t_end": float(parts[1]),
                        "dt": float(parts[2]),
                    }
                    idx += 1
                continue
            if key == "*CLOAD":
                idx += 1
                cloads = current_step.setdefault("cloads", [])
                while idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    target = parts[0].upper()
                    dof = int(parts[1])
                    val = float(parts[2])
                    cloads.append((target, dof, val))
                    idx += 1
                continue
            if key == "*BOUNDARY":
                idx += 1
                bcs = []
                while idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    target = parts[0].upper()
                    dof1 = int(parts[1])
                    dof2 = int(parts[2])
                    bcs.append((target, dof1, dof2))
                    idx += 1
                current_step.setdefault("boundary", []).extend(bcs)
                continue
            if key == "*MASS":
                target = opts.get("NSET", "").upper()
                idx += 1
                if idx < len(lines) and not lines[idx].startswith("*"):
                    val = float(lines[idx].split(",")[0])
                    mass_defs.append((target, val))
                    idx += 1
                continue
            if key == "*DAMPING":
                damping_def["zeta"] = float(opts.get("ZETA", "0.02"))
                damping_def["T0"] = float(opts.get("T0", "0.5"))
                idx += 1
                continue
            if key == "*BASEACCEL":
                idx += 1
                if idx < len(lines) and not lines[idx].startswith("*"):
                    parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                    current_step["baseaccel"] = {"dir": parts[0].upper(), "expr": parts[1].strip('"')}
                    idx += 1
                continue
            if key == "*END STEP":
                current_step = None
                idx += 1
                continue
        idx += 1

    # Build nodes
    dm = DofManager()
    node_ids = sorted(nodes_raw.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    nodes = []
    for nid in node_ids:
        x, y = nodes_raw[nid]
        nodes.append(Node(x, y, dm.new_trans(), dm.new_rot()))

    # Build elements
    elements = []
    elements_meta = []
    elset_elem_indices: Dict[str, List[int]] = {}
    for elem in elements_raw:
        etype = elem["type"].upper()
        if etype == "BEAM2D":
            elset = elem["elset"].upper()
            sec = sections[elset]
            E = materials[sec.material]["E"]
            ni = node_index[int(elem["ni"])]; nj = node_index[int(elem["nj"])]
            elements.append(Beam2D(ni, nj, E=E, A=sec.A, I=sec.I, nodes=nodes))
            elements_meta.append({"eid": elem["eid"], "type": "BEAM2D", "ni": ni, "nj": nj, "prop": elset})
            elset_elem_indices.setdefault(elset, []).append(len(elements) - 1)

    hinges = []
    hinge_elsets: List[str] = []
    for elem in elements_raw:
        etype = elem["type"].upper()
        if etype == "HINGE_NM":
            elset = elem["elset"].upper()
            params = hinge_nm_params[elset]
            surface = nm_surfaces[params["SURFACE"]]
            ni = node_index[int(elem["ni"])]; nj = node_index[int(elem["nj"])]
            hinge = ColumnHingeNMRot(surface=surface, k0=params["K0"], alpha_post=params["ALPHA_POST"])
            hinges.append(RotSpringElementNM(ni, nj, hinge, nodes))
            hinge_elsets.append(elset)
        if etype == "HINGE_MTHETA":
            elset = elem["elset"].upper()
            params = hinge_mtheta_params[elset]
            ni = node_index[int(elem["ni"])]; nj = node_index[int(elem["nj"])]
            hinge = SHMBeamHinge1D(
                K0_0=params["K0"],
                My_0=params["MY"],
                alpha_post=params["ALPHA_POST"],
                cK=params["CK"],
                cMy=params["CMY"],
            )
            hinges.append(RotSpringElementMTheta(ni, nj, hinge, nodes))
            hinge_elsets.append(elset)

    fixed_dofs = []
    if steps and steps[0].get("boundary"):
        for target, dof1, dof2 in steps[0]["boundary"]:
            node_list = _resolve_nodeset(target, nsets)
            for nid in node_list:
                ni = node_index[nid]
                nd = nodes[ni]
                for dof in range(dof1, dof2 + 1):
                    if dof == 1:
                        fixed_dofs.append(nd.dof_u[0])
                    elif dof == 2:
                        fixed_dofs.append(nd.dof_u[1])
                    elif dof == 3:
                        fixed_dofs.append(nd.dof_th)

    fixed = np.array(sorted(set(fixed_dofs)), dtype=int)
    ndof = dm.ndof
    mass = np.zeros(ndof)
    C = np.zeros(ndof)
    p0 = np.zeros(ndof)

    model = Model(
        nodes=nodes,
        elements=elements,
        hinges=hinges,
        fixed_dofs=fixed,
        mass_diag=mass,
        C_diag=C,
        load_const=p0,
        elements_meta=elements_meta,
        options=ModelOptions(),
    )

    col_hinge_groups = []
    for idx, elset in enumerate(hinge_elsets):
        if "COL" not in elset:
            continue
        elset_upper = elset.upper()
        if "BASE_L" in elset_upper:
            beam_elset = "COL_L"
            beam_idx = elset_elem_indices[beam_elset][0]
        elif "TOP_L" in elset_upper:
            beam_elset = "COL_L"
            beam_idx = elset_elem_indices[beam_elset][-1]
        elif "BASE_R" in elset_upper:
            beam_elset = "COL_R"
            beam_idx = elset_elem_indices[beam_elset][0]
        elif "TOP_R" in elset_upper:
            beam_elset = "COL_R"
            beam_idx = elset_elem_indices[beam_elset][-1]
        else:
            continue
        col_hinge_groups.append((idx, beam_idx, +1))
    model.col_hinge_groups = col_hinge_groups

    def story_stiffness_linear() -> float:
        nd = model.ndof()
        fd = model.free_dofs()
        u_comm = np.zeros(nd)
        u_trial = np.zeros(nd)
        for e in model.elements:
            e.geometry = "linear"
        model.update_column_yields(u_comm)
        K, _, _ = model.assemble(u_trial, u_comm)
        f = np.zeros(nd)
        ux2 = model.nodes[2].dof_u[0]
        ux3 = model.nodes[3].dof_u[0]
        f[ux2] = 0.5
        f[ux3] = 0.5
        f_free = f[fd]
        u_free = np.linalg.solve(K + 1e-14 * np.eye(fd.size), f_free)
        u = np.zeros(nd); u[fd] = u_free
        u_top = 0.5 * (u[ux2] + u[ux3])
        return float(1.0 / u_top)

    if mass_defs:
        for target, val in mass_defs:
            node_list = _resolve_nodeset(target, nsets)
            for nid in node_list:
                ni = node_index[nid]
                nd = nodes[ni]
                model.mass_diag[nd.dof_u[0]] += val
    else:
        K_story = story_stiffness_linear()
        T0 = float(damping_def.get("T0", 0.5))
        omega0 = 2.0 * np.pi / T0
        M_total = K_story / (omega0 ** 2)
        model.mass_diag[model.nodes[2].dof_u[0]] = 0.5 * M_total
        model.mass_diag[model.nodes[3].dof_u[0]] = 0.5 * M_total

    zeta = float(damping_def.get("zeta", 0.02))
    T0 = float(damping_def.get("T0", 0.5))
    omega0 = 2.0 * np.pi / T0
    model.C_diag[:] = 2.0 * zeta * omega0 * model.mass_diag

    plan = AnalysisPlan()
    for step in steps:
        load = np.zeros(ndof)
        for target, dof, val in step.get("cloads", []):
            node_list = _resolve_nodeset(target, nsets)
            for nid in node_list:
                ni = node_index[nid]
                nd = nodes[ni]
                if dof == 1:
                    load[nd.dof_u[0]] += val
                elif dof == 2:
                    load[nd.dof_u[1]] += val
                elif dof == 3:
                    load[nd.dof_th] += val

        geometry = "corotational" if step.get("nlgeom", "NO") == "YES" else "linear"
        if "static" in step:
            plan.steps.append(StaticStep(name=step["name"], load_const=load,
                                         geometry=geometry,
                                         max_iter=step["static"]["max_iter"],
                                         tol=step["static"]["tol"]))
        if "hht" in step:
            base_expr = step.get("baseaccel", {}).get("expr", "0.0")
            plan.steps.append(HHTStep(name=step["name"], load_const=load,
                                      geometry=geometry,
                                      t_end=step["hht"]["t_end"], dt=step["hht"]["dt"],
                                      alpha=step["hht"]["alpha"],
                                      base_accel_expr=base_expr))

    return model, plan
