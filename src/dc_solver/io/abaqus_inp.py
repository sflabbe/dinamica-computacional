"""Minimal Abaqus-like .inp parser for 2D frame problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Iterable

import numpy as np

from dc_solver.fem.nodes import Node, DofManager
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model


@dataclass
class PartData:
    name: str
    nodes: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    elements: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    elsets: Dict[str, List[int]] = field(default_factory=dict)
    nsets: Dict[str, List[int]] = field(default_factory=dict)
    elset_generate: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    nset_generate: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    beam_section: Dict[str, Tuple[str, float, float, str]] = field(default_factory=dict)


@dataclass
class MaterialData:
    name: str
    E: float = 0.0
    nu: float = 0.0
    density: float = 0.0


@dataclass
class StepData:
    name: str
    kind: str
    nlgeom: bool = False
    time_period: float = 0.0
    dt: float = 0.0
    gravity: Optional[Tuple[float, float]] = None
    accel_bc: Optional[Tuple[str, int, float, Optional[str]]] = None
    cloads: List[Tuple[str, int, float]] = field(default_factory=list)


@dataclass
class ModelData:
    part: PartData
    material: MaterialData
    assembly_translation: Tuple[float, float] = (0.0, 0.0)
    steps: List[StepData] = field(default_factory=list)
    amplitudes: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    boundaries: List[Tuple[str, int, int, float]] = field(default_factory=list)


def _parse_keyword(line: str) -> Tuple[str, Dict[str, str]]:
    raw = line.strip().lstrip("*")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    key = parts[0].upper()
    opts = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().lower()] = v.strip()
    return key, opts


def _expand_includes(path: str, warning_cb: Optional[Callable[[str], None]] = None) -> List[Tuple[str, int, str]]:
    base = Path(path)
    if not base.exists():
        raise ValueError(f"Input file not found: {path}")
    seen = set()

    def _read_file(file_path: Path) -> List[Tuple[str, int, str]]:
        if file_path in seen:
            msg = f"[inp] Warning: include cycle detected for '{file_path}'."
            if warning_cb is not None:
                warning_cb(msg)
            return []
        seen.add(file_path)
        lines: List[Tuple[str, int, str]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                if line.lstrip().startswith("*"):
                    key, opts = _parse_keyword(line)
                    if key == "INCLUDE":
                        include_name = opts.get("input") or opts.get("file")
                        if not include_name:
                            raise ValueError(f"{file_path}:{line_no}: *Include missing input= filename.")
                        include_path = (file_path.parent / include_name).resolve()
                        lines.extend(_read_file(include_path))
                        continue
                lines.append((line, line_no, str(file_path)))
        return lines

    return _read_file(base.resolve())


def _as_int(value: str, source: str, line_no: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_no}: invalid integer '{value}'.") from exc


def _as_float(value: str, source: str, line_no: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_no}: invalid float '{value}'.") from exc


def parse_inp(path: str, warning_cb: Optional[Callable[[str], None]] = None) -> ModelData:
    part = None
    material = None
    steps: List[StepData] = []
    amplitudes: Dict[str, List[Tuple[float, float]]] = {}
    boundaries: List[Tuple[str, int, int, float]] = []
    assembly_translation = (0.0, 0.0)

    current_keyword = None
    current_opts = {}
    current_part = None
    current_material = None
    current_step: Optional[StepData] = None
    current_amp: Optional[str] = None

    supported = {
        "PART", "NODE", "ELEMENT", "NSET", "ELSET", "BEAM SECTION", "MATERIAL",
        "ELASTIC", "DENSITY", "ASSEMBLY", "INSTANCE", "END INSTANCE", "END ASSEMBLY",
        "STEP", "STATIC", "DYNAMIC", "DLOAD", "CLOAD", "BOUNDARY", "AMPLITUDE", "END STEP",
        "INCLUDE", "END PART", "OUTPUT", "NODE OUTPUT", "ELEMENT OUTPUT",
    }
    for raw, line_no, source in _expand_includes(path, warning_cb=warning_cb):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            key, opts = _parse_keyword(line)
            if key not in supported:
                msg = f"[inp] Warning: unsupported keyword '*{key}' ignored."
                if warning_cb is not None:
                    warning_cb(msg)
                else:
                    print(msg)
                current_keyword = None
                continue
            current_keyword = key
            current_opts = opts
            current_amp = None
            if key == "PART":
                current_part = PartData(name=opts.get("name", "PART-1"))
                part = current_part
            elif key == "MATERIAL":
                current_material = MaterialData(name=opts.get("name", "MAT-1"))
                material = current_material
            elif key == "STEP":
                current_step = StepData(
                    name=opts.get("name", "STEP-1"),
                    kind="STATIC",
                    nlgeom=opts.get("nlgeom", "NO").upper() == "YES",
                )
                steps.append(current_step)
            elif key == "AMPLITUDE":
                current_amp = opts.get("name", "AMP-1")
                amplitudes[current_amp] = []
            continue

        if current_keyword == "NODE":
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) < 3:
                raise ValueError(f"{source}:{line_no}: *Node requires id, x, y.")
            nid = _as_int(fields[0], source, line_no)
            x = _as_float(fields[1], source, line_no)
            y = _as_float(fields[2], source, line_no)
            current_part.nodes[nid] = (x, y)
        elif current_keyword == "ELEMENT":
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) < 3:
                raise ValueError(f"{source}:{line_no}: *Element requires id, n1, n2.")
            eid = _as_int(fields[0], source, line_no)
            n1 = _as_int(fields[1], source, line_no)
            n2 = _as_int(fields[2], source, line_no)
            current_part.elements[eid] = (n1, n2)
            elset = current_opts.get("elset", "")
            if elset:
                current_part.elsets.setdefault(elset, []).append(eid)
        elif current_keyword in ("NSET", "ELSET"):
            target = current_part.nsets if current_keyword == "NSET" else current_part.elsets
            name = current_opts.get("nset" if current_keyword == "NSET" else "elset", "")
            if "generate" in current_opts:
                values = [v.strip() for v in line.split(",") if v.strip()]
                if len(values) >= 3:
                    start = _as_int(values[0], source, line_no)
                    end = _as_int(values[1], source, line_no)
                    step = _as_int(values[2], source, line_no)
                    if current_keyword == "NSET":
                        current_part.nset_generate[name] = (start, end, step)
                    else:
                        current_part.elset_generate[name] = (start, end, step)
            else:
                ids = [_as_int(v.strip(), source, line_no) for v in line.split(",") if v.strip()]
                target.setdefault(name, []).extend(ids)
        elif current_keyword == "BEAM SECTION":
            elset = current_opts.get("elset", "")
            mat = current_opts.get("material", "")
            section = current_opts.get("section", "")
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) < 2:
                raise ValueError(f"{source}:{line_no}: *Beam Section requires b, h.")
            b = _as_float(fields[0], source, line_no)
            h = _as_float(fields[1], source, line_no)
            current_part.beam_section[elset] = (section, b, h, mat)
        elif current_keyword == "ELASTIC":
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if not fields:
                raise ValueError(f"{source}:{line_no}: *Elastic requires at least E.")
            current_material.E = _as_float(fields[0], source, line_no)
            current_material.nu = _as_float(fields[1], source, line_no) if len(fields) > 1 else 0.0
        elif current_keyword == "DENSITY":
            current_material.density = _as_float(line.split(",")[0].strip(), source, line_no)
        elif current_keyword == "INSTANCE":
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) >= 2:
                assembly_translation = (
                    _as_float(fields[0], source, line_no),
                    _as_float(fields[1], source, line_no),
                )
        elif current_keyword == "BOUNDARY":
            if current_opts.get("type", "").upper() == "ACCELERATION":
                fields = [f.strip() for f in line.split(",") if f.strip()]
                if len(fields) >= 3 and current_step is not None:
                    set_name = fields[0]
                    dof1 = _as_int(fields[1], source, line_no)
                    dof2 = _as_int(fields[2], source, line_no)
                    value = _as_float(fields[3], source, line_no) if len(fields) > 3 else 1.0
                    amp_name = current_opts.get("amplitude", None)
                    current_step.accel_bc = (set_name, dof1, value, amp_name)
            else:
                fields = [f.strip() for f in line.split(",") if f.strip()]
                if len(fields) >= 3:
                    set_name = fields[0]
                    dof1 = _as_int(fields[1], source, line_no)
                    dof2 = _as_int(fields[2], source, line_no)
                    value = _as_float(fields[3], source, line_no) if len(fields) > 3 else 0.0
                    boundaries.append((set_name, dof1, dof2, value))
        elif current_keyword == "STATIC":
            if current_step is not None:
                current_step.kind = "STATIC"
        elif current_keyword == "DYNAMIC":
            if current_step is not None:
                current_step.kind = "DYNAMIC"
                fields = [f.strip() for f in line.split(",") if f.strip()]
                if len(fields) >= 2:
                    current_step.dt = _as_float(fields[0], source, line_no)
                    current_step.time_period = _as_float(fields[1], source, line_no)
        elif current_keyword == "DLOAD":
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) >= 5 and fields[1].upper() == "GRAV":
                g = _as_float(fields[2], source, line_no)
                gx = _as_float(fields[3], source, line_no) * g
                gy = _as_float(fields[4], source, line_no) * g
                if current_step is not None:
                    current_step.gravity = (gx, gy)
        elif current_keyword == "CLOAD":
            if current_step is None:
                continue
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) >= 3:
                target = fields[0]
                dof = _as_int(fields[1], source, line_no)
                value = _as_float(fields[2], source, line_no)
                current_step.cloads.append((target, dof, value))
        elif current_keyword == "AMPLITUDE" and current_amp:
            fields = [f.strip() for f in line.split(",") if f.strip()]
            if len(fields) >= 2:
                amplitudes[current_amp].append(
                    (_as_float(fields[0], source, line_no), _as_float(fields[1], source, line_no))
                )
        else:
            continue

    if part is None or material is None:
        raise ValueError("Missing *Part or *Material section.")

    return ModelData(
        part=part,
        material=material,
        assembly_translation=assembly_translation,
        steps=steps,
        amplitudes=amplitudes,
        boundaries=boundaries,
    )


def _expand_generate(start: int, end: int, step: int) -> List[int]:
    return list(range(start, end + 1, step))


def _node_map_from_part(data: ModelData) -> Dict[int, int]:
    return {nid: idx for idx, (nid, _) in enumerate(sorted(data.part.nodes.items()))}


def _collect_nsets(data: ModelData) -> Dict[str, List[int]]:
    nsets = {name: ids[:] for name, ids in data.part.nsets.items()}
    for name, gen in data.part.nset_generate.items():
        nsets[name] = _expand_generate(*gen)
    return nsets


def build_model(data: ModelData, nlgeom: bool = False) -> Model:
    dm = DofManager()
    node_map: Dict[int, int] = {}
    nodes: List[Node] = []

    tx, ty = data.assembly_translation
    for nid, (x, y) in sorted(data.part.nodes.items()):
        node_map[nid] = len(nodes)
        nodes.append(Node(x + tx, y + ty, dm.new_trans(), dm.new_rot()))

    beams: List[FrameElementLinear2D] = []
    elsets = {name: ids[:] for name, ids in data.part.elsets.items()}
    for name, gen in data.part.elset_generate.items():
        elsets[name] = _expand_generate(*gen)

    for eid, (n1, n2) in sorted(data.part.elements.items()):
        n1i = node_map[n1]
        n2i = node_map[n2]
        elset = None
        for name, ids in elsets.items():
            if eid in ids:
                elset = name
                break
        section = data.part.beam_section.get(elset, None)
        if section is None:
            raise ValueError(f"Missing *Beam Section for elset {elset}")
        _, b, h, _ = section
        A = b * h
        I = b * (h ** 3) / 12.0
        beams.append(FrameElementLinear2D(n1i, n2i, E=data.material.E, A=A, I=I, nodes=nodes))

    nd = dm.ndof
    mass = np.zeros(nd)
    C = np.zeros(nd)
    load = np.zeros(nd)

    rho = data.material.density
    for e in beams:
        L, _, _ = e._geom()
        m = rho * e.A * L
        dofs = e.dofs()
        mass[dofs[0]] += 0.5 * m
        mass[dofs[3]] += 0.5 * m

    fixed_dofs: List[int] = []
    nsets = _collect_nsets(data)
    for name, dof1, dof2, value in data.boundaries:
        node_ids = nsets.get(name, [])
        for nid in node_ids:
            idx = node_map.get(nid, None)
            if idx is None:
                continue
            dofs = nodes[idx].dof_u + (nodes[idx].dof_th,)
            for dof in range(dof1, dof2 + 1):
                if dof == 1:
                    fixed_dofs.append(dofs[0])
                elif dof == 2:
                    fixed_dofs.append(dofs[1])
                elif dof == 3:
                    fixed_dofs.append(dofs[2])

    model = Model(
        nodes=nodes,
        beams=beams,
        hinges=[],
        fixed_dofs=np.array(sorted(set(fixed_dofs)), dtype=int),
        mass_diag=mass,
        C_diag=C,
        load_const=load,
        col_hinge_groups=[],
        nlgeom=nlgeom,
    )
    return model


def apply_cloads(model: Model, data: ModelData, step: StepData) -> None:
    nsets = _collect_nsets(data)
    node_map = _node_map_from_part(data)
    for target, dof, value in step.cloads:
        if target in nsets:
            node_ids = nsets[target]
        else:
            try:
                node_ids = [int(target)]
            except ValueError as exc:
                raise ValueError(f"Unknown node or nset '{target}' in *CLOAD.") from exc
        for nid in node_ids:
            idx = node_map.get(nid, None)
            if idx is None:
                continue
            node = model.nodes[idx]
            dofs = node.dof_u + (node.dof_th,)
            if dof == 1:
                model.load_const[dofs[0]] += value
            elif dof == 2:
                model.load_const[dofs[1]] += value
            elif dof == 3:
                model.load_const[dofs[2]] += value
            else:
                raise ValueError(f"Unsupported DOF {dof} in *CLOAD.")


def apply_gravity(model: Model, data: ModelData, gravity: Tuple[float, float]) -> None:
    gx, gy = gravity
    for e in model.beams:
        w = (e.A * data.material.density * gx, e.A * data.material.density * gy)
        f_g = e.equiv_nodal_load_global(w)
        dofs = e.dofs()
        for a, ia in enumerate(dofs):
            model.load_const[ia] += f_g[a]


def amplitude_series(amplitude: List[Tuple[float, float]], dt: float, t_end: float) -> np.ndarray:
    if not amplitude:
        return np.zeros(int(t_end / dt) + 1)
    amp = np.array(amplitude, dtype=float)
    t = np.arange(0.0, t_end + 1e-12, dt)
    return np.interp(t, amp[:, 0], amp[:, 1])
