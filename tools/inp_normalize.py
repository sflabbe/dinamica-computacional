"""Best-effort normalizer for Abaqus-like input decks."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class InpBlock:
    keyword: str
    header: str
    lines: List[str]


def _parse_keyword(line: str) -> Tuple[str, Dict[str, str]]:
    raw = line.strip().lstrip("*")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    key = parts[0].upper()
    opts: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().lower()] = v.strip()
    return key, opts


def _split_blocks(lines: Sequence[str]) -> List[InpBlock]:
    blocks: List[InpBlock] = []
    current: Optional[InpBlock] = None
    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.lstrip().startswith("**"):
            continue
        if line.lstrip().startswith("*"):
            key, _ = _parse_keyword(line)
            if current is not None:
                blocks.append(current)
            current = InpBlock(keyword=key, header=line, lines=[])
        else:
            if current is None:
                current = InpBlock(keyword="TEXT", header="", lines=[])
            current.lines.append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def _collect_defined_sets(blocks: Sequence[InpBlock]) -> Tuple[set[str], set[str]]:
    nsets: set[str] = set()
    elsets: set[str] = set()
    for block in blocks:
        if block.keyword in {"NSET", "ELSET"}:
            _, opts = _parse_keyword(block.header)
            name = opts.get("nset") if block.keyword == "NSET" else opts.get("elset")
            if name:
                (nsets if block.keyword == "NSET" else elsets).add(name)
        if block.keyword == "ELEMENT":
            _, opts = _parse_keyword(block.header)
            elset = opts.get("elset")
            if elset:
                elsets.add(elset)
    return nsets, elsets


def _collect_referenced_sets(blocks: Sequence[InpBlock]) -> Tuple[set[str], set[str]]:
    nsets: set[str] = set()
    elsets: set[str] = set()
    for block in blocks:
        if block.keyword in {"BOUNDARY", "CLOAD"}:
            for line in block.lines:
                fields = [f.strip() for f in line.split(",") if f.strip()]
                if fields:
                    nsets.add(fields[0])
        if block.keyword == "DLOAD":
            for line in block.lines:
                fields = [f.strip() for f in line.split(",") if f.strip()]
                if fields:
                    elsets.add(fields[0])
        if block.keyword == "BEAM SECTION":
            _, opts = _parse_keyword(block.header)
            elset = opts.get("elset")
            if elset:
                elsets.add(elset)
    return nsets, elsets


def _find_amplitude_blocks(blocks: Sequence[InpBlock]) -> Dict[str, InpBlock]:
    amps: Dict[str, InpBlock] = {}
    for block in blocks:
        if block.keyword == "AMPLITUDE":
            _, opts = _parse_keyword(block.header)
            name = opts.get("name", "")
            if name:
                amps[name] = block
    return amps


def _find_amplitude_refs(lines: Sequence[str]) -> List[str]:
    refs: List[str] = []
    for line in lines:
        if not line or line.lstrip().startswith("**"):
            continue
        if line.lstrip().startswith("*"):
            _, opts = _parse_keyword(line)
            amp = opts.get("amplitude")
            if amp:
                refs.append(amp)
    return refs


def find_unused_amplitudes(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    blocks = _split_blocks(lines)
    amps = _find_amplitude_blocks(blocks)
    refs = set(_find_amplitude_refs(lines))
    return sorted([name for name in amps if name not in refs])


def rename_sets(lines: Iterable[str], mapping: Dict[str, str]) -> List[str]:
    renamed: List[str] = []
    for raw in lines:
        line = raw
        if line.lstrip().startswith("*"):
            for old, new in mapping.items():
                line = re.sub(rf"\b{re.escape(old)}\b", new, line)
        else:
            for old, new in mapping.items():
                line = re.sub(rf"\b{re.escape(old)}\b", new, line)
        renamed.append(line)
    return renamed


def normalize_inp(
    input_path: Path,
    output_dir: Path,
    set_mapping: Dict[str, str],
) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = rename_sets(lines, set_mapping)
    blocks = _split_blocks(lines)

    amplitudes = _find_amplitude_blocks(blocks)
    amp_refs = set(_find_amplitude_refs(lines))
    unused_amps = sorted([name for name in amplitudes if name not in amp_refs])
    defined_nsets, defined_elsets = _collect_defined_sets(blocks)
    ref_nsets, ref_elsets = _collect_referenced_sets(blocks)
    undefined_nsets = sorted([name for name in ref_nsets if name not in defined_nsets])
    undefined_elsets = sorted([name for name in ref_elsets if name not in defined_elsets])

    if unused_amps:
        print(f"Unused amplitudes: {', '.join(unused_amps)}")
    if undefined_nsets:
        print(f"Undefined nsets: {', '.join(undefined_nsets)}")
    if undefined_elsets:
        print(f"Undefined elsets: {', '.join(undefined_elsets)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    amp_dir = output_dir / "amplitudes"
    amp_dir.mkdir(parents=True, exist_ok=True)

    geometry_blocks: List[InpBlock] = []
    material_blocks: List[InpBlock] = []
    section_blocks: List[InpBlock] = []
    assembly_blocks: List[InpBlock] = []
    step_blocks: List[InpBlock] = []

    in_step = False
    current_step: List[InpBlock] = []
    for block in blocks:
        if block.keyword == "STEP":
            in_step = True
            current_step = [block]
            continue
        if in_step:
            current_step.append(block)
            if block.keyword == "END STEP":
                step_blocks.extend(current_step)
                in_step = False
            continue
        if block.keyword in {"PART", "NODE", "ELEMENT", "NSET", "ELSET", "END PART"}:
            geometry_blocks.append(block)
        elif block.keyword in {"MATERIAL", "ELASTIC", "DENSITY"}:
            material_blocks.append(block)
        elif block.keyword in {"BEAM SECTION"}:
            section_blocks.append(block)
        elif block.keyword in {"ASSEMBLY", "INSTANCE", "END INSTANCE", "END ASSEMBLY"}:
            assembly_blocks.append(block)
        elif block.keyword == "AMPLITUDE":
            _, opts = _parse_keyword(block.header)
            name = opts.get("name", "AMP")
            if name not in unused_amps:
                amp_path = amp_dir / f"{name}.inp"
                amp_path.write_text("\n".join([block.header] + block.lines) + "\n", encoding="utf-8")
        else:
            geometry_blocks.append(block)

    def _write_blocks(path: Path, blocks_to_write: Sequence[InpBlock]) -> None:
        content: List[str] = []
        for block in blocks_to_write:
            if block.header:
                content.append(block.header)
            content.extend(block.lines)
        if content:
            path.write_text("\n".join(content) + "\n", encoding="utf-8")

    _write_blocks(output_dir / "geometry.inp", geometry_blocks)
    _write_blocks(output_dir / "materials.inp", material_blocks)
    _write_blocks(output_dir / "sections.inp", section_blocks)
    _write_blocks(output_dir / "assembly.inp", assembly_blocks)
    if step_blocks:
        step_path = output_dir / "steps.inp"
        _write_blocks(step_path, step_blocks)

    main_lines = [
        "** INPUT DECK V2 - NORMALIZED",
        "** UNITS: SI (m, N, kg, s)",
        "*Include, input=geometry.inp",
        "*Include, input=materials.inp",
        "*Include, input=sections.inp",
        "*Include, input=assembly.inp",
    ]
    if step_blocks:
        main_lines.append("*Include, input=steps.inp")
    for amp in amp_refs:
        if (amp_dir / f"{amp}.inp").exists():
            main_lines.append(f"*Include, input=amplitudes/{amp}.inp")
    (output_dir / "main.inp").write_text("\n".join(main_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Abaqus-like .inp files into v2 structure.")
    parser.add_argument("input", help="Path to .inp file")
    parser.add_argument("--output-dir", default="normalized_inp", help="Output directory")
    parser.add_argument(
        "--map",
        nargs="*",
        default=[],
        help="Set rename mapping in the form Old=New",
    )
    args = parser.parse_args()
    mapping = {}
    for item in args.map:
        if "=" in item:
            old, new = item.split("=", 1)
            mapping[old] = new
    normalize_inp(Path(args.input), Path(args.output_dir), mapping)


if __name__ == "__main__":
    main()
