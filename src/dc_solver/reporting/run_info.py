"""Lightweight run-info export (txt + json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import os
import platform
import sys
from datetime import datetime


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def build_run_info(
    *,
    job: str,
    output_dir: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "job": str(job),
        "timestamp": _now_iso(),
        "cwd": os.getcwd(),
        "output_dir": str(output_dir),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    if meta:
        info["meta"] = meta
    return info


def write_run_info(
    out_dir: Path,
    *,
    base_name: str = "runinfo",
    info: Dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{base_name}.json"
    txt_path = out_dir / f"{base_name}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    flat = _flatten(info)
    lines = [f"{k}: {flat[k]}" for k in sorted(flat.keys())]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
