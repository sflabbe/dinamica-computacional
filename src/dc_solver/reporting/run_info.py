"""Lightweight run-info export (txt + json) with JSON-safe serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import json
import os
import platform
import sys
from datetime import datetime, date
import numpy as np


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_jsonable(obj: Any) -> Any:
    """Convert objects to JSON-serializable types recursively.

    Handles:
    - np.ndarray -> list
    - np.generic (np.float64, np.int64, etc.) -> float/int
    - Path -> str
    - datetime/date -> isoformat string
    - set/tuple -> list
    - dict with non-string keys -> dict with str keys
    - Recursively processes lists and dicts

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    Any
        JSON-serializable version of obj

    Examples
    --------
    >>> to_jsonable(np.array([1.0, 2.0, 3.0]))
    [1.0, 2.0, 3.0]
    >>> to_jsonable(np.float64(3.14))
    3.14
    >>> to_jsonable(Path("/home/user"))
    '/home/user'
    """
    # None, bool, int, float, str are already JSON-safe
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy arrays -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy scalars -> python types
    if isinstance(obj, np.generic):
        item = obj.item()
        # item() returns a python scalar, but recurse to be safe
        return to_jsonable(item)

    # Path -> str
    if isinstance(obj, Path):
        return str(obj)

    # datetime/date -> isoformat
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()

    # set/tuple -> list (recursively)
    if isinstance(obj, (set, tuple)):
        return [to_jsonable(item) for item in obj]

    # list -> list (recursively)
    if isinstance(obj, list):
        return [to_jsonable(item) for item in obj]

    # dict -> dict (recursively, with str keys)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Fallback: try str() conversion
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


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
    """Write run info to both JSON and text formats.

    Automatically applies to_jsonable() to handle numpy arrays, Path objects, etc.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{base_name}.json"
    txt_path = out_dir / f"{base_name}.txt"

    # Convert to JSON-safe format
    info_safe = to_jsonable(info)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(info_safe, f, indent=2, ensure_ascii=False)

    flat = _flatten(info_safe)
    lines = [f"{k}: {flat[k]}" for k in sorted(flat.keys())]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
