"""Thin wrapper to run Problema 4 portal frame demo."""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from problems.problema4_portico import main


if __name__ == "__main__":
    main()
