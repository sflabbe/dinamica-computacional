from __future__ import annotations

import sys

from dc_solver.run import run_inp


def main() -> None:
    if len(sys.argv) < 2:
        path = "inputs/portal_problem4.inp"
    else:
        path = sys.argv[1]
    run_inp(path)


if __name__ == "__main__":
    main()
