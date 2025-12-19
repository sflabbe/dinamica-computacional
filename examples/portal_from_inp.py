from __future__ import annotations

import sys

from dinamica_computacional.io.abaqus_like import read_inp
from dinamica_computacional.core.analysis import run_analysis


def main() -> None:
    if len(sys.argv) < 2:
        path = "inputs/portal_problem4.inp"
    else:
        path = sys.argv[1]
    model, plan = read_inp(path)
    results = run_analysis(model, plan)
    results.save_plots(outfile_prefix="problem4")


if __name__ == "__main__":
    main()
