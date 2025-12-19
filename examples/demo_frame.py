"""Run the portal frame problem using the dc_solver input runner."""

from __future__ import annotations

import argparse

from dc_solver.run import run_inp


def main() -> None:
    parser = argparse.ArgumentParser(description="Portal frame demo using new engine + input format.")
    parser.add_argument(
        "--input",
        default="inputs/portal_problem4.inp",
        help="Path to Abaqus-like input file.",
    )
    parser.add_argument(
        "--prefix",
        default="problem4",
        help="Output prefix for generated plots.",
    )
    args = parser.parse_args()

    run_inp(args.input)


if __name__ == "__main__":
    main()
