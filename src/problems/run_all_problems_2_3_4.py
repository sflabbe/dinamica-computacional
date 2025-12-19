"""Run problems 2, 3, and 4 scripts and store outputs."""

from __future__ import annotations

from problems import problema2_secciones_nm
from problems import problema2_hinge_nm_verification
from problems import problema3_shm_verification
from problems import problema4_portico


def main() -> None:
    problema2_secciones_nm.main()
    problema2_hinge_nm_verification.main()
    problema3_shm_verification.main()
    problema4_portico.main()


if __name__ == "__main__":
    main()
