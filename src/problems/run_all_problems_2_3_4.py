"""Run problems 2, 3, 4, and 5 scripts and store outputs."""

from __future__ import annotations

from problems import problema2_secciones_nm
from problems import problema2_interaccion
from problems import problema3_shm_verification
from problems import problema4_portico
from problems import problema5_fiber_section_interaction


def main() -> None:
    problema2_secciones_nm.main()
    problema2_interaccion.main()
    problema3_shm_verification.main()
    problema4_portico.main()
    problema5_fiber_section_interaction.main()


if __name__ == "__main__":
    main()
