## Fiber ##

# Gravity (fiber) con verbose
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --gravity --gravity-verbose

# IDA con HHT (default)
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --integrator newmark
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --integrator hht
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --integrator explicit

# P-Delta
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --gravity --gravity-verbose --nlgeom --integrator newmark
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --integrator newmark --nlgeom
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --integrator hht --nlgeom
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge fiber --integrator explicit --nlgeom

## SHM ##

# Gravity (fiber) con verbose
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --gravity --gravity-verbose

# IDA con HHT (default)
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --integrator newmark
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --integrator hht
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --integrator explicit

# P-Delta
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --gravity --gravity-verbose --nlgeom --integrator newmark
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --integrator newmark --nlgeom
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --integrator hht --nlgeom
PYTHONPATH=src python -u -m problems.problema4_portico --beam-hinge shm --integrator explicit --nlgeom
