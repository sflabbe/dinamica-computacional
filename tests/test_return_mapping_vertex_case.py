from __future__ import annotations

import numpy as np

from plastic_hinge.return_mapping import project_onto_polytope_2d


def test_return_mapping_vertex_case():
    # Triangle: x>=0, y>=0, x+y<=1
    # Constraints in A x <= b form
    A = np.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
    ])
    b = np.array([0.0, 0.0, 1.0])

    x0 = np.array([-0.2, 1.2])
    res = project_onto_polytope_2d(x0, A, b, W=np.eye(2))
    assert np.allclose(res.x, np.array([0.0, 1.0]), atol=1e-10)
    assert np.any(np.isclose(A @ res.x, b, atol=1e-10))
