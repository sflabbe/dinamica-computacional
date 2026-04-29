import numpy as np
import plotly.graph_objects as go

from dc_solver.post.plotly_charts import (
    plot_base_shear_drift,
    plot_drift_time_history,
    plot_energy_balance,
    plot_frame_state,
    plot_hinge_mtheta,
    plot_mode_shape,
    plot_response_spectrum,
)
from dc_solver.post.results import DynamicResult, FrameStateResult, HingeTimeHistory


def test_plotly_functions_return_figures():
    state = FrameStateResult(
        label="s",
        u=np.zeros(4),
        scale=1.0,
        node_xy_ref=np.zeros((2, 2)),
        node_xy_def=np.ones((2, 2)),
        node_umag=np.zeros(2),
        member_sigma_max=np.zeros(1),
        drift=0.0,
    )
    h = HingeTimeHistory(0, "k", "h", np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.1]), np.array([0.0, 0.0]))
    dyn = DynamicResult(
        t=np.array([0.0, 1.0]),
        ag=np.zeros(2),
        drift=np.array([0.0, 0.1]),
        Vb=np.array([0.0, 1.0]),
        u=np.zeros((2, 1)),
        energy={"Ek": np.array([1.0, 0.9])},
    )
    funcs = [
        plot_frame_state(state),
        plot_drift_time_history(dyn),
        plot_base_shear_drift(dyn),
        plot_hinge_mtheta(h),
        plot_energy_balance(dyn),
        plot_mode_shape(),
        plot_response_spectrum(),
    ]
    assert all(isinstance(fig, go.Figure) for fig in funcs)
