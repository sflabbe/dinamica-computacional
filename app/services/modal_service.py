from __future__ import annotations

from dc_solver.modal.modal_analysis import run_modal_analysis

from .session_schema import AnalysisSettings


def run_modal_case(model, settings: AnalysisSettings):
    return run_modal_analysis(model, n_modes=int(settings.n_modes))


def modal_summary_table(modal_result) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for i, (freq, period, ratio, cum) in enumerate(
        zip(
            modal_result.freq_hz,
            modal_result.periods,
            modal_result.effective_modal_mass_ratio,
            modal_result.cumulative_mass_ratio,
        ),
        start=1,
    ):
        out.append(
            {
                "mode": float(i),
                "freq_hz": float(freq),
                "period_s": float(period),
                "mass_ratio": float(ratio),
                "cum_mass_ratio": float(cum),
            }
        )
    return out
