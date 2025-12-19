"""Formatting helpers for Abaqus-like log files."""

from __future__ import annotations

from datetime import datetime

_MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def format_date(dt: datetime) -> str:
    month = _MONTHS[dt.month - 1]
    return f"{dt.day:02d}-{month}-{dt.year:04d}"


def format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def format_float(value: float) -> str:
    if value == 0:
        return "0.000"
    abs_val = abs(value)
    if abs_val >= 1.0e3 or abs_val < 1.0e-2:
        return f"{value:.3E}"
    return f"{value:.4f}"


def format_sta_row(
    step: int,
    inc: int,
    attempt: int,
    converged: bool,
    severe_iters: int,
    equil_iters: int,
    total_iters: int,
    total_time: float,
    step_time: float,
    inc_time: float,
) -> str:
    att = f"{attempt}U" if not converged else f"{attempt}"
    return (
        f"{step:>5d}"
        f"{inc:>5d}"
        f"{att:>5s}"
        f"{severe_iters:>8d}"
        f"{equil_iters:>12d}"
        f"{total_iters:>12d}"
        f"{format_float(total_time):>12s}"
        f"{format_float(step_time):>12s}"
        f"{format_float(inc_time):>12s}"
    )


def sta_header() -> str:
    return (
        " STEP  INC  ATT  SEVERE DISCON ITERS  EQUIL ITERS  TOTAL ITERS"
        "  TOTAL TIME  STEP TIME  INC OF TIME"
    )
