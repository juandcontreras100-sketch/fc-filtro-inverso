"""Compatibilidad para importaciones históricas desde ``inverse_filter``."""

from src.dsp.inverse_filter import epsilon_profile, estimate_h_inverse
from src.dsp.numerics import (
    _cubic_spline_interpolate,
    calculate_condition_number,
    calculate_filter_energy,
    calculate_group_delay,
    estimate_error_bounds,
    estimate_h_inverse_gradient,
    find_cutoff_frequency,
    smooth_coherence_spline,
)

__all__ = [
    "epsilon_profile",
    "estimate_h_inverse",
    "calculate_condition_number",
    "estimate_error_bounds",
    "estimate_h_inverse_gradient",
    "calculate_group_delay",
    "smooth_coherence_spline",
    "_cubic_spline_interpolate",
    "calculate_filter_energy",
    "find_cutoff_frequency",
]