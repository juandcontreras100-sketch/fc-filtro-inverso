"""Utilidades de procesamiento digital de señal."""

from .inverse_filter import (
    OverlapAddFFTEqualizer,
    epsilon_profile,
    estimate_h_inverse,
)
from .numerics import (
    calculate_condition_number,
    calculate_filter_energy,
    calculate_group_delay,
    estimate_error_bounds,
    estimate_h_inverse_gradient,
    find_cutoff_frequency,
    smooth_coherence_spline,
)

__all__ = [
    "OverlapAddFFTEqualizer",
    "epsilon_profile",
    "estimate_h_inverse",
    "calculate_condition_number",
    "estimate_error_bounds",
    "estimate_h_inverse_gradient",
    "calculate_group_delay",
    "smooth_coherence_spline",
    "calculate_filter_energy",
    "find_cutoff_frequency",
]
