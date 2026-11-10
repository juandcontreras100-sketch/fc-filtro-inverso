"""Soft limiter for the output stage."""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.8
DEFAULT_CEILING = 0.98


def soft_limit(
    samples: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    ceiling: float = DEFAULT_CEILING,
    *,
    knee: float | None = None,
) -> np.ndarray:
    """Compress peaks smoothly while preserving samples below the threshold.

    ``knee`` is accepted for callers using the alternate ceiling-minus-knee
    convention; when supplied, it sets ``threshold = ceiling - knee``.
    """
    if knee is not None:
        if threshold != DEFAULT_THRESHOLD:
            raise ValueError("use threshold or knee, not both")
        if not np.isfinite(knee) or not 0.0 < knee < ceiling:
            raise ValueError("knee must be in (0, ceiling)")
        threshold = ceiling - knee
    if not np.isfinite(threshold) or not 0.0 < threshold < ceiling <= 1.0:
        raise ValueError("require 0 < threshold < ceiling <= 1")

    values = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")

    magnitude = np.abs(values)
    over = magnitude > threshold
    if not np.any(over):
        return values

    knee_width = ceiling - threshold
    limited = values.copy()
    limited[over] = np.sign(values[over]) * (
        threshold
        + knee_width * np.tanh((magnitude[over] - threshold) / knee_width)
    )
    return limited
