"""Compara métodos numéricos y evalúa la linealidad de la ruta de audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from generar_medicion_demo import generate_measurement
from src.dsp.inverse_filter import epsilon_profile, estimate_h_inverse
from src.dsp.numerics import (
    calculate_condition_number,
    calculate_filter_energy,
    calculate_group_delay,
    estimate_h_inverse_gradient,
    find_cutoff_frequency,
)


def analyze_measurement(
    h_response: np.ndarray,
    coherence: np.ndarray,
    *,
    sample_rate: float,
    epsilon_base: float,
    max_gain_db: float,
    iterations: int = 2_000,
) -> dict[str, float]:
    """Calcula métricas numéricas para una respuesta compleja medida."""
    h = np.asarray(h_response, dtype=np.complex128)
    gamma_squared = np.asarray(coherence, dtype=np.float64)
    if h.ndim != 1 or h.size < 3 or h.shape != gamma_squared.shape:
        raise ValueError("H y coherencia deben tener la misma forma 1-D y al menos 3 bins")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(gamma_squared)):
        raise ValueError("H y coherencia deben contener solo valores finitos")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate debe ser finita y positiva")
    if iterations <= 0:
        raise ValueError("iterations debe ser positivo")

    fft_size = 2 * (h.size - 1)
    delta_f = sample_rate / fft_size
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    epsilon = epsilon_profile(h, gamma_squared, epsilon_base)
    power = np.abs(h) ** 2
    analytic_unbounded = np.conj(h) / (power + epsilon)
    learning_rate = 0.5 / float(np.max(power + epsilon))
    gradient = estimate_h_inverse_gradient(
        h,
        gamma_squared,
        epsilon_base,
        iterations,
        learning_rate,
        initial_guess=np.zeros_like(h),
    )
    inverse = estimate_h_inverse(h, gamma_squared, epsilon_base, max_gain_db)
    condition = calculate_condition_number(h, epsilon)
    group_delay = calculate_group_delay(inverse, delta_f)
    filter_energy = calculate_filter_energy(inverse, delta_f)
    magnitude = np.abs(inverse)
    cutoff_target = float((np.min(magnitude) + np.max(magnitude)) / 2.0)
    cutoff_frequency = find_cutoff_frequency(
        frequencies,
        magnitude,
        cutoff_target,
        tol=max(delta_f * 1e-8, 1e-10),
    )

    return {
        "gradient_max_error": float(np.max(np.abs(gradient - analytic_unbounded))),
        "condition_min": float(np.min(condition)),
        "condition_median": float(np.median(condition)),
        "condition_max": float(np.max(condition)),
        "group_delay_rms_ms": float(np.sqrt(np.mean(group_delay**2)) * 1_000.0),
        "group_delay_peak_ms": float(np.max(np.abs(group_delay)) * 1_000.0),
        "filter_energy": filter_energy,
        "cutoff_target_magnitude": cutoff_target,
        "cutoff_frequency_hz": cutoff_frequency,
    }


def compare_transfer_linearity(
    h_low: np.ndarray,
    h_high: np.ndarray,
) -> dict[str, float]:
    """Compara H(f) medida a dos amplitudes; diferencias indican no linealidad."""
    low = np.asarray(h_low, dtype=np.complex128)
    high = np.asarray(h_high, dtype=np.complex128)
    if low.ndim != 1 or low.shape != high.shape or low.size == 0:
        raise ValueError("Las dos respuestas deben ser vectores 1-D de la misma forma")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Las respuestas deben contener solo valores finitos")

    scale = max(float(np.max(np.abs(low))), float(np.max(np.abs(high))))
    floor = max(scale * 1e-12, 1e-12)
    valid = np.maximum(np.abs(low), np.abs(high)) > max(scale * 1e-6, floor)
    if not np.any(valid):
        raise ValueError("No hay bins con señal suficiente para comparar")

    low_valid = low[valid]
    high_valid = high[valid]
    relative_error = np.abs(high_valid - low_valid) / np.maximum(
        np.maximum(np.abs(low_valid), np.abs(high_valid)), floor
    )
    magnitude_difference_db = 20.0 * np.log10(
        np.maximum(np.abs(high_valid), floor) / np.maximum(np.abs(low_valid), floor)
    )
    phase_difference = np.angle(high_valid * np.conj(low_valid), deg=True)

    return {
        "bins_compared": float(np.count_nonzero(valid)),
        "relative_error_mean": float(np.mean(relative_error)),
        "relative_error_max": float(np.max(relative_error)),
        "magnitude_difference_db_mean": float(np.mean(np.abs(magnitude_difference_db))),
        "magnitude_difference_db_max": float(np.max(np.abs(magnitude_difference_db))),
        "phase_difference_deg_max": float(np.max(np.abs(phase_difference))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análisis numérico del filtro inverso")
    parser.add_argument("--h-response", type=Path, help="medición H(f); si falta, usa la demo")
    parser.add_argument("--coherence", type=Path, help="archivo gamma²(f), requiere --h-response")
    parser.add_argument("--sample-rate", type=float, default=48_000.0)
    parser.add_argument("--fft-size", type=int, default=2_048)
    parser.add_argument("--epsilon-base", type=float, default=0.02)
    parser.add_argument("--max-gain-db", type=float, default=12.0)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--linearity-low", type=Path, help="H(f) medida con amplitud baja")
    parser.add_argument("--linearity-high", type=Path, help="H(f) medida con amplitud alta")
    return parser.parse_args()


def _load_vector(path: Path, name: str) -> np.ndarray:
    values = np.load(path)
    if values.ndim != 1:
        raise ValueError(f"{name} debe contener un vector 1-D")
    return values


def main() -> int:
    args = parse_args()
    if bool(args.linearity_low) != bool(args.linearity_high):
        print("Use --linearity-low y --linearity-high juntos.", file=sys.stderr)
        return 2
    if args.coherence is not None and args.h_response is None:
        print("--coherence requiere --h-response.", file=sys.stderr)
        return 2

    try:
        if args.h_response is None:
            h_response, coherence = generate_measurement(args.fft_size, args.sample_rate)
        else:
            h_response = _load_vector(args.h_response, "H")
            coherence = (
                _load_vector(args.coherence, "coherence")
                if args.coherence is not None
                else np.ones(h_response.size, dtype=np.float64)
            )
        metrics = analyze_measurement(
            h_response,
            coherence,
            sample_rate=args.sample_rate,
            epsilon_base=args.epsilon_base,
            max_gain_db=args.max_gain_db,
            iterations=args.iterations,
        )
        print("Análisis del filtro")
        print(f"Error máximo gradiente vs. analítica: {metrics['gradient_max_error']:.6g}")
        print(
            "Número de condición (mín/mediana/máx): "
            f"{metrics['condition_min']:.4g} / {metrics['condition_median']:.4g} / "
            f"{metrics['condition_max']:.4g}"
        )
        print(
            "Retardo de grupo RMS/pico: "
            f"{metrics['group_delay_rms_ms']:.6g} / {metrics['group_delay_peak_ms']:.6g} ms"
        )
        print(f"Energía integrada del filtro: {metrics['filter_energy']:.6g}")
        print(
            "Corte por bisección al módulo "
            f"{metrics['cutoff_target_magnitude']:.4g}: "
            f"{metrics['cutoff_frequency_hz']:.3f} Hz"
        )

        if args.linearity_low is not None and args.linearity_high is not None:
            linearity = compare_transfer_linearity(
                _load_vector(args.linearity_low, "linearity-low"),
                _load_vector(args.linearity_high, "linearity-high"),
            )
            print("Linealidad entre amplitudes")
            print(f"Bins comparados: {int(linearity['bins_compared'])}")
            print(
                "Error relativo medio/máximo: "
                f"{linearity['relative_error_mean']:.4g} / {linearity['relative_error_max']:.4g}"
            )
            print(
                "Diferencia de magnitud máxima: "
                f"{linearity['magnitude_difference_db_max']:.4g} dB"
            )
            print(
                "Diferencia de fase máxima: "
                f"{linearity['phase_difference_deg_max']:.4g} grados"
            )
    except (OSError, ValueError) as error:
        print(f"No se pudo completar el análisis: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
