"""Genera una medición sintética para probar la carga de calibración."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import DEFAULT_FFT_SIZE_SAMPLES, DEFAULT_SAMPLE_RATE_HZ

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera una medición demo para el filtro inverso")
    parser.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE_SAMPLES)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    return parser.parse_args()


def generate_measurement(
    fft_size: int = DEFAULT_FFT_SIZE_SAMPLES,
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve H(f) y gamma² con las muescas demo configuradas."""
    if not isinstance(fft_size, (int, np.integer)) or fft_size < 4 or fft_size % 2:
        raise ValueError("fft_size debe ser un entero par mayor o igual a 4")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate debe ser finita y positiva")
    if sample_rate / 2.0 <= 4_000.0:
        raise ValueError("sample_rate debe superar 8 kHz para representar la muesca de 4 kHz")

    bins = fft_size // 2 + 1
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    magnitude = np.ones(bins)
    for center, depth_db, width_hz in (
        (1_000.0, -18.0, 250.0),
        (4_000.0, -9.0, 500.0),
    ):
        magnitude *= 10.0 ** (
            depth_db / 20.0 * np.exp(-((frequencies - center) / width_hz) ** 2)
        )

    response = magnitude.astype(np.complex128)
    response[0] = abs(response[0])
    response[-1] = abs(response[-1])
    coherence = np.ones(bins, dtype=np.float64)
    return response, coherence


def main() -> int:
    args = parse_args()
    try:
        h_response, coherence = generate_measurement(args.fft_size, args.sample_rate)
    except ValueError as error:
        print(f"Configuración no válida: {error}", file=sys.stderr)
        return 2

    h_path = Path("medicion_sintetica_H.npy")
    coherence_path = Path("medicion_sintetica_gamma2.npy")
    np.save(h_path, h_response)
    np.save(coherence_path, coherence)
    h_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "sample_rate_hz": args.sample_rate,
                "fft_size_samples": args.fft_size,
                "frequency_range_hz": [20.0, min(20_000.0, args.sample_rate / 2.0)],
                "synthetic": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{h_path}: {h_response.size} bins a {args.sample_rate:g} Hz, "
        "muescas de -18 dB en 1 kHz y -9 dB en 4 kHz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
