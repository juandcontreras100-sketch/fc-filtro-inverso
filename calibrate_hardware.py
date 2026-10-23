"""Mide la respuesta compleja de un circuito con un barrido logarítmico.

El resultado ``medicion_H.npy`` contiene los bins RFFT que consume ``main.py``.
Use el mismo ``--sample-rate`` y ``--fft-size`` en ambos programas.

Ejemplo:
    python calibrate_hardware.py --device "Mi interfaz" --fft-size 2048
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibración H(f) por Log Sine Sweep")
    parser.add_argument("--sample-rate", type=float, default=48_000.0)
    parser.add_argument("--fft-size", type=int, default=2_048, help="debe coincidir con main.py")
    parser.add_argument("--duration", type=float, default=5.0, help="duración del sweep en segundos")
    parser.add_argument("--start-freq", type=float, default=20.0)
    parser.add_argument("--stop-freq", type=float, default=20_000.0)
    parser.add_argument("--amplitude", type=float, default=0.15, help="pico digital; manténgalo bajo")
    parser.add_argument("--pre-roll", type=float, default=0.5, help="silencio antes del sweep, en segundos")
    parser.add_argument("--tail", type=float, default=0.75, help="silencio tras el sweep, en segundos")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--device", help="índice o nombre del dispositivo sounddevice")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--record-channel", type=int, default=0, help="canal grabado que viene del circuito")
    parser.add_argument("--output", type=Path, default=Path("medicion_H.npy"))
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.fft_size < 4 or args.fft_size % 2:
        raise ValueError("--fft-size debe ser un entero par mayor o igual a 4")
    if args.duration <= 0 or args.sample_rate <= 0:
        raise ValueError("--duration y --sample-rate deben ser positivos")
    if not 0 < args.start_freq < args.stop_freq < args.sample_rate / 2:
        raise ValueError("las frecuencias deben cumplir 0 < inicio < fin < Nyquist")
    if not 0 < args.amplitude <= 1.0:
        raise ValueError("--amplitude debe estar en (0, 1]")
    if args.pre_roll < 0 or args.tail < 0 or args.countdown < 0:
        raise ValueError("pre-roll, tail y countdown no pueden ser negativos")
    if args.input_channels < 1 or not 0 <= args.record_channel < args.input_channels:
        raise ValueError("--record-channel debe pertenecer a --input-channels")


def log_sine_sweep(sample_rate: float, duration: float, start_freq: float, stop_freq: float, amplitude: float) -> np.ndarray:
    """Genera un sweep exponencial con fundidos para evitar clics de borde."""
    count = int(round(duration * sample_rate))
    if count < 2:
        raise ValueError("el sweep debe tener al menos dos muestras")
    t = np.arange(count, dtype=np.float64) / sample_rate
    ratio_log = np.log(stop_freq / start_freq)
    phase = 2.0 * np.pi * start_freq * duration / ratio_log * (np.exp(t * ratio_log / duration) - 1.0)
    sweep = amplitude * np.sin(phase)

    fade_samples = min(int(round(0.02 * sample_rate)), count // 2)
    if fade_samples:
        fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_samples) / fade_samples)
        sweep[:fade_samples] *= fade
        sweep[-fade_samples:] *= fade[::-1]
    return sweep


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def estimate_start_sample(recorded: np.ndarray, reference: np.ndarray) -> int:
    """Localiza el sweep en la grabación mediante correlación FFT.

    Se alinea y, por tanto, se elimina el retardo constante de conversores y
    buffers. Conservarlo obligaría al inversor de tiempo real a implementar un
    avance no causal.
    """
    fft_length = next_power_of_two(recorded.size + reference.size - 1)
    correlation = np.fft.irfft(
        np.fft.rfft(recorded, fft_length) * np.conj(np.fft.rfft(reference, fft_length)),
        fft_length,
    )
    lag = int(np.argmax(np.abs(correlation)))
    if lag > fft_length // 2:
        lag -= fft_length
    return lag


def estimate_transfer(
    reference: np.ndarray,
    recorded: np.ndarray,
    sample_rate: float,
    fft_size: int,
    start_freq: float,
    stop_freq: float,
) -> tuple[np.ndarray, int]:
    """Calcula H(f) regularizada y la interpola a los bins del ecualizador."""
    start = estimate_start_sample(recorded, reference)
    end = start + reference.size
    if start < 0 or end > recorded.size:
        raise RuntimeError(
            "No se pudo capturar el sweep completo. Aumente --pre-roll/--tail "
            "o compruebe la ruta de audio."
        )

    response = recorded[start:end]
    analysis_size = next_power_of_two(reference.size)
    window = np.hanning(reference.size)
    source_fft = np.fft.rfft(reference * window, analysis_size)
    response_fft = np.fft.rfft(response * window, analysis_size)
    source_power = np.abs(source_fft) ** 2
    # División de Wiener muy suave: evita amplificar bins donde el sweep tiene
    # energía despreciable sin sesgar de forma apreciable la banda útil.
    epsilon = 1e-10 * max(float(np.max(source_power)), 1.0)
    transfer = response_fft * np.conj(source_fft) / (source_power + epsilon)

    measured_frequencies = np.fft.rfftfreq(analysis_size, d=1.0 / sample_rate)
    target_frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    target = np.ones(target_frequencies.size, dtype=np.complex128)
    usable = (target_frequencies >= start_freq) & (target_frequencies <= stop_freq)
    power_at_target = np.interp(target_frequencies, measured_frequencies, source_power)
    # Los extremos fundidos del sweep pueden tener energía prácticamente nula.
    # Es más seguro dejar esos bins en bypass que estimar una razón dominada por
    # ruido y luego pedir al inversor que la amplifique.
    reliable = usable & (power_at_target > 100.0 * epsilon)
    target[reliable] = np.interp(target_frequencies[reliable], measured_frequencies, transfer.real) + 1j * np.interp(
        target_frequencies[reliable], measured_frequencies, transfer.imag
    )
    return target, start


def main() -> int:
    args = parse_args()
    try:
        validate(args)
        import sounddevice as sd
    except (ModuleNotFoundError, ValueError) as error:
        print(f"Configuración no disponible: {error}", file=sys.stderr)
        if isinstance(error, ModuleNotFoundError):
            print("Instale sounddevice con: pip install sounddevice numpy", file=sys.stderr)
        return 2

    sweep = log_sine_sweep(
        args.sample_rate, args.duration, args.start_freq, args.stop_freq, args.amplitude
    )
    pre_samples = int(round(args.pre_roll * args.sample_rate))
    tail_samples = int(round(args.tail * args.sample_rate))
    playback = np.concatenate((np.zeros(pre_samples), sweep, np.zeros(tail_samples))).astype(np.float32)

    print("Calibración preparada.")
    print("Baje el volumen antes de continuar y evite saturación en la entrada.")
    for remaining in range(args.countdown, 0, -1):
        print(f"Emitiendo sweep en {remaining}...", flush=True)
        time.sleep(1)

    try:
        recorded_channels = sd.playrec(
            playback[:, np.newaxis],
            samplerate=args.sample_rate,
            channels=args.input_channels,
            dtype="float32",
            device=args.device,
            blocking=True,
        )
    except sd.PortAudioError as error:
        print(f"No se pudo abrir el stream full-dúplex: {error}", file=sys.stderr)
        return 1

    recorded = np.asarray(recorded_channels[:, args.record_channel], dtype=np.float64)
    if np.max(np.abs(recorded)) >= 0.999:
        print("Advertencia: la entrada alcanzó saturación; reduzca el volumen y repita.", file=sys.stderr)

    try:
        h_response, start_sample = estimate_transfer(
            sweep,
            recorded,
            args.sample_rate,
            args.fft_size,
            args.start_freq,
            args.stop_freq,
        )
    except RuntimeError as error:
        print(f"Calibración fallida: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, h_response)
    print(f"Sweep alineado en la muestra {start_sample} ({start_sample / args.sample_rate:.4f} s).")
    print(f"Respuesta compleja guardada en: {args.output.resolve()}")
    print(f"Use: python main.py --sample-rate {args.sample_rate:g} --fft-size {args.fft_size} --h-response {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
