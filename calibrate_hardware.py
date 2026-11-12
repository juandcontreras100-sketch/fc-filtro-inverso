"""Mide la respuesta compleja de un circuito con un barrido logarítmico.

El resultado ``medicion_H.npy`` contiene los bins RFFT que consume ``main.py``.
Use el mismo ``--sample-rate`` y ``--fft-size`` en ambos programas.

Ejemplo:
    python calibrate_hardware.py --device "Mi interfaz" --fft-size 2048
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.config import DEFAULT_FFT_SIZE_SAMPLES, DEFAULT_SAMPLE_RATE_HZ

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibración H(f) por Log Sine Sweep")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument(
        "--fft-size",
        type=int,
        default=DEFAULT_FFT_SIZE_SAMPLES,
        help="debe coincidir con main.py",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="duración del sweep en segundos")
    parser.add_argument("--start-freq", type=float, default=20.0)
    parser.add_argument("--stop-freq", type=float, default=20_000.0)
    parser.add_argument("--amplitude", type=float, default=0.15, help="pico digital; manténgalo bajo")
    parser.add_argument("--pre-roll", type=float, default=0.5, help="silencio antes del sweep, en segundos")
    parser.add_argument("--tail", type=float, default=0.75, help="silencio tras el sweep, en segundos")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--device", help="índice o nombre del dispositivo sounddevice")
    parser.add_argument("--sweeps", type=int, default=4, help="cantidad de barridos para promediar (mínimo 2)")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--record-channel", type=int, default=0, help="canal grabado que viene del circuito")
    parser.add_argument("--output", type=Path, default=Path("medicion_H.npy"))
    parser.add_argument(
        "--farina-output",
        type=Path,
        help="guardar respuesta impulsional Farina diagnóstica (no usada por el ecualizador)",
    )
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
    if args.sweeps < 2:
        raise ValueError("--sweeps debe ser al menos 2")


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


def estimate_transfer_ensemble(
    reference: np.ndarray,
    recordings: list[np.ndarray],
    sample_rate: float,
    fft_size: int,
    start_freq: float,
    stop_freq: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Promedia barridos alineados y estima coherencia cuadrática por bin."""
    if len(recordings) < 2:
        raise ValueError("Se requieren al menos dos barridos para estimar coherencia")

    analysis_size = next_power_of_two(reference.size)
    window = np.hanning(reference.size)
    source_fft = np.fft.rfft(reference * window, analysis_size)
    source_power = np.abs(source_fft) ** 2
    epsilon = 1e-10 * max(float(np.max(source_power)), 1.0)

    starts: list[int] = []
    response_spectra = []
    for recorded in recordings:
        start = estimate_start_sample(recorded, reference)
        end = start + reference.size
        if start < 0 or end > recorded.size:
            raise RuntimeError(
                "No se pudo capturar un sweep completo. Aumente --pre-roll/--tail "
                "o compruebe la ruta de audio."
            )
        response = recorded[start:end]
        starts.append(start)
        response_spectra.append(np.fft.rfft(response * window, analysis_size))

    response_spectra_array = np.asarray(response_spectra)
    cross_power = np.mean(response_spectra_array * np.conj(source_fft), axis=0)
    response_power = np.mean(np.abs(response_spectra_array) ** 2, axis=0)
    transfer = cross_power / (source_power + epsilon)
    coherence = np.abs(cross_power) ** 2 / (
        source_power * response_power + epsilon**2
    )
    coherence = np.clip(coherence, 0.0, 1.0)

    measured_frequencies = np.fft.rfftfreq(analysis_size, d=1.0 / sample_rate)
    target_frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    target = np.ones(target_frequencies.size, dtype=np.complex128)
    target_coherence = np.zeros(target_frequencies.size, dtype=np.float64)
    usable = (target_frequencies >= start_freq) & (target_frequencies <= stop_freq)
    power_at_target = np.interp(target_frequencies, measured_frequencies, source_power)
    reliable = usable & (power_at_target > 100.0 * epsilon)
    target[reliable] = np.interp(
        target_frequencies[reliable], measured_frequencies, transfer.real
    ) + 1j * np.interp(
        target_frequencies[reliable], measured_frequencies, transfer.imag
    )
    target_coherence[reliable] = np.interp(
        target_frequencies[reliable], measured_frequencies, coherence
    )
    return target, target_coherence, starts


def estimate_impulse_response_farina(
    reference: np.ndarray,
    recorded: np.ndarray,
    sample_rate: float,
    start_freq: float,
    stop_freq: float,
) -> tuple[np.ndarray, int]:
    """Deconvoluciona una captura con la inversa exponencial de Farina.

    Es una salida diagnóstica opcional; no reemplaza la estimación H(f) usada
    por el ecualizador.
    """
    reference = np.asarray(reference, dtype=np.float64)
    recorded = np.asarray(recorded, dtype=np.float64)
    if reference.ndim != 1 or recorded.ndim != 1 or reference.size < 2:
        raise ValueError("reference y recorded deben ser vectores 1-D y reference >= 2")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(recorded)):
        raise ValueError("reference y recorded deben contener solo valores finitos")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate debe ser finita y positiva")
    if not 0.0 < start_freq < stop_freq < sample_rate / 2.0:
        raise ValueError("las frecuencias deben cumplir 0 < inicio < fin < Nyquist")

    duration = reference.size / sample_rate
    time_axis = np.arange(reference.size, dtype=np.float64) / sample_rate
    inverse_sweep = reference[::-1] * np.exp(
        -time_axis * np.log(stop_freq / start_freq) / duration
    )
    reference_conv_size = reference.size + inverse_sweep.size - 1
    recorded_conv_size = recorded.size + inverse_sweep.size - 1
    fft_size = next_power_of_two(max(reference_conv_size, recorded_conv_size))
    inverse_spectrum = np.fft.rfft(inverse_sweep, fft_size)
    reference_response = np.fft.irfft(
        np.fft.rfft(reference, fft_size) * inverse_spectrum,
        fft_size,
    )[:reference_conv_size]
    normalization = float(np.max(np.abs(reference_response)))
    if normalization <= np.finfo(np.float64).eps:
        raise ValueError("No se pudo normalizar la inversa Farina")

    impulse_response = np.fft.irfft(
        np.fft.rfft(recorded, fft_size) * inverse_spectrum,
        fft_size,
    )[:recorded_conv_size] / normalization
    peak_sample = int(np.argmax(np.abs(impulse_response)))
    return impulse_response, peak_sample


def validate_calibration_quality(
    recordings: list[np.ndarray],
    coherence: np.ndarray,
    sample_rate: float,
    fft_size: int,
) -> dict[str, float]:
    """Rechaza capturas silenciosas, saturadas o con coherencia insuficiente."""
    if len(recordings) < 2:
        raise ValueError("Se requieren al menos dos grabaciones")
    if not np.isfinite(sample_rate) or sample_rate <= 20_000.0:
        raise ValueError("sample_rate debe ser finita y mayor que 20 kHz")
    if not isinstance(fft_size, (int, np.integer)) or fft_size < 4 or fft_size % 2:
        raise ValueError("fft_size debe ser un entero par mayor o igual a 4")

    gamma_squared = np.asarray(coherence, dtype=np.float64)
    if gamma_squared.ndim != 1 or gamma_squared.size != fft_size // 2 + 1:
        raise ValueError("coherence debe tener fft_size // 2 + 1 bins")
    if not np.all(np.isfinite(gamma_squared)):
        raise ValueError("coherence debe contener solo valores finitos")

    rms_levels = []
    for index, recording in enumerate(recordings, start=1):
        samples = np.asarray(recording, dtype=np.float64)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError(f"grabación {index} debe ser un vector 1-D no vacío")
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"grabación {index} contiene valores no finitos")

        peak = float(np.max(np.abs(samples)))
        if peak >= 0.999:
            raise ValueError(
                f"grabación {index} alcanzó saturación (pico {peak:.4f}); "
                "reduzca el nivel y repita"
            )

        rms = float(np.sqrt(np.mean(samples * samples)))
        rms_dbfs = 20.0 * np.log10(max(rms, 1e-12))
        if rms_dbfs <= -60.0:
            raise ValueError(
                f"señal de entrada demasiado baja en la grabación {index} "
                f"({rms_dbfs:.2f} dBFS); no se guardó calibración"
            )
        rms_levels.append(rms_dbfs)

    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    band = (frequencies >= 100.0) & (frequencies <= 10_000.0)
    if not np.any(band):
        raise ValueError("fft_size no ofrece bins suficientes entre 100 Hz y 10 kHz")
    median_coherence = float(np.median(gamma_squared[band]))
    if median_coherence <= 0.8:
        raise ValueError(
            "coherencia mediana insuficiente entre 100 Hz y 10 kHz "
            f"({median_coherence:.3f}; se requiere > 0.8)"
        )

    return {
        "minimum_rms_dbfs": float(min(rms_levels)),
        "median_coherence_100_10000_hz": median_coherence,
    }


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

    recordings = []
    for sweep_index in range(args.sweeps):
        print(f"Barrido {sweep_index + 1}/{args.sweeps}...", flush=True)
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
            print(
                f"Advertencia: el barrido {sweep_index + 1} alcanzó saturación; "
                "reduzca el nivel y repita.",
                file=sys.stderr,
            )
        recordings.append(recorded)

    try:
        h_response, coherence, starts = estimate_transfer_ensemble(
            sweep,
            recordings,
            args.sample_rate,
            args.fft_size,
            args.start_freq,
            args.stop_freq,
        )
    except (RuntimeError, ValueError) as error:
        print(f"Calibración fallida: {error}", file=sys.stderr)
        return 1

    try:
        quality = validate_calibration_quality(
            recordings, coherence, args.sample_rate, args.fft_size
        )
    except ValueError as error:
        print(f"Calibración rechazada: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, h_response)
    coherence_stem = args.output.stem.replace("_H", "_gamma2")
    if coherence_stem == args.output.stem:
        coherence_stem = f"{args.output.stem}_gamma2"
    coherence_path = args.output.with_name(f"{coherence_stem}.npy")
    np.save(coherence_path, coherence)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "sample_rate_hz": args.sample_rate,
                "fft_size_samples": args.fft_size,
                "frequency_range_hz": [args.start_freq, args.stop_freq],
                "synthetic": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for sweep_index, start_sample in enumerate(starts, start=1):
        print(
            f"Barrido {sweep_index} alineado en la muestra {start_sample} "
            f"({start_sample / args.sample_rate:.4f} s)."
        )
    print(f"Respuesta compleja guardada en: {args.output.resolve()}")
    print(f"Coherencia guardada en: {coherence_path.resolve()}")
    print(
        "Calidad: "
        f"RMS mínimo {quality['minimum_rms_dbfs']:.2f} dBFS, "
        "coherencia mediana "
        f"{quality['median_coherence_100_10000_hz']:.3f}."
    )
    if args.farina_output is not None:
        try:
            impulse_response, peak_sample = estimate_impulse_response_farina(
                sweep,
                recordings[0],
                args.sample_rate,
                args.start_freq,
                args.stop_freq,
            )
        except ValueError as error:
            print(f"No se pudo calcular la respuesta impulsional Farina: {error}", file=sys.stderr)
        else:
            args.farina_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.farina_output, impulse_response)
            print(
                f"Respuesta impulsional Farina guardada en: {args.farina_output.resolve()} "
                f"(pico en muestra {peak_sample})."
            )
    print(
        f"Use: python main.py --sample-rate {args.sample_rate:g} --fft-size "
        f"{args.fft_size} --h-response {args.output} --coherence {coherence_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
