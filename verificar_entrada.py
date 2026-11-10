"""Verifica la captura del PCM2902 sin reproducir ni guardar audio.

Por defecto acepta voz intermitente: basta con que un bloque de 100 ms supere
el umbral. Para un tono continuo de 1 kHz desde un teléfono, usá
``--tone 1000``. Para exigir RMS global sin tono, usá ``--require-continuous``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _dbfs(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-12))


def _tone_metrics(
    data: np.ndarray,
    *,
    sample_rate: int,
    tone_frequency: float,
    tolerance: float,
) -> dict[str, float]:
    """Estima la componente coherente en ``tone_frequency`` y su dominante."""
    count = data.size
    window = np.hanning(count)
    fft_size = 1 << (count - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(data * window, fft_size))
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    band = np.abs(frequencies - tone_frequency) <= tolerance
    if not np.any(band):
        raise ValueError("tone_frequency fuera de la banda analizada")

    band_indices = np.flatnonzero(band)
    peak_index = int(band_indices[int(np.argmax(spectrum[band]))])
    peak_frequency = float(frequencies[peak_index])

    time_axis = np.arange(count, dtype=np.float64) / sample_rate
    window_sum = float(np.sum(window))
    projection = (
        2.0
        * np.sum(
            window * data * np.exp(-2.0j * np.pi * peak_frequency * time_axis)
        )
        / window_sum
    )
    tone_rms = abs(projection) / np.sqrt(2.0)
    tone_dbfs = max(-240.0, _dbfs(tone_rms))

    noise_band = (
        (frequencies >= 100.0)
        & (frequencies <= 20_000.0)
        & ~band
    )
    noise_floor_db = (
        20.0 * np.log10(max(float(np.median(spectrum[noise_band])), 1e-12))
        if np.any(noise_band)
        else -240.0
    )
    tone_peak_db = 20.0 * np.log10(max(float(spectrum[peak_index]), 1e-12))
    dominant_frequency = float(frequencies[int(np.argmax(spectrum))])

    return {
        "tone_frequency_hz": float(tone_frequency),
        "tone_peak_frequency_hz": peak_frequency,
        "tone_dbfs": float(tone_dbfs),
        "tone_peak_db": float(tone_peak_db),
        "noise_floor_db": float(noise_floor_db),
        "tone_snr_db": float(tone_peak_db - noise_floor_db),
        "dominant_frequency_hz": dominant_frequency,
    }


def analyze_capture(
    samples: np.ndarray,
    *,
    sample_rate: int,
    threshold_dbfs: float = -60.0,
    require_continuous: bool = False,
    tone_frequency: float | None = None,
    tone_tolerance: float = 50.0,
) -> dict[str, object]:
    """Calcula métricas de nivel y decide si la entrada tiene señal útil.

    Por defecto evalúa el máximo RMS por bloques de 100 ms, para detectar voz
    intermitente. Con ``require_continuous=True`` exige que el RMS global
    supere el umbral, útil para tonos continuos o pruebas de loopback.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("muestras debe ser un vector 1-D no vacío")
    if not np.all(np.isfinite(data)):
        raise ValueError("muestras debe contener solo valores finitos")
    if sample_rate <= 0 or not np.isfinite(threshold_dbfs):
        raise ValueError("sample_rate y threshold_dbfs deben ser válidos")
    if tone_frequency is not None:
        if not np.isfinite(tone_frequency) or not 0 < tone_frequency < sample_rate / 2:
            raise ValueError("tone_frequency debe estar entre 0 y Nyquist")
        if not np.isfinite(tone_tolerance) or tone_tolerance <= 0:
            raise ValueError("tone_tolerance debe ser positiva")

    rms_dbfs = max(-240.0, _dbfs(np.sqrt(np.mean(data * data))))
    peak = float(np.max(np.abs(data)))
    clipped_samples = int(np.count_nonzero(np.abs(data) >= 0.999))
    frame_size = max(1, sample_rate // 10)
    frame_levels = [
        max(-240.0, _dbfs(np.sqrt(np.mean(frame * frame))))
        for frame in (data[start : start + frame_size] for start in range(0, data.size, frame_size))
    ]
    max_frame_rms_dbfs = max(frame_levels) if frame_levels else -240.0
    frames_above_threshold = sum(
        1 for level in frame_levels if level > threshold_dbfs
    )
    signal_fraction = (
        frames_above_threshold / len(frame_levels) if frame_levels else 0.0
    )
    tone_metrics = (
        _tone_metrics(
            data,
            sample_rate=sample_rate,
            tone_frequency=tone_frequency,
            tolerance=tone_tolerance,
        )
        if tone_frequency is not None
        else None
    )

    if tone_metrics is not None:
        criterion_dbfs = tone_metrics["tone_dbfs"]
        criterion_name = f"tono {tone_frequency:.0f} Hz"
    elif require_continuous:
        criterion_dbfs = rms_dbfs
        criterion_name = "RMS global"
    else:
        criterion_dbfs = max_frame_rms_dbfs
        criterion_name = "máximo por bloque"

    if clipped_samples:
        status = "FAIL"
        message = "Se detectó saturación; reduce la ganancia de entrada."
    elif tone_metrics is not None and criterion_dbfs <= threshold_dbfs:
        status = "FAIL"
        message = (
            f"No se detectó el tono de {tone_frequency:.0f} Hz: "
            f"componente a {criterion_dbfs:.2f} dBFS "
            f"(se requiere > {threshold_dbfs:.1f} dBFS; "
            f"dominante {tone_metrics['dominant_frequency_hz']:.1f} Hz)."
        )
    elif criterion_dbfs <= threshold_dbfs:
        status = "FAIL"
        message = (
            f"Señal demasiado baja ({criterion_name} {criterion_dbfs:.2f} dBFS; "
            f"se requiere > {threshold_dbfs:.1f} dBFS). "
            f"RMS global {rms_dbfs:.2f} dBFS, máximo por bloque "
            f"{max_frame_rms_dbfs:.2f} dBFS."
        )
    elif tone_metrics is not None:
        status = "PASS"
        message = (
            f"Tono de {tone_frequency:.0f} Hz detectado a "
            f"{criterion_dbfs:.2f} dBFS (umbral > {threshold_dbfs:.1f} dBFS; "
            f"dominante {tone_metrics['dominant_frequency_hz']:.1f} Hz)."
        )
    else:
        status = "PASS"
        message = (
            f"La entrada capturó señal por encima del umbral "
            f"({criterion_name} {criterion_dbfs:.2f} dBFS; "
            f"{frames_above_threshold}/{len(frame_levels)} bloques)."
        )

    return {
        "status": status,
        "message": message,
        "sample_rate": int(sample_rate),
        "frames": int(data.size),
        "duration_seconds": float(data.size / sample_rate),
        "rms_dbfs": float(rms_dbfs),
        "max_frame_rms_dbfs": float(max_frame_rms_dbfs),
        "frames_above_threshold": int(frames_above_threshold),
        "signal_fraction": float(signal_fraction),
        "criterion": (
            "tone"
            if tone_metrics is not None
            else "rms_global" if require_continuous else "max_frame_rms"
        ),
        "require_continuous": bool(require_continuous),
        "tone_detected": bool(
            tone_metrics is not None and criterion_dbfs > threshold_dbfs
        ),
        "peak": peak,
        "peak_dbfs": float(max(-240.0, _dbfs(peak))),
        "clipped_samples": clipped_samples,
        "threshold_dbfs": float(threshold_dbfs),
        "frame_rms_dbfs_100ms": frame_levels,
        **({} if tone_metrics is None else tone_metrics),
    }


def _write_svg(report: dict[str, object], path: Path) -> None:
    levels = report["frame_rms_dbfs_100ms"]
    assert isinstance(levels, list)
    width, height = 900, 360
    left, top, plot_width, plot_height = 70, 30, 790, 270
    floor_db, ceiling_db = -120.0, 0.0
    points = []
    for index, level in enumerate(levels):
        x = left + (plot_width * index / max(1, len(levels) - 1))
        y = top + plot_height * (ceiling_db - max(floor_db, min(ceiling_db, level))) / (
            ceiling_db - floor_db
        )
        points.append(f"{x:.1f},{y:.1f}")
    threshold = float(report["threshold_dbfs"])
    threshold_y = top + plot_height * (ceiling_db - threshold) / (ceiling_db - floor_db)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="20" font-family="sans-serif" font-size="16">Nivel RMS por bloques de 100 ms</text>',
    ]
    for db in range(-120, 1, 20):
        y = top + plot_height * (ceiling_db - db) / (ceiling_db - floor_db)
        svg.append(f'<path d="M {left} {y:.1f} H {left + plot_width}" stroke="#ddd"/>')
        svg.append(f'<text x="12" y="{y + 4:.1f}" font-family="sans-serif" font-size="12">{db} dBFS</text>')
    svg.extend(
        [
            f'<path d="M {left} {threshold_y:.1f} H {left + plot_width}" stroke="#c33" stroke-dasharray="6 4"/>',
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#1769aa" stroke-width="2"/>',
            f'<text x="70" y="335" font-family="sans-serif" font-size="14">{report["status"]}: {report["message"]}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")


def _windows_input_state() -> tuple[str, float, bool] | None:
    """Devuelve nombre, volumen porcentual y mute del endpoint USB PnP."""
    try:
        from pycaw.pycaw import AudioUtilities
    except ImportError:
        return None
    for device in AudioUtilities.GetAllDevices():
        name = str(device.FriendlyName)
        device_id = str(device.id)
        if "USB PnP Sound Device" in name and device_id.startswith("{0.0.1."):
            try:
                return name, float(device.volume_percent), bool(device.EndpointVolume.GetMute())
            except Exception:
                return None
    return None


def _find_input_device(sd: object, requested: str | None) -> int:
    devices = sd.query_devices()
    if requested is not None:
        try:
            index = int(requested)
        except ValueError:
            index = next(
                (
                    i
                    for i, device in enumerate(devices)
                    if requested.casefold() in device["name"].casefold()
                    and device["max_input_channels"] > 0
                ),
                -1,
            )
        if (
            index < 0
            or index >= len(devices)
            or devices[index]["max_input_channels"] < 1
        ):
            raise ValueError(f"No se encontró una entrada de audio válida: {requested}")
        return index

    for index, device in enumerate(devices):
        if "USB PnP" in device["name"] and device["max_input_channels"] > 0:
            return index
    raise ValueError("No se encontró la entrada USB PnP del PCM2902")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="índice o parte del nombre de la entrada")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--threshold-dbfs", type=float, default=-60.0)
    parser.add_argument(
        "--require-continuous",
        action="store_true",
        help="exigir RMS global; por defecto basta un bloque de 100 ms",
    )
    parser.add_argument(
        "--tone",
        type=float,
        metavar="HZ",
        help="verificar un tono continuo a esa frecuencia (por ejemplo 1000)",
    )
    parser.add_argument(
        "--tone-tolerance",
        type=float,
        default=50.0,
        help="margen alrededor de --tone para buscar el pico, en Hz",
    )
    parser.add_argument("--report", type=Path, default=Path("verificacion/entrada.json"))
    parser.add_argument("--plot", type=Path, default=Path("verificacion/entrada.svg"))
    args = parser.parse_args(argv)
    if args.sample_rate <= 0 or args.duration <= 0:
        parser.error("--sample-rate y --duration deben ser positivos")

    try:
        import sounddevice as sd
    except ImportError as error:
        print(f"No se pudo importar sounddevice: {error}", file=sys.stderr)
        return 2
    try:
        device_index = _find_input_device(sd, args.device)
        device = sd.query_devices(device_index)
        sd.check_input_settings(
            device=device_index,
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
        )
    except (ValueError, sd.PortAudioError) as error:
        print(f"No se pudo preparar la entrada: {error}", file=sys.stderr)
        return 2

    print(f"Entrada: {device_index} — {device['name']}")
    state = _windows_input_state()
    if state is None:
        print("Windows: no se pudo consultar el volumen (pycaw no disponible).")
    else:
        name, volume, muted = state
        print(f"Windows: {name} | volumen {volume:.0f} % | mute {muted}")
        if muted or volume < 5.0:
            print(
                "ADVERTENCIA: la entrada está muteada o casi en cero. "
                "Subila en mmsys.cpl → Grabación → Micrófono → Niveles."
            )
    print(f"Preparando captura de {args.duration:g} s; no se reproducirá audio.")
    for remaining in (3, 2, 1):
        print(f"Grabando en {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print(
        f"Grabando {args.duration:g} s. Hablá o soplá al micrófono ahora.          ",
        flush=True,
    )
    try:
        captured = sd.rec(
            int(round(args.duration * args.sample_rate)),
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
            blocking=True,
        )
    except sd.PortAudioError as error:
        print(f"Falló la captura de entrada: {error}", file=sys.stderr)
        return 2

    report = analyze_capture(
        captured[:, 0],
        sample_rate=args.sample_rate,
        threshold_dbfs=args.threshold_dbfs,
        require_continuous=args.require_continuous,
        tone_frequency=args.tone,
        tone_tolerance=args.tone_tolerance,
    )
    report.update(
        {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "device_index": device_index,
            "device_name": device["name"],
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_svg(report, args.plot)
    print(f"{report['status']}: {report['message']}")
    print(
        f"RMS={report['rms_dbfs']:.2f} dBFS, "
        f"máximo por bloque={report['max_frame_rms_dbfs']:.2f} dBFS, "
        f"pico={report['peak_dbfs']:.2f} dBFS"
    )
    if args.tone is not None:
        print(
            f"Tono {report['tone_frequency_hz']:.0f} Hz: "
            f"{report['tone_dbfs']:.2f} dBFS | "
            f"dominante {report['dominant_frequency_hz']:.1f} Hz | "
            f"SNR espectral {report['tone_snr_db']:.1f} dB"
        )
    print(f"Reporte: {args.report.resolve()}")
    print(f"Gráfico: {args.plot.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
