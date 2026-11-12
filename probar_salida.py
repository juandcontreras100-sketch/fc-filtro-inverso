"""Prueba la salida del PCM2902 con un tono de 1 kHz.

Uso:
    py probar_salida.py
    py probar_salida.py --duration 3 --amplitude 0.1
    py probar_salida.py --no-prompt
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def build_tone(
    sample_rate: int,
    duration: float,
    frequency: float,
    amplitude: float,
    fade_seconds: float = 0.02,
) -> np.ndarray:
    """Genera un tono senoidal con fundido de entrada y salida."""
    if sample_rate <= 0:
        raise ValueError("sample_rate debe ser positivo")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration debe ser positiva")
    if not np.isfinite(frequency) or not 0 < frequency < sample_rate / 2:
        raise ValueError("frequency debe estar entre 0 y Nyquist")
    if not np.isfinite(amplitude) or not 0 < amplitude <= 1:
        raise ValueError("amplitude debe estar entre 0 y 1")
    if not np.isfinite(fade_seconds) or not 0 <= fade_seconds <= duration / 2:
        raise ValueError("fade_seconds debe estar entre 0 y duration/2")

    count = max(1, int(round(sample_rate * duration)))
    time_axis = np.arange(count, dtype=np.float64) / sample_rate
    tone = amplitude * np.sin(2.0 * np.pi * frequency * time_axis)

    fade_samples = min(int(round(fade_seconds * sample_rate)), count // 2)
    if fade_samples > 0:
        fade = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
        tone[:fade_samples] *= fade
        tone[-fade_samples:] *= fade[::-1]

    return tone.astype(np.float32)


def _find_output_device(sd: object, requested: str | None) -> int:
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
                    and device["max_output_channels"] > 0
                ),
                -1,
            )
        if (
            index < 0
            or index >= len(devices)
            or devices[index]["max_output_channels"] < 1
        ):
            raise ValueError(f"No se encontró una salida de audio válida: {requested}")
        return index

    for index, device in enumerate(devices):
        if "USB PnP" in device["name"] and device["max_output_channels"] > 0:
            return index
    raise ValueError("No se encontró la salida USB PnP del PCM2902")


def _windows_output_state(
    name_hint: str = "USB PnP Sound Device",
) -> tuple[str, float, bool] | None:
    """Devuelve nombre, volumen porcentual y mute de una salida de Windows."""
    try:
        from pycaw.pycaw import AudioUtilities
    except ImportError:
        return None
    for device in AudioUtilities.GetAllDevices():
        name = str(device.FriendlyName)
        device_id = str(device.id)
        if name_hint.casefold() in name.casefold() and device_id.startswith("{0.0.0."):
            try:
                return name, float(device.volume_percent), bool(device.EndpointVolume.GetMute())
            except Exception:
                return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="índice o parte del nombre de la salida")
    parser.add_argument("--frequency", type=float, default=1_000.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="amplitud lineal; 0.1 equivale a 120 dB",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="reproducir sin pedir confirmación auditiva",
    )
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except ImportError as error:
        print(f"No se pudo importar sounddevice: {error}", file=sys.stderr)
        return 2

    try:
        tone = build_tone(
            args.sample_rate, args.duration, args.frequency, args.amplitude
        )
        device_index = _find_output_device(sd, args.device)
        device = sd.query_devices(device_index)
        sd.check_output_settings(
            device=device_index,
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
        )
    except (ValueError, sd.PortAudioError) as error:
        print(f"No se pudo preparar la salida: {error}", file=sys.stderr)
        return 2

    print(f"Salida: {device_index} — {device['name']}")
    is_usb_pnp = "USB PnP" in device["name"]
    state = _windows_output_state() if is_usb_pnp else None
    if state is not None:
        name, volume, muted = state
        print(f"Windows: {name} | volumen {volume:.0f} % | mute {muted}")
        if muted or volume < 5.0:
            print(
                "ADVERTENCIA: la salida está muteada o casi en cero. "
                "Subila antes de continuar."
            )
    elif is_usb_pnp:
        print("Windows: no se pudo consultar el volumen (pycaw no disponible).")

    if is_usb_pnp:
        print("Conectá auriculares al jack VERDE del PCM2902.")
    else:
        print(f"Reproduciendo por: {device['name']}.")
    print(
        f"Vas a escuchar un tono de {args.frequency:.0f} Hz durante "
        f"{args.duration:g} s."
    )
    for remaining in (3, 2, 1):
        print(f"Tono en {remaining}...", end="\r", flush=True)
        time.sleep(1)
    print("Reproduciendo...          ")

    try:
        sd.play(
            tone,
            samplerate=args.sample_rate,
            device=device_index,
            blocking=True,
        )
    except sd.PortAudioError as error:
        print(f"Falló la reproducción: {error}", file=sys.stderr)
        print("Cerrá la app si está abierta y volvé a intentar.", file=sys.stderr)
        return 2

    print("Tono reproducido.")
    if args.no_prompt:
        return 0

    answer = input("¿Escuchaste el tono en el jack verde? [s/n]: ").strip().casefold()
    if answer in {"s", "si", "sí", "y", "yes"}:
        print("PASS: la salida del PCM2902 reproduce audio.")
        return 0

    print(
        "FAIL: no se confirmó el tono. Revisá el jack verde, los auriculares, "
        "el volumen/mute de Windows y que la app no esté ocupando la salida."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
