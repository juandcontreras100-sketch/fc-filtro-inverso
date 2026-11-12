"""Prueba la entrada del PCM2902 con una fuente externa conocida.

Reproduce un tono de 1 kHz por la salida de auriculares de la laptop y lo
graba por la entrada USB PnP (PCM2902). El cable debe ir de la salida de
auriculares de la laptop al jack ROSA del PCM2902.

Uso:
    py probar_entrada_fuente.py
    py probar_entrada_fuente.py --output-device 5 --duration 5
    py probar_entrada_fuente.py --yes
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from probar_salida import build_tone
from verificar_entrada import DEFAULT_THRESHOLD_DB, analyze_capture


def _find_usb_pnp_input(sd: object) -> int:
    for index, device in enumerate(sd.query_devices()):
        if "USB PnP" in device["name"] and device["max_input_channels"] > 0:
            return index
    raise ValueError("No se encontró la entrada USB PnP del PCM2902")


def _find_laptop_output(sd: object, requested: str | None) -> int:
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
            raise ValueError(f"No se encontró una salida válida: {requested}")
        return index

    preferred = ("qualcomm", "speakers", "altavoces")
    for index, device in enumerate(devices):
        name = device["name"].casefold()
        if (
            device["max_output_channels"] > 0
            and "usb pnp" not in name
            and any(token in name for token in preferred)
        ):
            return index
    raise ValueError(
        "No se encontró la salida de la laptop. Usá --output-device con el índice."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-device",
        help="índice o parte del nombre de la salida de la laptop",
    )
    parser.add_argument("--frequency", type=float, default=1_000.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.01,
        help="amplitud lineal del tono; 0.01 equivale a 100 dB",
    )
    parser.add_argument("--threshold-db", type=float, default=DEFAULT_THRESHOLD_DB)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="no esperar Enter; el cable ya debe estar conectado",
    )
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except ImportError as error:
        print(f"No se pudo importar sounddevice: {error}", file=sys.stderr)
        return 2

    try:
        input_index = _find_usb_pnp_input(sd)
        output_index = _find_laptop_output(sd, args.output_device)
        input_device = sd.query_devices(input_index)
        output_device = sd.query_devices(output_index)
        sd.check_input_settings(
            device=input_index,
            samplerate=48_000,
            channels=1,
            dtype="float32",
        )
        sd.check_output_settings(
            device=output_index,
            samplerate=48_000,
            channels=2,
            dtype="float32",
        )
    except (ValueError, sd.PortAudioError) as error:
        print(f"No se pudo preparar la prueba: {error}", file=sys.stderr)
        return 2

    print("Prueba de entrada con fuente externa")
    print(f"  entrada {input_index}: {input_device['name']}")
    print(f"  salida  {output_index}: {output_device['name']}")
    print()
    print("1. Conectá el cable de la salida de auriculares de la laptop al jack ROSA.")
    print("2. Bajá el volumen de la laptop a ~30 %.")
    print("3. No conectes el TL072 ni la protoboard al PCM2902.")
    if not args.yes:
        input("Presiona Enter cuando esté conectado... ")

    try:
        tone = build_tone(48_000, args.duration + 1.0, args.frequency, args.amplitude)
    except ValueError as error:
        print(f"Parámetros inválidos: {error}", file=sys.stderr)
        return 2

    stereo = np.column_stack([tone, tone])
    try:
        sd.play(stereo, samplerate=48_000, device=output_index, blocking=False)
        time.sleep(0.5)
        recorded = sd.rec(
            int(round(args.duration * 48_000)),
            samplerate=48_000,
            channels=1,
            dtype="float32",
            device=input_index,
            blocking=True,
        )
    except sd.PortAudioError as error:
        print(f"Falló la prueba de audio: {error}", file=sys.stderr)
        print("Cerrá la app si está abierta y volvé a intentar.", file=sys.stderr)
        return 2
    finally:
        sd.stop()

    report = analyze_capture(
        np.asarray(recorded[:, 0], dtype=np.float64),
        sample_rate=48_000,
        threshold_db=args.threshold_db,
        tone_frequency=args.frequency,
    )

    print()
    print(f"RMS: {report['rms_db']:.2f} dB")
    print(f"Pico: {report['peak_db']:.2f} dB")
    print(
        f"Tono {args.frequency:.0f} Hz: {report['tone_db']:.2f} dB | "
        f"dominante {report['dominant_frequency_hz']:.1f} Hz | "
        f"SNR espectral {report['tone_snr_db']:.1f} dB"
    )
    print(f"{report['status']}: {report['message']}")

    if report["status"] == "PASS":
        print()
        print("PASS: la entrada del PCM2902 recibe el tono de la laptop.")
        print("El ADC y el jack rosa funcionan. El problema está en la salida")
        print("del PCM2902, en el cable verde o en el circuito.")
        return 0

    print()
    print("FAIL: la entrada del PCM2902 no recibe el tono de la laptop.")
    print("Revisá: cable de la laptop al jack rosa, volumen de la laptop,")
    print("nivel/mute de grabación de Windows y que el cable esté bien insertado.")
    print("Si hay saturación, repetí con --amplitude 0.005.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
