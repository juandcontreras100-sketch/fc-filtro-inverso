"""Prueba de loopback del PCM2902: salida verde -> entrada rosa.

Sirve para separar dos problemas:
- Si el loopback devuelve el tono de 1 kHz, el PCM2902, sus jacks y Windows
  están bien; el problema está en el circuito (micrófono/TL072) o en el cable
  que llega al PCM2902.
- Si el loopback queda en silencio, el problema está en el PCM2902, el jack,
  el cable o el nivel/mute de grabación de Windows.

Uso:
    py probar_loopback.py
    py probar_loopback.py --amplitude 0.005
    py probar_loopback.py --yes
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from probar_salida import build_tone
from verificar_entrada import DEFAULT_THRESHOLD_DB, analyze_capture


def find_usb_pnp(kind: str) -> int:
    import sounddevice as sd

    channel_key = f"max_{kind}_channels"
    for index, device in enumerate(sd.query_devices()):
        if "USB PnP" in device["name"] and device[channel_key] > 0:
            return index
    raise SystemExit(
        "No se encontro el dispositivo 'USB PnP Sound Device' con "
        f"{'entrada' if kind == 'input' else 'salida'} de audio."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=1_000.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.02,
        help="amplitud lineal del tono; 0.02 equivale a 106 dB",
    )
    parser.add_argument("--threshold-db", type=float, default=DEFAULT_THRESHOLD_DB)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="no esperar Enter; el cable ya debe estar conectado",
    )
    args = parser.parse_args(argv)

    import sounddevice as sd

    input_index = find_usb_pnp("input")
    output_index = find_usb_pnp("output")
    sd.default.device = [input_index, output_index]

    print("Prueba de loopback del PCM2902")
    print(f"  entrada {input_index}: {sd.query_devices(input_index)['name']}")
    print(f"  salida  {output_index}: {sd.query_devices(output_index)['name']}")
    print()
    print("1. Cerra la ventana de la app si esta abierta.")
    print("2. Conecta un cable 3.5 mm macho-macho de la salida VERDE a la entrada ROSA.")
    print("3. Deja el volumen de salida moderado (~50 %) para no saturar la entrada.")
    if not args.yes:
        input("Presiona Enter cuando este conectado... ")

    rate = 48_000
    try:
        tone = build_tone(rate, args.duration, args.frequency, args.amplitude)
    except ValueError as error:
        print(f"Parametros invalidos: {error}", file=sys.stderr)
        return 2

    try:
        recorded = sd.playrec(
            tone[:, np.newaxis],
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocking=True,
        )
    except sd.PortAudioError as error:
        print(f"No se pudo abrir el stream: {error}", file=sys.stderr)
        print("Cerra la app si esta abierta y volve a intentar.", file=sys.stderr)
        return 2

    report = analyze_capture(
        np.asarray(recorded[:, 0], dtype=np.float64),
        sample_rate=rate,
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
        print("PASS: el PCM2902 recibe el tono del loopback.")
        print("La interfaz, los jacks y Windows estan bien. El problema esta en")
        print("el circuito (microfono/TL072) o en el cable que llega al PCM2902.")
        return 0

    print()
    print("FAIL: el PCM2902 no recibe el tono del loopback.")
    print("Revisa: cable verde->rosa, jack rosa, nivel/mute de grabacion de Windows,")
    print("formato del endpoint y que el volumen de salida no este en cero.")
    print("Si hay saturacion, repeti con --amplitude 0.005.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
