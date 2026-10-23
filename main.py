"""Ecualizacion FFT full-duplex en tiempo real.

Ejemplos:
    python main.py
    python main.py --h-response medicion_H.npy --coherence medicion_gamma2.npy

``medicion_H.npy`` debe ser un vector complejo de ``fft_size // 2 + 1``
bins, medido a la misma frecuencia de muestreo que el dispositivo. Si no se
proporciona, el programa arranca en modo *bypass*: sirve para comprobar el
microfono, los altavoces y las graficas sin usar una correccion medida.

La interfaz completa esta en ``src/ui/`` y el motor de audio en
``src/audio/``; este archivo solo arranca todo y conecta los argumentos de
linea de comandos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ecualizador FFT full-duplex con espectros en vivo"
    )
    parser.add_argument("--sample-rate", type=float, default=48_000.0)
    parser.add_argument(
        "--fft-size", type=int, default=2_048, help="par; el salto es fft-size/2"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="canales del dispositivo full-duplex"
    )
    parser.add_argument("--device", help="indice o nombre de dispositivo sounddevice")
    parser.add_argument(
        "--h-response", type=Path, help="archivo .npy con la respuesta compleja H(f)"
    )
    parser.add_argument(
        "--coherence", type=Path, help="archivo .npy con gamma2(f); requiere --h-response"
    )
    parser.add_argument("--epsilon-base", type=float, default=0.02)
    parser.add_argument("--max-gain-db", type=float, default=12.0)
    return parser.parse_args()


def resolve_device(value: str | None) -> Any:
    """Un indice numerico se usa como indice; cualquier otra cosa, como nombre."""
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


def available_devices() -> list[tuple[Any, str]]:
    """Dispositivos que pueden abrir entrada y salida al mismo tiempo."""
    import sounddevice as sd

    found: list[tuple[Any, str]] = [(None, "Sistema (entrada y salida por defecto)")]
    for index, device in enumerate(sd.query_devices()):
        has_input = device["max_input_channels"] > 0
        has_output = device["max_output_channels"] > 0
        if not (has_input and has_output):
            continue
        kind = "duplex" if has_input and has_output else ("entrada" if has_input else "salida")
        found.append((index, f"{index}: {device['name']} ({kind})"))
    return found


def main() -> int:
    args = parse_args()

    try:
        import sounddevice as sd
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets  # noqa: F401 - asegura el binding de Qt
    except ModuleNotFoundError as error:
        print(
            "Falta una dependencia. Instale: "
            "pip install sounddevice pyqtgraph PySide6 numpy",
            file=sys.stderr,
        )
        print(error, file=sys.stderr)
        return 2

    if args.fft_size < 4 or args.fft_size % 2:
        print("--fft-size debe ser un entero par mayor o igual a 4", file=sys.stderr)
        return 2
    if args.channels < 1:
        print("--channels debe ser al menos 1", file=sys.stderr)
        return 2
    if args.coherence and not args.h_response:
        print("--coherence requiere tambien --h-response", file=sys.stderr)
        return 2

    from src.audio.engine import AudioEngine
    from src.ui.window import MainWindow

    engine = AudioEngine(
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        channels=args.channels,
        device=resolve_device(args.device),
    )
    engine.max_gain_db = args.max_gain_db
    engine.epsilon_base = args.epsilon_base

    if args.h_response:
        try:
            engine.load_calibration(args.h_response, args.coherence)
        except (OSError, ValueError) as error:
            print(f"No se pudo cargar la calibracion: {error}", file=sys.stderr)
            return 2
        engine.bypass = False
        engine.recompute_inverse()

    app = pg.mkQApp("Ecualizador FFT")
    window = MainWindow(engine)
    window.controls.populate_devices(available_devices())
    window.controls.set_device(engine.device)
    window.show()

    try:
        engine.open()
    except sd.PortAudioError as error:
        print(f"No se pudo abrir el stream de audio: {error}", file=sys.stderr)
        return 1

    try:
        return app.exec()
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
