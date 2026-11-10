"""Calibra el circuito usando el par de audio USB PnP (PCM2902).

Uso:
    py calibrar_demo.py
    py calibrar_demo.py --amplitude 0.05 --output .\verificacion\circuito\medicion_H.npy
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_usb_pnp(kind: str) -> int:
    """Devuelve el primer dispositivo USB PnP con canales del tipo pedido."""
    import sounddevice as sd

    channel_key = f"max_{kind}_channels"
    for index, device in enumerate(sd.query_devices()):
        if "USB PnP" in device["name"] and device[channel_key] > 0:
            return index
    raise SystemExit(
        "No se encontro el dispositivo 'USB PnP Sound Device' con "
        f"{'entrada' if kind == 'input' else 'salida'} de audio. "
        "Conecta el PCM2902 por USB y volve a intentar."
    )


def main() -> None:
    import sounddevice as sd

    input_index = find_usb_pnp("input")
    output_index = find_usb_pnp("output")
    sd.default.device = [
        str(sd.query_devices(input_index)["name"]),
        str(sd.query_devices(output_index)["name"]),
    ]

    print("Calibracion con el par USB PnP:")
    print(f"  entrada {input_index}: {sd.query_devices(input_index)['name']}")
    print(f"  salida  {output_index}: {sd.query_devices(output_index)['name']}")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "calibrate_hardware.py"), run_name="__main__")


if __name__ == "__main__":
    main()
