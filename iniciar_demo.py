"""Abre la interfaz usando el par de audio USB PnP (PCM2902).

Este lanzador evita depender de los dispositivos por defecto de Windows:
fuerza entrada y salida al PCM2902 y luego ejecuta ``main.py``.

Uso:
    py iniciar_demo.py
    py iniciar_demo.py --h-response medicion_H.npy --coherence medicion_gamma2.npy
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
    # Nombres, no índices: se resuelven al abrir el stream y sobreviven a
    # cambios de índice cuando Bluetooth aparece o desaparece.
    sd.default.device = [
        str(sd.query_devices(input_index)["name"]),
        str(sd.query_devices(output_index)["name"]),
    ]

    print("Audio de la demo:")
    print(f"  entrada {input_index}: {sd.query_devices(input_index)['name']}")
    print(f"  salida  {output_index}: {sd.query_devices(output_index)['name']}")
    print("Si la entrada queda en silencio, revisa el cableado del TL072 al PCM2902.")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
