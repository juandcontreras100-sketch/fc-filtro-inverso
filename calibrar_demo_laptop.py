"""Calibra el par acústico de la laptop (micrófono + parlantes).

El barrido se reproduce por los parlantes y se graba con el micrófono de la
laptop. Mide la respuesta combinada parlantes + ambiente + micrófono, que es
la cadena real de la demo de respaldo.

Los archivos se guardan en ``verificacion/laptop/`` para no pisar la medición
sintética de la raíz.

Uso:
    py calibrar_demo_laptop.py
    py calibrar_demo_laptop.py --amplitude 0.05 --sweeps 4
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "verificacion" / "laptop" / "medicion_H.npy"


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)

    import sounddevice as sd

    from iniciar_demo_laptop import resolve_laptop_pair

    devices = list(sd.query_devices())
    hostapis = list(sd.query_hostapis())
    input_index, output_index = resolve_laptop_pair(devices, hostapis)
    sd.default.device = [
        str(devices[input_index]["name"]),
        str(devices[output_index]["name"]),
    ]

    print("Calibración acústica con el par de la laptop:")
    print(f"  entrada {input_index}: {devices[input_index]['name']}")
    print(f"  salida  {output_index}: {devices[output_index]['name']}")
    print()
    print("Cerrá la app si está abierta y dejá la laptop en silencio.")
    print("El barrido se reproduce por los parlantes y se graba con el micrófono.")
    print("Empezá con volumen bajo para evitar acople.")
    print()

    if "--output" not in arguments:
        arguments.extend(["--output", str(DEFAULT_OUTPUT)])

    sys.argv = [str(ROOT / "calibrate_hardware.py"), *arguments]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "calibrate_hardware.py"), run_name="__main__")


if __name__ == "__main__":
    main()
