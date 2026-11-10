"""Abre la interfaz usando el micrófono y los parlantes de la laptop.

Este lanzador evita depender de los dispositivos por defecto de Windows:
fuerza por nombre el micrófono y los parlantes de la laptop, y luego ejecuta
``main.py``. Sirve como demo de respaldo cuando el PCM2902 o la protoboard no
están verificados.

Los índices de ``sounddevice`` cambian cuando Bluetooth se conecta o
desconecta; por eso se resuelven por nombre y no se hardcodean.

Uso:
    py iniciar_demo_laptop.py
    py iniciar_demo_laptop.py --check
    py iniciar_demo_laptop.py --h-response medicion_H.npy --coherence medicion_gamma2.npy
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent

MIC_TOKENS = ("microphone array", "micrófono")
SPEAKER_TOKENS = ("speakers (qualcomm", "speakers", "altavoces")
EXCLUDED_TOKENS = ("usb pnp", "bluetooth", "huawei", "freebuds")


def _contains_any(name: str, tokens: Sequence[str]) -> bool:
    normalized = name.casefold()
    return any(token in normalized for token in tokens)


def find_device(
    devices: Sequence[dict[str, Any]],
    hostapis: Sequence[dict[str, Any]],
    *,
    kind: str,
    name_tokens: Sequence[str],
    excluded_tokens: Sequence[str] = EXCLUDED_TOKENS,
) -> int | None:
    """Devuelve el índice del mejor dispositivo cuyo nombre coincide.

    Descarta el PCM2902 y cualquier auricular Bluetooth. Prefiere el host API
    MME y, dentro de él, el orden de ``name_tokens``. Devuelve ``None`` si no
    hay coincidencias.
    """
    channel_key = f"max_{kind}_channels"
    best: tuple[int, int, int] | None = None
    for index, device in enumerate(devices):
        if int(device[channel_key]) < 1:
            continue
        name = str(device["name"])
        if _contains_any(name, excluded_tokens):
            continue
        position = next(
            (
                order
                for order, token in enumerate(name_tokens)
                if token in name.casefold()
            ),
            None,
        )
        if position is None:
            continue
        api_name = str(hostapis[int(device["hostapi"])]["name"]).casefold()
        candidate = (0 if api_name == "mme" else 1, position, index)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]


def has_bluetooth_output(devices: Sequence[dict[str, Any]]) -> bool:
    """Indica si Windows todavía expone una salida Bluetooth."""
    return any(
        int(device["max_output_channels"]) > 0
        and _contains_any(str(device["name"]), ("bluetooth", "huawei", "freebuds"))
        for device in devices
    )


def resolve_laptop_pair(
    devices: Sequence[dict[str, Any]],
    hostapis: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Resuelve (micrófono, parlantes) de la laptop por nombre."""
    microphone = find_device(
        devices, hostapis, kind="input", name_tokens=MIC_TOKENS
    )
    speaker = find_device(
        devices, hostapis, kind="output", name_tokens=SPEAKER_TOKENS
    )
    if microphone is None or speaker is None:
        missing = []
        if microphone is None:
            missing.append("micrófono de la laptop")
        if speaker is None:
            missing.append("parlantes de la laptop")
        raise SystemExit(
            "No se encontró: "
            + " ni ".join(missing)
            + ". Verificá que el audio de la laptop esté habilitado en Windows."
        )
    return microphone, speaker


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check" in arguments
    if check_only:
        arguments.remove("--check")

    import sounddevice as sd

    devices = list(sd.query_devices())
    hostapis = list(sd.query_hostapis())
    input_index, output_index = resolve_laptop_pair(devices, hostapis)

    # Se guardan los nombres, no los índices: PortAudio los resuelve al abrir
    # el stream. Así un cambio de índices por Bluetooth no desvía el audio.
    sd.default.device = [
        str(devices[input_index]["name"]),
        str(devices[output_index]["name"]),
    ]

    print("Audio de la demo (laptop):")
    print(f"  entrada {input_index}: {devices[input_index]['name']}")
    print(f"  salida  {output_index}: {devices[output_index]['name']}")
    if has_bluetooth_output(devices):
        print(
            "ADVERTENCIA: Windows todavía ve una salida Bluetooth. "
            "Apagá esos auriculares para que no se roben el audio."
        )
    print("En la app, dejá 'Interfaz' en 'Sistema' y FFT en 2048.")

    if check_only:
        print("--check: no se abre la interfaz.")
        return

    sys.argv = [str(ROOT / "main.py"), *arguments]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
