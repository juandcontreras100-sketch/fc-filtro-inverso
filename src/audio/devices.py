"""Seleccion de dispositivos de audio, sin depender de PortAudio.

En Windows la entrada y la salida de una misma interfaz (por ejemplo el
ESP32-S3 como tarjeta de sonido USB) aparecen casi siempre como **dos
dispositivos separados**: "Microphone (X)" con solo entrada y "Speakers (X)"
con solo salida. Un filtro que exija entrada *y* salida en el mismo dispositivo
no encuentra nada. Aqui se emparejan por el nombre que comparten y se ofrecen
como ``(indice_entrada, indice_salida)``, que es lo que ``sd.Stream`` acepta.

``build_device_options`` recibe las listas ya consultadas, asi se puede probar
con datos de cualquier PC.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DEFAULT_LABEL = "Sistema (entrada y salida por defecto)"
DEFAULT_INPUT_LABEL = "Entrada por defecto del sistema"
DEFAULT_OUTPUT_LABEL = "Salida por defecto del sistema"

# Orden de preferencia: WASAPI y MME toleran bien los cambios de formato.
_HOST_API_ORDER = ("WASAPI", "MME", "DIRECTSOUND", "ASIO")

# WDM-KS no se ofrece automaticamente: expone cada endpoint varias veces
# (Output 1, Output 2...) y es la API mas delicada con sounddevice. Si hiciera
# falta, se puede pedir a mano con --input-device y --output-device.
_EXCLUDED_HOST_APIS = ("WDM-KS",)

# MME corta los nombres a 31 caracteres; un nombre mas corto no esta truncado.
_MME_NAME_LIMIT = 31

# Prefijo comun minimo (en caracteres) para fiarse de un emparejamiento cuando
# los nombres no son identicos ni uno es el truncado del otro.
_MIN_COMMON_PREFIX = 12


def parse_device(value: str | None) -> int | str | None:
    """Un indice numerico se usa como indice; cualquier otra cosa, como nombre."""
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


def combine_device_arg(
    device: str | None,
    input_device: str | None = None,
    output_device: str | None = None,
) -> Any:
    """Construye el argumento ``device`` de ``sd.Stream`` desde la linea de comandos."""
    if device is not None and (input_device is not None or output_device is not None):
        raise ValueError("--device no se puede combinar con --input-device/--output-device")
    if device is not None:
        return parse_device(device)
    if input_device is None and output_device is None:
        return None
    return (parse_device(input_device), parse_device(output_device))


def _pairing_key(name: str) -> str:
    """Parte del nombre que identifica a la interfaz: el texto tras el primer ' ('.

    ``"Speakers (ESP32-S3 USB Audio)"`` -> ``"esp32-s3 usb audio"``. Los nombres
    MME vienen truncados a 31 caracteres y los de WASAPI repiten la interfaz con
    sufijos distintos, asi que se compara por prefijo comun.
    """
    _head, separator, tail = name.partition(" (")
    if not separator:
        return ""
    return tail.rstrip(") ").strip().lower()


def _same_interface(first: str, second: str) -> bool:
    key_a, key_b = _pairing_key(first), _pairing_key(second)
    shortest = min(len(key_a), len(key_b))
    if shortest == 0:
        return False
    common = 0
    for char_a, char_b in zip(key_a, key_b):
        if char_a != char_b:
            break
        common += 1

    if common == shortest:
        # Una clave es prefijo de la otra: iguales, o la corta es un nombre
        # truncado por MME. Si no parece truncado ("Speaker" frente a
        # "Speaker Headset") son interfaces distintas.
        if len(key_a) == len(key_b):
            return shortest >= 4
        shorter_name = first if len(key_a) < len(key_b) else second
        return len(shorter_name) >= _MME_NAME_LIMIT - 1 and shortest >= 4

    # Divergen a mitad de nombre (sufijos de WASAPI): vale con un prefijo largo.
    return common >= _MIN_COMMON_PREFIX


def _api_rank(host_name: str) -> int:
    upper = host_name.upper()
    for rank, token in enumerate(_HOST_API_ORDER):
        if token in upper:
            return rank
    return len(_HOST_API_ORDER)


def build_device_options(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, str]]:
    """Opciones ``(valor, etiqueta)`` para el combo de interfaces.

    Siempre empieza por el dispositivo del sistema (``None``). Despues van los
    dispositivos full-duplex reales y las parejas entrada+salida emparejadas por
    nombre dentro de la misma API de audio.
    """
    options: list[tuple[int | tuple[int, int], str, int, int]] = []

    def host_name(device: Mapping[str, Any]) -> str:
        return str(hostapis[int(device["hostapi"])]["name"])

    def is_offered(device: Mapping[str, Any]) -> bool:
        upper = host_name(device).upper()
        return not any(token in upper for token in _EXCLUDED_HOST_APIS)

    for index, device in enumerate(devices):
        if not is_offered(device):
            continue
        if device["max_input_channels"] > 0 and device["max_output_channels"] > 0:
            api = host_name(device)
            label = f"{index}: {device['name']} (duplex, {api})"
            options.append((index, label, _api_rank(api), index))

    inputs = [
        (i, d) for i, d in enumerate(devices)
        if is_offered(d) and d["max_input_channels"] > 0 and d["max_output_channels"] == 0
    ]
    outputs = [
        (i, d) for i, d in enumerate(devices)
        if is_offered(d) and d["max_output_channels"] > 0 and d["max_input_channels"] == 0
    ]
    for in_index, in_device in inputs:
        for out_index, out_device in outputs:
            if in_device["hostapi"] != out_device["hostapi"]:
                continue
            if not _same_interface(str(in_device["name"]), str(out_device["name"])):
                continue
            api = host_name(in_device)
            label = (
                f"{in_index}+{out_index}: {in_device['name']} / "
                f"{out_device['name']} ({api})"
            )
            options.append(((in_index, out_index), label, _api_rank(api), in_index))

    options.sort(key=lambda option: (option[2], option[3]))
    return [(None, DEFAULT_LABEL)] + [(value, label) for value, label, _rank, _i in options]


def _device_host_name(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
    index: int,
) -> str:
    return str(hostapis[int(devices[index]["hostapi"])]["name"])


def _is_offered(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
    index: int,
) -> bool:
    upper = _device_host_name(devices, hostapis, index).upper()
    return not any(token in upper for token in _EXCLUDED_HOST_APIS)


def _device_option(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
    index: int,
) -> tuple[int, str, int, int]:
    api = _device_host_name(devices, hostapis, index)
    label = f"{index}: {devices[index]['name']} ({api})"
    return index, label, _api_rank(api), index


def _valid_index(devices: Sequence[Mapping[str, Any]], index: Any) -> bool:
    return isinstance(index, int) and 0 <= index < len(devices)


def build_input_options(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, str]]:
    """Opciones de microfono: solo dispositivos con canales de entrada.

    Entrada y salida deben vivir en la misma API para abrir un unico
    ``sd.Stream``; la salida se filtra con ``build_output_options``.
    """
    options = [
        _device_option(devices, hostapis, index)
        for index, device in enumerate(devices)
        if _is_offered(devices, hostapis, index)
        and int(device["max_input_channels"]) > 0
    ]
    options.sort(key=lambda option: (option[2], option[3]))
    return [(None, DEFAULT_INPUT_LABEL)] + [
        (value, label) for value, label, _rank, _index in options
    ]


def build_output_options(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
    input_index: int | None,
) -> list[tuple[Any, str]]:
    """Opciones de altavoz limitadas a la API del microfono elegido."""
    allowed_api = None
    if _valid_index(devices, input_index):
        allowed_api = int(devices[input_index]["hostapi"])

    options = []
    for index, device in enumerate(devices):
        if not _is_offered(devices, hostapis, index):
            continue
        if int(device["max_output_channels"]) <= 0:
            continue
        if allowed_api is not None and int(device["hostapi"]) != allowed_api:
            continue
        options.append(_device_option(devices, hostapis, index))
    options.sort(key=lambda option: (option[2], option[3]))
    return [(None, DEFAULT_OUTPUT_LABEL)] + [
        (value, label) for value, label, _rank, _index in options
    ]


def device_name(
    devices: Sequence[Mapping[str, Any]],
    index: Any,
) -> str | None:
    """Nombre estable de un dispositivo: nunca devuelve indices al perfil."""
    if index is None:
        return None
    if isinstance(index, str):
        return index
    if not _valid_index(devices, index):
        return None
    return str(devices[index]["name"])


def find_device_by_name(
    devices: Sequence[Mapping[str, Any]],
    name: str | None,
) -> int | None:
    """Busca un dispositivo por nombre para sobrevivir a cambios de indice."""
    if not name:
        return None
    target = str(name).strip().casefold()
    for index, device in enumerate(devices):
        if str(device["name"]).strip().casefold() == target:
            return index
    return None


def _host_api_default(
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
    api_index: int,
    key: str,
    *,
    needs_input: bool = False,
    needs_output: bool = False,
) -> int | None:
    """Devuelve el dispositivo por defecto de una API si sirve para ese rol."""
    try:
        default = int(hostapis[api_index].get(key, -1))
    except (IndexError, TypeError, ValueError):
        return None
    if not _valid_index(devices, default) or not _is_offered(devices, hostapis, default):
        return None
    device = devices[default]
    if needs_input and int(device["max_input_channels"]) <= 0:
        return None
    if needs_output and int(device["max_output_channels"]) <= 0:
        return None
    return default


def device_pair(
    input_index: int | None,
    output_index: int | None,
    devices: Sequence[Mapping[str, Any]],
    hostapis: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int | None] | None:
    """Construye el ``device`` de ``sd.Stream`` manteniendo la misma API.

    Si un lado queda en "por defecto del sistema", se resuelve al dispositivo
    por defecto de la API del otro lado. Asi el par nunca cruza MME/WASAPI.
    """
    input_valid = _valid_index(devices, input_index)
    output_valid = _valid_index(devices, output_index)
    if not input_valid and not output_valid:
        return None
    if input_valid and output_valid:
        return (input_index, output_index)
    if input_valid:
        api_index = int(devices[input_index]["hostapi"])
        return (
            input_index,
            _host_api_default(
                devices,
                hostapis,
                api_index,
                "default_output_device",
                needs_output=True,
            ),
        )
    api_index = int(devices[output_index]["hostapi"])
    return (
        _host_api_default(
            devices,
            hostapis,
            api_index,
            "default_input_device",
            needs_input=True,
        ),
        output_index,
    )


def available_devices() -> list[tuple[Any, str]]:
    """Consulta PortAudio y devuelve las opciones para el combo."""
    import sounddevice as sd

    return build_device_options(list(sd.query_devices()), list(sd.query_hostapis()))


def describe_devices() -> str:
    """Texto de diagnostico: todos los dispositivos y las parejas detectadas."""
    import sounddevice as sd

    devices = list(sd.query_devices())
    hostapis = list(sd.query_hostapis())
    lines = ["Dispositivos (indice: nombre [API] entradas/salidas):"]
    for index, device in enumerate(devices):
        api = hostapis[int(device["hostapi"])]["name"]
        lines.append(
            f"  {index:>3}: {device['name']} [{api}] "
            f"{device['max_input_channels']} in / {device['max_output_channels']} out"
        )
    lines.append("")
    lines.append("Opciones que ofrecera la aplicacion:")
    for value, label in build_device_options(devices, hostapis):
        lines.append(f"  {value!s:>10}  {label}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_devices())
