"""Persistencia de la calibracion SPL, sin dependencias de audio ni Qt."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "calibracion_spl.json"


@dataclass(frozen=True)
class SplCalibration:
    offset_db: float
    date: str
    device_name: str


def _invalid(path: Path, detail: str) -> ValueError:
    return ValueError(f"Calibracion SPL invalida en {path}: {detail}")


def load_spl_calibration(
    path: str | Path = CALIBRATION_PATH,
) -> SplCalibration | None:
    """Carga una calibracion; un archivo ausente significa sin calibrar."""
    calibration_path = Path(path)
    try:
        text = calibration_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeError as error:
        raise _invalid(calibration_path, "el archivo no es UTF-8") from error

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise _invalid(calibration_path, "JSON mal formado") from error

    if not isinstance(data, dict):
        raise _invalid(calibration_path, "la raiz debe ser un objeto")
    missing = {"offset_db", "date", "device_name"} - data.keys()
    if missing:
        raise _invalid(
            calibration_path, f"faltan campos: {', '.join(sorted(missing))}"
        )

    offset_db = data["offset_db"]
    if (
        isinstance(offset_db, bool)
        or not isinstance(offset_db, (int, float))
        or not math.isfinite(offset_db)
    ):
        raise _invalid(calibration_path, "offset_db debe ser un numero finito")

    date = data["date"]
    if not isinstance(date, str) or not date.strip():
        raise _invalid(calibration_path, "date debe ser una fecha ISO 8601 con zona UTC")
    try:
        parsed_date = datetime.fromisoformat(
            date[:-1] + "+00:00" if date.endswith("Z") else date
        )
    except ValueError as error:
        raise _invalid(
            calibration_path, "date debe ser una fecha ISO 8601 con zona UTC"
        ) from error
    if parsed_date.tzinfo is None or parsed_date.utcoffset() != timedelta(0):
        raise _invalid(calibration_path, "date debe incluir la zona UTC")

    device_name = data["device_name"]
    if not isinstance(device_name, str) or not device_name.strip():
        raise _invalid(calibration_path, "device_name debe ser texto no vacio")

    return SplCalibration(float(offset_db), date, device_name)


def save_spl_calibration(
    offset_db: float,
    device_name: str,
    path: str | Path = CALIBRATION_PATH,
) -> SplCalibration:
    """Guarda y devuelve una calibracion fechada en UTC."""
    if (
        isinstance(offset_db, bool)
        or not isinstance(offset_db, (int, float))
        or not math.isfinite(offset_db)
    ):
        raise ValueError("offset_db debe ser un numero finito")
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("device_name debe ser texto no vacio")

    calibration = SplCalibration(
        offset_db=float(offset_db),
        date=datetime.now(timezone.utc).isoformat(),
        device_name=device_name,
    )
    payload = {
        "offset_db": calibration.offset_db,
        "date": calibration.date,
        "device_name": calibration.device_name,
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return calibration


def input_device_name(device: Any) -> str:
    """Resuelve el nombre de la entrada, tambien para un par full-duplex."""
    import sounddevice as sd

    if isinstance(device, (tuple, list)):
        device = device[0]
    info = sd.query_devices(kind="input") if device is None else sd.query_devices(device)
    return info["name"]
