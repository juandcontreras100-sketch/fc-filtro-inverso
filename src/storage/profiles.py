"""Perfil del usuario y de sus dispositivos de audio.

El objetivo es recordar al usuario y sus dispositivos **por nombre**, porque
los indices de PortAudio cambian al reconectar USB o Bluetooth. El archivo es
tolerante: si falta o esta corrupto, la aplicacion arranca sin perfil.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILES_PATH = Path(__file__).resolve().parents[2] / "perfiles.json"

_DOCUMENT_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")
_FOLDER_PATTERN = re.compile(r"[^a-z0-9]+")


def _clean_text(value: Any) -> str:
    return " ".join(str(value).split())


def _folder_token(value: str) -> str:
    """Convierte texto a una sola pieza segura para un nombre de carpeta."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    normalized = _FOLDER_PATTERN.sub("_", without_accents.casefold())
    return normalized.strip("_")


@dataclass(frozen=True)
class UserProfile:
    """Usuario identificado por nombre y documento, sin contrasena."""

    name: str
    document: str
    input_device: str | None = None
    output_device: str | None = None

    def __post_init__(self) -> None:
        name = _clean_text(self.name)
        document = _clean_text(self.document)
        if not name:
            raise ValueError("El nombre es obligatorio")
        if not document:
            raise ValueError("El documento es obligatorio")
        if not _DOCUMENT_PATTERN.fullmatch(document):
            raise ValueError(
                "El documento solo admite letras, numeros, punto y guion"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "document", document)

    @property
    def folder_name(self) -> str:
        """Nombre de carpeta seguro: ``juan_perez_1234567``."""
        name = _folder_token(self.name) or "usuario"
        document = _folder_token(self.document) or "sin_documento"
        return f"{name}_{document}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "usuario": {"nombre": self.name, "documento": self.document},
            "dispositivos": {
                "entrada": self.input_device,
                "salida": self.output_device,
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> UserProfile | None:
        if not isinstance(data, dict):
            return None
        user = data.get("usuario")
        if not isinstance(user, dict):
            return None
        devices = data.get("dispositivos")
        if not isinstance(devices, dict):
            devices = {}
        try:
            return cls(
                name=user.get("nombre", ""),
                document=user.get("documento", ""),
                input_device=devices.get("entrada"),
                output_device=devices.get("salida"),
            )
        except (TypeError, ValueError):
            return None


class ProfileStore:
    """Lee y escribe ``perfiles.json`` sin romper el arranque."""

    def __init__(self, path: str | Path = PROFILES_PATH) -> None:
        self.path = Path(path)

    def load(self) -> UserProfile | None:
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return UserProfile.from_dict(data)

    def save(self, profile: UserProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(self.path)
