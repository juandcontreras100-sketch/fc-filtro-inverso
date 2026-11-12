"""Usuario de la sesion (nombre y documento) y sus dispositivos preferidos.

El usuario se identifica al abrir el programa. Cada grabacion se guarda en una
carpeta unica por usuario, ``sesiones/<nombre>_<documento>/``, que acumula todas
sus grabaciones. Los dispositivos se recuerdan por **nombre**, porque los
indices de PortAudio cambian (Bluetooth, USB).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_FILE = "perfiles.json"


@dataclass(frozen=True)
class UserProfile:
    name: str
    document: str

    def __post_init__(self) -> None:
        name = " ".join(self.name.split())
        document = "".join(self.document.split())
        if not name:
            raise ValueError("El nombre es obligatorio")
        if not document:
            raise ValueError("El numero de documento es obligatorio")
        if not re.fullmatch(r"[0-9A-Za-z.\-]+", document):
            raise ValueError("El documento solo admite letras, numeros, punto y guion")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "document", document)

    @property
    def folder_name(self) -> str:
        """Nombre de carpeta seguro: ``juan_perez_1234567890``."""
        ascii_name = (
            unicodedata.normalize("NFKD", self.name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "usuario"
        document = re.sub(r"[^0-9A-Za-z]+", "", self.document)
        return f"{slug}_{document}"

    def as_dict(self) -> dict[str, str]:
        return {"nombre": self.name, "documento": self.document}


class ProfileStore:
    """Ultimo usuario y dispositivos usados, en ``perfiles.json``."""

    def __init__(self, path: str | Path = PROFILE_FILE) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def last_user(self) -> UserProfile | None:
        data = self.load()
        try:
            return UserProfile(str(data["nombre"]), str(data["documento"]))
        except (KeyError, ValueError):
            return None

    def last_devices(self) -> tuple[str | None, str | None]:
        data = self.load()
        mic, speaker = data.get("microfono"), data.get("altavoz")
        return (
            mic if isinstance(mic, str) else None,
            speaker if isinstance(speaker, str) else None,
        )

    def save(
        self,
        user: UserProfile,
        microphone: str | None,
        speaker: str | None,
    ) -> None:
        data = {**user.as_dict(), "microfono": microphone, "altavoz": speaker}
        try:
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass  # recordar al usuario es una comodidad, nunca debe romper el arranque
