"""Limitador suave para la salida, antes del DAC.

Con ganancias altas del filtro inverso la salida puede superar +/-1.0 y el
driver la recorta de golpe (distorsion dura y fuerte en los audifonos). Este
limitador deja pasar sin cambios todo lo que este por debajo de ``threshold`` y
comprime con una curva tanh lo que lo supera, de modo que la salida nunca pase
de ``ceiling``.

La curva es impar, monotona, continua y con pendiente 1 en el umbral, asi que
no introduce saltos ni invierte la forma de onda.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.8
DEFAULT_CEILING = 0.98


def soft_limit(
    samples: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    ceiling: float = DEFAULT_CEILING,
) -> np.ndarray:
    """Devuelve ``samples`` comprimidos suavemente para que |y| <= ``ceiling``."""
    if not 0.0 < threshold < ceiling <= 1.0:
        raise ValueError("se requiere 0 < threshold < ceiling <= 1")

    values = np.asarray(samples, dtype=np.float64)
    magnitude = np.abs(values)
    over = magnitude > threshold
    if not np.any(over):
        return values

    knee = ceiling - threshold
    limited = values.copy()
    limited[over] = np.sign(values[over]) * (
        threshold + knee * np.tanh((magnitude[over] - threshold) / knee)
    )
    return limited
