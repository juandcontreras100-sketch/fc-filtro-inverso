"""Contexto fisico y colores de zonas para niveles SPL."""

from src.ui.theme import SPL_DANGER, SPL_SAFE, SPL_WARNING


def spl_context_text(level_spl: float) -> str:
    """Describe el nivel SPL con una referencia fisica breve."""
    if level_spl < 45:
        return "~30 dB: susurro"
    if level_spl < 70:
        return "~60 dB: conversacion normal"
    if level_spl < 88:
        return "~80 dB: calle con trafico"
    return "~90 dB o mas: grito"


def spl_zone_color(level_spl: float) -> str:
    """Devuelve el token de color correspondiente a la zona SPL."""
    if level_spl < 70:
        return SPL_SAFE
    if level_spl <= 85:
        return SPL_WARNING
    return SPL_DANGER
