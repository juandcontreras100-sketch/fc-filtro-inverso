"""Paleta, tipografia y hoja de estilos de la interfaz."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6 import QtGui

# Escala de grises oscuros con contraste progresivo (fondo -> panel -> control)
BACKGROUND = "#12141a"
PANEL = "#1a1d26"
SURFACE = "#232733"
SURFACE_HOVER = "#2c3242"
PANEL_EDGE = "#2e3444"
GRID = "#262b38"

# Tipografia y jerarquia de texto (contraste WCAG AA sobre BACKGROUND/PANEL)
TEXT = "#eef1f7"
MUTED = "#949db3"

# Acentos semanticos de senal y estado
INPUT = "#ff9f43"
INPUT_FILL = (255, 159, 67, 28)
OUTPUT = "#2ed573"
OUTPUT_FILL = (46, 213, 115, 28)
FILTER = "#5f7cfa"
CLIP = "#ff4757"
WARNING = "#ffa502"
CURSOR = "#f5f7fa"
PEAK = "#ffffff"

ACCENT = "#4c8dff"

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"


def mono_font(size: int = 9, bold: bool = False) -> QtGui.QFont:
    """Fuente monoespaciada para que los valores numericos no bailen."""
    font = QtGui.QFont(FONT_MONO, size)
    font.setStyleHint(QtGui.QFont.Monospace)
    font.setFixedPitch(True)
    font.setBold(bold)
    return font


def style_plot(plot: pg.PlotItem, title: str | None = None) -> None:
    """Aplica la paleta oscura y tipografia consistente a un PlotItem."""
    if title is not None:
        plot.setTitle(title, color=TEXT, size="10pt")
    axis_pen = pg.mkPen(PANEL_EDGE, width=1)
    text_pen = pg.mkPen(MUTED)
    for name in ("left", "bottom", "right"):
        axis = plot.getAxis(name)
        if axis is not None:
            axis.setPen(axis_pen)
            axis.setTextPen(text_pen)


# Ancho fijo de los ejes laterales: con el mismo ancho a izquierda y derecha, el
# recuadro de cada grafica mide lo mismo y sus titulos quedan alineados.
AXIS_WIDTH = 66


def reserve_right_axis(plot: pg.PlotItem) -> None:
    """Muestra un eje derecho sin marcas ni numeros, solo para reservar su ancho."""
    plot.showAxis("right")
    axis = plot.getAxis("right")
    axis.setStyle(showValues=False, tickLength=0)
    axis.setPen(pg.mkPen(PANEL_EDGE, width=1))


def fit_plot(plot: pg.PlotItem) -> None:
    """Quita holguras: ejes de ancho fijo, sin relleno interno ni margenes."""
    for name in ("left", "right"):
        axis = plot.getAxis(name)
        if axis is not None and axis.isVisible():
            axis.setWidth(AXIS_WIDTH)
    view = plot.getViewBox()
    if hasattr(view, "setDefaultPadding"):
        view.setDefaultPadding(0.0)
    plot.layout.setContentsMargins(0, 0, 0, 0)
    plot.layout.setSpacing(0)


def status_badge_style(state: str) -> str:
    """Estilo tipo pastilla (badge) segun el estado operativo del filtro."""
    palette: dict[str, tuple[str, str, str]] = {
        "active": ("#132a20", OUTPUT, OUTPUT),
        "warning": ("#2b2112", WARNING, WARNING),
        "bypass": (SURFACE, PANEL_EDGE, MUTED),
    }
    bg, border, fg = palette.get(state, palette["bypass"])
    return (
        f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
        "border-radius: 5px; padding: 3px 6px; font-weight: 600;"
    )


def stylesheet() -> str:
    """Hoja de estilos QSS global para toda la aplicacion."""
    return f"""
    QMainWindow, QWidget {{
        background-color: {BACKGROUND};
        color: {TEXT};
        font-family: "{FONT_UI}", sans-serif;
        font-size: 9pt;
    }}
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QGroupBox {{
        background-color: {PANEL};
        border: 1px solid {PANEL_EDGE};
        border-radius: 6px;
        margin-top: 16px;
        padding: 14px 6px 6px 6px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 4px;
        color: {ACCENT};
        background-color: transparent;
    }}
    QLabel {{
        background-color: transparent;
        color: {TEXT};
    }}
    QPushButton {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {PANEL_EDGE};
        border-radius: 5px;
        padding: 4px 8px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {SURFACE_HOVER};
        border-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background-color: {PANEL};
    }}
    QPushButton:disabled {{
        background-color: {PANEL};
        color: #555c6e;
        border-color: #222632;
    }}
    QPushButton#primaryButton {{
        background-color: #1c315e;
        border-color: {ACCENT};
        color: {TEXT};
    }}
    QPushButton#primaryButton:hover {{
        background-color: #254078;
    }}
    QPushButton#recordButton:checked {{
        background-color: #3a1820;
        border-color: {CLIP};
        color: {CLIP};
    }}
    QComboBox, QDoubleSpinBox {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {PANEL_EDGE};
        border-radius: 5px;
        padding: 2px 6px;
        min-height: 18px;
    }}
    QComboBox:hover, QDoubleSpinBox:hover {{
        border-color: {ACCENT};
    }}
    QComboBox QAbstractItemView {{
        background-color: {PANEL};
        color: {TEXT};
        selection-background-color: {SURFACE_HOVER};
        selection-color: {TEXT};
        border: 1px solid {PANEL_EDGE};
    }}
    QCheckBox {{
        background-color: transparent;
        spacing: 6px;
    }}
    QSlider {{
        background-color: transparent;
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {SURFACE};
        border: 1px solid {PANEL_EDGE};
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {FILTER};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {TEXT};
        border: 1px solid {ACCENT};
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSplitter::handle {{
        background-color: {PANEL_EDGE};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    QStatusBar {{
        background-color: {PANEL};
        color: {MUTED};
        border-top: 1px solid {PANEL_EDGE};
    }}
    QToolTip {{
        background-color: {PANEL};
        color: {TEXT};
        border: 1px solid {ACCENT};
        padding: 4px 6px;
    }}
    QScrollBar:vertical {{
        background: {BACKGROUND};
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {PANEL_EDGE};
        min-height: 24px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """


def apply_theme(target: Any) -> None:
    """Aplica la hoja de estilos oscura a una ventana o aplicacion Qt."""
    target.setStyleSheet(stylesheet())
