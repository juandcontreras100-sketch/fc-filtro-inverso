"""Lectura de valores bajo el cursor del raton.

Une las curvas de entrada y salida con una linea vertical sincronizada y
devuelve una etiqueta con la frecuencia y los niveles exactos.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from src.ui import theme


class CursorReadout:
    """Linea de cursor compartida entre varios ``PlotItem``."""

    def __init__(self, plots: Sequence[pg.PlotItem]) -> None:
        self._plots = list(plots)
        self._lines: list[pg.InfiniteLine] = []
        for plot in self._plots:
            line = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(theme.CURSOR, width=1, style=QtCore.Qt.DashLine),
            )
            line.setVisible(False)
            plot.addItem(line)
            self._lines.append(line)

        self._frequencies = np.zeros(0)
        self._spectra: dict[str, np.ndarray] = {}

    def set_spectra(
        self,
        frequencies: np.ndarray,
        spectra: dict[str, np.ndarray],
    ) -> None:
        self._frequencies = frequencies
        self._spectra = spectra

    def handle_mouse(self, position: QtCore.QPointF) -> str | None:
        """Actualiza las lineas y devuelve el texto de lectura, o ``None``."""
        for plot, line in zip(self._plots, self._lines):
            view = plot.getViewBox()
            if view.sceneBoundingRect().contains(position):
                point = view.mapSceneToView(position)
                line.setPos(point.x())
                line.setVisible(True)
                return self._describe(point.x())
        for line in self._lines:
            line.setVisible(False)
        return None

    def _describe(self, frequency: float) -> str | None:
        if self._frequencies.size == 0:
            return None
        parts = [f"{frequency:,.0f} Hz".replace(",", ".")]
        for name, values in self._spectra.items():
            if values.size != self._frequencies.size:
                continue
            level = float(np.interp(frequency, self._frequencies, values))
            parts.append(f"{name} {level:+.1f} dB")
        return "  |  ".join(parts)
