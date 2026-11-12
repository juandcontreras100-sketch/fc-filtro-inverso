"""Lectura de valores bajo el cursor del raton.

Una linea vertical sincronizada recorre **todas** las graficas a la vez y, en
cada curva, un punto con su nivel en dB. Ademas devuelve una etiqueta de texto
con la frecuencia y los niveles exactos.

Las series con marcador se declaran con ``series``: una lista de
``(indice_de_grafica, clave_del_espectro, color)``. Las claves que empiezan por
``_`` solo dibujan marcador y no aparecen en el texto de la etiqueta.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from src.ui import theme


class _Marker:
    """Punto y etiqueta de dB sobre una curva de una grafica."""

    def __init__(self, plot: pg.PlotItem, key: str, color: str) -> None:
        self.key = key
        self.plot = plot
        self.dot = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(theme.BACKGROUND, width=1),
            brush=pg.mkBrush(color),
        )
        self.label = pg.TextItem(color=color, anchor=(0.0, 1.0), fill=theme.BACKGROUND)
        self.label.setFont(theme.mono_font(8, bold=True))
        for item in (self.dot, self.label):
            item.setZValue(20)
            item.setVisible(False)
            plot.addItem(item)

    def show_at(self, x: float, y: float, text: str, flip: bool) -> None:
        self.dot.setData([x], [y])
        self.label.setText(text)
        # En la mitad derecha el texto se ancla por su derecha para no salirse.
        self.label.setAnchor((1.0, 1.0) if flip else (0.0, 1.0))
        self.label.setPos(x, y)
        self.dot.setVisible(True)
        self.label.setVisible(True)

    def hide(self) -> None:
        self.dot.setVisible(False)
        self.label.setVisible(False)


class CursorReadout:
    """Linea de cursor y marcadores de dB compartidos entre varios ``PlotItem``."""

    def __init__(
        self,
        plots: Sequence[pg.PlotItem],
        log_x: bool = False,
        series: Sequence[tuple[int, str, str]] = (),
    ) -> None:
        self._plots = list(plots)
        self._log_x = bool(log_x)
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

        self._markers = [
            _Marker(self._plots[plot_index], key, color)
            for plot_index, key, color in series
        ]
        self._frequencies = np.zeros(0)
        self._spectra: dict[str, np.ndarray] = {}
        self._units: dict[str, str] = {}

    def set_spectra(
        self,
        frequencies: np.ndarray,
        spectra: dict[str, np.ndarray],
        units: dict[str, str] | None = None,
    ) -> None:
        self._frequencies = frequencies
        self._spectra = spectra
        self._units = {} if units is None else units

    def handle_mouse(self, position: QtCore.QPointF) -> str | None:
        """Actualiza lineas y marcadores; devuelve el texto de lectura o ``None``."""
        for plot in self._plots:
            view = plot.getViewBox()
            if view.sceneBoundingRect().contains(position):
                point = view.mapSceneToView(position)
                return self._move_to(point.x(), view)
        self._hide_all()
        return None

    # -------------------------------------------------------------- internos
    def _move_to(self, x: float, view: object) -> str | None:
        frequency = 10.0 ** x if self._log_x else x
        for line in self._lines:
            line.setPos(x)
            line.setVisible(True)
        self._update_markers(x, frequency, view)
        return self._describe(frequency)

    def _update_markers(self, x: float, frequency: float, view: object) -> None:
        if self._frequencies.size == 0:
            self._hide_markers()
            return
        flip = self._in_right_half(x, view)
        for marker in self._markers:
            values = self._spectra.get(marker.key)
            if values is None or values.size != self._frequencies.size:
                marker.hide()
                continue
            level = float(np.interp(frequency, self._frequencies, values))
            unit = self._units.get(marker.key, "dB")
            marker.show_at(x, level, f" {level:+.1f} {unit} ", flip)

    @staticmethod
    def _in_right_half(x: float, view: object) -> bool:
        view_range = getattr(view, "viewRange", None)
        if view_range is None:
            return False
        low, high = view_range()[0]
        return x > (low + high) / 2.0

    def _hide_markers(self) -> None:
        for marker in self._markers:
            marker.hide()

    def _hide_all(self) -> None:
        for line in self._lines:
            line.setVisible(False)
        self._hide_markers()

    def _describe(self, frequency: float) -> str | None:
        if self._frequencies.size == 0:
            return None
        parts = [f"{frequency:,.0f} Hz".replace(",", ".")]
        for name, values in self._spectra.items():
            if name.startswith("_") or values.size != self._frequencies.size:
                continue
            level = float(np.interp(frequency, self._frequencies, values))
            unit = self._units.get(name, "dB")
            parts.append(f"{name} {level:+.1f} {unit}")
        return "  |  ".join(parts)
