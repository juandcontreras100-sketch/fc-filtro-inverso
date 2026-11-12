"""Lectura de valores bajo el cursor del raton.

Comparte una linea vertical entre varias graficas y, para cada curva visible,
dibuja un marcador y una etiqueta con el nivel en la unidad correspondiente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from src.ui import theme

# Separacion de la etiqueta respecto del marcador, en pixeles de pantalla.
_LABEL_GAP_X = 9.0
_LABEL_GAP_Y = 12.0
_LABEL_FILL = (18, 20, 26, 220)


@dataclass
class _CursorCurve:
    """Elementos de cursor que pertenecen a una curva concreta."""

    name: str
    color: str
    plot: pg.PlotItem
    view: Any
    marker: pg.ScatterPlotItem
    label: pg.TextItem
    # Clave en ``set_spectra``; permite que una curva muestre otra unidad.
    data_key: str


class CursorReadout:
    """Linea de cursor compartida entre varios ``PlotItem``.

    ``curves`` permite indicar, por grafica, que espectros se dibujan y de que
    color. Si se omite, solo se mueve la linea y se mantiene el texto de
    lectura, que es lo que necesitan las pruebas con plots falsos.
    """

    def __init__(
        self,
        plots: Sequence[pg.PlotItem],
        log_x: bool = False,
        curves: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self._plots = list(plots)
        self._log_x = bool(log_x)
        self._lines: list[pg.InfiniteLine] = []
        self._curves: list[_CursorCurve] = []

        for index, plot in enumerate(self._plots):
            line = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(theme.CURSOR, width=1, style=QtCore.Qt.DashLine),
            )
            line.setVisible(False)
            plot.addItem(line)
            self._lines.append(line)

            specs = [] if curves is None or index >= len(curves) else curves[index]
            for spec in specs:
                self._add_curve(plot, spec)

        self._frequencies = np.zeros(0)
        self._labels: np.ndarray | None = None
        self._spectra: dict[str, np.ndarray] = {}
        self._units: dict[str, str] = {}
        self._names: dict[str, str] = {}

    def _add_curve(self, plot: pg.PlotItem, spec: Mapping[str, Any]) -> None:
        name = str(spec["name"])
        data_key = str(spec.get("data", name))
        color = str(spec.get("color", theme.CURSOR))
        view = spec.get("view") or plot.getViewBox()

        marker = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(theme.BACKGROUND, width=1),
            brush=pg.mkBrush(color),
        )
        marker.setZValue(100)
        marker.setVisible(False)
        view.addItem(marker, ignoreBounds=True)

        label = pg.TextItem(
            "",
            color=color,
            anchor=(1.0, 0.5),
            fill=pg.mkBrush(*_LABEL_FILL),
            border=pg.mkPen(color, width=1),
        )
        label.setZValue(101)
        label.setVisible(False)
        view.addItem(label, ignoreBounds=True)

        self._curves.append(
            _CursorCurve(
                name=name,
                color=color,
                plot=plot,
                view=view,
                marker=marker,
                label=label,
                data_key=data_key,
            )
        )

    def set_spectra(
        self,
        frequencies: np.ndarray,
        spectra: dict[str, np.ndarray],
        units: dict[str, str] | None = None,
        labels: np.ndarray | None = None,
        names: dict[str, str] | None = None,
    ) -> None:
        """Actualiza los datos que se leen al mover el cursor.

        ``labels`` se usa en la vista de bandas: el eje X son indices y esta
        matriz traduce cada indice a su frecuencia central en Hz.
        """
        self._frequencies = np.asarray(frequencies, dtype=np.float64)
        self._spectra = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in spectra.items()
        }
        self._units = {} if units is None else units
        self._names = {} if names is None else names
        self._labels = None if labels is None else np.asarray(labels, dtype=np.float64)

    def handle_mouse(self, position: QtCore.QPointF) -> str | None:
        """Actualiza lineas, marcadores y etiquetas; devuelve el texto de lectura."""
        target = None
        for plot, line in zip(self._plots, self._lines):
            view = plot.getViewBox()
            if view.sceneBoundingRect().contains(position):
                target = (plot, line, view)
                break

        if target is None:
            self.hide()
            return None

        _plot, _line, target_view = target
        point = target_view.mapSceneToView(position)
        x = float(point.x())
        frequency = self._frequency_at(x)

        for line in self._lines:
            line.setPos(x)
            line.setVisible(True)

        for curve in self._curves:
            values = self._spectra.get(curve.data_key)
            if values is None:
                # Sin datos propios se usa la serie del mismo nombre.
                values = self._spectra.get(curve.name)
            if values is None or values.size == 0:
                curve.marker.setVisible(False)
                curve.label.setVisible(False)
                continue

            y = self._value_at(frequency, values)
            if y is None:
                curve.marker.setVisible(False)
                curve.label.setVisible(False)
                continue

            curve.marker.setData([x], [y])
            curve.marker.setVisible(True)
            self._place_label(curve, x, y)

        return self._describe(frequency)

    def hide(self) -> None:
        """Oculta lineas, marcadores y etiquetas sin borrar los datos."""
        for line in self._lines:
            line.setVisible(False)
        for curve in self._curves:
            curve.marker.setVisible(False)
            curve.label.setVisible(False)

    # Alias usado por la vista de barras, que no dibuja cursor.
    clear = hide

    def _frequency_at(self, x: float) -> float:
        if self._labels is not None and self._labels.size:
            index = int(round(x))
            index = max(0, min(index, self._labels.size - 1))
            return float(self._labels[index])
        if self._log_x:
            return float(10.0 ** x)
        return float(x)

    def _value_at(self, frequency: float, values: np.ndarray) -> float | None:
        domain = self._labels if self._labels is not None else self._frequencies
        if domain.size == 0 or values.size != domain.size:
            return None
        return float(np.interp(frequency, domain, values))

    def _place_label(self, curve: _CursorCurve, x: float, y: float) -> None:
        """Posiciona la etiqueta a un lado del punto; la voltea en la mitad derecha."""
        view_range = curve.view.viewRange()[0]
        right_half = x >= (view_range[0] + view_range[1]) / 2.0
        curve.label.setAnchor((0.0, 0.5) if right_half else (1.0, 0.5))

        try:
            pixel_width, pixel_height = curve.view.viewPixelSize()
        except Exception:  # pragma: no cover - plots falsos sin transformacion
            pixel_width = pixel_height = 0.0
        direction = 1.0 if right_half else -1.0
        label_x = x + direction * _LABEL_GAP_X * pixel_width
        label_y = y + _LABEL_GAP_Y * pixel_height
        curve.label.setPos(label_x, label_y)

        unit = self._units.get(curve.data_key) or self._units.get(curve.name, "dB")
        curve.label.setText(f"{y:+.1f} {unit}")
        curve.label.setVisible(True)

    def _describe(self, frequency: float) -> str | None:
        if self._frequencies.size == 0:
            return None
        parts = [f"{frequency:,.0f} Hz".replace(",", ".")]
        domain = self._labels if self._labels is not None else self._frequencies
        for name, values in self._spectra.items():
            if values.size != domain.size:
                continue
            level = float(np.interp(frequency, domain, values))
            unit = self._units.get(name, "dB")
            label = self._names.get(name, name)
            parts.append(f"{label} {level:+.1f} {unit}")
        return "  |  ".join(parts)
