"""Cascada temporal del espectro de salida.

Cada fila de la cascada es un bloque del ecualizador; las columnas son los
bins de la FFT. El tiempo avanza hacia abajo, de modo que la fila mas reciente
queda pegada al borde inferior del area de graficas.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from src.ui import theme

from src.config import (
    DBFS_CEILING_DB,
    DBFS_FLOOR_DB,
    FREQUENCY_MAX_HZ,
    FREQUENCY_MIN_HZ,
)

DB_FLOOR = DBFS_FLOOR_DB
DB_CEILING = DBFS_CEILING_DB


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    """Historial del espectro: filas = tiempo, columnas = frecuencia."""

    def __init__(self, parent=None, rows: int = 160) -> None:
        super().__init__(parent)
        self.setBackground(theme.BACKGROUND)
        self._rows = rows
        self._bins = 0
        self._nyquist = 1.0
        self._seconds_per_row = 1.0
        self._levels = (DB_FLOOR, DB_CEILING)
        self._buffer = np.full((rows, 2), DB_FLOOR, dtype=np.float32)

        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setColorMap(pg.colormap.get("viridis"))
        self._image.setLevels((DB_FLOOR, DB_CEILING))

        self.ci.layout.setContentsMargins(2, 2, 2, 2)
        self.ci.layout.setSpacing(2)
        self.plot = self.addPlot()
        self.plot.getViewBox().setDefaultPadding(0.0)
        theme.style_plot(self.plot, "Cascada de la salida (abajo = mas reciente)")
        self.plot.setLabel("bottom", "Frecuencia lineal", units="kHz")
        self.plot.setLabel("left", "Tiempo hacia atras", units="s")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.addItem(self._image)

        self._colorbar = self.plot.addColorBar(
            self._image,
            colorMap="viridis",
            values=(DB_FLOOR, DB_CEILING),
            limits=(DB_FLOOR, DB_CEILING),
        )
        self._colorbar.setLabel("right", "Nivel", units="dBFS")
        self._colorbar.setMaximumWidth(70)

    # --------------------------------------------------------------- formato
    def reconfigure(
        self,
        bins: int,
        nyquist: float,
        seconds_per_row: float,
        db_floor: float = DB_FLOOR,
        db_ceiling: float = DB_CEILING,
    ) -> None:
        """Reinicia la cascada cuando cambia el formato del stream."""
        self._bins = bins
        self._nyquist = nyquist
        self._seconds_per_row = max(seconds_per_row, 1e-6)
        self._levels = (db_floor, db_ceiling)
        self._buffer = np.full((self._rows, bins), db_floor, dtype=np.float32)

        self._image.setLevels(self._levels)
        nyquist_khz = nyquist / 1_000.0
        self._image.setRect(
            0.0, 0.0, nyquist_khz, self._rows * self._seconds_per_row
        )
        self._image.setImage(self._buffer, axisOrder="row-major", levels=self._levels)

        # Igual que las graficas de espectro: el eje arranca en el bin 1 para
        # no dejar un hueco vacio entre 0 Hz y el primer dato representable.
        first_bin = nyquist / max(bins - 1, 1)
        lower = max(FREQUENCY_MIN_HZ, first_bin)
        upper = min(FREQUENCY_MAX_HZ, nyquist)
        self.plot.setXRange(lower / 1_000.0, upper / 1_000.0, padding=0)
        self.plot.setYRange(0.0, self._rows * self._seconds_per_row, padding=0)

    # ----------------------------------------------------------------- datos
    def push(self, spectrum_db: np.ndarray) -> None:
        """Agrega una fila al final del historial y desplaza el resto."""
        if self._bins == 0 or spectrum_db.size != self._bins:
            return
        self._buffer[:-1, :] = self._buffer[1:, :]
        self._buffer[-1, :] = spectrum_db.astype(np.float32, copy=False)
        self._image.setImage(self._buffer, axisOrder="row-major", levels=self._levels)

    def clear(self) -> None:
        self._buffer.fill(DB_FLOOR)
        self._image.setImage(self._buffer, axisOrder="row-major", levels=self._levels)
