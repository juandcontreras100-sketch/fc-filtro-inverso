"""Medidores de nivel de entrada y salida con deteccion de saturacion."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from src.dsp.spectrum import CLIP_DB
from src.ui import theme

DB_MIN = -60.0
DB_MAX = 0.0
TICK_DBS = (-48.0, -36.0, -24.0, -12.0, -6.0)


def db_to_fraction(db: float) -> float:
    """Convierte dBFS a una posicion normalizada en la barra."""
    if db <= DB_MIN:
        return 0.0
    if db >= DB_MAX:
        return 1.0
    return (db - DB_MIN) / (DB_MAX - DB_MIN)


class LevelMeter(QtWidgets.QWidget):
    """Barra horizontal con relleno RMS, marca de pico y lectura numerica."""

    def __init__(self, title: str, color: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._color = color
        self._rms_db = DB_MIN
        self._peak_db = DB_MIN
        self.setMinimumHeight(50)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_levels(self, rms_db: float, peak_db: float) -> None:
        self._rms_db = rms_db
        self._peak_db = peak_db
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.PANEL_EDGE), 1))
        painter.setBrush(QtGui.QColor(theme.PANEL))
        painter.drawRoundedRect(rect, 6, 6)

        bar = QtCore.QRectF(rect.left() + 8, rect.top() + 25, max(rect.width() - 16, 1), 14)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(theme.SURFACE))
        painter.drawRoundedRect(bar, 3, 3)

        rms_width = bar.width() * db_to_fraction(self._rms_db)
        if rms_width > 1.0:
            painter.setBrush(QtGui.QColor(self._color))
            painter.drawRoundedRect(
                QtCore.QRectF(bar.left(), bar.top(), rms_width, bar.height()), 3, 3
            )

        # Marcas de referencia en la escala de dBFS
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.PANEL_EDGE), 1))
        for tick_db in TICK_DBS:
            tick_x = bar.left() + bar.width() * db_to_fraction(tick_db)
            painter.drawLine(
                QtCore.QPointF(tick_x, bar.bottom() - 3),
                QtCore.QPointF(tick_x, bar.bottom()),
            )

        peak_x = bar.left() + bar.width() * db_to_fraction(self._peak_db)
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.PEAK), 2))
        painter.drawLine(
            QtCore.QPointF(peak_x, bar.top() - 1),
            QtCore.QPointF(peak_x, bar.bottom() + 1),
        )

        title_font = painter.font()
        title_font.setPointSize(9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(theme.TEXT))
        painter.drawText(
            QtCore.QRectF(rect.left() + 8, rect.top() + 4, rect.width() - 16, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            self._title,
        )

        painter.setFont(theme.mono_font(8))
        painter.setPen(QtGui.QColor(theme.MUTED))
        painter.drawText(
            QtCore.QRectF(rect.left() + 8, rect.top() + 4, rect.width() - 16, 16),
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
            f"Pico {self._peak_db:+.1f} / RMS {self._rms_db:+.1f} dB",
        )


class ClipIndicator(QtWidgets.QWidget):
    """LED que se enciende cuando la salida alcanza saturacion."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._hot = False
        self._release_at = 0.0
        self.setFixedSize(78, 20)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def trigger(self) -> None:
        self._hot = True
        self._release_at = time.monotonic() + 1.5
        self.update()

    def _tick(self) -> None:
        if self._hot and time.monotonic() >= self._release_at:
            self._hot = False
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)

        border_color = theme.CLIP if self._hot else theme.PANEL_EDGE
        bg_color = "#3a1820" if self._hot else theme.PANEL
        painter.setPen(QtGui.QPen(QtGui.QColor(border_color), 1))
        painter.setBrush(QtGui.QColor(bg_color))
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QtCore.Qt.NoPen)
        color = QtGui.QColor(theme.CLIP if self._hot else theme.PANEL_EDGE)
        painter.setBrush(color)
        painter.drawEllipse(QtCore.QRectF(rect.left() + 6, rect.top() + 5, 9, 9))

        painter.setFont(theme.mono_font(8, bold=True))
        painter.setPen(QtGui.QColor(theme.CLIP if self._hot else theme.MUTED))
        painter.drawText(
            QtCore.QRectF(rect.left() + 20, rect.top(), rect.width() - 22, rect.height()),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "CLIP",
        )


class MeterPanel(QtWidgets.QWidget):
    """Conjunto de medidores: entrada, salida e indicador de saturacion."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        title = QtWidgets.QLabel("NIVELES EN TIEMPO REAL")
        title.setStyleSheet(
            f"color: {theme.ACCENT}; font-weight: 700; font-size: 8pt;"
        )
        self.clip = ClipIndicator()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.clip)
        layout.addLayout(header)

        self.input_meter = LevelMeter("Entrada", theme.INPUT)
        self.output_meter = LevelMeter("Salida", theme.OUTPUT)
        layout.addWidget(self.input_meter)
        layout.addWidget(self.output_meter)

    def set_input(self, rms_db: float, peak_db: float) -> None:
        self.input_meter.set_levels(rms_db, peak_db)

    def set_output(self, rms_db: float, peak_db: float, would_clip: bool = False) -> None:
        self.output_meter.set_levels(rms_db, peak_db)
        if would_clip or peak_db >= CLIP_DB:
            self.clip.trigger()
