"""Medidores de nivel de entrada y salida con deteccion de saturacion."""

from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from src.ui import theme

DB_MIN = -60.0
DB_MAX = 0.0
CLIP_DB = -0.1


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
        self.setMinimumHeight(48)
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
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(theme.PANEL))
        painter.drawRoundedRect(rect, 5, 5)

        bar = QtCore.QRectF(rect.left() + 8, rect.top() + 24, max(rect.width() - 16, 1), 15)
        painter.setBrush(QtGui.QColor(theme.PANEL_EDGE))
        painter.drawRoundedRect(bar, 3, 3)

        rms_width = bar.width() * db_to_fraction(self._rms_db)
        if rms_width > 1.0:
            painter.setBrush(QtGui.QColor(self._color))
            painter.drawRoundedRect(
                QtCore.QRectF(bar.left(), bar.top(), rms_width, bar.height()), 3, 3
            )

        peak_x = bar.left() + bar.width() * db_to_fraction(self._peak_db)
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.PEAK), 2))
        painter.drawLine(QtCore.QPointF(peak_x, bar.top() - 1), QtCore.QPointF(peak_x, bar.bottom() + 1))

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(theme.TEXT))
        painter.drawText(
            QtCore.QRectF(rect.left() + 8, rect.top() + 3, rect.width() - 16, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            self._title,
        )
        painter.setPen(QtGui.QColor(theme.MUTED))
        painter.drawText(
            QtCore.QRectF(rect.left() + 8, rect.top() + 3, rect.width() - 16, 16),
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
            f"{self._peak_db:+.1f} / {self._rms_db:+.1f} dB",
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

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(theme.PANEL))
        painter.drawRoundedRect(rect, 4, 4)

        color = QtGui.QColor(theme.CLIP if self._hot else theme.PANEL_EDGE)
        painter.setBrush(color)
        painter.drawEllipse(QtCore.QRectF(rect.left() + 5, rect.top() + 5, 10, 10))

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(theme.CLIP if self._hot else theme.MUTED))
        painter.drawText(
            QtCore.QRectF(rect.left() + 19, rect.top(), rect.width() - 22, rect.height()),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "CLIP",
        )


class MeterPanel(QtWidgets.QWidget):
    """Conjunto de medidores: entrada, salida e indicador de saturacion."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.input_meter = LevelMeter("Entrada", theme.INPUT)
        self.output_meter = LevelMeter("Salida", theme.OUTPUT)
        layout.addWidget(self.input_meter)
        layout.addWidget(self.output_meter)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.clip = ClipIndicator()
        row.addWidget(self.clip)
        row.addStretch(1)
        layout.addLayout(row)

    def set_input(self, rms_db: float, peak_db: float) -> None:
        self.input_meter.set_levels(rms_db, peak_db)

    def set_output(self, rms_db: float, peak_db: float) -> None:
        self.output_meter.set_levels(rms_db, peak_db)
        if peak_db >= CLIP_DB:
            self.clip.trigger()
