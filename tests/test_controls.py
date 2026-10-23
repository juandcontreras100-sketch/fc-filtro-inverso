import os

# Las pruebas de interfaz no deben abrir una ventana real.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from src.audio.engine import AudioEngine
from src.ui.controls import ControlPanel


class ControlPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-controles")

    def setUp(self):
        self.panel = ControlPanel()

    def test_the_mode_combo_offers_bypass_and_ecualizado(self):
        self.assertEqual(self.panel.mode_combo.count(), 2)
        self.assertTrue(self.panel.mode_combo.itemData(0))
        self.assertFalse(self.panel.mode_combo.itemData(1))

    def test_changing_the_mode_emits_the_signal(self):
        received: list[bool] = []
        self.panel.modeChanged.connect(received.append)

        self.panel.mode_combo.setCurrentIndex(1)
        self.assertEqual(received, [False])

        self.panel.mode_combo.setCurrentIndex(0)
        self.assertEqual(received, [False, True])

    def test_changing_the_gain_emits_its_value(self):
        received: list[float] = []
        self.panel.gainChanged.connect(received.append)

        self.panel.gain_slider.setValue(20)
        self.assertEqual(received, [20.0])

    def test_changing_the_epsilon_emits_its_value(self):
        received: list[float] = []
        self.panel.epsilonChanged.connect(received.append)

        self.panel.epsilon_spin.setValue(0.05)
        self.assertAlmostEqual(received[-1], 0.05, places=6)

    def test_populating_the_devices_does_not_emit_a_change(self):
        received: list[object] = []
        self.panel.deviceChanged.connect(received.append)

        self.panel.populate_devices([(None, "Sistema"), (3, "3: Altavoces")])
        self.assertEqual(self.panel.device_combo.count(), 2)
        self.assertEqual(received, [])

    def test_setting_the_device_programmatically_does_not_emit(self):
        received: list[object] = []
        self.panel.deviceChanged.connect(received.append)

        self.panel.populate_devices([(None, "Sistema"), (3, "3: Altavoces")])
        self.panel.set_device(3)
        self.assertEqual(self.panel.device_combo.currentData(), 3)
        self.assertEqual(received, [])

    def test_the_format_combo_reports_sample_rate_and_fft_size(self):
        received: list[tuple[float, int]] = []
        self.panel.formatChanged.connect(lambda r, s: received.append((r, s)))

        self.panel.fft_combo.setCurrentIndex(self.panel.fft_combo.findData(4096))
        expected_rate = self.panel.rate_combo.currentData()
        self.assertEqual(received[-1], (expected_rate, 4096))

    def test_the_panel_drives_the_engine(self):
        engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        engine.h_response = np.ones(engine.bin_count, dtype=np.complex128)
        engine.h_response[64] = 0.05
        engine.bypass = False
        engine.recompute_inverse()
        boosted = float(np.max(np.abs(engine.equalizer.h_inverse)))

        panel = self.panel
        panel.modeChanged.connect(lambda bypass: self._set_mode(engine, bypass))
        panel.gainChanged.connect(lambda db: self._set_gain(engine, db))

        # El combo arranca en "Bypass"; se pasa a "Ecualizado" y de vuelta.
        panel.mode_combo.setCurrentIndex(1)
        self.assertGreater(float(np.max(np.abs(engine.equalizer.h_inverse))), 1.0)

        # Bypass deja el filtro plano.
        panel.mode_combo.setCurrentIndex(0)
        self.assertTrue(np.allclose(engine.equalizer.h_inverse, 1.0))

        # Ecualizado aplica el filtro inverso de la medicion.
        panel.mode_combo.setCurrentIndex(1)
        self.assertGreater(float(np.max(np.abs(engine.equalizer.h_inverse))), 1.0)

        # Bajar la ganancia maxima recorta el realce de la muesca.
        panel.gain_slider.setValue(3)
        self.assertLess(float(np.max(np.abs(engine.equalizer.h_inverse))), boosted)

    @staticmethod
    def _set_mode(engine: AudioEngine, bypass: bool) -> None:
        engine.bypass = bypass
        engine.recompute_inverse()

    @staticmethod
    def _set_gain(engine: AudioEngine, db: float) -> None:
        engine.max_gain_db = db
        engine.recompute_inverse()
