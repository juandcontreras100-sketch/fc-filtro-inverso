import os

# Las pruebas de interfaz no deben abrir una ventana real.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from src.audio.engine import AudioEngine, SpectrumFrame
from src.storage.recorder import SessionRecorder
from src.ui import window as window_module
from src.ui.controls import ControlPanel
from src.ui.readout import CursorReadout
from src.ui.window import MainWindow


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


class MainWindowSpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-espectro-ventana")

    def setUp(self):
        self.window = MainWindow(AudioEngine(sample_rate=48_000.0, fft_size=512))

    def tearDown(self):
        self.window.close()

    def test_the_default_exponential_averaging_factor_is_0_8(self):
        self.assertAlmostEqual(self.window.spectrum_averaging, 0.8, places=6)

    def test_exponential_averaging_weights_previous_and_current_spectra(self):
        previous = np.array([0.0, -10.0, -20.0])
        current = np.array([-30.0, -10.0, 0.0])

        smoothed = self.window._smooth_spectrum(previous, current)

        np.testing.assert_allclose(smoothed, 0.8 * previous + 0.2 * current)
        np.testing.assert_allclose(previous, [0.0, -10.0, -20.0])


class MainWindowLogPlotsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-ejes-log")

    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.window = MainWindow(self.engine)

    def tearDown(self):
        self.window.close()

    def test_frequency_plots_use_a_log_axis_from_20_hz_to_20_khz(self):
        for plot in (self.window.input_plot, self.window.output_plot):
            self.assertTrue(plot.ctrl.logXCheck.isChecked())
            self.assertFalse(plot.ctrl.logYCheck.isChecked())

        x_min, x_max = self.window.input_plot.viewRange()[0]

        self.assertAlmostEqual(x_min, np.log10(20.0), places=5)
        self.assertAlmostEqual(x_max, np.log10(20_000.0), places=5)

    def test_third_plot_shows_measured_inverse_and_product(self):
        bins = self.engine.bin_count
        self.engine.h_response = np.ones(bins, dtype=np.complex128)
        self.engine.bypass = False
        self.engine.recompute_inverse()

        self.window._update_filter_curve()

        _frequencies, measured = self.window.h_measured_curve.getData()
        _frequencies, inverse = self.window.h_inverse_curve.getData()
        _frequencies, product = self.window.h_product_curve.getData()

        self.assertEqual(measured.size, bins - 1)
        np.testing.assert_allclose(measured, 0.0, atol=1e-6)
        # El inverso Wiener/Tikhonov queda apenas por debajo de 0 dB.
        np.testing.assert_allclose(inverse, 0.0, atol=1.0)
        np.testing.assert_allclose(product, 0.0, atol=1.0)

    def test_third_plot_is_flat_in_bypass_and_without_calibration(self):
        bins = self.engine.bin_count
        self.engine.h_response = np.full(bins, 10.0, dtype=np.complex128)
        self.engine.bypass = True
        self.engine.recompute_inverse()

        self.window._update_filter_curve()
        self._assert_flat_curves()

        self.engine.h_response = None
        self.engine.bypass = False
        self.engine.recompute_inverse()

        self.window._update_filter_curve()
        self._assert_flat_curves()

    def test_spectrogram_time_axis_uses_real_seconds_per_row(self):
        spectrogram = self.window.spectrogram

        self.assertAlmostEqual(
            spectrogram._seconds_per_row, self.engine.seconds_per_row, places=9
        )

        y_min, y_max = spectrogram.plot.viewRange()[1]
        expected = spectrogram._rows * self.engine.seconds_per_row

        self.assertAlmostEqual(y_min, 0.0, places=9)
        self.assertAlmostEqual(y_max, expected, places=9)

    def _assert_flat_curves(self) -> None:
        for curve in (
            self.window.h_measured_curve,
            self.window.h_inverse_curve,
            self.window.h_product_curve,
        ):
            _frequencies, values = curve.getData()
            np.testing.assert_allclose(values, 0.0, atol=1e-9)


class _FakeViewBox:
    def sceneBoundingRect(self):
        return QtCore.QRectF(0.0, 0.0, 100.0, 100.0)

    def mapSceneToView(self, _position):
        return QtCore.QPointF(np.log10(1_000.0), 0.0)


class _FakePlot:
    def addItem(self, _item) -> None:
        pass

    def getViewBox(self):
        return _FakeViewBox()


class CursorReadoutLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-lectura-log")

    def test_log_cursor_reports_the_frequency_in_hz(self):
        readout = CursorReadout([_FakePlot()], log_x=True)
        readout.set_spectra(
            np.array([100.0, 1_000.0, 10_000.0]),
            {"entrada": np.array([-10.0, -20.0, -30.0])},
        )

        text = readout.handle_mouse(QtCore.QPointF(50.0, 50.0))

        self.assertIsNotNone(text)
        self.assertIn("1.000 Hz", text)
        self.assertIn("entrada -20.0 dB", text)


class ControlPanelSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-sesion-controles")

    def setUp(self):
        self.panel = ControlPanel()

    def test_record_button_toggles_its_text_and_emits_the_state(self):
        received: list[bool] = []
        self.panel.recordToggled.connect(received.append)

        self.panel.record_button.click()
        self.assertEqual(received, [True])
        self.assertEqual(self.panel.record_button.text(), "Detener")

        self.panel.record_button.click()
        self.assertEqual(received, [True, False])
        self.assertEqual(self.panel.record_button.text(), "Grabar")

    def test_set_recording_updates_the_button_without_emitting(self):
        received: list[bool] = []
        self.panel.recordToggled.connect(received.append)

        self.panel.set_recording(True)
        self.assertTrue(self.panel.record_button.isChecked())
        self.assertEqual(self.panel.record_button.text(), "Detener")

        self.panel.set_recording(False)
        self.assertFalse(self.panel.record_button.isChecked())
        self.assertEqual(self.panel.record_button.text(), "Grabar")
        self.assertEqual(received, [])

    def test_export_button_emits_a_request(self):
        received: list[bool] = []
        self.panel.exportRequested.connect(lambda: received.append(True))

        self.panel.export_button.click()

        self.assertEqual(received, [True])


class MainWindowSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-sesion-ventana")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.window = MainWindow(self.engine)

    def tearDown(self):
        self.window.close()

    def _patch_recorder(self):
        """Redirige la sesión a un directorio temporal para no ensuciar el repo."""
        return mock.patch.object(
            window_module,
            "SessionRecorder",
            side_effect=lambda engine: SessionRecorder(engine, root_dir=self.root),
        )

    def test_recording_toggle_creates_and_closes_a_session(self):
        with self._patch_recorder():
            self.window.controls.record_button.click()
            recorder = self.window._recorder
            self.assertIsNotNone(recorder)
            self.assertTrue(recorder.is_recording)
            self.assertEqual(self.window.controls.record_button.text(), "Detener")
            directory = recorder.directory
            self.assertTrue((directory / "metadata.json").exists())
            self.assertTrue((directory / "niveles.csv").exists())

            self.window.controls.record_button.click()

        self.assertIsNone(self.window._recorder)
        self.assertEqual(self.window.controls.record_button.text(), "Grabar")
        self.assertTrue((directory / "espectros.npz").exists())
        self.assertFalse(recorder.is_recording)

    def test_refresh_feeds_the_recorder_from_the_gui_thread(self):
        hop_size = self.engine.hop_size
        fft_size = self.engine.fft_size
        frame = SpectrumFrame(
            input_samples=np.full(hop_size, 0.1),
            output_samples=np.full(hop_size, 0.05),
            would_clip=True,
            input_window=np.full(fft_size, 0.1),
            output_window=np.full(fft_size, 0.05),
        )

        with self._patch_recorder():
            self.window._on_record_toggled(True)
            directory = self.window._recorder.directory
            self.engine.display_queue.put(frame)
            self.window._refresh()
            self.window._on_record_toggled(False)

        with (directory / "niveles.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["clip"]), 1)

        with np.load(directory / "espectros.npz") as data:
            self.assertEqual(data["input_db"].shape, (1, self.engine.bin_count))

    def test_export_capture_writes_a_png_and_the_current_csv(self):
        self.window.show()
        self.window.resize(800, 600)

        png_path, csv_path = self.window._export_capture(self.root / "captura.png")

        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 0)
        self.assertTrue(csv_path.exists())
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(
            reader.fieldnames,
            [
                "frequency_hz",
                "input_db",
                "output_db",
                "h_measured_db",
                "h_inverse_db",
                "h_product_db",
            ],
        )
        self.assertEqual(len(rows), self.engine.bin_count)
