import os

# Las pruebas de interfaz no deben abrir una ventana real.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from src.audio.engine import AudioEngine, SpectrumFrame
from src.config import FREQUENCY_MAX_HZ
from src.dsp.spectrum import DB_FULL_SCALE
from src.storage.profiles import UserProfile
from src.storage.recorder import SessionRecorder
from src.ui import theme
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

    def test_the_control_panel_uses_8_px_between_its_groups(self):
        self.assertEqual(self.panel.layout().spacing(), 8)

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

    def test_the_spectrum_combo_starts_on_linea_and_emits_on_change(self):
        self.assertEqual(self.panel.spectrum_combo.currentData(), "linea")

        received: list[str] = []
        self.panel.spectrumStyleChanged.connect(received.append)

        self.panel.spectrum_combo.setCurrentIndex(1)
        self.assertEqual(received, ["barras"])

        # set_* programmatico no emite la senal.
        self.panel.set_spectrum_style("linea")
        self.assertEqual(self.panel.spectrum_combo.currentData(), "linea")
        self.assertEqual(received, ["barras"])

    def test_the_panel_drives_the_engine(self):
        engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        engine.h_response = np.ones(engine.bin_count, dtype=np.complex128)
        engine.h_response[64] = 0.05
        engine.coherence = np.ones(engine.bin_count, dtype=np.float64)
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

    def test_frequency_plots_use_a_log_axis_to_the_shared_frequency_maximum(self):
        for plot in (self.window.input_plot, self.window.output_plot):
            self.assertTrue(plot.ctrl.logXCheck.isChecked())
            self.assertFalse(plot.ctrl.logYCheck.isChecked())

        x_min, x_max = self.window.input_plot.viewRange()[0]
        first_bin_khz = (
            self.engine.sample_rate / self.engine.fft_size / 1_000.0
        )

        self.assertGreaterEqual(x_min, np.log10(first_bin_khz))
        self.assertLess(x_min, x_max)
        self.assertAlmostEqual(
            x_max, np.log10(FREQUENCY_MAX_HZ / 1_000.0), places=5
        )

    def test_measured_inverse_and_product_stay_available_for_the_csv(self):
        bins = self.engine.bin_count
        self.engine.h_response = np.ones(bins, dtype=np.complex128)
        self.engine.bypass = False
        self.engine.recompute_inverse()

        self.window._update_filter_curve()

        self.assertEqual(self.window._measured_db.size, bins)
        np.testing.assert_allclose(self.window._measured_db, DB_FULL_SCALE, atol=1e-6)
        # El inverso Wiener/Tikhonov queda apenas por debajo de la ganancia unitaria.
        np.testing.assert_allclose(self.window._inverse_db, DB_FULL_SCALE, atol=1.0)
        np.testing.assert_allclose(self.window._product_db, DB_FULL_SCALE, atol=1.0)

    def test_compare_plot_overlays_input_and_output_in_db(self):
        combo = self.window.controls.third_plot_combo
        combo.setCurrentIndex(combo.findData("comparacion"))
        bins = self.engine.bin_count
        self.window._input_db = np.full(bins, 120.0)
        self.window._output_db = np.full(bins, 105.0)

        self.window._update_input_display()

        frequencies, input_values = self.window.compare_input_curve.getOriginalDataset()
        _frequencies, output_values = (
            self.window.compare_output_curve.getOriginalDataset()
        )

        np.testing.assert_allclose(
            frequencies, self.window._frequencies[1:] / 1_000.0
        )
        np.testing.assert_allclose(input_values, 120.0)
        np.testing.assert_allclose(output_values, 105.0)

    def test_response_plot_shows_calibration_but_not_uncalibrated_placeholders(self):
        bins = self.engine.bin_count
        self.engine.h_response = np.full(bins, 10.0, dtype=np.complex128)
        self.engine.bypass = True
        self.engine.recompute_inverse()

        self.window._update_filter_curve()
        np.testing.assert_allclose(
            self.window._measured_db, DB_FULL_SCALE + 20.0, atol=1e-6
        )
        np.testing.assert_allclose(self.window._inverse_db, DB_FULL_SCALE, atol=1e-6)
        np.testing.assert_allclose(
            self.window._product_db, DB_FULL_SCALE + 20.0, atol=1e-6
        )

        self.engine.h_response = None
        self.engine.bypass = False
        self.engine.recompute_inverse()

        self.window._update_filter_curve()
        self.assertEqual(self.window._measured_db.size, 0)
        self.assertEqual(self.window._inverse_db.size, 0)
        self.assertEqual(self.window._product_db.size, 0)
        self.assertTrue(self.window.response_message.isVisible())

    def test_spectrogram_time_axis_uses_real_seconds_per_row(self):
        spectrogram = self.window.spectrogram

        self.assertAlmostEqual(
            spectrogram._seconds_per_row, self.engine.seconds_per_row, places=9
        )

        y_min, y_max = spectrogram.plot.viewRange()[1]
        expected = spectrogram._rows * self.engine.seconds_per_row

        self.assertAlmostEqual(y_min, 0.0, places=9)
        self.assertAlmostEqual(y_max, expected, places=9)

class MainWindowSpectrumStyleTests(unittest.TestCase):
    """Alternancia entre la vista de linea y la de barras de 1/3 de octava."""

    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-vista-espectro")

    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.window = MainWindow(self.engine)

    def tearDown(self):
        self.window.close()

    def _feed_frame(self):
        hop_size = self.engine.hop_size
        frame = SpectrumFrame(
            input_samples=np.full(hop_size, 0.1),
            output_samples=np.full(hop_size, 0.05),
            would_clip=False,
            input_window=np.full(self.engine.fft_size, 0.1),
            output_window=np.full(self.engine.fft_size, 0.05),
        )
        self.engine.display_queue.put(frame)
        self.window._refresh()

    def test_bars_mode_hides_the_curves_and_draws_bands(self):
        self._feed_frame()

        self.window.controls.spectrum_combo.setCurrentIndex(1)

        self.assertEqual(self.window._spectrum_style, "barras")
        self.assertFalse(self.window.input_curve.isVisible())
        self.assertFalse(self.window.output_curve.isVisible())
        self.assertFalse(self.window.filter_curve.isVisible())
        self.assertTrue(self.window.input_bars.isVisible())
        self.assertTrue(self.window.output_bars.isVisible())
        self.assertTrue(self.window.filter_band_curve.isVisible())

        band_count = self.window.input_bars.opts["x"].size
        self.assertGreater(band_count, 0)
        self.assertEqual(
            self.window.output_bars.opts["x"].size, band_count
        )
        self.assertEqual(self.window.filter_band_curve.getData()[0].size, band_count)

        # El eje pasa a indices de banda con ticks de frecuencia central.
        self.assertFalse(self.window.input_plot.ctrl.logXCheck.isChecked())
        ticks = self.window.input_plot.getAxis("bottom")._tickLevels
        self.assertIsNotNone(ticks)
        self.assertEqual(len(ticks[0]), band_count)
        self.assertEqual(ticks[0][0], (0.0, "100"))

        # El cursor se desactiva en modo barras: el aviso lo indica.
        self.assertTrue(
            all(not line.isVisible() for line in self.window.readout._lines)
        )
        self.assertIn("modo Linea", self.window.readout_label.text())

    def test_line_mode_restores_the_log_axis(self):
        self.window.controls.spectrum_combo.setCurrentIndex(1)
        self.window.controls.spectrum_combo.setCurrentIndex(0)

        self.assertTrue(self.window.input_curve.isVisible())
        self.assertTrue(self.window.output_curve.isVisible())
        self.assertTrue(self.window.filter_curve.isVisible())
        self.assertFalse(self.window.input_bars.isVisible())
        self.assertFalse(self.window.output_bars.isVisible())
        self.assertFalse(self.window.filter_band_curve.isVisible())

        self.assertTrue(self.window.input_plot.ctrl.logXCheck.isChecked())
        self.assertIsNone(self.window.input_plot.getAxis("bottom")._tickLevels)
        x_min, x_max = self.window.input_plot.viewRange()[0]
        first_bin_khz = (
            self.engine.sample_rate / self.engine.fft_size / 1_000.0
        )

        self.assertGreaterEqual(x_min, np.log10(first_bin_khz))
        self.assertLess(x_min, x_max)
        self.assertAlmostEqual(
            x_max, np.log10(FREQUENCY_MAX_HZ / 1_000.0), places=5
        )


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
            {"entrada": np.array([110.0, 100.0, 90.0])},
        )

        text = readout.handle_mouse(QtCore.QPointF(50.0, 50.0))

        self.assertIsNotNone(text)
        self.assertIn("1.000 Hz", text)
        self.assertIn("entrada 100.0 dB", text)


class CursorReadoutWindowTests(unittest.TestCase):
    """Cursor compartido, marcadores y etiquetas en las tres graficas."""

    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-cursor-ventana")

    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.window = MainWindow(self.engine)
        combo = self.window.controls.third_plot_combo
        combo.setCurrentIndex(combo.findData("comparacion"))
        self.window.show()
        self.window.resize(1_200, 800)
        self.app.processEvents()
        self._feed_frame()

    def tearDown(self):
        self.window.close()

    def _feed_frame(self) -> None:
        hop_size = self.engine.hop_size
        fft_size = self.engine.fft_size
        self.engine.display_queue.put(
            SpectrumFrame(
                input_samples=np.full(hop_size, 0.1),
                output_samples=np.full(hop_size, 0.05),
                would_clip=False,
                input_window=np.full(fft_size, 0.1),
                output_window=np.full(fft_size, 0.05),
            )
        )
        self.window._refresh()
        self.app.processEvents()

    def _scene_position(self, plot, frequency: float, y: float = DB_FULL_SCALE - 40.0):
        return plot.getViewBox().mapViewToScene(
            QtCore.QPointF(np.log10(frequency / 1_000.0), y)
        )

    def _curve(self, plot, name: str):
        return next(
            curve
            for curve in self.window.readout._curves
            if curve.plot is plot and curve.name == name
        )

    def test_mouse_move_places_the_cursor_in_every_plot(self):
        scene_position = self._scene_position(self.window.input_plot, 1_000.0)

        text = self.window.readout.handle_mouse(scene_position)

        self.assertIsNotNone(text)
        self.assertEqual(len(self.window.readout._lines), 3)
        for line in self.window.readout._lines:
            self.assertTrue(line.isVisible())
            self.assertAlmostEqual(
                line.value(), np.log10(1_000.0 / 1_000.0), places=5
            )

    def test_each_curve_gets_a_marker_and_a_label_with_its_own_color(self):
        scene_position = self._scene_position(self.window.input_plot, 1_000.0)

        self.window.readout.handle_mouse(scene_position)

        frequencies = self.window._frequencies
        expected_input = np.interp(1_000.0, frequencies, self.window._input_db)
        expected_output = np.interp(1_000.0, frequencies, self.window._output_db)

        input_curve = self._curve(self.window.input_plot, "entrada")
        output_curve = self._curve(self.window.output_plot, "salida")
        compare_input = self._curve(self.window.compare_plot, "entrada")
        compare_output = self._curve(self.window.compare_plot, "salida")

        for curve in (input_curve, output_curve, compare_input, compare_output):
            self.assertTrue(curve.marker.isVisible())
            self.assertTrue(curve.label.isVisible())

        _x, y_values = input_curve.marker.getData()
        self.assertAlmostEqual(float(y_values[0]), expected_input, places=5)
        self.assertEqual(
            input_curve.label.toPlainText(),
            f"{expected_input:.1f} dB",
        )

        _x, y_values = output_curve.marker.getData()
        self.assertAlmostEqual(float(y_values[0]), expected_output, places=5)
        self.assertEqual(
            output_curve.label.toPlainText(),
            f"{expected_output:.1f} dB",
        )
        self.assertEqual(input_curve.color, theme.INPUT)
        self.assertEqual(output_curve.color, theme.OUTPUT)
        self.assertEqual(compare_input.color, theme.INPUT)
        self.assertEqual(compare_output.color, theme.OUTPUT)

    def test_label_flips_to_the_right_half_of_the_plot(self):
        curve = self._curve(self.window.input_plot, "entrada")

        self.window.readout.handle_mouse(
            self._scene_position(self.window.input_plot, 200.0)
        )
        left_anchor = curve.label.anchor.x()
        left_x = curve.label.pos().x()

        self.window.readout.handle_mouse(
            self._scene_position(self.window.input_plot, 15_000.0)
        )
        right_anchor = curve.label.anchor.x()
        right_x = curve.label.pos().x()

        self.assertEqual(left_anchor, 1.0)
        self.assertEqual(right_anchor, 0.0)
        self.assertLess(left_x, np.log10(200.0 / 1_000.0))
        self.assertGreater(right_x, np.log10(15_000.0 / 1_000.0))

    def test_input_label_uses_spl_when_spl_display_is_active(self):
        self.engine.spl_offset_db = 120.0
        self.window.controls.spl_checkbox.setChecked(True)
        self.window._update_input_display()

        # En SPL el eje Y arranca en 0 dB SPL aunque el piso de dB equivalga a menos.
        self.window.readout.handle_mouse(
            self._scene_position(self.window.input_plot, 1_000.0, y=60.0)
        )

        input_curve = self._curve(self.window.input_plot, "entrada")
        self.assertTrue(input_curve.label.toPlainText().endswith("dB SPL"))

    def test_compare_labels_stay_uncalibrated_when_spl_display_is_active(self):
        self.engine.spl_offset_db = 120.0
        self.window.controls.spl_checkbox.setChecked(True)
        self.window._update_input_display()

        self.window.readout.handle_mouse(
            self._scene_position(self.window.compare_plot, 1_000.0)
        )

        compare_input = self._curve(self.window.compare_plot, "entrada")
        label = compare_input.label.toPlainText()
        self.assertTrue(label.endswith(" dB"))
        self.assertNotIn("SPL", label)

    def test_leaving_the_graphics_hides_lines_markers_and_labels(self):
        self.window.readout.handle_mouse(
            self._scene_position(self.window.input_plot, 1_000.0)
        )

        self.window.readout.handle_mouse(QtCore.QPointF(-10_000.0, -10_000.0))

        for line in self.window.readout._lines:
            self.assertFalse(line.isVisible())
        for curve in self.window.readout._curves:
            self.assertFalse(curve.marker.isVisible())
            self.assertFalse(curve.label.isVisible())


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
            side_effect=lambda engine, **kwargs: SessionRecorder(
                engine, root_dir=self.root, **kwargs
            ),
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

    def test_the_recording_gets_the_logged_in_user_and_devices(self):
        profile = UserProfile("Juan Perez", "1234567")
        devices = {"entrada": "Microfono", "salida": "Altavoz"}
        window = MainWindow(self.engine, user=profile, devices=devices)
        self.addCleanup(window.close)

        with self._patch_recorder():
            window._on_record_toggled(True)
            directory = window._recorder.directory
            window._on_record_toggled(False)

        self.assertEqual(directory.parent.name, "juan_perez_1234567")
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["usuario"]["nombre"], "Juan Perez")
        self.assertEqual(metadata["dispositivos"], devices)

    def test_the_export_dialog_defaults_to_the_user_folder(self):
        profile = UserProfile("Juan Perez", "1234567")
        window = MainWindow(self.engine, user=profile)
        self.addCleanup(window.close)

        with mock.patch.object(
            window_module.QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=(str(self.root / "captura.png"), ""),
        ) as dialog:
            window._on_export_requested()

        default = Path(dialog.call_args.args[2])
        self.assertEqual(default, Path("sesiones") / "juan_perez_1234567" / "captura.png")
        self.assertTrue((self.root / "captura.png").exists())

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
                "provenance",
            ],
        )
        self.assertEqual(len(rows), self.engine.bin_count)
        self.assertEqual(rows[0]["provenance"], "SIN CALIBRACIÓN")
        self.assertEqual(rows[0]["h_measured_db"], "")

    def test_synthetic_calibration_provenance_reaches_csv_and_plot_title(self):
        response_path = self.root / "demo_H.npy"
        coherence_path = self.root / "demo_gamma2.npy"
        np.save(
            response_path,
            np.ones(self.engine.bin_count, dtype=np.complex128),
        )
        np.save(
            coherence_path,
            np.ones(self.engine.bin_count, dtype=np.float64),
        )
        response_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "sample_rate_hz": self.engine.sample_rate,
                    "fft_size_samples": self.engine.fft_size,
                    "frequency_range_hz": [20.0, 20_000.0],
                    "synthetic": True,
                }
            ),
            encoding="utf-8",
        )
        self.engine.load_calibration(response_path, coherence_path)
        self.window._update_calibration_banner()
        self.window._update_filter_curve()
        csv_path = self.root / "demo.csv"
        self.window._write_current_csv(csv_path)

        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["provenance"], "DATOS SINTÉTICOS")
        self.assertFalse(self.window.calibration_banner.isHidden())
        self.assertIn("DATOS SINTÉTICOS", self.window.response_plot.titleLabel.text)
