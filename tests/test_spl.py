import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from src.audio.engine import AudioEngine, SpectrumFrame
from src.dsp.spectrum import (
    DB_CEILING,
    DB_FLOOR,
    DB_FULL_SCALE,
    a_weighting_db,
    db_to_spl,
    level_db,
    total_level_db,
)
from src.ui.controls import ControlPanel
from src.ui.window import MainWindow


class SplSpectrumTests(unittest.TestCase):
    def test_full_scale_sine_with_94_db_reference_reads_90_99_spl(self):
        sample_rate = 48_000.0
        samples = np.sin(
            2.0 * np.pi * 1_000.0 * np.arange(4_800) / sample_rate
        )

        total_db = total_level_db(samples)
        total_spl = db_to_spl(total_db, 94.0)

        self.assertAlmostEqual(total_db, DB_FULL_SCALE - 3.0103, delta=0.001)
        self.assertAlmostEqual(total_spl, 90.99, delta=0.1)

    def test_reference_94_and_measured_114_give_offset_120(self):
        measured_db = 114.0
        reference_spl = 94.0

        offset_db = reference_spl - measured_db + DB_FULL_SCALE

        self.assertAlmostEqual(offset_db, 120.0, places=6)
        self.assertAlmostEqual(db_to_spl(measured_db, offset_db), 94.0, places=6)

    def test_db_to_spl_converts_arrays_without_mutating_them(self):
        values_db = np.array([110.0, 120.0, 130.0])

        values_spl = db_to_spl(values_db, 94.0)

        np.testing.assert_allclose(values_spl, [64.0, 74.0, 84.0])
        np.testing.assert_allclose(values_db, [110.0, 120.0, 130.0])

    def test_a_weighting_matches_iec_anchor_values(self):
        frequencies = np.array([1_000.0, 100.0, 10_000.0])
        expected = np.array([0.0, -19.1, -2.5])

        values = a_weighting_db(frequencies)

        for actual, target in zip(values, expected):
            with self.subTest(target=target):
                self.assertAlmostEqual(float(actual), float(target), delta=0.2)

    def test_z_total_matches_the_time_domain_rms_by_parseval(self):
        samples = np.random.default_rng(1234).standard_normal(2_048)

        total = total_level_db(samples, weighting="Z")
        expected, _peak = level_db(samples)

        self.assertAlmostEqual(total, expected, places=9)

    def test_a_total_leaves_an_exact_1khz_sine_unattenuated(self):
        sample_rate = 48_000.0
        samples = np.sin(
            2.0 * np.pi * 1_000.0 * np.arange(4_800) / sample_rate
        )

        total_z = total_level_db(samples, weighting="Z")
        total_a = total_level_db(samples, weighting="A", sample_rate=sample_rate)

        self.assertAlmostEqual(total_a, total_z, delta=1e-6)

    def test_empty_block_has_the_db_floor(self):
        self.assertEqual(total_level_db(np.zeros(0)), DB_FLOOR)

    def test_unknown_weighting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "weighting"):
            total_level_db(np.ones(16), weighting="Q")


class SplEngineStateTests(unittest.TestCase):
    def test_spl_display_defaults_to_uncalibrated_and_z_weighting(self):
        engine = AudioEngine(sample_rate=48_000.0, fft_size=512)

        self.assertIsNone(engine.spl_offset_db)
        self.assertEqual(engine.weighting, "Z")


class SplControlPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-controles-spl")

    def setUp(self):
        self.panel = ControlPanel()

    def test_spl_controls_emit_their_values(self):
        display_events: list[bool] = []
        offset_events: list[float] = []
        weighting_events: list[str] = []
        self.panel.splDisplayToggled.connect(display_events.append)
        self.panel.splOffsetChanged.connect(offset_events.append)
        self.panel.weightingChanged.connect(weighting_events.append)

        self.panel.spl_checkbox.setChecked(True)
        self.panel.spl_offset_spin.setValue(121.5)
        self.panel.weighting_combo.setCurrentIndex(
            self.panel.weighting_combo.findData("A")
        )

        self.assertEqual(display_events, [True])
        self.assertEqual(offset_events, [121.5])
        self.assertEqual(weighting_events, ["A"])

    def test_programmatic_spl_state_does_not_emit_signals(self):
        display_events: list[bool] = []
        offset_events: list[float] = []
        weighting_events: list[str] = []
        self.panel.splDisplayToggled.connect(display_events.append)
        self.panel.splOffsetChanged.connect(offset_events.append)
        self.panel.weightingChanged.connect(weighting_events.append)

        self.panel.set_spl_display(True)
        self.panel.set_spl_offset(94.0)
        self.panel.set_weighting("A")

        self.assertTrue(self.panel.spl_checkbox.isChecked())
        self.assertEqual(self.panel.spl_offset_spin.value(), 94.0)
        self.assertEqual(self.panel.weighting_combo.currentData(), "A")
        self.assertEqual(display_events, [])
        self.assertEqual(offset_events, [])
        self.assertEqual(weighting_events, [])


class SplMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-ventana-spl")

    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.engine.spl_offset_db = 94.0
        self.window = MainWindow(self.engine)

    def tearDown(self):
        self.window.close()

    def _refresh_with_sine(self) -> np.ndarray:
        fft_size = self.engine.fft_size
        hop_size = self.engine.hop_size
        time = np.arange(fft_size) / self.engine.sample_rate
        samples = np.sin(2.0 * np.pi * 3_000.0 * time)
        self.engine.display_queue.put(
            SpectrumFrame(
                input_samples=samples[-hop_size:].copy(),
                output_samples=np.zeros(hop_size),
                input_window=samples,
                output_window=np.zeros(fft_size),
            )
        )
        self.window._refresh()
        return samples[-hop_size:]

    def test_spl_mode_converts_only_the_input_display_and_axis(self):
        self.window.controls.spl_checkbox.setChecked(True)
        self._refresh_with_sine()

        y_min, y_max = self.window.input_plot.viewRange()[1]
        self.assertAlmostEqual(y_min, 0.0, places=6)
        self.assertAlmostEqual(y_max, DB_CEILING - DB_FULL_SCALE + 94.0, places=6)
        self.assertEqual(self.window.input_plot.getAxis("left").labelUnits, "dB SPL")

        output_min, output_max = self.window.output_plot.viewRange()[1]
        self.assertAlmostEqual(output_min, DB_FLOOR, places=6)
        self.assertAlmostEqual(output_max, DB_CEILING, places=6)
        self.assertEqual(self.window.output_plot.getAxis("left").labelUnits, "dB")

        _frequencies, displayed_input = self.window.input_curve.getOriginalDataset()
        np.testing.assert_allclose(
            displayed_input,
            db_to_spl(self.window._input_db[1:], 94.0),
        )
        self.assertAlmostEqual(
            float(np.max(self.window._input_db)), DB_FULL_SCALE, delta=0.1
        )

        text = self.window.readout._describe(3_000.0)
        self.assertIn("dB SPL", text)
        self.assertIn("salida", text)
        self.assertIn("entrada (sin calibrar)", text)

        input_rms = self.window.meters.input_meter._rms_db
        self.assertAlmostEqual(
            input_rms, DB_FULL_SCALE + 20.0 * np.log10(1.0 / np.sqrt(2.0)), places=6
        )

    def test_total_level_uses_the_selected_a_weighting_and_spl_offset(self):
        self.window.controls.spl_checkbox.setChecked(True)
        self.window.controls.weighting_combo.setCurrentIndex(
            self.window.controls.weighting_combo.findData("A")
        )
        input_samples = self._refresh_with_sine()

        expected = db_to_spl(
            total_level_db(input_samples, weighting="A", sample_rate=48_000.0),
            94.0,
        )

        self.assertEqual(self.engine.weighting, "A")
        self.assertEqual(
            self.window.controls.total_level_label.text(),
            f"{expected:.1f} dB SPL (A)",
        )

    def test_total_and_input_stay_in_db_when_spl_is_not_selected(self):
        input_samples = self._refresh_with_sine()
        expected = total_level_db(input_samples, weighting="Z")

        self.assertEqual(
            self.window.controls.total_level_label.text(),
            f"{expected:.1f} dB (Z)",
        )
        self.assertEqual(self.window.input_plot.getAxis("left").labelUnits, "dB")
        _frequencies, displayed_input = self.window.input_curve.getOriginalDataset()
        np.testing.assert_allclose(displayed_input, self.window._input_db[1:])

    def test_disabling_spl_restores_the_db_axis_and_cursor(self):
        self.window.controls.spl_checkbox.setChecked(True)
        self._refresh_with_sine()

        self.window.controls.spl_checkbox.setChecked(False)

        y_min, y_max = self.window.input_plot.viewRange()[1]
        self.assertAlmostEqual(y_min, DB_FLOOR, places=6)
        self.assertAlmostEqual(y_max, DB_CEILING, places=6)
        self.assertEqual(self.window.input_plot.getAxis("left").labelUnits, "dB")
        self.assertNotIn("dB SPL", self.window.readout._describe(3_000.0))


if __name__ == "__main__":
    unittest.main()
