import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np

from src.audio import engine as engine_module
from src.audio.engine import AudioEngine, CalibrationDiscardedError


class AudioEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.bins = self.engine.bin_count

    def _write_measurement(
        self,
        bins: int,
        name: str = "medicion_H.npy",
        response: np.ndarray | None = None,
    ) -> Path:
        folder = Path(tempfile.mkdtemp())
        path = folder / name
        values = np.ones(bins, dtype=np.complex128) if response is None else response
        np.save(path, values)
        coherence_stem = path.stem.replace("_H", "_gamma2")
        if coherence_stem == path.stem:
            coherence_stem = f"{path.stem}_gamma2"
        np.save(path.with_name(f"{coherence_stem}.npy"), np.ones(bins))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "sample_rate_hz": self.engine.sample_rate,
                    "fft_size_samples": self.engine.fft_size,
                    "frequency_range_hz": [20.0, 20_000.0],
                    "synthetic": False,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_engine_starts_flat_and_in_bypass(self):
        self.assertTrue(self.engine.bypass)
        self.assertFalse(self.engine.is_calibrated)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_hop_size_is_half_the_fft_size(self):
        self.assertEqual(self.engine.hop_size, 256)
        self.assertEqual(self.engine.bin_count, 257)

    def test_loading_a_measurement_switches_to_the_ecualizado_filter(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)
        self.engine.bypass = False
        self.engine.recompute_inverse()

        self.assertTrue(self.engine.is_calibrated)
        self.assertAlmostEqual(self.engine.calibration_path, path)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0, atol=0.05))

    def test_loading_a_measurement_with_the_wrong_length_is_rejected(self):
        path = self._write_measurement(self.bins + 10)
        with self.assertRaisesRegex(ValueError, "bins"):
            self.engine.load_calibration(path)
        self.assertFalse(self.engine.is_calibrated)

    def test_bypass_keeps_the_filter_flat_even_with_a_measurement(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)
        self.engine.bypass = True
        self.engine.recompute_inverse()

        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_the_inverse_compensates_a_notch(self):
        h = np.ones(self.bins, dtype=np.complex128)
        h[64] = 0.1  # notch profundo
        path = self._write_measurement(self.bins, response=h)

        self.engine.load_calibration(path)
        self.engine.bypass = False
        self.engine.max_gain_db = 18.0
        self.engine.recompute_inverse()

        gain = abs(self.engine.equalizer.h_inverse[64])
        self.assertGreater(gain, 3.0)
        self.assertLessEqual(gain, 10.0 ** (18.0 / 20.0) + 1e-9)

    def test_even_a_huge_out_of_band_peak_does_not_collapse_the_filter(self):
        h = np.ones(self.bins, dtype=np.complex128)
        h[-1] = 100.0  # artefacto ultrasónico en Nyquist
        path = self._write_measurement(self.bins, response=h)

        self.engine.load_calibration(path)
        self.engine.bypass = False
        self.engine.recompute_inverse()

        inverse = np.asarray(self.engine.equalizer.h_inverse)
        self.assertTrue(np.allclose(inverse[:-1], 1.0, atol=0.1))
        self.assertLess(abs(inverse[-1]), 0.02)

    def test_clear_calibration_returns_to_a_flat_filter(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)
        self.engine.clear_calibration()

        self.assertFalse(self.engine.is_calibrated)
        self.assertIsNone(self.engine.calibration_path)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_changing_the_fft_size_discards_an_incompatible_measurement(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)

        with patch.object(self.engine, "restart") as restart:
            with self.assertRaises(CalibrationDiscardedError):
                self.engine.configure(fft_size=1024)
        restart.assert_called_once()

        self.assertFalse(self.engine.is_calibrated)
        self.assertEqual(self.engine.fft_size, 1024)
        self.assertEqual(self.engine.hop_size, 512)

    def test_changing_sample_rate_discards_calibration_and_restarts_stream(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)

        with patch.object(self.engine, "restart") as restart:
            with self.assertRaises(CalibrationDiscardedError):
                self.engine.configure(sample_rate=44_100.0)

        restart.assert_called_once()
        self.assertFalse(self.engine.is_calibrated)
        self.assertEqual(self.engine.sample_rate, 44_100.0)

    def test_recompute_inverse_bumps_the_revision(self):
        before = self.engine.revision
        self.engine.recompute_inverse()
        self.assertGreater(self.engine.revision, before)

    # ---- configure(): el stream debe quedar siempre consistente --------------
    def test_discarding_the_calibration_still_restarts_the_stream(self):
        # Regresion: antes se lanzaba el error ANTES de restart(), dejando el
        # stream con el blocksize viejo y el ecualizador nuevo (salida muda).
        self.engine.load_calibration(self._write_measurement(self.bins))

        with mock.patch.object(self.engine, "restart") as restart:
            with self.assertRaises(CalibrationDiscardedError):
                self.engine.configure(fft_size=1024)

        restart.assert_called_once()
        self.assertEqual(self.engine.hop_size, 512)
        self.assertEqual(self.engine.equalizer.fft_size, 1024)

    def test_the_discard_error_is_a_dedicated_value_error(self):
        self.engine.load_calibration(self._write_measurement(self.bins))

        with mock.patch.object(self.engine, "restart"):
            with self.assertRaises(engine_module.CalibrationDiscardedError) as caught:
                self.engine.configure(fft_size=1024)

        self.assertIsInstance(caught.exception, ValueError)

    def test_reapplying_the_same_format_keeps_the_calibration(self):
        self.engine.load_calibration(self._write_measurement(self.bins))

        with mock.patch.object(self.engine, "restart") as restart:
            self.engine.configure(sample_rate=48_000.0, fft_size=512)

        restart.assert_called_once()
        self.assertTrue(self.engine.is_calibrated)

    def test_changing_the_fft_size_without_calibration_just_restarts(self):
        with mock.patch.object(self.engine, "restart") as restart:
            self.engine.configure(fft_size=1024)

        restart.assert_called_once()
        self.assertEqual(self.engine.bin_count, 513)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_callback_sanitizes_non_finite_input_and_reports_it(self):
        hop_size = self.engine.hop_size
        output = np.empty((hop_size, 1), dtype=np.float32)
        self.engine._callback(
            np.zeros((hop_size, 1), dtype=np.float32),
            output,
            hop_size,
            None,
            None,
        )

        input_block = np.zeros((hop_size, 1), dtype=np.float32)
        input_block[0, 0] = np.nan
        input_block[1, 0] = np.inf
        input_block[2, 0] = -np.inf
        self.engine._callback(input_block, output, hop_size, None, None)

        self.assertTrue(np.all(np.isfinite(output)))
        frame = self.engine.display_queue.get_nowait()
        self.assertTrue(np.all(np.isfinite(frame.input_samples)))
        self.assertTrue(np.array_equal(frame.input_samples[:3], np.zeros(3)))
        self.assertIn("no finit", self.engine.status_queue.get_nowait())

    def test_callback_contains_processing_errors_and_silences_output(self):
        hop_size = self.engine.hop_size
        input_block = np.zeros((hop_size, 1), dtype=np.float32)
        output = np.full((hop_size, 1), 1.0, dtype=np.float32)

        with patch.object(self.engine.equalizer, "process_block", side_effect=RuntimeError("DSP fallo")):
            self.engine._callback(input_block, output, hop_size, None, None)

        self.assertTrue(np.array_equal(output, np.zeros_like(output)))
        self.assertIn("DSP fallo", self.engine.status_queue.get_nowait())

    def test_callback_limits_dac_output_and_tracks_pre_limiter_clipping(self):
        hop_size = self.engine.hop_size
        input_block = np.zeros((hop_size, 1), dtype=np.float32)
        output = np.zeros((hop_size, 1), dtype=np.float32)
        unbounded_block = np.full(hop_size, 1.2, dtype=np.float64)

        with patch.object(self.engine.equalizer, "process_block", return_value=unbounded_block):
            self.engine._callback(input_block, output, hop_size, None, None)

        self.assertLessEqual(float(np.max(np.abs(output))), 0.980001)
        frame = self.engine.display_queue.get_nowait()
        self.assertTrue(frame.would_clip)
        self.assertFalse(frame.input_would_clip)
        self.assertGreater(float(np.max(np.abs(frame.output_samples))), 1.0)

    def test_callback_tracks_input_clipping_separately_from_output(self):
        hop_size = self.engine.hop_size
        input_block = np.full((hop_size, 1), 1.0, dtype=np.float32)
        output = np.zeros((hop_size, 1), dtype=np.float32)
        filtered_block = np.full(hop_size, 0.25, dtype=np.float64)

        with patch.object(
            self.engine.equalizer, "process_block", return_value=filtered_block
        ):
            self.engine._callback(input_block, output, hop_size, None, None)
            self.engine.display_queue.get_nowait()
            input_block.fill(0.0)
            self.engine._callback(input_block, output, hop_size, None, None)

        frame = self.engine.display_queue.get_nowait()
        self.assertTrue(frame.input_would_clip)
        self.assertFalse(frame.would_clip)

    def test_callback_pairs_wola_output_with_the_input_hop_that_produced_it(self):
        hop_size = self.engine.hop_size
        first_input = np.full((hop_size, 1), 0.25, dtype=np.float32)
        next_input = np.full((hop_size, 1), 0.5, dtype=np.float32)
        output = np.zeros((hop_size, 1), dtype=np.float32)

        self.engine._callback(first_input, output, hop_size, None, None)
        self.engine._callback(next_input, output, hop_size, None, None)

        frame = self.engine.display_queue.get_nowait()
        self.assertTrue(np.allclose(frame.input_samples, 0.25))
