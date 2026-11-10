import unittest
import tempfile
from pathlib import Path

import numpy as np

from verificar_entrada import _write_svg, analyze_capture


class InputVerificationTests(unittest.TestCase):
    def test_detects_measurable_audio_above_threshold(self):
        sample_rate = 48_000
        t = np.arange(sample_rate) / sample_rate
        samples = 0.02 * np.sin(2.0 * np.pi * 1_000.0 * t)

        report = analyze_capture(samples, sample_rate=sample_rate)

        self.assertEqual(report["status"], "PASS")
        self.assertGreater(report["rms_dbfs"], -60.0)
        self.assertEqual(report["sample_rate"], sample_rate)

    def test_rejects_silence_as_unverified_input(self):
        report = analyze_capture(np.zeros(4_800), sample_rate=48_000)

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["rms_dbfs"], -240.0)
        self.assertIn("demasiado baja", report["message"])

    def test_detects_intermittent_speech_using_max_frame_level(self):
        sample_rate = 48_000
        samples = np.zeros(sample_rate * 10)
        t = np.arange(sample_rate // 10) / sample_rate
        burst = 0.01 * np.sin(2.0 * np.pi * 1_000.0 * t)
        samples[sample_rate : sample_rate + burst.size] = burst

        report = analyze_capture(samples, sample_rate=sample_rate)

        self.assertEqual(report["status"], "PASS")
        self.assertLess(report["rms_dbfs"], -60.0)
        self.assertGreater(report["max_frame_rms_dbfs"], -60.0)
        self.assertGreater(report["frames_above_threshold"], 0)
        self.assertLess(report["signal_fraction"], 0.5)

    def test_continuous_mode_still_rejects_intermittent_speech(self):
        sample_rate = 48_000
        samples = np.zeros(sample_rate * 10)
        t = np.arange(sample_rate // 10) / sample_rate
        burst = 0.01 * np.sin(2.0 * np.pi * 1_000.0 * t)
        samples[sample_rate : sample_rate + burst.size] = burst

        report = analyze_capture(
            samples, sample_rate=sample_rate, require_continuous=True
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertIn("RMS global", report["message"])

    def test_reports_clipping_without_marking_capture_as_valid(self):
        samples = np.zeros(4_800)
        samples[0] = 1.0

        report = analyze_capture(samples, sample_rate=48_000)

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["clipped_samples"], 1)
        self.assertIn("saturación", report["message"])

    def test_rejects_invalid_capture(self):
        with self.assertRaisesRegex(ValueError, "muestras"):
            analyze_capture(np.array([[0.1, 0.2]]), sample_rate=48_000)

        with self.assertRaisesRegex(ValueError, "solo valores finitos"):
            analyze_capture(np.array([0.1, np.nan]), sample_rate=48_000)

    def test_detects_known_tone_by_coherent_projection(self):
        sample_rate = 48_000
        t = np.arange(sample_rate) / sample_rate
        samples = 0.02 * np.sin(2.0 * np.pi * 1_000.0 * t)

        report = analyze_capture(
            samples, sample_rate=sample_rate, tone_frequency=1_000.0
        )

        self.assertEqual(report["status"], "PASS")
        self.assertAlmostEqual(report["tone_dbfs"], -36.99, delta=0.5)
        self.assertAlmostEqual(report["dominant_frequency_hz"], 1_000.0, delta=2.0)

    def test_rejects_noise_without_the_expected_tone(self):
        sample_rate = 48_000
        rng = np.random.default_rng(0)
        samples = 0.01 * rng.standard_normal(sample_rate)

        report = analyze_capture(
            samples, sample_rate=sample_rate, tone_frequency=1_000.0
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertLess(report["tone_dbfs"], -60.0)

    def test_rejects_a_tone_at_a_different_frequency(self):
        sample_rate = 48_000
        t = np.arange(sample_rate) / sample_rate
        samples = 0.02 * np.sin(2.0 * np.pi * 6_000.0 * t)

        report = analyze_capture(
            samples, sample_rate=sample_rate, tone_frequency=1_000.0
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertLess(report["tone_dbfs"], -60.0)
        self.assertAlmostEqual(report["dominant_frequency_hz"], 6_000.0, delta=2.0)

    def test_writes_an_svg_level_plot(self):
        report = analyze_capture(np.zeros(4_800), sample_rate=48_000)
        path = Path(tempfile.mkdtemp()) / "entrada.svg"

        _write_svg(report, path)

        self.assertIn("<svg", path.read_text(encoding="utf-8"))
        self.assertIn("FAIL", path.read_text(encoding="utf-8"))
