import unittest

import numpy as np

from calibrate_hardware import (
    estimate_transfer,
    estimate_transfer_ensemble,
    estimate_impulse_response_farina,
    log_sine_sweep,
    validate_calibration_quality,
)


class CalibrationTests(unittest.TestCase):
    def test_calibration_quality_accepts_a_clean_measurable_signal(self):
        sample_rate = 48_000.0
        fft_size = 512
        sweep = log_sine_sweep(sample_rate, 0.5, 500.0, 10_000.0, 0.15)
        delays = (127, 311)
        recordings = [
            np.concatenate((np.zeros(delay), 0.5 * sweep, np.zeros(500)))
            for delay in delays
        ]
        _, coherence, _ = estimate_transfer_ensemble(
            sweep,
            recordings,
            sample_rate,
            fft_size,
            500.0,
            10_000.0,
        )

        quality = validate_calibration_quality(
            recordings, coherence, sample_rate, fft_size
        )

        self.assertGreater(quality["minimum_rms_db"], 80.0)
        self.assertGreater(quality["median_coherence_100_10000_hz"], 0.8)

    def test_calibration_quality_rejects_silence(self):
        sample_rate = 48_000.0
        fft_size = 2048
        coherence = np.ones(fft_size // 2 + 1)
        recordings = [np.zeros(96_000), np.zeros(96_000)]

        with self.assertRaisesRegex(ValueError, "señal de entrada demasiado baja"):
            validate_calibration_quality(recordings, coherence, sample_rate, fft_size)

    def test_calibration_quality_rejects_clipping(self):
        sample_rate = 48_000.0
        fft_size = 2048
        coherence = np.ones(fft_size // 2 + 1)
        recordings = [np.full(96_000, 1.0), np.full(96_000, 1.0)]

        with self.assertRaisesRegex(ValueError, "saturación"):
            validate_calibration_quality(recordings, coherence, sample_rate, fft_size)

    def test_calibration_quality_rejects_low_median_coherence(self):
        sample_rate = 48_000.0
        fft_size = 2048
        coherence = np.zeros(fft_size // 2 + 1)
        recordings = [np.full(96_000, 0.05), np.full(96_000, 0.05)]

        with self.assertRaisesRegex(ValueError, "coherencia mediana"):
            validate_calibration_quality(recordings, coherence, sample_rate, fft_size)

    def test_transfer_recovers_gain_after_capture_delay(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)
        delay = 321
        gain = 0.6
        recorded = np.concatenate((np.zeros(delay), gain * sweep, np.zeros(500)))

        transfer, start = estimate_transfer(
            sweep,
            recorded,
            sample_rate=sample_rate,
            fft_size=256,
            start_freq=500.0,
            stop_freq=10_000.0,
        )
        frequencies = np.fft.rfftfreq(256, d=1.0 / sample_rate)
        # Se excluye el borde de 10 kHz: allí termina el fundido del sweep y
        # el calibrador lo deja correctamente en bypass si no hay energía.
        in_band = (frequencies >= 500.0) & (frequencies <= 9_000.0)

        self.assertEqual(start, delay)
        self.assertTrue(np.allclose(transfer[in_band], gain, rtol=1e-3, atol=1e-6))
        self.assertEqual(transfer[0], 1.0)  # fuera de la banda medida: bypass

    def test_ensemble_averages_transfer_and_reports_high_coherence(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)
        gains = (0.5, 0.6, 0.7)
        delays = (127, 311, 503)
        recordings = [
            np.concatenate((np.zeros(delay), gain * sweep, np.zeros(500)))
            for gain, delay in zip(gains, delays)
        ]

        transfer, coherence, starts = estimate_transfer_ensemble(
            sweep,
            recordings,
            sample_rate=sample_rate,
            fft_size=512,
            start_freq=500.0,
            stop_freq=10_000.0,
        )

        frequencies = np.fft.rfftfreq(512, d=1.0 / sample_rate)
        in_band = (frequencies >= 500.0) & (frequencies <= 9_000.0)
        self.assertEqual(starts, list(delays))
        self.assertTrue(np.allclose(transfer[in_band], np.mean(gains), rtol=1e-3))
        self.assertTrue(np.all(coherence[in_band] > 0.98))
        self.assertTrue(np.all(coherence[in_band] < 1.0))
        self.assertEqual(transfer[0], 1.0)
        self.assertEqual(coherence[0], 0.0)

    def test_incoherent_noise_lowers_ensemble_coherence(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)
        rng = np.random.default_rng(17)
        recordings = [
            np.concatenate(
                (np.zeros(delay), 0.6 * sweep + rng.normal(0.0, 0.05, sweep.size), np.zeros(500))
            )
            for delay in (127, 311, 503, 701)
        ]

        _, coherence, _ = estimate_transfer_ensemble(
            sweep,
            recordings,
            sample_rate=sample_rate,
            fft_size=512,
            start_freq=500.0,
            stop_freq=10_000.0,
        )

        frequencies = np.fft.rfftfreq(512, d=1.0 / sample_rate)
        in_band = (frequencies >= 1_000.0) & (frequencies <= 8_000.0)
        self.assertTrue(np.all(coherence[in_band] < 1.0))
        self.assertTrue(np.all(coherence[in_band] >= 0.0))

    def test_ensemble_requires_at_least_two_sweeps(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)
        recording = np.concatenate((np.zeros(100), sweep, np.zeros(100)))

        with self.assertRaisesRegex(ValueError, "al menos dos barridos"):
            estimate_transfer_ensemble(
                sweep,
                [recording],
                sample_rate=sample_rate,
                fft_size=256,
                start_freq=500.0,
                stop_freq=10_000.0,
            )

    def test_farina_deconvolution_recovers_delayed_linear_gain(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)
        delay = 137
        gain = 0.6
        recorded = np.concatenate((np.zeros(delay), gain * sweep, np.zeros(500)))

        impulse, peak_sample = estimate_impulse_response_farina(
            sweep,
            recorded,
            sample_rate=sample_rate,
            start_freq=500.0,
            stop_freq=10_000.0,
        )

        self.assertEqual(peak_sample, delay + sweep.size - 1)
        self.assertAlmostEqual(float(impulse[peak_sample]), gain, places=5)

    def test_farina_deconvolution_rejects_invalid_frequency_band(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)

        with self.assertRaisesRegex(ValueError, "frecuencias"):
            estimate_impulse_response_farina(
                sweep,
                sweep,
                sample_rate=sample_rate,
                start_freq=10_000.0,
                stop_freq=500.0,
            )
