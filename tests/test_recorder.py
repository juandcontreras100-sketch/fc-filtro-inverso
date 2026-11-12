"""Pruebas de la grabacion de sesiones en disco."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio.engine import AudioEngine
from src.storage.profiles import UserProfile
from src.storage.recorder import SessionRecorder

CSV_HEADER = [
    "time_s",
    "input_rms_db",
    "input_peak_db",
    "output_rms_db",
    "output_peak_db",
    "clip",
]


class SessionRecorderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.bins = self.engine.bin_count
        self.clock_values = [0.0]

    def _clock(self) -> float:
        return self.clock_values[-1]

    def _make_recorder(self) -> SessionRecorder:
        recorder = SessionRecorder(self.engine, root_dir=self.root, clock=self._clock)
        self.addCleanup(recorder.stop)
        return recorder

    @staticmethod
    def _levels(frame: int) -> dict:
        return {
            "input_rms_db": -20.0 - frame,
            "input_peak_db": -10.0 - frame,
            "output_rms_db": -21.0 - frame,
            "output_peak_db": -11.0 - frame,
            "clip": bool(frame % 2),
        }

    def _spectra(self, base: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.bins, base, dtype=np.float64),
            np.full(self.bins, base + 1.0, dtype=np.float64),
        )

    def test_metadata_matches_the_engine_state(self):
        calibration = self.root / "medicion_temporal.npy"
        np.save(calibration, np.ones(self.bins, dtype=np.complex128))
        np.save(
            self.root / "medicion_temporal_gamma2.npy",
            np.ones(self.bins, dtype=np.float64),
        )
        calibration.with_suffix(".json").write_text(
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
        self.engine.device = (1, 7)
        self.engine.epsilon_base = 0.03
        self.engine.max_gain_db = 9.0
        self.engine.bypass = False
        self.engine.load_calibration(calibration)

        recorder = self._make_recorder()
        directory = recorder.start()
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["sample_rate"], self.engine.sample_rate)
        self.assertEqual(metadata["fft_size"], self.engine.fft_size)
        self.assertEqual(metadata["device"], [1, 7])
        self.assertEqual(metadata["epsilon_base"], self.engine.epsilon_base)
        self.assertEqual(metadata["max_gain_db"], self.engine.max_gain_db)
        self.assertEqual(metadata["db_full_scale_db"], 140.0)
        self.assertEqual(metadata["calibration_file"], calibration.name)
        self.assertEqual(metadata["mode"], "ecualizado")

    def test_levels_csv_has_one_row_per_frame(self):
        recorder = self._make_recorder()
        recorder.start()

        for frame in range(2):
            self.clock_values.append(0.25 + 0.5 * frame)
            input_db, output_db = self._spectra(base=-30.0 - frame)
            recorder.record_frame(
                input_db=input_db,
                output_db=output_db,
                **self._levels(frame),
            )
        directory = recorder.stop()

        with (directory / "niveles.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, CSV_HEADER)

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["time_s"]), 0.25, places=6)
        self.assertAlmostEqual(float(rows[1]["time_s"]), 0.75, places=6)
        self.assertAlmostEqual(float(rows[0]["input_rms_db"]), -20.0, places=6)
        self.assertAlmostEqual(float(rows[0]["input_peak_db"]), -10.0, places=6)
        self.assertAlmostEqual(float(rows[0]["output_rms_db"]), -21.0, places=6)
        self.assertAlmostEqual(float(rows[0]["output_peak_db"]), -11.0, places=6)
        self.assertEqual(int(rows[0]["clip"]), 0)
        self.assertEqual(int(rows[1]["clip"]), 1)
        self.assertAlmostEqual(float(rows[1]["input_rms_db"]), -21.0, places=6)

    def test_spectra_npz_keeps_frames_and_frequencies(self):
        recorder = self._make_recorder()
        recorder.start()

        self.clock_values.append(0.05)
        input_db, output_db = self._spectra(base=-40.0)
        recorder.record_frame(
            input_db=input_db,
            output_db=output_db,
            **self._levels(0),
        )
        # La grabadora debe copiar los arreglos: mutar los originales no
        # puede cambiar lo que se guarda al detener.
        input_db[:] = 99.0
        output_db[:] = 99.0
        directory = recorder.stop()

        with np.load(directory / "espectros.npz") as data:
            frequencies = data["frequencies"]
            stored_input = data["input_db"]
            stored_output = data["output_db"]

        expected = np.fft.rfftfreq(self.engine.fft_size, 1.0 / self.engine.sample_rate)
        self.assertTrue(np.allclose(frequencies, expected))
        self.assertEqual(stored_input.shape, (1, self.bins))
        self.assertEqual(stored_output.shape, (1, self.bins))
        self.assertTrue(np.allclose(stored_input[0], -40.0))
        self.assertTrue(np.allclose(stored_output[0], -39.0))

    def test_spectra_are_decimated_to_about_ten_frames_per_second(self):
        recorder = self._make_recorder()
        recorder.start()

        for index in range(1, 31):
            self.clock_values.append(index / 30.0)
            input_db, output_db = self._spectra(base=-50.0)
            recorder.record_frame(
                input_db=input_db,
                output_db=output_db,
                **self._levels(index),
            )
        directory = recorder.stop()

        with np.load(directory / "espectros.npz") as data:
            frames = data["input_db"].shape[0]
        self.assertGreaterEqual(frames, 9)
        self.assertLessEqual(frames, 12)

    def test_periodic_flush_writes_readable_files_before_stop(self):
        recorder = self._make_recorder()
        directory = recorder.start()

        self.clock_values.append(0.5)
        input_db, output_db = self._spectra(base=-60.0)
        recorder.record_frame(
            input_db=input_db,
            output_db=output_db,
            **self._levels(0),
        )
        self.clock_values.append(2.5)
        input_db, output_db = self._spectra(base=-61.0)
        recorder.record_frame(
            input_db=input_db,
            output_db=output_db,
            **self._levels(1),
        )

        npz_path = directory / "espectros.npz"
        self.assertTrue(npz_path.exists())
        with np.load(npz_path) as data:
            self.assertEqual(data["input_db"].shape[1], self.bins)
            self.assertGreaterEqual(data["input_db"].shape[0], 1)

        with (directory / "niveles.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)

    def test_stop_twice_returns_none_the_second_time(self):
        recorder = self._make_recorder()
        recorder.start()
        self.assertTrue(recorder.is_recording)
        self.clock_values.append(0.1)
        input_db, output_db = self._spectra(base=-70.0)
        recorder.record_frame(
            input_db=input_db,
            output_db=output_db,
            **self._levels(0),
        )

        directory = recorder.stop()
        self.assertIsNotNone(directory)
        self.assertFalse(recorder.is_recording)
        self.assertIsNone(recorder.stop())

    def test_spectrum_with_wrong_size_is_ignored_but_level_row_is_kept(self):
        recorder = self._make_recorder()
        recorder.start()
        self.clock_values.append(0.1)
        recorder.record_frame(
            input_db=np.zeros(self.bins + 3),
            output_db=np.zeros(self.bins - 1),
            **self._levels(0),
        )
        directory = recorder.stop()

        with (directory / "niveles.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)

        with np.load(directory / "espectros.npz") as data:
            self.assertEqual(data["input_db"].shape, (0, self.bins))
            self.assertEqual(data["output_db"].shape, (0, self.bins))

    def test_start_while_recording_raises(self):
        recorder = self._make_recorder()
        recorder.start()
        with self.assertRaises(RuntimeError):
            recorder.start()

    def test_a_logged_in_session_lives_in_the_user_folder(self):
        profile = UserProfile("Juan Perez", "1234567")
        recorder = SessionRecorder(
            self.engine,
            root_dir=self.root,
            clock=self._clock,
            user=profile,
        )
        self.addCleanup(recorder.stop)

        directory = recorder.start()

        self.assertEqual(directory.parent, self.root / "juan_perez_1234567")
        self.assertEqual(directory.parent.parent, self.root)
        self.assertRegex(directory.name, r"^\d{4}-\d{2}-\d{2}_\d{6}$")

    def test_metadata_includes_the_user_and_devices(self):
        profile = UserProfile("Juan Perez", "1234567")
        devices = {"entrada": "Microfono USB", "salida": "Altavoz USB"}
        recorder = SessionRecorder(
            self.engine,
            root_dir=self.root,
            clock=self._clock,
            user=profile,
            devices=devices,
        )
        self.addCleanup(recorder.stop)

        directory = recorder.start()
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            metadata["usuario"],
            {"nombre": "Juan Perez", "documento": "1234567"},
        )
        self.assertEqual(metadata["dispositivos"], devices)

    def test_without_a_user_the_session_stays_in_the_root(self):
        recorder = self._make_recorder()

        directory = recorder.start()
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )

        self.assertEqual(directory.parent, self.root)
        self.assertIsNone(metadata["usuario"])
        self.assertEqual(metadata["dispositivos"], {})


if __name__ == "__main__":
    unittest.main()
