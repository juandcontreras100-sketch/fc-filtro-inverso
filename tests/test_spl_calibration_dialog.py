import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from src.ui.calibration import CalibrationDialog


class SplCalibrationDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-calibracion-spl")

    def setUp(self):
        self.sample_rate = 1_000.0
        self.dialog = CalibrationDialog(self.sample_rate)
        self.dialog.show()

    def tearDown(self):
        self.dialog.close()

    def _start_measurement(self):
        if self.dialog.stacked_widget.currentIndex() == 0:
            self.dialog.next_button.click()
        self.dialog.start_button.click()

    def _constant_block(self, rms_db, size):
        amplitude = 10.0 ** (rms_db / 20.0)
        return np.full(size, amplitude, dtype=np.float64)

    def _finish_valid_measurement(self):
        self._start_measurement()
        self.dialog.add_block(self._constant_block(-26.0, int(self.sample_rate)))

    def test_dialog_has_three_steps_and_wraps_all_instruction_labels(self):
        self.assertEqual(self.dialog.stacked_widget.count(), 3)
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 0)
        self.assertIn("94 dB SPL", self.dialog.preparation_label.text())
        self.assertIn("1 kHz", self.dialog.preparation_label.text())

        self.dialog.next_button.click()
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)
        self.assertTrue(self.dialog.start_button.isVisible())
        self.assertIn("1 segundo", self.dialog.measurement_label.text())

        self._finish_valid_measurement()
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 2)
        self.assertIn("Offset", self.dialog.result_label.text())
        self.assertTrue(
            all(label.wordWrap() for label in self.dialog.findChildren(QtWidgets.QLabel))
        )

    def test_valid_measurement_computes_offset_and_apply_accepts(self):
        accepted = []
        self.dialog.accepted.connect(lambda: accepted.append(True))
        self._start_measurement()
        block = self._constant_block(-26.0, 250)

        for _ in range(4):
            self.dialog.add_block(block)

        self.assertAlmostEqual(self.dialog.offset_db, 120.0, places=6)
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 2)
        self.assertIn("120.00", self.dialog.result_label.text())

        self.dialog.apply_button.click()
        self.assertEqual(accepted, [True])
        self.assertEqual(self.dialog.result(), QtWidgets.QDialog.Accepted)

    def test_reference_spl_is_adjustable(self):
        self.dialog.reference_spin.setValue(100.0)

        self._finish_valid_measurement()

        self.assertAlmostEqual(self.dialog.offset_db, 126.0, places=6)

    def test_rms_at_or_below_minus_70_dbfs_is_rejected_with_gain_advice(self):
        for rms_db in (-70.0, -71.0):
            with self.subTest(rms_db=rms_db):
                self._start_measurement()
                self.dialog.add_block(
                    self._constant_block(rms_db, int(self.sample_rate))
                )

                self.assertIsNone(self.dialog.offset_db)
                self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)
                self.assertIn("Sube la ganancia", self.dialog.status_label.text())
                self.assertIn("acerca la referencia", self.dialog.status_label.text())

    def test_peak_at_or_above_minus_1_dbfs_is_rejected_with_clipping_advice(self):
        for peak_db in (-1.0, 0.0):
            with self.subTest(peak_db=peak_db):
                self._start_measurement()
                self.dialog.add_block(
                    self._constant_block(peak_db, int(self.sample_rate))
                )

                self.assertIsNone(self.dialog.offset_db)
                self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)
                self.assertIn("Baja la ganancia", self.dialog.status_label.text())
                self.assertIn("saturacion", self.dialog.status_label.text())

    def test_block_rms_variation_at_or_above_1_db_is_rejected(self):
        for second_db in (-25.0, -24.0):
            with self.subTest(second_db=second_db):
                self._start_measurement()
                self.dialog.add_block(self._constant_block(-26.0, 500))
                self.dialog.add_block(self._constant_block(second_db, 500))

                self.assertIsNone(self.dialog.offset_db)
                self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)
                self.assertIn("Manten estable la referencia", self.dialog.status_label.text())

    def test_accept_is_blocked_until_a_valid_measurement_exists(self):
        accepted = []
        self.dialog.accepted.connect(lambda: accepted.append(True))

        self.dialog.accept()
        self.assertEqual(accepted, [])
        self.assertNotEqual(self.dialog.result(), QtWidgets.QDialog.Accepted)

        self._start_measurement()
        self.dialog.add_block(self._constant_block(-80.0, int(self.sample_rate)))
        self.dialog.accept()

        self.assertEqual(accepted, [])
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)
        self.assertIsNone(self.dialog.offset_db)

    def test_measurement_requires_one_second_of_samples_and_ignores_idle_blocks(self):
        self.dialog.next_button.click()
        self.dialog.add_block(self._constant_block(-26.0, int(self.sample_rate)))
        self.assertIsNone(self.dialog.offset_db)
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)

        self.dialog.start_button.click()
        self.dialog.add_block(self._constant_block(-26.0, 999))
        self.assertIsNone(self.dialog.offset_db)
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 1)

        self.dialog.add_block(self._constant_block(-26.0, 1))
        self.assertAlmostEqual(self.dialog.offset_db, 120.0, places=6)
        self.assertEqual(self.dialog.stacked_widget.currentIndex(), 2)

        self.dialog.add_block(self._constant_block(-10.0, 100))
        self.assertAlmostEqual(self.dialog.offset_db, 120.0, places=6)


if __name__ == "__main__":
    unittest.main()
