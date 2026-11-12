import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

import pyqtgraph as pg

from src.ui.meters import MeterPanel


class MeterPanelClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-medidores-clip")

    def test_input_and_output_clip_latches_are_independent(self):
        panel = MeterPanel()

        panel.set_input(-12.0, -0.5, would_clip=True)
        panel.set_output(-18.0, -6.0, would_clip=False)
        self.assertTrue(panel.input_clip._hot)
        self.assertFalse(panel.output_clip._hot)

        panel.set_input(-12.0, -6.0, would_clip=False)
        panel.set_output(-18.0, -0.5, would_clip=True)
        self.assertTrue(panel.input_clip._hot)
        self.assertTrue(panel.output_clip._hot)


if __name__ == "__main__":
    unittest.main()
