import unittest

from src.ui.spl_context import spl_context_text, spl_zone_color
from src.ui.theme import SPL_DANGER, SPL_SAFE, SPL_WARNING


class SplContextTextTests(unittest.TestCase):
    def test_anchors_and_boundaries_return_the_exact_context(self):
        cases = {
            30.0: "~30 dB: susurro",
            45.0: "~60 dB: conversacion normal",
            60.0: "~60 dB: conversacion normal",
            70.0: "~80 dB: calle con trafico",
            80.0: "~80 dB: calle con trafico",
            85.0: "~80 dB: calle con trafico",
            88.0: "~90 dB o mas: grito",
            90.0: "~90 dB o mas: grito",
        }

        for level_spl, expected in cases.items():
            with self.subTest(level_spl=level_spl):
                self.assertEqual(spl_context_text(level_spl), expected)


class SplZoneColorTests(unittest.TestCase):
    def test_anchors_and_boundaries_return_theme_tokens(self):
        cases = {
            30.0: SPL_SAFE,
            45.0: SPL_SAFE,
            60.0: SPL_SAFE,
            70.0: SPL_WARNING,
            80.0: SPL_WARNING,
            85.0: SPL_WARNING,
            88.0: SPL_DANGER,
            90.0: SPL_DANGER,
        }

        for level_spl, expected in cases.items():
            with self.subTest(level_spl=level_spl):
                self.assertEqual(spl_zone_color(level_spl), expected)


if __name__ == "__main__":
    unittest.main()
