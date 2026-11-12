import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from src.ui.spl_persistence import (
    SplCalibration,
    input_device_name,
    load_spl_calibration,
    save_spl_calibration,
)


class SplPersistenceTests(unittest.TestCase):
    def test_roundtrip_preserves_offset_date_and_device_with_exact_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibracion.json"

            saved = save_spl_calibration(118.25, "Microfono USB", path)
            loaded = load_spl_calibration(path)

            self.assertEqual(loaded, saved)
            self.assertEqual(loaded.offset_db, 118.25)
            self.assertEqual(loaded.device_name, "Microfono USB")
            parsed_date = datetime.fromisoformat(saved.date.replace("Z", "+00:00"))
            self.assertIsNotNone(parsed_date.tzinfo)
            self.assertEqual(parsed_date.utcoffset(), timedelta(0))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload), {"offset_db", "date", "device_name"}
            )
            self.assertEqual(payload["date"], saved.date)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ausente.json"

            self.assertIsNone(load_spl_calibration(path))

    def test_malformed_json_raises_a_clear_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibracion.json"
            path.write_text("{esto no es JSON", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SPL"):
                load_spl_calibration(path)

    def test_invalid_fields_raise_value_error(self):
        valid = {
            "offset_db": 120.0,
            "date": "2026-11-12T10:30:00+00:00",
            "device_name": "Entrada USB",
        }
        invalid_payloads = [
            [],
            {key: value for key, value in valid.items() if key != "offset_db"},
            {**valid, "offset_db": "120"},
            {**valid, "offset_db": math.inf},
            {**valid, "date": "ayer"},
            {**valid, "date": "2026-11-12T10:30:00"},
            {**valid, "device_name": None},
            {**valid, "device_name": ""},
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibracion.json"
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_spl_calibration(path)

    def test_save_rejects_non_finite_offsets_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibracion.json"
            for offset in (math.nan, math.inf, -math.inf):
                with self.subTest(offset=offset):
                    with self.assertRaisesRegex(ValueError, "offset_db"):
                        save_spl_calibration(offset, "Entrada USB", path)
                    self.assertFalse(path.exists())


class InputDeviceNameTests(unittest.TestCase):
    def test_duplex_tuple_and_list_resolve_the_input_device(self):
        sounddevice = SimpleNamespace(
            query_devices=Mock(return_value={"name": "Microfono USB"})
        )

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            self.assertEqual(input_device_name((4, 9)), "Microfono USB")
            sounddevice.query_devices.assert_called_once_with(4)
            sounddevice.query_devices.reset_mock()
            self.assertEqual(input_device_name([5, 8]), "Microfono USB")
            sounddevice.query_devices.assert_called_once_with(5)

    def test_none_resolves_the_default_input(self):
        sounddevice = SimpleNamespace(
            query_devices=Mock(return_value={"name": "Entrada predeterminada"})
        )

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            self.assertEqual(input_device_name(None), "Entrada predeterminada")

        sounddevice.query_devices.assert_called_once_with(kind="input")

    def test_index_and_name_resolve_that_device(self):
        sounddevice = SimpleNamespace(
            query_devices=Mock(return_value={"name": "Entrada consultada"})
        )

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            self.assertEqual(input_device_name(3), "Entrada consultada")
            sounddevice.query_devices.assert_called_once_with(3)
            sounddevice.query_devices.reset_mock()
            self.assertEqual(input_device_name("USB"), "Entrada consultada")
            sounddevice.query_devices.assert_called_once_with("USB")


class MainSplStartupTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimpleNamespace(spl_offset_db=None)
        self.qt_widgets = SimpleNamespace(
            QMessageBox=SimpleNamespace(warning=Mock())
        )

    def test_cli_offset_has_absolute_priority_over_file_and_device_warning(self):
        with (
            patch.object(main, "load_spl_calibration") as load,
            patch.object(main, "input_device_name") as device_name,
        ):
            main.apply_spl_calibration(
                self.engine, 117.5, (3, 4), self.qt_widgets
            )

        self.assertEqual(self.engine.spl_offset_db, 117.5)
        load.assert_not_called()
        device_name.assert_not_called()
        self.qt_widgets.QMessageBox.warning.assert_not_called()

    def test_file_offset_is_applied_without_warning_for_the_same_device(self):
        calibration = SplCalibration(119.0, "2026-11-12T10:30:00+00:00", "USB")

        with (
            patch.object(main, "load_spl_calibration", return_value=calibration),
            patch.object(main, "input_device_name", return_value="USB"),
        ):
            main.apply_spl_calibration(
                self.engine, None, (3, 4), self.qt_widgets
            )

        self.assertEqual(self.engine.spl_offset_db, 119.0)
        self.qt_widgets.QMessageBox.warning.assert_not_called()

    def test_device_change_warns_in_spanish_and_preserves_the_offset(self):
        calibration = SplCalibration(
            119.0, "2026-11-12T10:30:00+00:00", "Microfono anterior"
        )

        with (
            patch.object(main, "load_spl_calibration", return_value=calibration),
            patch.object(main, "input_device_name", return_value="Microfono actual"),
        ):
            main.apply_spl_calibration(self.engine, None, 3, self.qt_widgets)

        self.assertEqual(self.engine.spl_offset_db, 119.0)
        self.qt_widgets.QMessageBox.warning.assert_called_once()
        message = self.qt_widgets.QMessageBox.warning.call_args.args[2]
        self.assertIn("recalibrar", message)
        self.assertIn("Microfono anterior", message)
        self.assertIn("Microfono actual", message)
        self.assertTrue(message.isascii())

    def test_invalid_file_warns_and_starts_uncalibrated(self):
        errors = io.StringIO()

        with (
            patch.object(main, "load_spl_calibration", side_effect=ValueError("JSON roto")),
            patch.object(main, "input_device_name") as device_name,
            redirect_stderr(errors),
        ):
            main.apply_spl_calibration(self.engine, None, None, self.qt_widgets)

        self.assertIsNone(self.engine.spl_offset_db)
        self.assertIn("JSON roto", errors.getvalue())
        device_name.assert_not_called()
        self.qt_widgets.QMessageBox.warning.assert_not_called()

    def test_spl_offset_argument_defaults_to_none_and_accepts_float(self):
        with patch.object(sys, "argv", ["main.py"]):
            self.assertIsNone(main.parse_args().spl_offset)
        with patch.object(sys, "argv", ["main.py", "--spl-offset", "121.25"]):
            self.assertEqual(main.parse_args().spl_offset, 121.25)


if __name__ == "__main__":
    unittest.main()
