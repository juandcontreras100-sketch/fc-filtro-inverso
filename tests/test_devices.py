"""Seleccion de dispositivos: sin PortAudio, con datos tomados de un PC real."""

import unittest

from src.audio.devices import build_device_options, combine_device_arg, parse_device

HOSTAPIS = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
    {"name": "Windows WDM-KS"},
]


def _dev(name, hostapi, n_in, n_out):
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": n_in,
        "max_output_channels": n_out,
    }


# Subconjunto de la salida real de sd.query_devices() de un portatil Qualcomm:
# ningun dispositivo es full-duplex y los nombres MME vienen truncados.
LAPTOP = [
    _dev("Microphone Array (Qualcomm(R) A", 0, 2, 0),  # 0
    _dev("Speakers (Qualcomm(R) Aqstic(TM", 0, 0, 8),  # 1
    _dev("Microphone Array (Qualcomm(R) Aqstic(TM) ACX Static Endpoints Audio Device)", 2, 2, 0),  # 2
    _dev("Speakers (Qualcomm(R) Aqstic(TM) Audio Adapter Device)", 2, 0, 2),  # 3
    _dev("Auriculares con micrófono ()", 3, 1, 0),  # 4
    _dev("Output ()", 3, 0, 1),  # 5
]

# Lo que Windows mostraria con el ESP32-S3 como tarjeta de sonido USB.
ESP32 = [
    _dev("Microphone (ESP32-S3 USB Audio)", 0, 1, 0),  # 6
    _dev("Speakers (ESP32-S3 USB Audio)", 0, 0, 2),  # 7
    _dev("Microphone (ESP32-S3 USB Audio)", 2, 1, 0),  # 8
    _dev("Speakers (ESP32-S3 USB Audio)", 2, 0, 2),  # 9
]


def _values(options):
    return [value for value, _label in options]


class BuildDeviceOptionsTests(unittest.TestCase):
    def test_the_first_option_is_always_the_system_default(self):
        options = build_device_options(LAPTOP, HOSTAPIS)

        self.assertIsNone(options[0][0])
        self.assertIn("Sistema", options[0][1])

    def test_no_duplex_devices_still_produces_usable_pairs(self):
        options = build_device_options(LAPTOP, HOSTAPIS)

        self.assertGreater(len(options), 1)

    def test_truncated_mme_names_are_paired(self):
        values = _values(build_device_options(LAPTOP, HOSTAPIS))

        self.assertIn((0, 1), values)

    def test_full_wasapi_names_are_paired(self):
        values = _values(build_device_options(LAPTOP, HOSTAPIS))

        self.assertIn((2, 3), values)

    def test_empty_parentheses_never_pair_unrelated_devices(self):
        values = _values(build_device_options(LAPTOP, HOSTAPIS))

        self.assertNotIn((4, 5), values)

    def test_pairs_never_cross_host_apis(self):
        devices = LAPTOP + ESP32
        values = _values(build_device_options(devices, HOSTAPIS))

        for in_index, out_index in values[1:]:
            self.assertEqual(devices[in_index]["hostapi"], devices[out_index]["hostapi"])

    def test_the_esp32_input_and_output_are_paired_by_their_shared_name(self):
        devices = LAPTOP + ESP32
        options = build_device_options(devices, HOSTAPIS)

        values = _values(options)
        self.assertIn((6, 7), values)
        self.assertIn((8, 9), values)
        self.assertNotIn((6, 9), values)
        self.assertNotIn((8, 7), values)

    def test_labels_name_the_host_api_and_both_devices(self):
        devices = LAPTOP + ESP32
        options = dict(build_device_options(devices, HOSTAPIS))

        label = options[(8, 9)]
        self.assertIn("WASAPI", label)
        self.assertIn("ESP32-S3 USB Audio", label)

    def test_wasapi_is_listed_before_mme(self):
        devices = LAPTOP + ESP32
        values = _values(build_device_options(devices, HOSTAPIS))[1:]
        apis = [devices[value[0]]["hostapi"] for value in values]

        self.assertLess(apis.index(2), apis.index(0))

    def test_wdm_ks_devices_are_not_offered_automatically(self):
        # WDM-KS repite cada endpoint (Output 1, Output 2...) y da combinaciones absurdas.
        devices = [
            _dev("Input (QCAUD Wave Speaker)", 3, 2, 0),
            _dev("Output 1 (QCAUD Wave Speaker)", 3, 0, 2),
            _dev("Interfaz (USB Audio)", 3, 2, 2),
        ]

        self.assertEqual(_values(build_device_options(devices, HOSTAPIS)), [None])

    def test_a_name_that_is_a_strict_prefix_of_another_is_not_a_truncation(self):
        devices = [
            _dev("Input (QCAUD Wave Speaker)", 2, 2, 0),
            _dev("Output 1 (QCAUD Wave Speaker Headset)", 2, 0, 2),
        ]

        self.assertEqual(_values(build_device_options(devices, HOSTAPIS)), [None])

    def test_a_real_duplex_device_is_offered_by_its_index(self):
        devices = [_dev("Interfaz USB", 0, 2, 2)]
        values = _values(build_device_options(devices, HOSTAPIS))

        self.assertIn(0, values)

    def test_a_duplex_device_is_not_also_paired_with_itself(self):
        devices = [_dev("Interfaz USB", 0, 2, 2)]
        values = _values(build_device_options(devices, HOSTAPIS))

        self.assertNotIn((0, 0), values)

    def test_empty_device_list_only_offers_the_default(self):
        options = build_device_options([], HOSTAPIS)

        self.assertEqual(_values(options), [None])


class ArgumentParsingTests(unittest.TestCase):
    def test_parse_device_handles_none_index_and_name(self):
        self.assertIsNone(parse_device(None))
        self.assertEqual(parse_device("7"), 7)
        self.assertEqual(parse_device("ESP32"), "ESP32")

    def test_a_single_device_argument_is_used_for_both_directions(self):
        self.assertEqual(combine_device_arg("3", None, None), 3)

    def test_input_and_output_arguments_build_a_pair(self):
        self.assertEqual(combine_device_arg(None, "6", "7"), (6, 7))

    def test_a_pair_can_mix_an_index_and_a_name(self):
        self.assertEqual(combine_device_arg(None, "6", "Speakers"), (6, "Speakers"))

    def test_nothing_given_means_the_system_default(self):
        self.assertIsNone(combine_device_arg(None, None, None))

    def test_only_one_side_keeps_the_other_on_the_system_default(self):
        self.assertEqual(combine_device_arg(None, "6", None), (6, None))

    def test_device_and_pair_arguments_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            combine_device_arg("3", "6", "7")


if __name__ == "__main__":
    unittest.main()
