"""Seleccion de dispositivos: sin PortAudio, con datos tomados de un PC real."""

import unittest

from src.audio.devices import (
    build_device_options,
    build_input_options,
    build_output_options,
    combine_device_arg,
    device_name,
    device_pair,
    find_device_by_name,
    parse_device,
)

HOSTAPIS = [
    {"name": "MME", "default_input_device": 0, "default_output_device": 1},
    {"name": "Windows DirectSound", "default_input_device": -1, "default_output_device": -1},
    {"name": "Windows WASAPI", "default_input_device": 2, "default_output_device": 3},
    {"name": "Windows WDM-KS", "default_input_device": -1, "default_output_device": -1},
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


class InputOutputOptionsTests(unittest.TestCase):
    def test_input_options_start_with_the_system_default(self):
        options = build_input_options(LAPTOP, HOSTAPIS)

        self.assertIsNone(options[0][0])
        self.assertIn("por defecto del sistema", options[0][1])

    def test_input_options_only_list_devices_with_input_channels(self):
        options = build_input_options(LAPTOP, HOSTAPIS)
        values = _values(options)

        self.assertIn(0, values)
        self.assertIn(2, values)
        self.assertNotIn(1, values)
        self.assertNotIn(3, values)

    def test_input_options_keep_wasapi_before_mme(self):
        options = build_input_options(LAPTOP, HOSTAPIS)
        values = _values(options)

        self.assertLess(values.index(2), values.index(0))

    def test_output_options_are_filtered_to_the_input_host_api(self):
        mme_outputs = _values(build_output_options(LAPTOP, HOSTAPIS, 0))
        wasapi_outputs = _values(build_output_options(LAPTOP, HOSTAPIS, 2))

        self.assertEqual(mme_outputs, [None, 1])
        self.assertEqual(wasapi_outputs, [None, 3])

    def test_output_options_without_an_input_list_every_api(self):
        values = _values(build_output_options(LAPTOP, HOSTAPIS, None))

        self.assertIn(1, values)
        self.assertIn(3, values)

    def test_output_options_exclude_wdm_ks(self):
        devices = LAPTOP + [
            _dev("Input (QCAUD Wave Speaker)", 3, 2, 0),
            _dev("Output (QCAUD Wave Speaker)", 3, 0, 2),
        ]

        self.assertEqual(_values(build_output_options(devices, HOSTAPIS, 6)), [None])

    def test_output_options_start_with_the_system_default(self):
        options = build_output_options(LAPTOP, HOSTAPIS, 2)

        self.assertIsNone(options[0][0])
        self.assertIn("por defecto del sistema", options[0][1])

    def test_a_pair_uses_the_selected_indices(self):
        self.assertEqual(device_pair(0, 3, LAPTOP, HOSTAPIS), (0, 3))

    def test_a_default_output_is_resolved_inside_the_input_host_api(self):
        self.assertEqual(device_pair(0, None, LAPTOP, HOSTAPIS), (0, 1))
        self.assertEqual(device_pair(2, None, LAPTOP, HOSTAPIS), (2, 3))

    def test_a_default_input_is_resolved_inside_the_output_host_api(self):
        self.assertEqual(device_pair(None, 3, LAPTOP, HOSTAPIS), (2, 3))

    def test_without_an_api_default_the_other_side_stays_on_the_system_default(self):
        self.assertEqual(device_pair(4, None, LAPTOP, HOSTAPIS), (4, None))
        self.assertEqual(device_pair(None, 5, LAPTOP, HOSTAPIS), (None, 5))

    def test_both_defaults_mean_a_none_device(self):
        self.assertIsNone(device_pair(None, None, LAPTOP, HOSTAPIS))

    def test_device_name_resolves_indices_names_and_defaults(self):
        self.assertIsNone(device_name(LAPTOP, None))
        self.assertEqual(device_name(LAPTOP, 3), LAPTOP[3]["name"])
        self.assertEqual(device_name(LAPTOP, "Mi tarjeta"), "Mi tarjeta")
        self.assertIsNone(device_name(LAPTOP, 99))

    def test_find_device_by_name_is_case_insensitive_and_tolerates_changes(self):
        name = LAPTOP[3]["name"]

        self.assertEqual(find_device_by_name(LAPTOP, name), 3)
        self.assertEqual(find_device_by_name(LAPTOP, name.lower()), 3)
        self.assertIsNone(find_device_by_name(LAPTOP, "No existe"))
        self.assertIsNone(find_device_by_name(LAPTOP, None))


if __name__ == "__main__":
    unittest.main()
