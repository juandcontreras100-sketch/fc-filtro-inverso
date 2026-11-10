import unittest

from iniciar_demo_laptop import (
    find_device,
    has_bluetooth_output,
    resolve_laptop_pair,
)

HOSTAPIS = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
]


def device(name, *, hostapi=0, inputs=0, outputs=0):
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


class LaptopLauncherTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            device("Micrófono (USB PnP Sound Device", inputs=1),
            device("Microphone Array (Qualcomm(R) Aqstic", inputs=2),
            device("Auriculares (HUAWEI FreeBuds SE", hostapi=1, outputs=8),
            device("Altavoces (USB PnP Sound Device", outputs=2),
            device("Speakers (Qualcomm(R) Aqstic(TM", outputs=8),
        ]

    def test_resolves_laptop_microphone_and_speakers_by_name(self):
        microphone, speaker = resolve_laptop_pair(self.devices, HOSTAPIS)

        self.assertEqual(microphone, 1)
        self.assertEqual(speaker, 4)

    def test_never_returns_pcm2902_or_bluetooth(self):
        microphone, speaker = resolve_laptop_pair(self.devices, HOSTAPIS)

        self.assertNotIn("USB PnP", str(self.devices[microphone]["name"]))
        self.assertNotIn("USB PnP", str(self.devices[speaker]["name"]))
        self.assertNotIn("HUAWEI", str(self.devices[speaker]["name"]))

    def test_prefers_mme_over_other_host_apis(self):
        devices = [
            device("Speakers (Qualcomm(R) Aqstic", hostapi=2, outputs=8),
            device("Speakers (Qualcomm(R) Aqstic", hostapi=0, outputs=8),
        ]

        index = find_device(
            devices,
            HOSTAPIS,
            kind="output",
            name_tokens=("speakers",),
        )

        self.assertEqual(index, 1)

    def test_prefers_microphone_array_over_generic_microphone(self):
        devices = [
            device("Micrófono (placa de sonido)", inputs=1),
            device("Microphone Array (laptop)", inputs=2),
        ]

        index = find_device(
            devices,
            HOSTAPIS,
            kind="input",
            name_tokens=("microphone array", "micrófono"),
        )

        self.assertEqual(index, 1)

    def test_reports_bluetooth_output(self):
        self.assertTrue(has_bluetooth_output(self.devices))

        without_bluetooth = [self.devices[1], self.devices[4]]
        self.assertFalse(has_bluetooth_output(without_bluetooth))

    def test_explains_when_the_pair_is_missing(self):
        with self.assertRaises(SystemExit) as raised:
            resolve_laptop_pair([], HOSTAPIS)

        self.assertIn("micrófono", str(raised.exception))
        self.assertIn("parlantes", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
