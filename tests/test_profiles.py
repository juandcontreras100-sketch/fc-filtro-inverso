"""Perfil de usuario, persistencia y carpeta unica de grabaciones."""

import json
import tempfile
import unittest
from pathlib import Path

from src.audio.engine import AudioEngine
from src.storage.profiles import ProfileStore, UserProfile
from src.storage.recorder import SessionRecorder


class UserProfileTests(unittest.TestCase):
    def test_normalizes_and_builds_folder_name(self):
        user = UserProfile("  Juan  Pérez ", " 1.234 567 ")
        self.assertEqual(user.name, "Juan Pérez")
        self.assertEqual(user.document, "1.234567")
        self.assertEqual(user.folder_name, "juan_perez_1234567")

    def test_rejects_empty_or_unsafe_values(self):
        for name, document in (("", "1"), ("Ana", ""), ("Ana", "../x")):
            with self.assertRaises(ValueError):
                UserProfile(name, document)


class ProfileStoreTests(unittest.TestCase):
    def test_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(Path(tmp) / "perfiles.json")
            self.assertIsNone(store.last_user())
            self.assertEqual(store.last_devices(), (None, None))
            store.save(UserProfile("Ana", "99"), "Mic X", "Spk Y")
            self.assertEqual(store.last_user(), UserProfile("Ana", "99"))
            self.assertEqual(store.last_devices(), ("Mic X", "Spk Y"))

    def test_corrupt_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perfiles.json"
            path.write_text("{no es json", encoding="utf-8")
            self.assertIsNone(ProfileStore(path).last_user())


class RecorderUserTests(unittest.TestCase):
    def test_recordings_accumulate_in_one_user_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = AudioEngine(sample_rate=48_000.0, fft_size=256)
            user = UserProfile("Ana Gómez", "55")
            first = SessionRecorder(engine, tmp, user=user, devices={"microfono": "M"})
            one = first.start()
            first.stop()
            second = SessionRecorder(engine, tmp, user=user)
            two = second.start()
            second.stop()
            self.assertEqual(one.parent, two.parent)
            self.assertEqual(one.parent.name, "ana_gomez_55")
            self.assertNotEqual(one, two)
            metadata = json.loads((one / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["usuario"], {"nombre": "Ana Gómez", "documento": "55"})
            self.assertEqual(metadata["dispositivos"], {"microfono": "M"})


if __name__ == "__main__":
    unittest.main()
