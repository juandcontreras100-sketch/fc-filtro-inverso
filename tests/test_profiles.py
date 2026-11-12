"""Perfiles de usuario: validacion, carpeta segura y persistencia tolerante."""

import json
import tempfile
import unittest
from pathlib import Path

from src.storage.profiles import ProfileStore, UserProfile


class UserProfileTests(unittest.TestCase):
    def test_name_and_document_are_normalized(self):
        profile = UserProfile("  Juan   Perez ", " 1234567 ")

        self.assertEqual(profile.name, "Juan Perez")
        self.assertEqual(profile.document, "1234567")

    def test_name_and_document_are_required(self):
        with self.assertRaises(ValueError):
            UserProfile("   ", "1234567")
        with self.assertRaises(ValueError):
            UserProfile("Juan", "   ")

    def test_document_only_accepts_letters_numbers_dot_and_hyphen(self):
        for document in ("12 345", "12/345", "12_345", "12,345"):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    UserProfile("Juan", document)

        self.assertEqual(UserProfile("Juan", "AB.12-3").document, "AB.12-3")

    def test_folder_name_removes_accents_and_unsafe_characters(self):
        profile = UserProfile("Juan Pérez", "12.345-6")

        self.assertEqual(profile.folder_name, "juan_perez_12_345_6")

    def test_folder_name_can_still_be_built_with_symbols_in_the_name(self):
        profile = UserProfile("!!", "1234567")

        self.assertTrue(profile.folder_name.endswith("_1234567"))


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "perfiles.json"
        self.store = ProfileStore(self.path)

    def test_a_missing_file_means_no_profile(self):
        self.assertIsNone(self.store.load())

    def test_save_and_load_round_trip_the_user_and_devices(self):
        profile = UserProfile(
            "Juan Pérez",
            "1234567",
            input_device="Microphone (USB Audio)",
            output_device="Speakers (USB Audio)",
        )

        self.store.save(profile)
        loaded = self.store.load()

        self.assertEqual(loaded, profile)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["usuario"], {"nombre": "Juan Pérez", "documento": "1234567"}
        )
        self.assertEqual(data["dispositivos"]["entrada"], "Microphone (USB Audio)")

    def test_a_corrupt_file_is_treated_as_no_profile(self):
        self.path.write_text("{esto no es json", encoding="utf-8")

        self.assertIsNone(self.store.load())

    def test_an_unknown_json_shape_is_treated_as_no_profile(self):
        self.path.write_text(json.dumps({"otra": "cosa"}), encoding="utf-8")

        self.assertIsNone(self.store.load())

    def test_an_invalid_stored_document_is_ignored(self):
        self.path.write_text(
            json.dumps(
                {
                    "usuario": {"nombre": "Juan", "documento": "12 34"},
                    "dispositivos": {"entrada": None, "salida": None},
                }
            ),
            encoding="utf-8",
        )

        self.assertIsNone(self.store.load())

    def test_save_creates_missing_parent_directories(self):
        nested = Path(self._tmp.name) / "sub" / "perfiles.json"

        ProfileStore(nested).save(UserProfile("Ana", "123"))

        self.assertTrue(nested.exists())


if __name__ == "__main__":
    unittest.main()
