import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rusto import RustO, InitializeConfig, OcrRunOptions, RustoBindingError

class TestRustOPythonBinding(unittest.TestCase):
    def test_imports_and_types(self):
        self.assertTrue(callable(RustO.initialize))
        self.assertTrue(callable(RustO.initialize_from_json))

    def test_invalid_json_initialization_raises_binding_error(self):
        with self.assertRaises(RustoBindingError):
            RustO.initialize_from_json("invalid-non-json")

if __name__ == "__main__":
    unittest.main()
