import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rusto import RustO, InitializeConfig, OcrRunOptions, RustoBindingError
from rusto.generated.rusto import ImageSource, OutputGranularity

class TestRustOPythonBinding(unittest.TestCase):
    def test_imports_and_types(self):
        self.assertTrue(callable(RustO.initialize))
        self.assertTrue(callable(RustO.initialize_from_json))

    def test_invalid_json_initialization_raises_binding_error(self):
        with self.assertRaises(RustoBindingError):
            RustO.initialize_from_json("invalid-non-json")

    def test_real_ocr_example1_image(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        det_path = os.path.join(base_dir, "models/PPOCR_v6/det.rten")
        rec_path = os.path.join(base_dir, "models/PPOCR_v6/rec.rten")
        dict_path = os.path.join(base_dir, "models/PPOCR_v6/dict.txt")
        img_path = os.path.join(base_dir, "models/test_images/example1.png")

        if not os.path.exists(det_path) or not os.path.exists(img_path):
            self.skipTest("Models or test image not present")

        config_json = json.dumps({
            "detection": {"model_path": det_path},
            "recognition": {"model_path": rec_path, "dict_path": dict_path}
        })
        engine = RustO.initialize_from_json(config_json)

        opts_structured = OcrRunOptions(
            output=OutputGranularity.LINES,
            line_y_threshold=None, word_x_threshold=None, text_score=0.3,
            classification=None, orientation=None, min_height=None, max_side_len=None,
            min_side_len=None, width_height_ratio=None, detection=None, postprocess=None,
            calibration=None, optimization=None
        )
        res = engine.detect_text(ImageSource.PATH(img_path), opts_structured)
        self.assertTrue(res.is_STRUCTURED())
        self.assertGreater(len(res.results), 0)
        texts = [r.text for r in res.results]
        self.assertTrue(any("INVOICE" in t for t in texts))

        opts_spatial = OcrRunOptions(
            output=OutputGranularity.SPATIAL,
            line_y_threshold=None, word_x_threshold=None, text_score=0.3,
            classification=None, orientation=None, min_height=None, max_side_len=None,
            min_side_len=None, width_height_ratio=None, detection=None, postprocess=None,
            calibration=None, optimization=None
        )
        spatial_res = engine.detect_text(ImageSource.PATH(img_path), opts_spatial)
        self.assertTrue(spatial_res.is_SPATIAL())
        self.assertIn("INVOICE", spatial_res.text)

if __name__ == "__main__":
    unittest.main()
