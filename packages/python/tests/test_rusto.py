import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rusto import (
    RustO,
    InitializeConfig,
    OcrRunOptions,
    OptimizationOptions,
    RustoBindingError,
    create_initialize_config_for_model,
    download_model,
    download_all_models,
)
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

    def test_real_ocr_indra_idcard_image(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        det_path = os.path.join(base_dir, "models/indra/det.rten")
        rec_path = os.path.join(base_dir, "models/indra/rec.rten")
        dict_path = os.path.join(base_dir, "models/indra/meta.json")
        img_path = os.path.join(base_dir, "models/test_images/idcard.jpg")

        if not os.path.exists(det_path) or not os.path.exists(img_path):
            self.skipTest("Indra models or test image not present")

        config_json = json.dumps({
            "detection": {"model_path": det_path},
            "recognition": {"model_path": rec_path, "dict_path": dict_path}
        })
        engine = RustO.initialize_from_json(config_json)

        opts = OcrRunOptions(output=OutputGranularity.LINES, text_score=0.3)
        res = engine.detect_text(ImageSource.PATH(img_path), opts)
        self.assertTrue(res.is_STRUCTURED())
        self.assertGreater(len(res.results), 0)
        joined = " ".join([r.text for r in res.results])
        self.assertTrue("UNITED" in joined or "PASSPORT" in joined or "BRITAIN" in joined)

    def test_run_options_with_configurable_opts(self):
        opts = OcrRunOptions(
            text_score=0.4,
            rec_batch_num=8,
            optimization=OptimizationOptions(enabled=True, target_max_side=960, crop_padding_x=0.05, crop_padding_y=0.08),
        )
        self.assertEqual(opts.rec_batch_num, 8)
        self.assertIsNotNone(opts.optimization)
        self.assertTrue(opts.optimization.enabled)
        self.assertEqual(opts.optimization.target_max_side, 960)

    def test_create_initialize_config_for_model(self):
        cfg = create_initialize_config_for_model(model="indraocr", lang="latin", model_path="/tmp/custom")
        self.assertEqual(cfg.model, "indraocr")
        self.assertEqual(cfg.lang, "latin")
        self.assertEqual(cfg.model_path, "/tmp/custom")

    def test_initialize_config_defaults(self):
        cfg = InitializeConfig(model="ppocrv6-tiny", model_path="/tmp/models")
        self.assertEqual(cfg.model, "ppocrv6-tiny")
        self.assertEqual(cfg.model_path, "/tmp/models")
        self.assertIsNone(cfg.template)
        self.assertIsNone(cfg.lang)

    def test_download_model_helpers_callable(self):
        self.assertTrue(callable(download_model))
        self.assertTrue(callable(download_all_models))
        self.assertTrue(callable(RustO.download_model))
        self.assertTrue(callable(RustO.download_all_models))

if __name__ == "__main__":
    unittest.main()
