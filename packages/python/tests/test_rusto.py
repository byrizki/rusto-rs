import unittest
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rusto import (
    RustO,
    DetectTextBuilder,
    InitializeConfig,
    OcrRunOptions,
    PreDetectionOptions,
    DetectionRunOptions,
    PostDetectionOptions,
    PreRecognitionOptions,
    RecognitionRunOptions,
    PostRecognitionOptions,
    RustoBindingError,
    create_initialize_config_for_model,
    download_model,
    version,
    is_ready,
)
from rusto.generated.rusto import ImageSource, OutputGranularity


class TestRustOPythonBinding(unittest.TestCase):
    def test_imports_and_types(self):
        self.assertTrue(callable(RustO.initialize))
        self.assertTrue(callable(RustO.initialize_from_json))
        self.assertTrue(callable(RustO.detect_text_builder))
        self.assertTrue(callable(InitializeConfig.indra))
        self.assertTrue(callable(InitializeConfig.ppv6))
        self.assertTrue(callable(InitializeConfig.for_model))

    def test_preset_classmethods(self):
        indra_cfg = InitializeConfig.indra()
        self.assertEqual(indra_cfg.model, "indraocr")
        self.assertEqual(indra_cfg.template, "indra")

        ppv6_cfg = InitializeConfig.ppv6()
        self.assertEqual(ppv6_cfg.model, "ppocrv6-tiny")
        self.assertEqual(ppv6_cfg.template, "ppv6")

        custom_indra = InitializeConfig.indra("det.rten", "rec.rten", "meta.json")
        self.assertEqual(custom_indra.template, "indra")
        self.assertEqual(custom_indra.detection.model_path, "det.rten")

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

        with RustO.initialize(InitializeConfig.ppv6(det_path, rec_path, dict_path)) as engine:
            res = (
                engine.detect_text(img_path)
                .with_post_recognition(score_threshold=0.3)
                .as_lines()
                .run()
            )
            self.assertTrue(res.is_STRUCTURED())
            self.assertGreater(len(res.results), 0)
            texts = [r.text for r in res.results]
            self.assertTrue(any("INVOICE" in t for t in texts))

            spatial_res = (
                engine.detect_text(img_path)
                .with_post_recognition(score_threshold=0.3)
                .as_spatial()
                .run()
            )
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

        with RustO.initialize(InitializeConfig.indra(det_path, rec_path, dict_path)) as engine:
            res = (
                engine.detect_text(img_path)
                .with_post_recognition(score_threshold=0.3)
                .as_lines()
                .run()
            )
            self.assertTrue(res.is_STRUCTURED())
            self.assertGreater(len(res.results), 0)
            joined = " ".join([r.text for r in res.results])
            self.assertTrue("UNITED" in joined or "PASSPORT" in joined or "BRITAIN" in joined)

    def test_run_options_with_configurable_opts(self):
        opts = OcrRunOptions(
            recognition=RecognitionRunOptions(batch_size=8),
            post_recognition=PostRecognitionOptions(score_threshold=0.4),
            pre_detection=PreDetectionOptions(target_max_side=960, crop_padding_x=0.05, crop_padding_y=0.08),
        )
        self.assertEqual(opts.recognition.batch_size, 8)
        self.assertIsNotNone(opts.pre_detection)
        self.assertEqual(opts.pre_detection.target_max_side, 960)

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
        self.assertTrue(callable(RustO.download_model))
        self.assertTrue(callable(version))
        self.assertTrue(callable(is_ready))
        self.assertEqual(version(), "0.4.0")

    def test_chronam_ad_page_with_layout(self):
        root = Path(__file__).resolve().parents[3]
        img_path = root / "tests/fixtures/chronam-ad-page.png"
        det_path = root / "models/PPOCR_v6_tiny/det.rten"
        rec_path = root / "models/PPOCR_v6_tiny/rec.rten"
        dict_path = root / "models/PPOCR_v6_tiny/dict.txt"
        layout_path = root / "models/PPOCR_v6_tiny/layout_cdla.onnx"

        if not (img_path.exists() and det_path.exists() and rec_path.exists() and layout_path.exists()):
            self.skipTest("Required fixture or model files missing")

        config = {
            "detection": {"modelPath": str(det_path), "enabled": True},
            "recognition": {"modelPath": str(rec_path), "dictPath": str(dict_path), "enabled": True},
            "layout": {"modelPath": str(layout_path), "enabled": True},
        }
        engine = RustO.initialize_from_json(json.dumps(config))
        opts = OcrRunOptions(
            output=OutputGranularity.LINES,
            layout_reading=True,
        )
        res = engine.detect_text(ImageSource.PATH(str(img_path)), opts)
        self.assertTrue(res.is_STRUCTURED())
        self.assertEqual(len(res.results), 238)
        self.assertEqual(res.results[0].text, "All Orders in our")
        self.assertIsNotNone(res.results[0].region_id)
        self.assertIsNotNone(res.results[0].layout_type)
        engine.close()

    def test_fluent_builder_chaining(self):
        root = Path(__file__).resolve().parents[3]
        img_path = root / "tests/fixtures/chronam-ad-page.png"
        det_path = root / "models/PPOCR_v6_tiny/det.rten"
        rec_path = root / "models/PPOCR_v6_tiny/rec.rten"
        dict_path = root / "models/PPOCR_v6_tiny/dict.txt"
        layout_path = root / "models/PPOCR_v6_tiny/layout_cdla.onnx"

        if not (img_path.exists() and det_path.exists() and rec_path.exists() and layout_path.exists()):
            self.skipTest("Required fixture or model files missing")

        config = {
            "detection": {"modelPath": str(det_path), "enabled": True},
            "recognition": {"modelPath": str(rec_path), "dictPath": str(dict_path), "enabled": True},
            "layout": {"modelPath": str(layout_path), "enabled": True},
        }
        engine = RustO.initialize_from_json(json.dumps(config))

        builder = (
            engine.detect_text(str(img_path))
            .with_layout_reading(True)
            .with_pre_detection(descreen_strength=0.1)
            .with_recognition(batch_size=6, workers=2, max_workers=4)
            .with_post_recognition(score_threshold=0.5)
            .as_lines()
        )
        self.assertIsInstance(builder, DetectTextBuilder)
        res = builder.run()
        self.assertTrue(res.is_STRUCTURED())
        self.assertGreater(len(res.results), 0)
        self.assertTrue(res.results[0].text.startswith("All Orders in"))

        spatial_res = (
            engine.detect_text()
            .image(str(img_path))
            .with_layout_reading(True)
            .as_spatial()
            .run()
        )
        self.assertTrue(spatial_res.is_SPATIAL())
        self.assertIn("Orders", spatial_res.text)
        engine.close()


if __name__ == "__main__":
    unittest.main()
