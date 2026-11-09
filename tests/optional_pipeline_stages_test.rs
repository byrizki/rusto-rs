use std::collections::HashMap;
use rusto::ArrayD;
use rusto::{
    CoreError, DetectTextResult, DocUnwarper, ImageSource, InferenceSession, InitializeConfig, LayoutConfig,
    LayoutDetector, LayoutType, OcrRunOptions, RustO, UnwarpConfig,
};

#[test]
fn test_config_json_parsing_for_optional_stages() {
    let json_config = r#"{
        "template": "ppv6",
        "detection": {
            "modelPath": "models/det.rten",
            "enabled": true
        },
        "recognition": {
            "modelPath": "models/rec.rten",
            "dictPath": "models/dict.txt",
            "enabled": true
        },
        "classification": {
            "modelPath": "models/cls.rten",
            "enabled": true,
            "thresh": 0.95
        },
        "orientation": {
            "modelPath": "models/orient.rten",
            "enabled": true,
            "thresh": 0.88
        },
        "unwarp": {
            "modelPath": "models/uvdoc.rten",
            "enabled": true
        },
        "layout": {
            "modelPath": "models/layout.rten",
            "enabled": true,
            "confThresh": 0.65,
            "targetSize": 800
        }
    }"#;

    let config = InitializeConfig::from_json(json_config).expect("Parse JSON InitializeConfig");

    assert!(config.global.use_cls);
    assert!(config.global.use_orient);
    assert!(config.global.use_unwarp);
    assert!(config.global.use_layout);

    let unwarp = config.unwarp.expect("unwarp config should exist");
    assert_eq!(unwarp.model_path.to_str().unwrap(), "models/uvdoc.rten");

    let layout = config.layout.expect("layout config should exist");
    assert_eq!(layout.model_path.to_str().unwrap(), "models/layout.rten");
    assert_eq!(layout.conf_thresh, 0.65);
    assert_eq!(layout.target_size, 800);

    let cls = config.cls.expect("cls config should exist");
    assert_eq!(cls.model_path.to_str().unwrap(), "models/cls.rten");
    assert_eq!(cls.cls_thresh, 0.95);

    let orient = config.orient.expect("orient config should exist");
    assert_eq!(orient.model_path.to_str().unwrap(), "models/orient.rten");
    assert_eq!(orient.orient_thresh, 0.88);
}

#[test]
fn test_ocr_run_options_overrides() {
    let options_json = r#"{
        "output": "lines",
        "lineClassification": true,
        "orientationDetection": true,
        "autoUnwarp": true,
        "layoutReading": true,
        "postRecognition": {
            "scoreThreshold": 0.75
        }
    }"#;

    let options: OcrRunOptions = nanoserde::DeJson::deserialize_json(options_json).expect("Deserialize OcrRunOptions");
    assert_eq!(options.line_classification, Some(true));
    assert_eq!(options.orientation_detection, Some(true));
    assert_eq!(options.auto_unwarp, Some(true));
    assert_eq!(options.layout_reading, Some(true));
    assert_eq!(options.post_recognition.as_ref().unwrap().score_threshold, Some(0.75));

    let reserialized = nanoserde::SerJson::serialize_json(&options);
    assert!(reserialized.contains("\"autoUnwarp\":true"));
    assert!(reserialized.contains("\"layoutReading\":true"));
    assert!(reserialized.contains("\"lineClassification\":true"));
    assert!(reserialized.contains("\"orientationDetection\":true"));
}

struct MockLayoutSession {
    output_boxes: Vec<[f32; 6]>,
}

impl InferenceSession for MockLayoutSession {
    fn run(&mut self, _input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        let n = self.output_boxes.len();
        let mut flat = Vec::with_capacity(n * 6);
        for b in &self.output_boxes {
            flat.extend_from_slice(b);
        }
        let arr = ArrayD::<f32>::from_data(&[1, n, 6], flat);
        Ok(arr)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        let mut map = HashMap::new();
        map.insert("output".to_string(), self.run(input)?);
        Ok(map)
    }

    fn run_with_inputs(
        &mut self,
        _inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        let dummy = ArrayD::zeros([1, 1, 1]);
        self.run_all(dummy)
    }

    fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    fn have_key(&self, _key: &str) -> bool {
        false
    }
}

#[test]
fn test_layout_detector_with_mock_session() {
    let mock_session = MockLayoutSession {
        output_boxes: vec![
            [0.0, 0.95, 10.0, 20.0, 200.0, 60.0],  // Title (class 0)
            [1.0, 0.88, 10.0, 70.0, 300.0, 200.0], // Text (class 1)
            [5.0, 0.92, 10.0, 210.0, 300.0, 400.0], // Table (class 5)
        ],
    };

    let layout_config = LayoutConfig::default("mock/layout.onnx".into());
    let mut detector = LayoutDetector::with_custom_session(layout_config, mock_session);

    let test_img = image::DynamicImage::new_rgb8(400, 500);
    let mat = rusto_core::image::Mat::new(test_img);

    let output = detector.detect(&mat).expect("Run layout detection");
    assert_eq!(output.regions.len(), 3);

    assert_eq!(output.regions[0].layout_type, LayoutType::Title);
    assert_eq!(output.regions[0].confidence, 0.95);

    assert_eq!(output.regions[1].layout_type, LayoutType::Table);
    assert_eq!(output.regions[1].confidence, 0.92);

    assert_eq!(output.regions[2].layout_type, LayoutType::Text);
    assert_eq!(output.regions[2].confidence, 0.88);
}

struct MockUnwarpSession;

impl InferenceSession for MockUnwarpSession {
    fn run(&mut self, _input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        // Mock UVDoc 2D displacement field mesh [1, 2, 45, 31]
        let total = 2 * 45 * 31;
        let mut grid_data = Vec::with_capacity(total);
        for c in 0..2 {
            for y in 0..45 {
                for x in 0..31 {
                    let val = if c == 0 {
                        (x as f32 / 30.0) * 2.0 - 1.0
                    } else {
                        (y as f32 / 44.0) * 2.0 - 1.0
                    };
                    grid_data.push(val);
                }
            }
        }
        let arr = ArrayD::<f32>::from_data(&[1, 2, 45, 31], grid_data);
        Ok(arr)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        let mut map = HashMap::new();
        map.insert("output".to_string(), self.run(input)?);
        Ok(map)
    }

    fn run_with_inputs(
        &mut self,
        _inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        let dummy = ArrayD::zeros([1, 1, 1]);
        self.run_all(dummy)
    }

    fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    fn have_key(&self, _key: &str) -> bool {
        false
    }
}

#[test]
fn test_doc_unwarper_with_mock_session() {
    let unwarp_config = UnwarpConfig::default("mock/uvdoc.onnx".into());
    let mut unwarper = DocUnwarper::with_custom_session(unwarp_config, MockUnwarpSession);

    let test_img = image::DynamicImage::new_rgb8(100, 150);
    let mat = rusto_core::image::Mat::new(test_img);

    let output = unwarper.unwarp(&mat).expect("Run unwarp");
    assert_eq!(output.unwarped_image.cols(), 100);
    assert_eq!(output.unwarped_image.rows(), 150);
}

#[test]
fn test_real_doc_unwarper_mnn() {
    let unwarp_path = "models/PPOCR_v6_tiny/uvdoc.mnn";
    let img_path = "models/test_images/example1_curved.png";
    if !std::path::Path::new(unwarp_path).exists() || !std::path::Path::new(img_path).exists() {
        return;
    }

    let unwarp_config = UnwarpConfig::default(unwarp_path.into());
    let mut unwarper = DocUnwarper::new(unwarp_config).expect("Init DocUnwarper with uvdoc.mnn");

    let img = image::open(img_path).expect("Open curved image");
    let mat = rusto_core::image::Mat::new(img);
    let output = unwarper.unwarp(&mat).expect("Run unwarp with uvdoc.mnn");

    assert_eq!(output.unwarped_image.cols(), mat.cols());
    assert_eq!(output.unwarped_image.rows(), mat.rows());
}

#[test]
fn test_real_doc_unwarper_onnx() {
    let unwarp_path = "models/PPOCR_v6_tiny/uvdoc.onnx";
    let img_path = "models/test_images/example1_curved.png";
    if !std::path::Path::new(unwarp_path).exists() || !std::path::Path::new(img_path).exists() {
        return;
    }

    let mnn_session = rusto::engine::MnnSession::from_path(
        std::path::Path::new(unwarp_path),
        &rusto_core::types::EngineConfig::default(),
    ).expect("Init MnnSession for uvdoc.onnx");
    let mut unwarper = DocUnwarper::with_custom_session(
        UnwarpConfig::default(unwarp_path.into()),
        mnn_session,
    );

    let img = image::open(img_path).expect("Open curved image");
    let mat = rusto_core::image::Mat::new(img);
    let output = unwarper.unwarp(&mat).expect("Run unwarp with uvdoc.onnx");

    assert_eq!(output.unwarped_image.cols(), mat.cols());
    assert_eq!(output.unwarped_image.rows(), mat.rows());
}

#[test]
fn test_rusto_layout_reading_option() {
    let det_path = "models/PPOCR_v6_tiny/det.rten";
    let rec_path = "models/PPOCR_v6_tiny/rec.rten";
    let dict_path = "models/PPOCR_v6_tiny/dict.txt";

    if !std::path::Path::new(det_path).exists() {
        return;
    }

    let config = InitializeConfig::ppv6(det_path, rec_path, dict_path);
    let mut ocr = RustO::initialize(config).expect("Initialize RustO");

    let mock_layout = LayoutDetector::with_custom_session(
        LayoutConfig::default("mock/layout.onnx".into()),
        MockLayoutSession {
            output_boxes: vec![
                [1.0, 0.97, 50.0, 50.0, 300.0, 100.0], // Text (class 1, score 0.97)
            ],
        },
    );

    ocr = ocr.with_layout(mock_layout);

    let img_path = "models/test_images/example1.png";
    if std::path::Path::new(img_path).exists() {
        let opts = OcrRunOptions::default().with_layout_reading(true);
        let result = ocr.detect_text(&ImageSource::Path(img_path.into()), &opts)
            .expect("detect_text with layout reading");
        assert!(matches!(result, DetectTextResult::Structured(_)));
    }
}

#[test]
fn test_real_layout_cdla_detection() {
    let model_path = "models/PPOCR_v6_tiny/layout_cdla.onnx";
    let img_path = "vendor/RapidDoc/demo/images/layout_original_image_01.png";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(img_path).exists() {
        return;
    }

    let config = LayoutConfig::default(model_path.into());
    let mut detector = LayoutDetector::new(config).expect("Load layout_cdla.onnx");

    let img = image::open(img_path).expect("open test image");
    let mat = rusto_core::image::Mat::new(img);

    let output = detector.detect(&mat).expect("Run layout detection");
    println!("Detected {} layout regions in {:.2} ms:", output.regions.len(), output.elapse * 1000.0);
    for r in &output.regions {
        println!("  {}: score={:.4}, bbox=[{}, {}, {}, {}]", r.layout_type.name(), r.confidence, r.bbox.x_min, r.bbox.y_min, r.bbox.x_max, r.bbox.y_max);
    }
    assert!(!output.regions.is_empty(), "Should detect layout regions");

    for r in &output.regions {
        assert!(r.confidence >= 0.4);
        assert!(r.bbox.x_max > r.bbox.x_min);
        assert!(r.bbox.y_max > r.bbox.y_min);
    }

    let has_regions = !output.regions.is_empty();
    assert!(has_regions, "Should detect layout regions");
}

#[test]
fn test_real_layout_publaynet_detection() {
    let model_path = "models/PPOCR_v6_tiny/layout_publaynet.onnx";
    let img_path = "vendor/RapidDoc/demo/images/layout_original_image_01.png";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(img_path).exists() {
        return;
    }

    let config = LayoutConfig::default(model_path.into());
    let mut detector = LayoutDetector::new(config).expect("Load layout_publaynet.onnx");

    let img = image::open(img_path).expect("open test image");
    let mat = rusto_core::image::Mat::new(img);

    let output = detector.detect(&mat).expect("Run layout detection");
    assert!(!output.regions.is_empty(), "Should detect layout regions");

    for r in &output.regions {
        assert!(r.confidence >= 0.4);
        assert!(r.bbox.x_max > r.bbox.x_min);
        assert!(r.bbox.y_max > r.bbox.y_min);
    }
}

#[test]
fn test_real_layout_cdla_detection_mnn() {
    let model_path = "models/PPOCR_v6_tiny/layout_cdla.mnn";
    let img_path = "vendor/RapidDoc/demo/images/layout_original_image_01.png";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(img_path).exists() {
        return;
    }

    let config = LayoutConfig::default(model_path.into());
    let mut detector = LayoutDetector::new(config).expect("Load layout_cdla.mnn");

    let img = image::open(img_path).expect("open test image");
    let mat = rusto_core::image::Mat::new(img);

    let output = detector.detect(&mat).expect("Run layout detection with layout_cdla.mnn");
    assert!(!output.regions.is_empty(), "Should detect layout regions with layout_cdla.mnn");
}

#[test]
fn test_real_layout_publaynet_detection_mnn() {
    let model_path = "models/PPOCR_v6_tiny/layout_publaynet.mnn";
    let img_path = "vendor/RapidDoc/demo/images/layout_original_image_01.png";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(img_path).exists() {
        return;
    }

    let config = LayoutConfig::default(model_path.into());
    let mut detector = LayoutDetector::new(config).expect("Load layout_publaynet.mnn");

    let img = image::open(img_path).expect("open test image");
    let mat = rusto_core::image::Mat::new(img);

    let output = detector.detect(&mat).expect("Run layout detection with layout_publaynet.mnn");
    assert!(!output.regions.is_empty(), "Should detect layout regions with layout_publaynet.mnn");
}

#[test]
fn test_issue26_dense_page_total_order_does_not_panic() {
    use rusto_core::layout::sort_lines_reading_order;
    use rusto::{Frame, TextResult};

    // Construct 153 staggered lines simulating the dense advertisement page from Issue #26.
    // The previous non-transitive comparator (top_diff < 10.0 -> left else top) panicked
    // with "user-provided comparison function does not correctly implement a total order"
    // when tops cycle: 0 -> 8 -> 16 -> 0 with staggered left coordinates.
    let mut lines = Vec::new();
    for i in 0..153 {
        let top = (i * 8 % 740) as f32;
        let left = ((153 - i) * 13 % 1000) as f32;
        let width = 60.0;
        let height = 12.0;
        let box_points = [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ];
        lines.push(TextResult {
            text: format!("line_{i}"),
            score: 0.95,
            box_points,
            frame: Frame::new(width, height, top, left),
            region_id: None,
            layout_type: None,
        });
    }

    // Must not panic on Rust's slice sort
    sort_lines_reading_order(&mut lines);
    assert_eq!(lines.len(), 153);
}

#[test]
fn test_multi_column_heuristic_reading_order() {
    use rusto_core::layout::sort_lines_reading_order;
    use rusto::{Frame, TextResult};

    // 2 columns of text:
    // Column 0: x in [50, 300], rows at y = 100, 200
    // Column 1: x in [500, 750], rows at y = 100, 200
    // Interleaved input simulating horizontal raster order:
    // col0_row0, col1_row0, col0_row1, col1_row1
    let mut lines = vec![
        TextResult {
            text: "col0_row0".into(),
            score: 0.99,
            box_points: [(50.0, 100.0), (300.0, 100.0), (300.0, 120.0), (50.0, 120.0)],
            frame: Frame::new(250.0, 20.0, 100.0, 50.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col1_row0".into(),
            score: 0.99,
            box_points: [(500.0, 100.0), (750.0, 100.0), (750.0, 120.0), (500.0, 120.0)],
            frame: Frame::new(250.0, 20.0, 100.0, 500.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col0_row1".into(),
            score: 0.99,
            box_points: [(50.0, 200.0), (300.0, 200.0), (300.0, 220.0), (50.0, 220.0)],
            frame: Frame::new(250.0, 20.0, 200.0, 50.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col1_row1".into(),
            score: 0.99,
            box_points: [(500.0, 200.0), (750.0, 200.0), (750.0, 220.0), (500.0, 220.0)],
            frame: Frame::new(250.0, 20.0, 200.0, 500.0),
            region_id: None,
            layout_type: None,
        },
    ];

    sort_lines_reading_order(&mut lines);
    let texts: Vec<String> = lines.into_iter().map(|l| l.text).collect();
    assert_eq!(texts, vec!["col0_row0", "col0_row1", "col1_row0", "col1_row1"]);
}

#[test]
fn test_column_aware_line_clustering_preserves_columns_and_merges_words() {
    use rusto_core::layout::cluster_and_sort_reading_order;
    use rusto::{Frame, TextResult};

    // Multi-column layout with intra-row word fragments:
    // Column 0 (x ~ 50..300):
    //   Row 0: "col0_w1" (50..150), "col0_w2" (160..280)
    //   Row 1: "col0_line2" (50..300)
    // Column 1 (x ~ 500..750):
    //   Row 0: "col1_header" (500..750)
    //   Row 1: "col1_w1" (500..600), "col1_w2" (610..750)
    let fragments = vec![
        TextResult {
            text: "col0_w1".into(),
            score: 0.99,
            box_points: [(50.0, 100.0), (150.0, 100.0), (150.0, 120.0), (50.0, 120.0)],
            frame: Frame::new(100.0, 20.0, 100.0, 50.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col1_header".into(),
            score: 0.98,
            box_points: [(500.0, 100.0), (750.0, 100.0), (750.0, 120.0), (500.0, 120.0)],
            frame: Frame::new(250.0, 20.0, 100.0, 500.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col0_w2".into(),
            score: 0.97,
            box_points: [(160.0, 100.0), (280.0, 100.0), (280.0, 120.0), (160.0, 120.0)],
            frame: Frame::new(120.0, 20.0, 100.0, 160.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col0_line2".into(),
            score: 0.96,
            box_points: [(50.0, 200.0), (300.0, 200.0), (300.0, 220.0), (50.0, 220.0)],
            frame: Frame::new(250.0, 20.0, 200.0, 50.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col1_w1".into(),
            score: 0.95,
            box_points: [(500.0, 200.0), (600.0, 200.0), (600.0, 220.0), (500.0, 220.0)],
            frame: Frame::new(100.0, 20.0, 200.0, 500.0),
            region_id: None,
            layout_type: None,
        },
        TextResult {
            text: "col1_w2".into(),
            score: 0.94,
            box_points: [(610.0, 200.0), (750.0, 200.0), (750.0, 220.0), (610.0, 220.0)],
            frame: Frame::new(140.0, 20.0, 200.0, 610.0),
            region_id: None,
            layout_type: None,
        },
    ];

    let clustered = cluster_and_sort_reading_order(fragments, Some(0.5), Some(0.8));
    assert_eq!(clustered.len(), 4);
    assert_eq!(clustered[0].text, "col0_w1 col0_w2");
    assert_eq!(clustered[1].text, "col0_line2");
    assert_eq!(clustered[2].text, "col1_header");
    assert_eq!(clustered[3].text, "col1_w1 col1_w2");
}

#[test]
fn test_issue25_multi_column_reading_order_and_region_ids() {
    use rusto_core::layout::{order_text_results_by_layout, BBox, LayoutRegion, LayoutType};
    use rusto::{Frame, TextResult};

    // Simulate 4 columns of ads side-by-side across a 1000px wide page
    let regions = vec![
        // Column 0: x in [0, 220], y in [100, 700]
        LayoutRegion {
            bbox: BBox { x_min: 10, y_min: 100, x_max: 220, y_max: 700 },
            layout_type: LayoutType::Text,
            confidence: 0.95,
        },
        // Column 1: x in [250, 470], y in [100, 700]
        LayoutRegion {
            bbox: BBox { x_min: 250, y_min: 100, x_max: 470, y_max: 700 },
            layout_type: LayoutType::Text,
            confidence: 0.95,
        },
        // Column 2: x in [500, 720], y in [100, 700]
        LayoutRegion {
            bbox: BBox { x_min: 500, y_min: 100, x_max: 720, y_max: 700 },
            layout_type: LayoutType::Text,
            confidence: 0.95,
        },
        // Column 3: x in [750, 970], y in [100, 700]
        LayoutRegion {
            bbox: BBox { x_min: 750, y_min: 100, x_max: 970, y_max: 700 },
            layout_type: LayoutType::Text,
            confidence: 0.95,
        },
    ];

    // Create 3 lines per column
    let mut raw_lines = Vec::new();
    for col in 0..4 {
        for row in 0..3 {
            let left = (col * 250 + 20) as f32;
            let top = (120 + row * 150) as f32;
            let width = 180.0;
            let height = 20.0;
            let box_points = [
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
            ];
            raw_lines.push(TextResult {
                text: format!("col_{col}_row_{row}"),
                score: 0.99,
                box_points,
                frame: Frame::new(width, height, top, left),
                region_id: None,
                layout_type: None,
            });
        }
    }

    // Interleave lines (as det produces when scanning in top-down raster order)
    let mut interleaved = Vec::new();
    for row in 0..3 {
        for col in 0..4 {
            interleaved.push(raw_lines[col * 3 + row].clone());
        }
    }

    let ordered = order_text_results_by_layout(&interleaved, &regions, 1000.0);
    assert_eq!(ordered.len(), 12);

    // Verify Column 0 lines come first, then Column 1, then Column 2, then Column 3
    let ordered_texts: Vec<String> = ordered.iter().map(|r| r.text.clone()).collect();
    assert_eq!(ordered_texts[0], "col_0_row_0");
    assert_eq!(ordered_texts[1], "col_0_row_1");
    assert_eq!(ordered_texts[2], "col_0_row_2");
    assert_eq!(ordered_texts[3], "col_1_row_0");
    assert_eq!(ordered_texts[4], "col_1_row_1");
    assert_eq!(ordered_texts[5], "col_1_row_2");
    assert_eq!(ordered_texts[6], "col_2_row_0");
    assert_eq!(ordered_texts[7], "col_2_row_1");
    assert_eq!(ordered_texts[8], "col_2_row_2");
    assert_eq!(ordered_texts[9], "col_3_row_0");
    assert_eq!(ordered_texts[10], "col_3_row_1");
    assert_eq!(ordered_texts[11], "col_3_row_2");

    // Verify region_id and layout_type are set on every line
    for (i, line) in ordered.iter().enumerate() {
        let expected_col = i / 3;
        assert_eq!(line.region_id, Some(expected_col as i32));
        assert_eq!(line.layout_type, Some(LayoutType::Text));
    }
}

#[test]
#[cfg(feature = "ffi")]
fn test_issue25_c_ffi_layout_api_and_results() {
    use rusto::ffi::*;
    use rusto::{InitializeConfig, RustO};
    use std::ffi::CString;

    let det_path = "models/PPOCR_v6_tiny/det.rten";
    let rec_path = "models/PPOCR_v6_tiny/rec.rten";
    let dict_path = "models/PPOCR_v6_tiny/dict.txt";

    if !std::path::Path::new(det_path).exists() {
        return;
    }

    let config = InitializeConfig::ppv6(det_path, rec_path, dict_path);
    let ocr = RustO::initialize(config).expect("Initialize RustO");
    let handle = Box::into_raw(Box::new(unsafe {
        // Construct ROCRHandle
        std::mem::transmute::<RustO, ROCRHandle>(ocr)
    }));

    let mut results_ptr: *mut CTextResult = std::ptr::null_mut();
    let mut count: usize = 0;
    let options = CString::new(r#"{"layoutReading": true}"#).unwrap();

    let pixels = vec![255u8; 100 * 100 * 3];
    let ret = unsafe {
        rocr_detect_text_rgb(
            handle,
            pixels.as_ptr(),
            pixels.len(),
            100,
            100,
            300,
            options.as_ptr(),
            &mut results_ptr,
            &mut count,
        )
    };
    assert_eq!(ret, 0);

    unsafe {
        rocr_free_results(results_ptr, count);
        rocr_free(handle);
    }
}

#[test]
#[cfg(feature = "ffi")]
fn test_issue25_chronam_dense_page_real_image_layout_and_c_ffi() {
    use rusto::ffi::*;
    use rusto::{DetectTextResult, ImageSource, OcrRunOptions, RustO};
    use std::ffi::CString;
    use std::path::Path;

    let img_path = "tests/fixtures/chronam-ad-page.png";
    let det_path = "models/PPOCR_v6_tiny/det.rten";
    let rec_path = "models/PPOCR_v6_tiny/rec.rten";
    let dict_path = "models/PPOCR_v6_tiny/dict.txt";
    let layout_path = "models/PPOCR_v6_tiny/layout_cdla.onnx";

    if !Path::new(img_path).exists()
        || !Path::new(det_path).exists()
        || !Path::new(rec_path).exists()
        || !Path::new(dict_path).exists()
        || !Path::new(layout_path).exists()
    {
        return;
    }

    rusto::init();

    let config_json = format!(
        r#"{{
            "detection": {{ "modelPath": "{}", "enabled": true }},
            "recognition": {{ "modelPath": "{}", "dictPath": "{}", "enabled": true }},
            "layout": {{ "modelPath": "{}", "enabled": true }}
        }}"#,
        det_path, rec_path, dict_path, layout_path
    );

    // 1. Rust Library API verification
    let init_config = rusto::InitializeConfig::from_json(&config_json).expect("Parse config JSON");
    let mut ocr = RustO::initialize(init_config).expect("Init RustO");

    let options = OcrRunOptions {
        layout_reading: Some(true),
        ..Default::default()
    };

    let res = ocr.detect_text(&ImageSource::Path(img_path.into()), &options)
        .expect("detect_text on chronam-ad-page");
    let DetectTextResult::Structured(lines) = res else {
        panic!("Expected structured text results");
    };

    assert_eq!(lines.len(), 249, "Dense chronicling america page must detect exactly 249 lines");
    assert!(lines[0].text.starts_with("All Orders in"));
    assert!(lines[0].region_id.is_some(), "region_id should be populated when layout is enabled");
    assert!(lines[0].layout_type.is_some(), "layout_type should be populated when layout is enabled");

    // 2. C FFI API verification
    let c_config = CString::new(config_json).unwrap();
    let handle = unsafe { rocr_initialize(c_config.as_ptr()) };
    assert!(!handle.is_null(), "rocr_initialize should return valid handle");

    let c_img_path = CString::new(img_path).unwrap();
    let c_options = CString::new(r#"{"layoutReading": true}"#).unwrap();

    let mut c_results: *mut CTextResult = std::ptr::null_mut();
    let mut c_count: usize = 0;

    let ret = unsafe {
        rocr_detect_text_file(
            handle,
            c_img_path.as_ptr(),
            c_options.as_ptr(),
            &mut c_results,
            &mut c_count,
        )
    };
    assert_eq!(ret, 0, "rocr_detect_text_file should succeed");
    assert_eq!(c_count, 249, "C FFI must return exactly 249 lines");
    assert!(!c_results.is_null());

    unsafe {
        let first = &*c_results;
        let first_text = std::ffi::CStr::from_ptr(first.text).to_string_lossy();
        assert!(first_text.starts_with("All Orders in"));
        assert!(first.region_id >= 0, "C FFI region_id should be non-negative");
        assert!(first.layout_type >= 0, "C FFI layout_type should be non-negative");

        rocr_free_results(c_results, c_count);
        rocr_free(handle);
    }
}

#[test]
fn test_newspaper_page_column_clustering_prevents_merging_and_explosion() {
    let det_path = "models/PPOCR_v6_tiny/det.rten";
    let rec_path = "models/PPOCR_v6_tiny/rec.rten";
    let dict_path = "models/PPOCR_v6_tiny/dict.txt";
    let img_path = "tests/fixtures/chronam-ad-page.png";

    if !std::path::Path::new(det_path).exists()
        || !std::path::Path::new(rec_path).exists()
        || !std::path::Path::new(dict_path).exists()
        || !std::path::Path::new(img_path).exists()
    {
        return;
    }

    let config = InitializeConfig::ppv6(det_path, rec_path, dict_path);
    let mut ocr = RustO::initialize(config).expect("Init PP-OCRv6");

    let options = OcrRunOptions {
        layout_reading: Some(true),
        ..Default::default()
    };

    let res = ocr.detect_text(&ImageSource::Path(img_path.into()), &options)
        .expect("detect_text with layout reading");
    let DetectTextResult::Structured(lines) = res else {
        panic!("Expected structured text results");
    };

    println!("Detected lines count: {}", lines.len());
    for (i, l) in lines.iter().enumerate() {
        if l.text.contains("Special Values") || l.text.contains("Extraordinary") || l.text.contains("Big Sale of") {
            println!("Line {}: '{}'", i, l.text);
        }
    }

    // 1. Line count should be in the ~150-250 range, NOT ~530 word-level fragments
    assert!(lines.len() >= 120 && lines.len() <= 300, "Lines count should be 120-300, got {}", lines.len());

    // 2. Cross-column headers MUST NOT be merged together
    for line in &lines {
        let has_special_values = line.text.contains("Special Values");
        let has_extraordinary = line.text.contains("Extraordinary");
        let has_big_sale = line.text.contains("Big Sale of Corsets") || line.text.contains("Big Sale");
        
        let header_count = (has_special_values as u8) + (has_extraordinary as u8) + (has_big_sale as u8);
        assert!(header_count <= 1, "Cross-column headers were merged into a single line: '{}'", line.text);
    }

    // Now test with Indra OCR
    let indra_det = "models/indra/det.rten";
    let indra_rec = "models/indra/rec.rten";
    let indra_meta = "models/indra/meta.json";

    if std::path::Path::new(indra_det).exists()
        && std::path::Path::new(indra_rec).exists()
        && std::path::Path::new(indra_meta).exists()
    {
        let indra_cfg = InitializeConfig::indra(indra_det, indra_rec, indra_meta);
        let mut indra_ocr = RustO::initialize(indra_cfg).expect("Init Indra OCR");

        let indra_res = indra_ocr.detect_text(&ImageSource::Path(img_path.into()), &options)
            .expect("detect_text with indra OCR");
        let DetectTextResult::Structured(indra_lines) = indra_res else {
            panic!("Expected structured text results for Indra");
        };

        println!("Indra detected lines count: {}", indra_lines.len());
        for (i, l) in indra_lines.iter().enumerate() {
            if l.text.contains("most fashionable") || l.text.contains("Floor") {
                println!("Indra Line {}: '{}'", i, l.text);
            }
        }

        // Verify Indra OCR does not merge "most fashionable" with "Floor"
        for line in &indra_lines {
            assert!(
                !(line.text.contains("most fashionable") && line.text.contains("Floor")),
                "Indra merged middle paragraph with right column 'Floor': '{}'",
                line.text
            );
        }
    }
}
