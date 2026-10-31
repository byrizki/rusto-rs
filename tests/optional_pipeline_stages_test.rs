use std::collections::HashMap;
use ndarray::{ArrayD, IxDyn};
use rusto::{
    DocUnwarper, EngineError, ImageSource, InferenceSession, InitializeConfig, LayoutConfig,
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
        "classification": true,
        "orientation": true,
        "unwarp": true,
        "layout": true,
        "textScore": 0.75
    }"#;

    let options: OcrRunOptions = serde_json::from_str(options_json).expect("Deserialize OcrRunOptions");
    assert_eq!(options.classification, Some(true));
    assert_eq!(options.orientation, Some(true));
    assert_eq!(options.unwarp, Some(true));
    assert_eq!(options.layout, Some(true));
    assert_eq!(options.text_score, Some(0.75));

    let reserialized = serde_json::to_string(&options).expect("Serialize OcrRunOptions");
    assert!(reserialized.contains("\"unwarp\":true"));
    assert!(reserialized.contains("\"layout\":true"));
    assert!(reserialized.contains("\"classification\":true"));
    assert!(reserialized.contains("\"orientation\":true"));
}

struct MockLayoutSession {
    output_boxes: Vec<[f32; 6]>,
}

impl InferenceSession for MockLayoutSession {
    fn run(&mut self, _input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        let n = self.output_boxes.len();
        let mut flat = Vec::with_capacity(n * 6);
        for b in &self.output_boxes {
            flat.extend_from_slice(b);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, n, 6]), flat)
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(arr)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let mut map = HashMap::new();
        map.insert("output".to_string(), self.run(input)?);
        Ok(map)
    }

    fn run_with_inputs(
        &mut self,
        _inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let dummy = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![0.0])
            .map_err(|e| EngineError::Engine(e.to_string()))?;
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
    let mat = rusto::image_impl::Mat::new(test_img);

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
    fn run(&mut self, _input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        // Mock UVDoc 2D displacement field mesh [1, 2, 45, 31]
        let shape = IxDyn(&[1, 2, 45, 31]);
        let total = 1 * 2 * 45 * 31;
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
        let arr = ArrayD::from_shape_vec(shape, grid_data)
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(arr)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let mut map = HashMap::new();
        map.insert("output".to_string(), self.run(input)?);
        Ok(map)
    }

    fn run_with_inputs(
        &mut self,
        _inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let dummy = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![0.0])
            .map_err(|e| EngineError::Engine(e.to_string()))?;
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
    let mat = rusto::image_impl::Mat::new(test_img);

    let output = unwarper.unwarp(&mat).expect("Run unwarp");
    assert_eq!(output.unwarped_image.cols(), 100);
    assert_eq!(output.unwarped_image.rows(), 150);
}

#[test]
fn test_rusto_detect_layout_api() {
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
        let regions = ocr.detect_layout(&ImageSource::Path(img_path.into()))
            .expect("detect_layout on image path");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].layout_type, LayoutType::Text);
        assert_eq!(regions[0].confidence, 0.97);
    }
}
