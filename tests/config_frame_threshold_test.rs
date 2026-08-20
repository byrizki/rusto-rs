use rusto::{DetectTextResult, DetectionRunOptions, Frame, ImageSource, InitializeConfig, OcrRunOptions, OutputGranularity, PostprocessRunOptions, PreprocessingRunOptions, TextResult};

#[test]
fn test_frame_from_points_upright() {
    let points = [(10.0, 20.0), (110.0, 20.0), (110.0, 50.0), (10.0, 50.0)];
    let frame = Frame::from_points(&points);

    assert_eq!(frame.left, 10.0);
    assert_eq!(frame.top, 20.0);
    assert_eq!(frame.width, 100.0);
    assert_eq!(frame.height, 30.0);
}

#[test]
fn test_frame_from_points_rotated() {
    // A tilted quad
    let points = [(15.0, 25.0), (105.0, 20.0), (115.0, 55.0), (8.0, 48.0)];
    let frame = Frame::from_points(&points);

    assert_eq!(frame.left, 8.0);
    assert_eq!(frame.top, 20.0);
    assert_eq!(frame.width, 115.0 - 8.0);
    assert_eq!(frame.height, 55.0 - 20.0);
}

#[test]
fn test_text_result_serde() {
    let frame = Frame::new(100.0, 30.0, 20.0, 10.0);
    let result = TextResult {
        text: "Hello RustO".to_string(),
        score: 0.98,
        box_points: [(10.0, 20.0), (110.0, 20.0), (110.0, 50.0), (10.0, 50.0)],
        frame,
    };

    let json = serde_json::to_string(&result).expect("Serialize TextResult");
    let deserialized: TextResult = serde_json::from_str(&json).expect("Deserialize TextResult");

    assert_eq!(result, deserialized);
    assert_eq!(deserialized.frame.width, 100.0);
    assert_eq!(deserialized.frame.height, 30.0);
    assert_eq!(deserialized.frame.top, 20.0);
    assert_eq!(deserialized.frame.left, 10.0);
}

#[test]
fn test_rusto_config_builders() {
    let config = InitializeConfig::new("det.mnn", "rec.mnn", "dict.txt")
        .with_det_thresh(0.25)
        .with_det_box_thresh(0.6)
        .with_text_score(0.7)
        .with_xy_threshold(0.8, 1.5)
        .with_min_height(25.0)
        .with_max_side_len(1500.0)
        .with_min_side_len(50.0);

    assert_eq!(config.det.thresh, 0.25);
    assert_eq!(config.det.box_thresh, 0.6);
    assert_eq!(config.global.text_score, 0.7);
    assert_eq!(config.global.y_threshold_multiplier, Some(0.8));
    assert_eq!(config.global.x_threshold_multiplier, Some(1.5));
    assert_eq!(config.global.min_height, 25.0);
    assert_eq!(config.global.max_side_len, 1500.0);
    assert_eq!(config.global.min_side_len, 50.0);
}

#[test]
fn test_rusto_preprocessing_run_options_are_public_and_serialize() {
    let options = OcrRunOptions {
        preprocessing: Some(PreprocessingRunOptions {
            min_height: Some(24.0),
            max_side_len: Some(1600.0),
            min_side_len: Some(48.0),
            width_height_ratio: Some(8.0),
            detection: Some(DetectionRunOptions {
                limit_side_len: Some(960),
                limit_type: Some("min".into()),
                postprocess: Some(PostprocessRunOptions { use_dilation: Some(false), ..Default::default() }),
                ..Default::default()
            }),
        }),
        ..Default::default()
    };
    options.validate().expect("valid runtime preprocessing");
    let json = serde_json::to_string(&options).expect("Serialize options");
    assert!(json.contains("\"preprocessing\""));
    assert!(json.contains("\"useDilation\":false"));
}

#[test]
fn test_rusto_preprocessing_run_options_reject_invalid_bounds() {
    let options = OcrRunOptions {
        preprocessing: Some(PreprocessingRunOptions { min_side_len: Some(100.0), max_side_len: Some(50.0), ..Default::default() }),
        ..Default::default()
    };
    assert!(options.validate().is_err());
}

#[test]
fn test_rusto_config_grouped_json() {
    let json_str = r#"{
        "template": "ppv5",
        "detection": {
            "modelPath": "models/custom_det.mnn",
            "thresh": 0.38,
            "boxThresh": 0.58,
            "unclipRatio": 1.8,
            "enabled": true
        },
        "recognition": {
            "modelPath": "models/custom_rec.mnn",
            "dictPath": "models/custom_dict.txt",
            "scoreThresh": 0.72,
            "returnWordBox": true,
            "enabled": true
        },
        "classification": {
            "modelPath": "models/v5_cls.mnn",
            "threshold": 0.88,
            "enabled": true
        },
        "layout": {
            "yThresholdMultiplier": 0.62,
            "xThresholdMultiplier": 1.25
        }
    }"#;

    let config = InitializeConfig::from_json(json_str).expect("Parse grouped JSON");
    assert_eq!(
        config.det.model_path.to_str().unwrap(),
        "models/custom_det.mnn"
    );
    assert_eq!(
        config.rec.model_path.to_str().unwrap(),
        "models/custom_rec.mnn"
    );
    assert_eq!(
        config.rec.rec_keys_path.unwrap().to_str().unwrap(),
        "models/custom_dict.txt"
    );
    assert_eq!(config.det.thresh, 0.38);
    assert_eq!(config.det.box_thresh, 0.58);
    assert_eq!(config.det.unclip_ratio, 1.8);
    assert_eq!(config.global.text_score, 0.72);
    assert!(config.global.return_word_box);
    assert!(config.global.use_cls);
    assert!(config.cls.is_some());
    assert_eq!(
        config.cls.as_ref().unwrap().model_path.to_str().unwrap(),
        "models/v5_cls.mnn"
    );
    assert_eq!(config.cls.as_ref().unwrap().cls_thresh, 0.88);
    assert_eq!(config.global.y_threshold_multiplier, Some(0.62));
    assert_eq!(config.global.x_threshold_multiplier, Some(1.25));
}

#[test]
fn test_rusto_config_templates_and_presets() {
    use rusto::{PPV3_MODEL_CONFIG, PPV4_MODEL_CONFIG, PPV5_MODEL_CONFIG, PPV6_MODEL_CONFIG};

    let default_config = InitializeConfig::default();
    assert_eq!(
        default_config.det.limit_side_len,
        PPV6_MODEL_CONFIG.det_limit_side_len
    );
    assert_eq!(default_config.det.box_thresh, 0.6);

    let v6_config = InitializeConfig::ppv6("det6.mnn", "rec6.mnn", "dict.txt");
    assert_eq!(
        v6_config.det.limit_side_len,
        PPV6_MODEL_CONFIG.det_limit_side_len
    );
    assert_eq!(v6_config.det.limit_type, "min");
    assert_eq!(v6_config.det.box_thresh, 0.6);
    assert_eq!(v6_config.det.unclip_ratio, 2.0);

    let v5_config = InitializeConfig::ppv5("det5.mnn", "rec5.mnn", "dict.txt");
    assert_eq!(
        v5_config.det.limit_side_len,
        PPV5_MODEL_CONFIG.det_limit_side_len
    );
    assert_eq!(v5_config.det.limit_type, "min");
    assert_eq!(v5_config.det.unclip_ratio, 2.0);

    let v4_config = InitializeConfig::ppv4("det4.mnn", "rec4.mnn", "dict.txt");
    assert_eq!(
        v4_config.det.limit_side_len,
        PPV4_MODEL_CONFIG.det_limit_side_len
    );
    assert_eq!(v4_config.det.limit_type, "max");
    assert_eq!(v4_config.det.unclip_ratio, 1.5);

    let v3_config = InitializeConfig::ppv3("det3.mnn", "rec3.mnn", "dict.txt");
    assert_eq!(
        v3_config.det.limit_side_len,
        PPV3_MODEL_CONFIG.det_limit_side_len
    );
    assert_eq!(v3_config.det.limit_type, "max");

    // Test JSON with template selection
    let json_str_v6 = r#"{
        "template": "ppv6",
        "detModelPath": "det6.mnn",
        "recModelPath": "rec6.mnn",
        "dictPath": "dict.txt"
    }"#;
    let json_config_v6 = InitializeConfig::from_json(json_str_v6).expect("Parse v6 template JSON");
    assert_eq!(json_config_v6.det.box_thresh, 0.6);

    let json_str = r#"{
        "template": "ppv4",
        "detModelPath": "det.mnn",
        "recModelPath": "rec.mnn",
        "dictPath": "dict.txt",
        "maxCandidates": 2000,
        "scoreMode": "slow"
    }"#;

    let json_config = InitializeConfig::from_json(json_str).expect("Parse template JSON");
    assert_eq!(json_config.det.limit_side_len, 960);
    assert_eq!(json_config.det.limit_type, "max");
}

#[test]
fn test_ppv6_inference_tiny() {
    use rusto::RustO;
    use std::path::Path;

    let det_path = "models/PPOCR_v6/det.mnn";
    let rec_path = "models/PPOCR_v6/rec.mnn";
    let dict_path = "models/PPOCR_v6/dict.txt";
    let img_path = "models/test_images/example1.png";

    if !Path::new(det_path).exists()
        || !Path::new(rec_path).exists()
        || !Path::new(img_path).exists()
    {
        return;
    }

    let config = InitializeConfig::ppv6(det_path, rec_path, dict_path).with_text_score(0.3);

    let mut ocr = match RustO::initialize(config) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Skipping OCR test: {:?}", e);
            return;
        }
    };

    let DetectTextResult::Structured(text_results) = ocr
        .detect_text(&ImageSource::Path(img_path.into()), &OcrRunOptions::default())
        .expect("Run PP-OCRv6 OCR")
    else {
        panic!("default OCR result must be structured")
    };
    assert!(!text_results.is_empty(), "Should detect text with PP-OCRv6");
    println!(
        "PP-OCRv6 detected {} text regions on example1.png",
        text_results.len()
    );
    for res in text_results.iter().take(5) {
        println!(
            "  '{}' (score: {:.2}, frame: [w={:.1}, h={:.1}, top={:.1}, left={:.1}])",
            res.text, res.score, res.frame.width, res.frame.height, res.frame.top, res.frame.left
        );
    }
}

#[test]
fn test_default_ocr_v6_spatial_text() {
    use rusto::RustO;
    use std::path::Path;

    let det_path = "models/PPOCR_v6/det.mnn";
    let rec_path = "models/PPOCR_v6/rec.mnn";
    let dict_path = "models/PPOCR_v6/dict.txt";
    let img_path = "models/test_images/example1.png";

    if !Path::new(det_path).exists()
        || !Path::new(rec_path).exists()
        || !Path::new(img_path).exists()
    {
        return;
    }

    // Default constructor uses PP-OCRv6
    let config = InitializeConfig::new(det_path, rec_path, dict_path);
    let mut ocr = RustO::initialize(config).expect("Initialize OCR with default v6 config");
    let DetectTextResult::Spatial(spatial_text) = ocr
        .detect_text(
            &ImageSource::Path(img_path.into()),
            &OcrRunOptions { output: OutputGranularity::Spatial, ..Default::default() },
        )
        .expect("Run default v6 OCR")
    else {
        panic!("spatial OCR result must be text")
    };
    assert!(!spatial_text.is_empty(), "Spatial text should not be empty");
    println!("--- Spatial Text Output (v6 Default) ---\n{}", spatial_text);
}

#[test]
fn test_ppv6_invoice_ocr() {
    use rusto::RustO;
    use std::path::Path;

    let det_path = "models/PPOCR_v6/det.mnn";
    let rec_path = "models/PPOCR_v6/rec.mnn";
    let dict_path = "models/PPOCR_v6/dict.txt";
    let img_path = "models/test_images/invoice1.jpg";

    if !Path::new(det_path).exists()
        || !Path::new(rec_path).exists()
        || !Path::new(img_path).exists()
    {
        return;
    }

    let config = InitializeConfig::new(det_path, rec_path, dict_path).with_xy_threshold(0.7, 1.4);

    let mut ocr = RustO::initialize(config).expect("Initialize OCR for invoice");
    let DetectTextResult::Structured(text_results) = ocr
        .detect_text(&ImageSource::Path(img_path.into()), &OcrRunOptions::default())
        .expect("Run OCR on invoice")
    else {
        panic!("default OCR result must be structured")
    };
    assert!(!text_results.is_empty(), "Should detect text on invoice");
    println!(
        "PP-OCRv6 detected {} items on invoice1.jpg",
        text_results.len()
    );
}
