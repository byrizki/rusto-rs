use rusto::{Frame, RustOConfig, RustOOutput, TextResult};

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
    let config = RustOConfig::new("det.mnn", "rec.mnn", "dict.txt")
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
fn test_rusto_config_flat_json() {
    let json_str = r#"{
        "detModelPath": "models/custom_det.mnn",
        "recModelPath": "models/custom_rec.mnn",
        "dictPath": "models/custom_dict.txt",
        "textScore": 0.65,
        "detThresh": 0.35,
        "detBoxThresh": 0.55,
        "yThresholdMultiplier": 0.75,
        "xThresholdMultiplier": 1.45,
        "useCls": true,
        "clsModelPath": "models/cls.mnn"
    }"#;

    let config = RustOConfig::from_json(json_str).expect("Parse flat JSON");
    assert_eq!(config.det.model_path.to_str().unwrap(), "models/custom_det.mnn");
    assert_eq!(config.rec.model_path.to_str().unwrap(), "models/custom_rec.mnn");
    assert_eq!(config.rec.rec_keys_path.unwrap().to_str().unwrap(), "models/custom_dict.txt");
    assert_eq!(config.global.text_score, 0.65);
    assert_eq!(config.det.thresh, 0.35);
    assert_eq!(config.det.box_thresh, 0.55);
    assert_eq!(config.global.y_threshold_multiplier, Some(0.75));
    assert_eq!(config.global.x_threshold_multiplier, Some(1.45));
    assert!(config.global.use_cls);
    assert!(config.cls.is_some());
    assert_eq!(config.cls.unwrap().model_path.to_str().unwrap(), "models/cls.mnn");
}

#[test]
fn test_rusto_output_to_text_results() {
    use rusto::image_impl::Point2f;

    let output = RustOOutput {
        boxes: vec![
            [
                Point2f::new(10.0, 20.0),
                Point2f::new(110.0, 20.0),
                Point2f::new(110.0, 50.0),
                Point2f::new(10.0, 50.0),
            ],
            [
                Point2f::new(10.0, 60.0),
                Point2f::new(80.0, 60.0),
                Point2f::new(80.0, 90.0),
                Point2f::new(10.0, 90.0),
            ],
        ],
        txts: vec!["Hello".to_string(), "World".to_string()],
        scores: vec![0.95, 0.90],
        word_results: vec![Vec::new(), Vec::new()],
        orientation: None,
        elapse_det: 0.05,
        elapse_rec: 0.05,
        elapse_orient: 0.0,
        debug_oriented_image: None,
        y_threshold_multiplier: Some(0.6),
        x_threshold_multiplier: Some(1.3),
    };

    let text_results = output.to_text_results();
    assert_eq!(text_results.len(), 2);

    assert_eq!(text_results[0].text, "Hello");
    assert_eq!(text_results[0].score, 0.95);
    assert_eq!(text_results[0].frame.left, 10.0);
    assert_eq!(text_results[0].frame.top, 20.0);
    assert_eq!(text_results[0].frame.width, 100.0);
    assert_eq!(text_results[0].frame.height, 30.0);

    assert_eq!(text_results[1].text, "World");
    assert_eq!(text_results[1].score, 0.90);
    assert_eq!(text_results[1].frame.left, 10.0);
    assert_eq!(text_results[1].frame.top, 60.0);
    assert_eq!(text_results[1].frame.width, 70.0);
    assert_eq!(text_results[1].frame.height, 30.0);

    // Spatial text uses configured multipliers
    let spatial = output.to_spatial_text(None, None);
    assert!(spatial.contains("Hello"));
    assert!(spatial.contains("World"));
}

#[test]
fn test_rusto_config_templates_and_presets() {
    use rusto::{PPV3_MODEL_CONFIG, PPV4_MODEL_CONFIG, PPV5_MODEL_CONFIG, PPV6_MODEL_CONFIG};

    let default_config = RustOConfig::default();
    assert_eq!(default_config.det.limit_side_len, PPV6_MODEL_CONFIG.det_limit_side_len);
    assert_eq!(default_config.det.box_thresh, 0.6);

    let v6_config = RustOConfig::ppv6("det6.mnn", "rec6.mnn", "dict.txt");
    assert_eq!(v6_config.det.limit_side_len, PPV6_MODEL_CONFIG.det_limit_side_len);
    assert_eq!(v6_config.det.limit_type, "min");
    assert_eq!(v6_config.det.box_thresh, 0.6);
    assert_eq!(v6_config.det.unclip_ratio, 2.0);

    let v5_config = RustOConfig::ppv5("det5.mnn", "rec5.mnn", "dict.txt");
    assert_eq!(v5_config.det.limit_side_len, PPV5_MODEL_CONFIG.det_limit_side_len);
    assert_eq!(v5_config.det.limit_type, "min");
    assert_eq!(v5_config.det.unclip_ratio, 2.0);

    let v4_config = RustOConfig::ppv4("det4.mnn", "rec4.mnn", "dict.txt");
    assert_eq!(v4_config.det.limit_side_len, PPV4_MODEL_CONFIG.det_limit_side_len);
    assert_eq!(v4_config.det.limit_type, "max");
    assert_eq!(v4_config.det.unclip_ratio, 1.5);

    let v3_config = RustOConfig::ppv3("det3.mnn", "rec3.mnn", "dict.txt");
    assert_eq!(v3_config.det.limit_side_len, PPV3_MODEL_CONFIG.det_limit_side_len);
    assert_eq!(v3_config.det.limit_type, "max");

    // Test JSON with template selection
    let json_str_v6 = r#"{
        "template": "ppv6",
        "detModelPath": "det6.mnn",
        "recModelPath": "rec6.mnn",
        "dictPath": "dict.txt"
    }"#;
    let json_config_v6 = RustOConfig::from_json(json_str_v6).expect("Parse v6 template JSON");
    assert_eq!(json_config_v6.det.box_thresh, 0.6);

    let json_str = r#"{
        "template": "ppv4",
        "detModelPath": "det.mnn",
        "recModelPath": "rec.mnn",
        "dictPath": "dict.txt",
        "maxCandidates": 2000,
        "scoreMode": "slow"
    }"#;

    let json_config = RustOConfig::from_json(json_str).expect("Parse template JSON");
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

    if !Path::new(det_path).exists() || !Path::new(rec_path).exists() || !Path::new(img_path).exists() {
        return;
    }

    let config = RustOConfig::ppv6(det_path, rec_path, dict_path)
        .with_text_score(0.3);

    let mut ocr = match RustO::new(config) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Skipping OCR test: {:?}", e);
            return;
        }
    };

    let result = ocr.run(img_path).expect("Run PP-OCRv6 OCR");
    let text_results = result.to_text_results();
    assert!(!text_results.is_empty(), "Should detect text with PP-OCRv6");
    println!("PP-OCRv6 detected {} text regions on example1.png", text_results.len());
    for res in text_results.iter().take(5) {
        println!("  '{}' (score: {:.2}, frame: [w={:.1}, h={:.1}, top={:.1}, left={:.1}])",
            res.text, res.score, res.frame.width, res.frame.height, res.frame.top, res.frame.left);
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

    if !Path::new(det_path).exists() || !Path::new(rec_path).exists() || !Path::new(img_path).exists() {
        return;
    }

    // Default constructor uses PP-OCRv6
    let config = RustOConfig::new(det_path, rec_path, dict_path);
    let mut ocr = RustO::new(config).expect("Initialize OCR with default v6 config");
    let result = ocr.run(img_path).expect("Run default v6 OCR");

    let spatial_text = result.to_spatial_text(None, None);
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

    if !Path::new(det_path).exists() || !Path::new(rec_path).exists() || !Path::new(img_path).exists() {
        return;
    }

    let config = RustOConfig::new(det_path, rec_path, dict_path)
        .with_xy_threshold(0.7, 1.4);

    let mut ocr = RustO::new(config).expect("Initialize OCR for invoice");
    let result = ocr.run(img_path).expect("Run OCR on invoice");
    let text_results = result.to_text_results();
    assert!(!text_results.is_empty(), "Should detect text on invoice");
    println!("PP-OCRv6 detected {} items on invoice1.jpg", text_results.len());
}




