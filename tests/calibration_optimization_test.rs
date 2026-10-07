use rusto::calibration::{expand_box_padding, CalibrationOptions, OptimizationOptions};
use rusto::image_impl::Point2f;
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::path::Path;

#[test]
fn test_calibration_options_serde() {
    let calib = CalibrationOptions::for_id_card()
        .with_descreen(true)
        .with_descreen_strength(1.25)
        .with_sharpen(true);

    let json = serde_json::to_string(&calib).expect("serialize calibration options");
    assert!(json.contains("\"descreenBackground\":true"));
    assert!(json.contains("\"descreenStrength\":1.25"));
    assert!(json.contains("\"sharpen\":true"));

    let deserialized: CalibrationOptions =
        serde_json::from_str(&json).expect("deserialize calibration options");
    assert_eq!(deserialized.descreen_strength, Some(1.25));
    assert_eq!(deserialized.descreen_background, Some(true));
}

#[test]
fn test_optimization_options_serde() {
    let opt = OptimizationOptions::for_id_card()
        .with_target_max_side(960)
        .with_crop_padding(0.06, 0.08);

    let json = serde_json::to_string(&opt).expect("serialize optimization options");
    assert!(json.contains("\"targetMaxSide\":960"));
    assert!(json.contains("\"cropPaddingX\":0.06"));
    assert!(json.contains("\"cropPaddingY\":0.08"));

    let deserialized: OptimizationOptions =
        serde_json::from_str(&json).expect("deserialize optimization options");
    assert_eq!(deserialized.target_max_side, Some(960));
    assert_eq!(deserialized.crop_padding_x, Some(0.06));
    assert_eq!(deserialized.crop_padding_y, Some(0.08));
}

#[test]
fn test_runtime_options_with_calibration_and_optimization() {
    let options = OcrRunOptions {
        calibration: Some(CalibrationOptions::for_id_card()),
        optimization: Some(OptimizationOptions::for_id_card()),
        ..Default::default()
    };

    options.validate().expect("options should be valid");
    let json = serde_json::to_string(&options).expect("serialize OcrRunOptions");
    assert!(json.contains("\"calibration\""));
    assert!(json.contains("\"optimization\""));
    assert!(json.contains("\"descreenBackground\":true"));
    assert!(json.contains("\"targetMaxSide\":800"));
}

#[test]
fn test_runtime_options_validation_rejections() {
    // Negative descreen strength
    let opt_neg = OcrRunOptions {
        calibration: Some(CalibrationOptions {
            descreen_strength: Some(-1.0),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(opt_neg.validate().is_err());

    // Invalid crop padding (> 1.0)
    let opt_overflow = OcrRunOptions {
        optimization: Some(OptimizationOptions {
            crop_padding_x: Some(1.5),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(opt_overflow.validate().is_err());
}

#[test]
fn test_expand_box_padding_math() {
    let b = [
        Point2f { x: 100.0, y: 100.0 }, // top-left
        Point2f { x: 200.0, y: 100.0 }, // top-right
        Point2f { x: 200.0, y: 150.0 }, // bottom-right
        Point2f { x: 100.0, y: 150.0 }, // bottom-left
    ];

    // Width = 100, Height = 50.
    // pad_x = 0.10 (+10px left/right), pad_y = 0.10 (+5px top/bottom)
    let expanded = expand_box_padding(&b, 0.10, 0.10, 500.0, 500.0);

    assert_eq!(expanded[0].x, 90.0);
    assert_eq!(expanded[0].y, 95.0);
    assert_eq!(expanded[1].x, 210.0);
    assert_eq!(expanded[1].y, 95.0);
    assert_eq!(expanded[2].x, 210.0);
    assert_eq!(expanded[2].y, 155.0);
    assert_eq!(expanded[3].x, 90.0);
    assert_eq!(expanded[3].y, 155.0);
}

#[test]
fn test_initialize_config_builders() {
    let config = InitializeConfig::default()
        .with_calibration(CalibrationOptions::composite())
        .with_optimization(OptimizationOptions::for_id_card());

    assert!(config.global.calibration.is_some());
    assert!(config.global.optimization.is_some());
    assert_eq!(
        config.global.optimization.as_ref().unwrap().target_max_side,
        Some(800)
    );
}

#[test]
fn test_ocr_inference_with_calibration_and_optimization() {
    let ktp_path = Path::new("ktp.jpeg");
    let models_dir = Path::new("models/PPOCR_v6_tiny");

    if !ktp_path.exists() || !models_dir.exists() {
        return;
    }

    let config = InitializeConfig::ppv6(
        models_dir.join("det.rten"),
        models_dir.join("rec.rten"),
        models_dir.join("dict.txt"),
    )
    .with_calibration(CalibrationOptions::for_id_card())
    .with_optimization(OptimizationOptions::for_id_card());

    let mut ocr = RustO::initialize(config).expect("failed to initialize OCR");

    let result = ocr
        .detect_text(&ImageSource::Path(ktp_path.to_path_buf()), &OcrRunOptions::default())
        .expect("detect text failed");

    if let DetectTextResult::Structured(lines) = result {
        assert!(!lines.is_empty(), "should detect lines on ktp.jpeg");

        // Verify key field recognition accuracy
        let all_text = lines
            .iter()
            .map(|l| l.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        assert!(
            all_text.contains("3217") || all_text.contains("JAKARTA"),
            "should recognize core KTP text"
        );
    }
}
