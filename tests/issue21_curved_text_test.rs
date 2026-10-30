use ndarray::Array4;
use rusto::{
    postprocess::DBPostProcess, DetectionRunOptions, OcrRunOptions, PostprocessRunOptions,
};

#[test]
fn test_postprocess_options_merge_and_validation() {
    let mut opts = OcrRunOptions::default();
    opts.postprocess = Some(PostprocessRunOptions {
        score_mode: Some("slow".into()),
        box_threshold: Some(0.5),
        ..Default::default()
    });
    opts.detection = Some(DetectionRunOptions {
        postprocess: Some(PostprocessRunOptions {
            threshold: Some(0.25),
            score_mode: Some("fast".into()),
            ..Default::default()
        }),
        ..Default::default()
    });

    // Valid options must pass validation
    assert!(opts.validate().is_ok());

    // Invalid score_mode must fail
    let mut invalid = opts.clone();
    invalid.postprocess.as_mut().unwrap().score_mode = Some("bogus".into());
    assert!(invalid.validate().is_err());

    let mut invalid_nested = opts.clone();
    invalid_nested.detection.as_mut().unwrap().postprocess.as_mut().unwrap().score_mode =
        Some("unsupported".into());
    assert!(invalid_nested.validate().is_err());
}

#[test]
fn test_curved_contour_scoring_fast_vs_slow() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 200x400 prediction probability map
    let h = 200usize;
    let w = 400usize;
    let mut pred = Array4::<f32>::zeros((1, 1, h, w));

    // Draw an arched text band: curved like a page curving into a book spine
    // Font height is 20px, sagitta (bulge) is 60px
    for x in 50..350 {
        let u = (x as f32 - 200.0) / 150.0;
        let arc_center_y = 120.0 - 60.0 * (1.0 - u * u); // bulges upwards to y = 60 at center
        let y_min = (arc_center_y - 10.0).round() as usize;
        let y_max = (arc_center_y + 10.0).round() as usize;
        for y in y_min..=y_max {
            if y < h {
                pred[[0, 0, y, x]] = 0.95;
            }
        }
    }

    // Fast mode: score is averaged over the min_area_rect (which covers from y=50 to y=130 over width 300)
    // The curved band only occupies ~25% of that rectangle, so average score is ~0.25 - 0.35 (< 0.6 box_thresh).
    let postprocess_fast = DBPostProcess::new(0.3, 0.6, 1000, 1.5, false, "fast");
    let (boxes_fast, _) = postprocess_fast.process(&pred, h as i32, w as i32)?;
    assert!(
        boxes_fast.is_empty(),
        "Fast score mode on curved contour must drop box when box_thresh=0.6"
    );

    // Slow mode: score is averaged over the actual polygon contour pixels (~0.95), so it is retained!
    let postprocess_slow = DBPostProcess::new(0.3, 0.6, 1000, 1.5, false, "slow");
    let (boxes_slow, scores_slow) = postprocess_slow.process(&pred, h as i32, w as i32)?;
    assert_eq!(
        boxes_slow.len(),
        1,
        "Slow score mode must retain the curved contour"
    );
    assert!(
        scores_slow[0] >= 0.90,
        "Slow score mode score must be high inside the contour polygon, got {}",
        scores_slow[0]
    );

    Ok(())
}

#[test]
fn test_default_score_mode_is_slow() {
    use rusto::InitializeConfig;
    use std::path::PathBuf;

    let init = InitializeConfig::default();
    assert_eq!(
        init.det.score_mode, "slow",
        "Default detection score_mode must be slow to prevent line dropping on curved text"
    );

    let v6 = InitializeConfig::ppv6("det.rten", "rec.rten", "dict.txt");
    assert_eq!(v6.det.score_mode, "slow");

    let det = rusto::types::DetConfig::default(PathBuf::from("det.rten"));
    assert_eq!(det.score_mode, "slow");
}

#[test]
fn test_end_to_end_curved_page_ocr() -> Result<(), Box<dyn std::error::Error>> {
    use rusto::{DetectTextResult, InitializeConfig, OcrRunOptions, RustO};
    use std::path::Path;

    let fixture_path = "tests/fixtures/curved_page.jpg";
    if !Path::new(fixture_path).exists() {
        return Ok(());
    }

    // Try finding available detection and recognition models
    let candidate_models = [
        (
            "models/PPOCR_v6_tiny/det.rten",
            "models/PPOCR_v6_tiny/rec.rten",
            "models/PPOCR_v6_tiny/dict.txt",
        ),
        (
            "models/PPOCR_v6/det.rten",
            "models/PPOCR_v6/rec.rten",
            "models/PPOCR_v6/dict.txt",
        ),
    ];

    let found_model = candidate_models.iter().find(|(det, rec, dict)| {
        Path::new(det).exists() && Path::new(rec).exists() && Path::new(dict).exists()
    });

    let (det_path, rec_path, dict_path) = match found_model {
        Some(m) => *m,
        None => return Ok(()),
    };

    let mut ocr = RustO::initialize(InitializeConfig::ppv6(det_path, rec_path, dict_path))?;
    let rgb = image::open(fixture_path)?.into_rgb8();
    let (w, h) = rgb.dimensions();

    // Default options must retain all lines (>= 10 boxes covering all 12 lines) without dropping 7 lines
    let result = ocr.detect_text_rgb(
        rgb.as_raw(),
        w,
        h,
        w as usize * 3,
        &OcrRunOptions::default(),
    )?;

    if let DetectTextResult::Structured(lines) = result {
        assert!(
            lines.len() >= 10,
            "Default options on curved page must detect lines without dropping, got {} lines",
            lines.len()
        );
        let full_text: String = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            full_text.contains("Alice") || full_text.contains("sister"),
            "Line 1 must be present in output"
        );
        assert!(
            full_text.contains("bank") || full_text.contains("peeped"),
            "Line 3 (previously dropped) must be present in output"
        );
        assert!(
            full_text.contains("IllegalState")
                || full_text.contains("capacity")
                || full_text.contains("queue"),
            "Line 7 (previously dropped) must be present in output"
        );
    } else {
        panic!("Expected structured text result");
    }

    Ok(())
}
