use std::path::PathBuf;

use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};

fn get_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn test_mnn_ocr_on_example1_image() {
    let repo_root = get_repo_root();
    let det_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    let rec_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    let dict_path = repo_root.join("models/PPOCR_v6_tiny/dict.txt");
    let img_path = repo_root.join("models/test_images/example1.png");

    assert!(det_path.exists(), "det.mnn exists: {}", det_path.display());
    assert!(rec_path.exists(), "rec.mnn exists: {}", rec_path.display());
    assert!(dict_path.exists(), "dict.txt exists: {}", dict_path.display());
    assert!(img_path.exists(), "example1.png exists: {}", img_path.display());

    // Initialize config pointing to MNN models and dict directly
    let config = InitializeConfig::ppv6(
        det_path.to_str().unwrap(),
        rec_path.to_str().unwrap(),
        dict_path.to_str().unwrap(),
    )
    .with_text_score(0.3);

    let mut ocr = RustO::initialize(config)
        .expect("Initialize RustO with pure-Rust MNN engines");

    println!("Running pure-Rust MNN OCR on example1.png...");
    let t0 = std::time::Instant::now();
    let result = ocr
        .detect_text(
            &ImageSource::Path(img_path),
            &OcrRunOptions::default(),
        )
        .expect("Run MNN OCR pipeline");
    let elapsed = t0.elapsed();

    match result {
        DetectTextResult::Structured(lines) => {
            println!(
                "Successfully recognized {} text lines in {:.2}s:",
                lines.len(),
                elapsed.as_secs_f64()
            );
            assert!(!lines.is_empty(), "Should detect text regions on example1.png");

            for (i, line) in lines.iter().enumerate().take(15) {
                println!(
                    "  [{:>2}] text: {:<30} (score: {:.2})",
                    i + 1,
                    line.text,
                    line.score
                );
            }

            // High-confidence phrase assertions from example1.png
            let full_text: String = lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            println!("\nAggregated recognized text snippet:\n{}", full_text);
            assert!(
                full_text.to_lowercase().contains("paddle")
                    || full_text.contains("OCR")
                    || full_text.contains("百度")
                    || lines.len() >= 5,
                "OCR recognition result must contain identifiable text tokens from example1.png"
            );
        }
        _ => panic!("Expected structured text result"),
    }
}

#[test]
fn test_mnn_ocr_on_idcard_image() {
    let repo_root = get_repo_root();
    let det_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    let rec_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    let dict_path = repo_root.join("models/PPOCR_v6_tiny/dict.txt");
    let img_path = repo_root.join("models/test_images/idcard.jpg");

    assert!(img_path.exists(), "idcard.jpg exists: {}", img_path.display());
    assert!(det_path.exists(), "det.mnn exists: {}", det_path.display());
    assert!(rec_path.exists(), "rec.mnn exists: {}", rec_path.display());
    assert!(dict_path.exists(), "dict.txt exists: {}", dict_path.display());

    let config = InitializeConfig::ppv6(
        det_path.to_str().unwrap(),
        rec_path.to_str().unwrap(),
        dict_path.to_str().unwrap(),
    )
    .with_calibration(rusto::CalibrationOptions::for_id_card())
    .with_optimization(rusto::OptimizationOptions::for_id_card())
    .with_text_score(0.2);

    let mut ocr = RustO::initialize(config)
        .expect("Initialize RustO with pure-Rust MNN engines");

    println!("Running pure-Rust MNN OCR on idcard.jpg...");
    let t0 = std::time::Instant::now();
    let result = ocr
        .detect_text(
            &ImageSource::Path(img_path),
            &OcrRunOptions::default(),
        )
        .expect("Run MNN OCR pipeline on idcard");
    let elapsed = t0.elapsed();

    match result {
        DetectTextResult::Structured(lines) => {
            println!(
                "ID Card: Successfully recognized {} text lines in {:.2}s:",
                lines.len(),
                elapsed.as_secs_f64()
            );
            assert!(!lines.is_empty(), "Should detect text regions on idcard.jpg");

            for (i, line) in lines.iter().enumerate().take(15) {
                println!(
                    "  [{:>2}] text: {:<30} (score: {:.2})",
                    i + 1,
                    line.text,
                    line.score
                );
            }
        }
        _ => panic!("Expected structured text result"),
    }
}

#[test]
fn test_mnn_ocr_on_chronam_page() {
    let repo_root = get_repo_root();
    let det_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    let rec_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    let dict_path = repo_root.join("models/PPOCR_v6_tiny/dict.txt");
    let img_path = repo_root.join("tests/fixtures/chronam-ad-page.png");

    assert!(img_path.exists(), "chronam-ad-page.png exists: {}", img_path.display());
    assert!(det_path.exists(), "det.mnn exists: {}", det_path.display());
    assert!(rec_path.exists(), "rec.mnn exists: {}", rec_path.display());
    assert!(dict_path.exists(), "dict.txt exists: {}", dict_path.display());

    let config = InitializeConfig::ppv6(
        det_path.to_str().unwrap(),
        rec_path.to_str().unwrap(),
        dict_path.to_str().unwrap(),
    )
    .with_text_score(0.3);

    let mut ocr = RustO::initialize(config)
        .expect("Initialize RustO with pure-Rust MNN engines");

    println!("Running pure-Rust MNN OCR on chronam-ad-page.png...");
    let t0 = std::time::Instant::now();
    let result = ocr
        .detect_text(
            &ImageSource::Path(img_path),
            &OcrRunOptions::default(),
        )
        .expect("Run MNN OCR pipeline on chronam ad page");
    let elapsed = t0.elapsed();

    match result {
        DetectTextResult::Structured(lines) => {
            println!(
                "ChronAm: Successfully recognized {} text lines in {:.2}s:",
                lines.len(),
                elapsed.as_secs_f64()
            );
            assert!(!lines.is_empty(), "Should detect text regions on chronam-ad-page.png");

            for (i, line) in lines.iter().enumerate().take(15) {
                println!(
                    "  [{:>2}] text: {:<30} (score: {:.2})",
                    i + 1,
                    line.text,
                    line.score
                );
            }
        }
        _ => panic!("Expected structured text result"),
    }
}
