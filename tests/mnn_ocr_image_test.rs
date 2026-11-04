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

    assert!(det_path.exists(), "det.mnn exists");
    assert!(rec_path.exists(), "rec.mnn exists");
    assert!(dict_path.exists(), "dict.txt exists");
    assert!(img_path.exists(), "example1.png exists");

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

            for (i, line) in lines.iter().enumerate().take(10) {
                println!(
                    "  [{:>2}] text: {:<30} (score: {:.2}, bbox: [w={:.1}, h={:.1}, top={:.1}, left={:.1}])",
                    i + 1,
                    line.text,
                    line.score,
                    line.frame.width,
                    line.frame.height,
                    line.frame.top,
                    line.frame.left
                );
            }

            // Verify recognizable text tokens
            let all_text: String = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ");
            println!("\nCombined detected text sample: {}", &all_text[..all_text.len().min(200)]);
            assert!(
                all_text.contains("INVOICE") || all_text.contains("Invoice") || all_text.contains("Material"),
                "Expected invoice keywords in recognized text, got: {}",
                all_text
            );
        }
        _ => panic!("Expected structured text result"),
    }
}

#[test]
fn test_mnn_ocr_on_ktp_image() {
    let repo_root = get_repo_root();
    let det_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    let rec_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    let dict_path = repo_root.join("models/PPOCR_v6_tiny/dict.txt");
    let img_path = repo_root.join("ktp.jpeg");

    if !img_path.exists() {
        eprintln!("ktp.jpeg does not exist, skipping");
        return;
    }

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

    println!("Running pure-Rust MNN OCR on ktp.jpeg...");
    let t0 = std::time::Instant::now();
    let result = ocr
        .detect_text(
            &ImageSource::Path(img_path),
            &OcrRunOptions::default(),
        )
        .expect("Run MNN OCR pipeline on ktp");
    let elapsed = t0.elapsed();

    match result {
        DetectTextResult::Structured(lines) => {
            println!(
                "KTP: Successfully recognized {} text lines in {:.2}s:",
                lines.len(),
                elapsed.as_secs_f64()
            );
            assert!(!lines.is_empty(), "Should detect text regions on ktp.jpeg");

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

    if !img_path.exists() {
        eprintln!("chronam-ad-page.png does not exist, skipping");
        return;
    }

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
        .expect("Run MNN OCR pipeline on chronam");
    let elapsed = t0.elapsed();

    match result {
        DetectTextResult::Structured(lines) => {
            println!(
                "Chronam: Successfully recognized {} text lines in {:.2}s:",
                lines.len(),
                elapsed.as_secs_f64()
            );
            assert!(!lines.is_empty(), "Should detect lines on chronam-ad-page.png");

            for (i, line) in lines.iter().enumerate().take(10) {
                println!(
                    "  [{:>3}] text: {:<35} (score: {:.2})",
                    i + 1,
                    line.text,
                    line.score
                );
            }
        }
        _ => panic!("Expected structured text result"),
    }
}

