use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rusto::{
    DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, OutputGranularity, RustO,
};

fn get_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn execute_ocr_run(
    mut ocr: RustO,
    img_path: &Path,
    granularity: OutputGranularity,
) -> (DetectTextResult, Duration) {
    let t0 = Instant::now();
    let opts = OcrRunOptions {
        output: granularity,
        ..Default::default()
    };
    let result = ocr
        .detect_text(&ImageSource::Path(img_path.to_path_buf()), &opts)
        .expect("OCR execution failed");
    (result, t0.elapsed())
}

#[test]
fn test_ppocr_v6_tiny_rten_vs_mnn_comparison() {
    let root = get_repo_root();
    let rten_det = root.join("models/PPOCR_v6_tiny/det.rten");
    let rten_rec = root.join("models/PPOCR_v6_tiny/rec.rten");
    let mnn_det = root.join("models/PPOCR_v6_tiny/det.mnn");
    let mnn_rec = root.join("models/PPOCR_v6_tiny/rec.mnn");
    let dict = root.join("models/PPOCR_v6_tiny/dict.txt");

    assert!(rten_det.exists(), "det.rten exists: {}", rten_det.display());
    assert!(rten_rec.exists(), "rec.rten exists: {}", rten_rec.display());
    assert!(mnn_det.exists(), "det.mnn exists: {}", mnn_det.display());
    assert!(mnn_rec.exists(), "rec.mnn exists: {}", mnn_rec.display());
    assert!(dict.exists(), "dict.txt exists: {}", dict.display());

    let test_images = [
        ("example1.png", root.join("models/test_images/example1.png")),
        ("idcard.jpg", root.join("models/test_images/idcard.jpg")),
        ("chronam-ad-page.png", root.join("tests/fixtures/chronam-ad-page.png")),
    ];

    println!("\n==================================================================================");
    println!("PP-OCR v6 Tiny: FULL PIPELINE COMPARISON (RTen vs Pure-Rust MNN)");
    println!("==================================================================================");

    for (img_name, img_path) in &test_images {
        if !img_path.exists() {
            println!("Skipping {} (not found)", img_name);
            continue;
        }

        println!("\n--- Test Fixture: {} ---", img_name);

        // 1. RTen Engine Run
        let rten_cfg = InitializeConfig::ppv6(
            rten_det.to_str().unwrap(),
            rten_rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.2)
        .with_rec_batch_num(16);

        let rten_ocr = RustO::initialize(rten_cfg).expect("Init RTen OCR");
        let (rten_res, rten_lat) = execute_ocr_run(rten_ocr, img_path, OutputGranularity::Lines);

        let (rten_lines, rten_avg_score, rten_sample) = match &rten_res {
            DetectTextResult::Structured(lines) => {
                let cnt = lines.len();
                let avg = if cnt > 0 {
                    lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                } else {
                    0.0
                };
                let sample = lines
                    .iter()
                    .take(5)
                    .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                    .collect::<Vec<_>>()
                    .join(", ");
                (cnt, avg, sample)
            }
            _ => (0, 0.0, String::new()),
        };

        // 2. Pure-Rust MNN Engine Run
        let mnn_cfg = InitializeConfig::ppv6(
            mnn_det.to_str().unwrap(),
            mnn_rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.2)
        .with_rec_batch_num(16);

        let mnn_ocr = RustO::initialize(mnn_cfg).expect("Init MNN OCR");
        let (mnn_res, mnn_lat) = execute_ocr_run(mnn_ocr, img_path, OutputGranularity::Lines);

        let (mnn_lines, mnn_avg_score, mnn_sample) = match &mnn_res {
            DetectTextResult::Structured(lines) => {
                let cnt = lines.len();
                let avg = if cnt > 0 {
                    lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                } else {
                    0.0
                };
                let sample = lines
                    .iter()
                    .take(5)
                    .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                    .collect::<Vec<_>>()
                    .join(", ");
                (cnt, avg, sample)
            }
            _ => (0, 0.0, String::new()),
        };

        println!(
            "{:<16} | {:<12} | {:<10} | {:<12}",
            "Engine", "Lines Found", "Avg Score", "Latency"
        );
        println!("{:-<16}-+-{:-<12}-+-{:-<10}-+-{:-<12}", "", "", "", "");
        println!(
            "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
            "RTen (ONNX)", rten_lines, rten_avg_score, rten_lat
        );
        println!(
            "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
            "Pure-Rust MNN", mnn_lines, mnn_avg_score, mnn_lat
        );

        // 3. Original Alibaba MNN C++ Engine Run (PP-OCRv6)
        #[cfg(feature = "mnn-cpp")]
        {
            let cpp_det = rusto_mnn::MnnCppSession::from_file(&mnn_det, 4).expect("Init MNN C++ Det");
            let cpp_rec = rusto_mnn::MnnCppSession::from_file(&mnn_rec, 4).expect("Init MNN C++ Rec");
            let cpp_cfg = InitializeConfig::ppv6(
                mnn_det.to_str().unwrap(),
                mnn_rec.to_str().unwrap(),
                dict.to_str().unwrap(),
            )
            .with_text_score(0.2)
            .with_rec_batch_num(16);
            let cpp_ocr = RustO::with_engine_sessions(
                cpp_cfg,
                rusto::EngineSession::from_mnn_cpp(cpp_det),
                rusto::EngineSession::from_mnn_cpp(cpp_rec),
            )
            .expect("Init MNN C++ OCR");
            let (cpp_res, cpp_lat) = execute_ocr_run(cpp_ocr, img_path, OutputGranularity::Lines);
            let (cpp_lines, cpp_avg_score, cpp_sample) = match &cpp_res {
                DetectTextResult::Structured(lines) => {
                    let cnt = lines.len();
                    let avg = if cnt > 0 {
                        lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                    } else {
                        0.0
                    };
                    let sample = lines
                        .iter()
                        .take(5)
                        .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                        .collect::<Vec<_>>()
                        .join(", ");
                    (cnt, avg, sample)
                }
                _ => (0, 0.0, String::new()),
            };

            println!(
                "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
                "Alibaba MNN C++", cpp_lines, cpp_avg_score, cpp_lat
            );
            println!("  MNN C++ sample: {}", cpp_sample);
        }

        #[cfg(feature = "webgpu")]
        {
            rusto_nn_webgpu::init_and_register().ok();
            let gpu_cfg = InitializeConfig::ppv6(
                mnn_det.to_str().unwrap(),
                mnn_rec.to_str().unwrap(),
                dict.to_str().unwrap(),
            )
            .with_text_score(0.2)
            .with_rec_batch_num(16);

            if let Ok(gpu_ocr) = RustO::initialize(gpu_cfg) {
                let (gpu_res, gpu_lat) = execute_ocr_run(gpu_ocr, img_path, OutputGranularity::Lines);
                let (gpu_lines, gpu_avg_score, gpu_sample) = match &gpu_res {
                    DetectTextResult::Structured(lines) => {
                        let cnt = lines.len();
                        let avg = if cnt > 0 {
                            lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                        } else {
                            0.0
                        };
                        let sample = lines
                            .iter()
                            .take(5)
                            .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                            .collect::<Vec<_>>()
                            .join(", ");
                        (cnt, avg, sample)
                    }
                    _ => (0, 0.0, String::new()),
                };
                println!(
                    "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
                    "Pure MNN WebGPU", gpu_lines, gpu_avg_score, gpu_lat
                );
                println!("  WebGPU sample:   {}", gpu_sample);
            }
            rusto_nn_webgpu::clear();
        }

        println!("  RTen sample:     {}", rten_sample);
        println!("  Pure MNN sample: {}", mnn_sample);

        // Verification logging
        if mnn_lines > 0 {
            let line_diff = (rten_lines as isize - mnn_lines as isize).abs();
            let max_lines = rten_lines.max(mnn_lines);
            let relative_diff = line_diff as f32 / max_lines as f32;
            println!(
                "Line count relative difference on {}: {:.2} (RTen={}, MNN={})",
                img_name, relative_diff, rten_lines, mnn_lines
            );
        } else {
            println!(
                "Notice: MNN detected 0 lines on {} (may require calibration/optimization)",
                img_name
            );
        }
    }
}

#[test]
fn test_indra_ocr_rten_vs_mnn_comparison() {
    let root = get_repo_root();
    let rten_det = root.join("models/indra/det.rten");
    let rten_rec = root.join("models/indra/rec.rten");
    let mnn_det = root.join("models/indra/det.mnn");
    let mnn_rec = root.join("models/indra/rec.mnn");
    let meta = root.join("models/indra/meta.json");

    assert!(rten_det.exists(), "indra det.rten exists: {}", rten_det.display());
    assert!(rten_rec.exists(), "indra rec.rten exists: {}", rten_rec.display());
    assert!(mnn_det.exists(), "indra det.mnn exists: {}", mnn_det.display());
    assert!(mnn_rec.exists(), "indra rec.mnn exists: {}", mnn_rec.display());
    assert!(meta.exists(), "indra meta.json exists: {}", meta.display());

    let test_images = [
        ("idcard.jpg", root.join("models/test_images/idcard.jpg")),
        ("example1.png", root.join("models/test_images/example1.png")),
    ];

    println!("\n==================================================================================");
    println!("Indra OCR: FULL PIPELINE COMPARISON (RTen vs Pure-Rust MNN)");
    println!("==================================================================================");

    for (img_name, img_path) in &test_images {
        if !img_path.exists() {
            println!("Skipping {} (not found)", img_name);
            continue;
        }

        println!("\n--- Test Fixture: {} ---", img_name);

        // 1. RTen Engine Run
        let rten_cfg = InitializeConfig::indra(
            rten_det.to_str().unwrap(),
            rten_rec.to_str().unwrap(),
            meta.to_str().unwrap(),
        );

        let rten_ocr = RustO::initialize(rten_cfg).expect("Init RTen Indra OCR");
        let (rten_res, rten_lat) = execute_ocr_run(rten_ocr, img_path, OutputGranularity::Lines);

        let (rten_lines, rten_avg_score, rten_sample, _rten_full) = match &rten_res {
            DetectTextResult::Structured(lines) => {
                let cnt = lines.len();
                let avg = if cnt > 0 {
                    lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                } else {
                    0.0
                };
                let sample = lines
                    .iter()
                    .take(5)
                    .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                    .collect::<Vec<_>>()
                    .join(", ");
                let full = lines
                    .iter()
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                (cnt, avg, sample, full)
            }
            _ => (0, 0.0, String::new(), String::new()),
        };

        // 2. Pure-Rust MNN Engine Run
        let mnn_cfg = InitializeConfig::indra(
            mnn_det.to_str().unwrap(),
            mnn_rec.to_str().unwrap(),
            meta.to_str().unwrap(),
        );

        let mnn_ocr = RustO::initialize(mnn_cfg).expect("Init MNN Indra OCR");
        let (mnn_res, mnn_lat) = execute_ocr_run(mnn_ocr, img_path, OutputGranularity::Lines);
        println!("MNN RES DEBUG: {:?}", match &mnn_res {
            DetectTextResult::Structured(lines) => format!("Structured with {} lines", lines.len()),
            DetectTextResult::Spatial(text) => format!("Spatial: {}", text),
        });

        let (mnn_lines, mnn_avg_score, mnn_sample, mnn_full) = match &mnn_res {
            DetectTextResult::Structured(lines) => {
                let cnt = lines.len();
                let avg = if cnt > 0 {
                    lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                } else {
                    0.0
                };
                let sample = lines
                    .iter()
                    .take(5)
                    .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                    .collect::<Vec<_>>()
                    .join(", ");
                let full = lines
                    .iter()
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                (cnt, avg, sample, full)
            }
            _ => (0, 0.0, String::new(), String::new()),
        };

        println!(
            "{:<16} | {:<12} | {:<10} | {:<12}",
            "Engine", "Lines Found", "Avg Score", "Latency"
        );
        println!("{:-<16}-+-{:-<12}-+-{:-<10}-+-{:-<12}", "", "", "", "");
        println!(
            "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
            "RTen (ONNX)", rten_lines, rten_avg_score, rten_lat
        );
        println!(
            "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
            "Pure-Rust MNN", mnn_lines, mnn_avg_score, mnn_lat
        );

        // 3. Original Alibaba MNN C++ Engine Run (Indra OCR)
        #[cfg(feature = "mnn-cpp")]
        {
            let cpp_det = rusto_mnn::MnnCppSession::from_file(&mnn_det, 4).expect("Init MNN C++ Indra Det");
            let cpp_rec = rusto_mnn::MnnCppSession::from_file(&mnn_rec, 4).expect("Init MNN C++ Indra Rec");
            let cpp_cfg = InitializeConfig::indra(
                mnn_det.to_str().unwrap(),
                mnn_rec.to_str().unwrap(),
                meta.to_str().unwrap(),
            );
            let cpp_ocr = RustO::with_engine_sessions(
                cpp_cfg,
                rusto::EngineSession::from_mnn_cpp(cpp_det),
                rusto::EngineSession::from_mnn_cpp(cpp_rec),
            )
            .expect("Init MNN C++ OCR");
            let (cpp_res, cpp_lat) = execute_ocr_run(cpp_ocr, img_path, OutputGranularity::Lines);
            let (cpp_lines, cpp_avg_score, cpp_sample) = match &cpp_res {
                DetectTextResult::Structured(lines) => {
                    let cnt = lines.len();
                    let avg = if cnt > 0 {
                        lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                    } else {
                        0.0
                    };
                    let sample = lines
                        .iter()
                        .take(5)
                        .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                        .collect::<Vec<_>>()
                        .join(", ");
                    (cnt, avg, sample)
                }
                _ => (0, 0.0, String::new()),
            };

            println!(
                "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
                "Alibaba MNN C++", cpp_lines, cpp_avg_score, cpp_lat
            );
            println!("  MNN C++ sample: {}", cpp_sample);
        }

        #[cfg(feature = "webgpu")]
        {
            std::env::set_var("MNN_WEBGPU", "1");
            let gpu_cfg = InitializeConfig::indra(
                mnn_det.to_str().unwrap(),
                mnn_rec.to_str().unwrap(),
                meta.to_str().unwrap(),
            );

            if let Ok(gpu_ocr) = RustO::initialize(gpu_cfg) {
                let (gpu_res, gpu_lat) = execute_ocr_run(gpu_ocr, img_path, OutputGranularity::Lines);
                let (gpu_lines, gpu_avg_score, gpu_sample) = match &gpu_res {
                    DetectTextResult::Structured(lines) => {
                        let cnt = lines.len();
                        let avg = if cnt > 0 {
                            lines.iter().map(|l| l.score).sum::<f32>() / cnt as f32
                        } else {
                            0.0
                        };
                        let sample = lines
                            .iter()
                            .take(5)
                            .map(|l| format!("\"{}\" ({:.2})", l.text.trim(), l.score))
                            .collect::<Vec<_>>()
                            .join(", ");
                        (cnt, avg, sample)
                    }
                    _ => (0, 0.0, String::new()),
                };
                println!(
                    "{:<16} | {:<12} | {:<10.2} | {:<12.2?}",
                    "Pure MNN WebGPU", gpu_lines, gpu_avg_score, gpu_lat
                );
                println!("  WebGPU sample:   {}", gpu_sample);
            }
            std::env::remove_var("MNN_WEBGPU");
        }

        println!("  RTen sample:     {}", rten_sample);
        println!("  Pure MNN sample: {}", mnn_sample);

        // Verification assertions
        assert!(mnn_lines > 0, "MNN must detect lines on {}", img_name);
        println!("  MNN full recognized text: {}", mnn_full.trim());
    }
}

#[test]
fn test_spatial_layout_mnn_verification() {
    let root = get_repo_root();
    let mnn_det = root.join("models/PPOCR_v6_tiny/det.mnn");
    let mnn_rec = root.join("models/PPOCR_v6_tiny/rec.mnn");
    let dict = root.join("models/PPOCR_v6_tiny/dict.txt");
    let img_path = root.join("models/test_images/example1.png");

    assert!(mnn_det.exists(), "det.mnn exists: {}", mnn_det.display());
    assert!(mnn_rec.exists(), "rec.mnn exists: {}", mnn_rec.display());
    assert!(dict.exists(), "dict.txt exists: {}", dict.display());
    assert!(img_path.exists(), "example1.png exists: {}", img_path.display());

    let mnn_cfg = InitializeConfig::ppv6(
        mnn_det.to_str().unwrap(),
        mnn_rec.to_str().unwrap(),
        dict.to_str().unwrap(),
    )
    .with_text_score(0.2);

    let mnn_ocr = RustO::initialize(mnn_cfg).expect("Init MNN OCR");
    let (mnn_res, lat) = execute_ocr_run(mnn_ocr, &img_path, OutputGranularity::Spatial);

    match mnn_res {
        DetectTextResult::Spatial(s) => {
            println!("MNN Spatial layout ({} bytes) in {:.2?}:", s.len(), lat);
            println!("{}", &s[..s.len().min(300)]);
            assert!(!s.is_empty(), "MNN must produce non-empty spatial text");
            assert!(
                s.contains("INVOICE") || s.contains("Invoice") || s.contains("Material"),
                "Expected invoice keywords in spatial text"
            );
        }
        _ => panic!("Expected Spatial result"),
    }
}

#[test]
#[cfg(feature = "rten")]
fn test_debug_compare_rten_vs_rusto_nn() {
    let root = get_repo_root();
    let rten_path = root.join("models/indra/rec.rten");
    assert!(rten_path.exists(), "indra rec.rten exists: {}", rten_path.display());

    let mut official_rten = rusto_rten::RtenSession::from_file(&rten_path, 1).expect("load official rten");
    let rusto_nn_model = rusto_nn::Model::from_file(&rten_path).expect("load rusto_nn model");
    let rusto_nn_session = rusto_nn::Session::new(&rusto_nn_model).expect("create rusto_nn session");

    // Create an input tensor simulating ink text on white background:
    // height = 96, width = 288. Background = 0.0 (no ink). Text bars = 1.0 (ink).
    let mut x_arr = ndarray::Array4::<f32>::zeros((1, 1, 96, 288));
    for y in 20..76 {
        for x in (30..250).step_by(10) {
            for dx in 0..4 {
                x_arr[[0, 0, y, x + dx]] = 0.9;
            }
        }
    }
    let widths = vec![288i32];

    let (off_ids, off_confs) = official_rten.run_linenet_rec(&x_arr, &widths).expect("run official rten");
    let x_t = rusto_nn::Tensor::from_floats(&[1, 1, 96, 288], x_arr.iter().copied().collect(), rusto_nn::TensorLayout::NCHW);
    let widths_t = rusto_nn::Tensor::from_ints(&[1], widths);
    let rusto_out = rusto_nn_session.run(&[("x", &x_t), ("widths", &widths_t)]).expect("run rusto_nn");
    let r_ids = rusto_out["ids"].to_i32_vec();
    let r_confs = rusto_out["conf"].to_f32_vec();

    assert_eq!(off_ids[0], r_ids, "Token IDs between official RTen and rusto-nn must match identically");
    for (&oc, &rc) in off_confs[0].iter().zip(r_confs.iter()) {
        let oc_f: f32 = oc;
        let rc_f: f32 = rc;
        assert!((oc_f - rc_f).abs() < 1e-4, "Confidence deviation too large: off={} vs rusto={}", oc, rc);
    }
}

