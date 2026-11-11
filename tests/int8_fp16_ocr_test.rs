use std::path::{Path, PathBuf};

use rusto::{
    models::resolve_base_pipeline, DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions,
    RustO,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ocr_lines(config: InitializeConfig, img: &Path) -> Vec<String> {
    let mut ocr = RustO::initialize(config).expect("Initialize RustO");
    match ocr
        .detect_text(&ImageSource::Path(img.to_path_buf()), &OcrRunOptions::default())
        .expect("Run OCR")
    {
        DetectTextResult::Structured(lines) => lines.into_iter().map(|l| l.text).collect(),
        _ => panic!("Expected structured OCR output"),
    }
}

fn assert_recognized_invoice(lines: &[String], label: &str) {
    let joined = lines.join(" ");
    println!("[{label}] ({} lines) {}", lines.len(), &joined);
    assert!(
        lines.len() >= 10,
        "{label} recognized too few lines ({}) on example1.png: {}",
        lines.len(),
        joined
    );
    let lower = joined.to_ascii_lowercase();
    assert!(
        lower.contains("invoice") && (lower.contains("ad4tech") || lower.contains("john doe")),
        "{label} missing expected invoice phrases: {}",
        joined
    );
}

#[test]
fn test_ppocrv6_model_resolution_tiny_fp16_and_int8() {
    let root = repo_root();
    let img = root.join("models/test_images/example1.png");
    assert!(img.exists(), "Missing test image: {}", img.display());

    let empty_cache = std::env::temp_dir().join("rusto_test_empty_model_cache_fp16_int8");
    for model_name in ["ppocrv6-tiny", "ppocrv6-tiny-fp16", "ppocrv6-tiny-int8"] {
        let (det, rec, dict) =
            resolve_base_pipeline(Some(model_name), None, None, None, None, None, &empty_cache)
                .unwrap_or_else(|e| panic!("resolve {model_name}: {e}"));
        let cfg = InitializeConfig::ppv6(
            det.to_str().unwrap(),
            rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.3);
        let lines = ocr_lines(cfg, &img);
        assert_recognized_invoice(&lines, &format!("{model_name} auto"));
    }
}

#[test]
fn test_ppocrv6_tiny_fp16_formats_ocr() {
    let root = repo_root();
    let img = root.join("models/test_images/example1.png");
    let dict = root.join("models/PPOCR_v6_tiny/dict.txt");

    for ext in ["mnn", "onnx"] {
        let det = root.join(format!("models/PPOCR_v6_tiny_fp16/det.{ext}"));
        let rec = root.join(format!("models/PPOCR_v6_tiny_fp16/rec.{ext}"));
        let cfg = InitializeConfig::ppv6(
            det.to_str().unwrap(),
            rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.3);
        let lines = ocr_lines(cfg, &img);
        assert_recognized_invoice(&lines, &format!("tiny-fp16 {ext}"));
    }
}

#[test]
fn test_ppocrv6_tiny_int8_formats_ocr() {
    let root = repo_root();
    let img = root.join("models/test_images/example1.png");
    let dict = root.join("models/PPOCR_v6_tiny/dict.txt");

    for ext in ["mnn", "onnx", "rten"] {
        let det = root.join(format!("models/PPOCR_v6_tiny_int8/det.{ext}"));
        let rec = root.join(format!("models/PPOCR_v6_tiny_int8/rec.{ext}"));
        let cfg = InitializeConfig::ppv6(
            det.to_str().unwrap(),
            rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.3);
        let lines = ocr_lines(cfg, &img);
        assert_recognized_invoice(&lines, &format!("tiny-int8 {ext}"));
    }
}

#[test]
fn test_ppocrv6_small_int8_ocr() {
    let root = repo_root();
    let img = root.join("models/test_images/example1.png");
    let dict = root.join("models/PPOCR_v6_small_int8/dict.txt");

    for ext in ["mnn", "rten"] {
        let det = root.join(format!("models/PPOCR_v6_small_int8/det.{ext}"));
        let rec = root.join(format!("models/PPOCR_v6_small_int8/rec.{ext}"));
        let cfg = InitializeConfig::ppv6(
            det.to_str().unwrap(),
            rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.3);
        let lines = ocr_lines(cfg, &img);
        assert_recognized_invoice(&lines, &format!("small-int8 {ext}"));
    }
}

#[test]
fn test_ppocrv6_medium_int8_ocr() {
    let root = repo_root();
    let img = root.join("models/test_images/example1.png");
    let dict = root.join("models/PPOCR_v6_medium_int8/dict.txt");

    for ext in ["mnn", "rten"] {
        let det = root.join(format!("models/PPOCR_v6_medium_int8/det.{ext}"));
        let rec = root.join(format!("models/PPOCR_v6_medium_int8/rec.{ext}"));
        let cfg = InitializeConfig::ppv6(
            det.to_str().unwrap(),
            rec.to_str().unwrap(),
            dict.to_str().unwrap(),
        )
        .with_text_score(0.3);
        let lines = ocr_lines(cfg, &img);
        assert_recognized_invoice(&lines, &format!("medium-int8 {ext}"));
    }
}

#[cfg(feature = "mnn-cpp")]
#[test]
fn test_mnn_cpp_vs_rusto_nn_fp16_and_int8_parity() {
    use rusto_core::tensor::ArrayD;
    use rusto_mnn::MnnCppSession;
    use rusto_nn::{Model, Session, Tensor, TensorLayout};

    let root = repo_root();
    let make_input = |shape: &[usize], seed: u64| -> Vec<f32> {
        let n: usize = shape.iter().product();
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
            })
            .collect()
    };

    for (label, rel, shape) in [
        ("tiny_fp16 det", "models/PPOCR_v6_tiny_fp16/det.mnn", vec![1usize, 3, 64, 64]),
        ("tiny_fp16 rec", "models/PPOCR_v6_tiny_fp16/rec.mnn", vec![1usize, 3, 48, 160]),
        ("tiny_int8 det", "models/PPOCR_v6_tiny_int8/det.mnn", vec![1usize, 3, 64, 64]),
        ("tiny_int8 rec", "models/PPOCR_v6_tiny_int8/rec.mnn", vec![1usize, 3, 48, 160]),
    ] {
        let path = root.join(rel);
        let data = make_input(&shape, 42);

        let nn_model = Model::from_file(&path).expect("rusto-nn load");
        let nn_session = Session::new(&nn_model).expect("rusto-nn session");
        let nn_out = nn_session
            .run_single(&Tensor::from_data(&shape, data.clone(), TensorLayout::NCHW))
            .expect("rusto-nn run");
        let nn_floats = nn_out.as_floats().expect("rusto-nn floats");

        let mut cpp_session = MnnCppSession::from_file(&path, 2).expect("mnn-cpp load");
        let cpp_in = ArrayD::from_shape_vec(shape.clone(), data).expect("ndarray input");
        let cpp_out = cpp_session.run(cpp_in).expect("mnn-cpp run");
        let cpp_floats = cpp_out.as_slice();

        assert_eq!(nn_out.shape(), cpp_out.shape(), "{label} shape mismatch");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (&a, &b) in nn_floats.iter().zip(cpp_floats.iter()) {
            let d: f32 = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
        }
        let mean_abs = (sum_abs / nn_floats.len().max(1) as f64) as f32;
        println!("[MNN C++ vs rusto-nn] {label}: max_abs={max_abs:.6} mean_abs={mean_abs:.6}");
        assert!(
            max_abs < 1e-2 && mean_abs < 1e-4,
            "{label} diverged from MNN C++: max_abs={max_abs}, mean_abs={mean_abs}"
        );
    }
}
