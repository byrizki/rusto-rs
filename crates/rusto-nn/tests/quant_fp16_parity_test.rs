use std::path::{Path, PathBuf};

use rusto_nn::{Model, Session, Tensor, TensorLayout};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn make_seeded_tensor(shape: &[usize], seed: u64) -> (Vec<usize>, Vec<f32>) {
    let n: usize = shape.iter().product();
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 33) as f32) / ((u32::MAX >> 1) as f32);
        data.push(u * 2.0 - 1.0);
    }
    (shape.to_vec(), data)
}

fn run_rusto_nn(path: &Path, shape: &[usize], data: &[f32]) -> (Vec<usize>, Vec<f32>) {
    assert!(path.exists(), "Missing model: {}", path.display());
    let model = Model::from_file(path).unwrap_or_else(|e| panic!("Load {}: {e}", path.display()));
    let session = Session::new(&model).unwrap_or_else(|e| panic!("Session {}: {e}", path.display()));
    let input = Tensor::from_data(shape, data.to_vec(), TensorLayout::NCHW);
    let out = session
        .run_single(&input)
        .unwrap_or_else(|e| panic!("Run {}: {e}", path.display()));
    let out_shape = out.shape().to_vec();
    let floats = out
        .as_floats()
        .unwrap_or_else(|| panic!("Float output expected for {}", path.display()))
        .to_vec();
    for &v in &floats {
        assert!(v.is_finite(), "Non-finite output {} in {}", v, path.display());
    }
    (out_shape, floats)
}

#[cfg(feature = "rten")]
fn run_rten_oracle(path: &Path, shape: &[usize], data: &[f32]) -> (Vec<usize>, Vec<f32>) {
    use rten::{ModelOptions, Value};
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor as RtenTensor;

    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("Read {}: {e}", path.display()));
    let model = ModelOptions::with_all_ops()
        .load(bytes)
        .unwrap_or_else(|e| panic!("rten load {}: {e}", path.display()));
    let in_id = model.input_ids()[0];
    let out_id = model.output_ids()[0];
    let tensor = RtenTensor::from_data(shape, data.to_vec());
    let mut outputs = model
        .run(vec![(in_id, tensor.view().into())], &[out_id], None)
        .unwrap_or_else(|e| panic!("rten run {}: {e}", path.display()));
    match outputs.remove(0) {
        Value::FloatTensor(t) => (t.shape().to_vec(), t.to_vec()),
        other => panic!("Unexpected rten output variant for {}: {:?}", path.display(), other),
    }
}

fn diff_stats(a: &[f32], b: &[f32], label: &str) -> (f32, f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "Length mismatch for {label}: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        assert!(
            x.is_finite() && y.is_finite(),
            "Non-finite value in {label}: {x} vs {y}"
        );
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
    }
    (max_abs, (sum_abs / a.len().max(1) as f64) as f32)
}

fn assert_softmax_rows(shape: &[usize], floats: &[f32], label: &str) {
    assert_eq!(
        shape.len(),
        3,
        "{label}: expected 3D rec output, got {shape:?}"
    );
    let t_steps = shape[1];
    let classes = shape[2];
    for t in 0..t_steps {
        let row = &floats[t * classes..(t + 1) * classes];
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "{label}: softmax sum at step {t} was {sum} (expected ~1.0)"
        );
    }
}

#[test]
fn test_ppocrv6_int8_det_parity_across_tiers() {
    let root = repo_root();
    let (shape, input) = make_seeded_tensor(&[1, 3, 64, 64], 12345);

    for (fp32_dir, int8_dir) in [
        ("PPOCR_v6_tiny", "PPOCR_v6_tiny_int8"),
        ("PPOCR_v6_small", "PPOCR_v6_small_int8"),
        ("PPOCR_v6_medium", "PPOCR_v6_medium_int8"),
    ] {
        let int8_base = root.join("models").join(int8_dir);
        let fp32_mnn_path = root.join("models").join(fp32_dir).join("det.mnn");

        // MNN weight-quantized IDSTQuan det vs FP32 MNN det
        let (mnn_shape, mnn_out) = run_rusto_nn(&int8_base.join("det.mnn"), &shape, &input);
        let (fp32_shape, fp32_out) = run_rusto_nn(&fp32_mnn_path, &shape, &input);
        assert_eq!(mnn_shape, vec![1, 1, 64, 64]);
        assert_eq!(mnn_shape, fp32_shape);
        let (mnn_max, mnn_mean) =
            diff_stats(&mnn_out, &fp32_out, &format!("{int8_dir}/det.mnn vs fp32"));
        println!("[{int8_dir}/det.mnn vs fp32] max_abs={mnn_max:.6} mean_abs={mnn_mean:.6}");
        assert!(
            mnn_mean < 0.05,
            "{int8_dir}/det.mnn diverged from fp32: mean_abs={mnn_mean}"
        );

        #[cfg(feature = "rten")]
        {
            let rten_path = int8_base.join("det.rten");
            let (nn_rten_shape, nn_rten_out) = run_rusto_nn(&rten_path, &shape, &input);
            let (oracle_shape, oracle_out) = run_rten_oracle(&rten_path, &shape, &input);
            assert_eq!(nn_rten_shape, oracle_shape);
            let (rten_max, rten_mean) =
                diff_stats(&nn_rten_out, &oracle_out, &format!("{int8_dir}/det.rten vs rten"));
            println!("[{int8_dir}/det.rten vs rten] max_abs={rten_max:.6} mean_abs={rten_mean:.6}");
            // Note: rten 0.27.0 diverges from ORT by ~1.8e-3 on tiny and ~0.057 (mean ~0.0016) on
            // the 122-layer medium det, whereas rusto-nn stays within ~1.2e-7 (tiny) / ~0.0053 (med) of ORT.
            let (max_tol, mean_tol) = if int8_dir == "PPOCR_v6_medium_int8" {
                (0.08, 3e-3)
            } else {
                (5e-3, 1e-4)
            };
            assert!(
                rten_max < max_tol && rten_mean < mean_tol,
                "{int8_dir}/det.rten oracle mismatch: max={rten_max}, mean={rten_mean}"
            );

            #[cfg(feature = "onnx")]
            {
                let (onnx_shape, onnx_out) =
                    run_rusto_nn(&int8_base.join("det.onnx"), &shape, &input);
                assert_eq!(onnx_shape, oracle_shape);
                let (nn_pair_max, nn_pair_mean) =
                    diff_stats(&onnx_out, &nn_rten_out, &format!("{int8_dir}/det onnx vs rten"));
                println!(
                    "[{int8_dir}/det onnx vs rten in rusto-nn] max_abs={nn_pair_max:.6} mean_abs={nn_pair_mean:.6}"
                );
                assert!(
                    nn_pair_max < 1e-5 && nn_pair_mean < 1e-6,
                    "{int8_dir}/det.onnx vs det.rten mismatch: max={nn_pair_max}"
                );
            }
        }
    }
}

#[test]
fn test_ppocrv6_int8_rec_parity_across_tiers() {
    let root = repo_root();
    let (shape, input) = make_seeded_tensor(&[1, 3, 48, 160], 67890);

    for int8_dir in [
        "PPOCR_v6_tiny_int8",
        "PPOCR_v6_small_int8",
        "PPOCR_v6_medium_int8",
    ] {
        let int8_base = root.join("models").join(int8_dir);

        let (mnn_shape, mnn_out) = run_rusto_nn(&int8_base.join("rec.mnn"), &shape, &input);
        assert_softmax_rows(&mnn_shape, &mnn_out, &format!("{int8_dir}/rec.mnn"));

        #[cfg(feature = "rten")]
        {
            let rten_path = int8_base.join("rec.rten");
            let (nn_rten_shape, nn_rten_out) = run_rusto_nn(&rten_path, &shape, &input);
            let (oracle_shape, oracle_out) = run_rten_oracle(&rten_path, &shape, &input);
            assert_eq!(nn_rten_shape, oracle_shape);
            assert_softmax_rows(&nn_rten_shape, &nn_rten_out, &format!("{int8_dir}/rec.rten"));
            let (rten_max, rten_mean) =
                diff_stats(&nn_rten_out, &oracle_out, &format!("{int8_dir}/rec.rten vs rten"));
            println!("[{int8_dir}/rec.rten vs rten] max_abs={rten_max:.6} mean_abs={rten_mean:.6}");
            assert!(
                rten_max < 2e-2 && rten_mean < 1e-4,
                "{int8_dir}/rec.rten oracle mismatch: max={rten_max}, mean={rten_mean}"
            );

            #[cfg(feature = "onnx")]
            {
                let (onnx_shape, onnx_out) =
                    run_rusto_nn(&int8_base.join("rec.onnx"), &shape, &input);
                assert_eq!(onnx_shape, oracle_shape);
                assert_softmax_rows(&onnx_shape, &onnx_out, &format!("{int8_dir}/rec.onnx"));
                let (nn_pair_max, nn_pair_mean) =
                    diff_stats(&onnx_out, &nn_rten_out, &format!("{int8_dir}/rec onnx vs rten"));
                println!(
                    "[{int8_dir}/rec onnx vs rten in rusto-nn] max_abs={nn_pair_max:.6} mean_abs={nn_pair_mean:.6}"
                );
                assert!(
                    nn_pair_max < 1e-5 && nn_pair_mean < 1e-6,
                    "{int8_dir}/rec.onnx vs rec.rten mismatch: max={nn_pair_max}"
                );
            }
        }
    }
}

#[test]
fn test_ppocrv6_tiny_fp16_parity() {
    let root = repo_root();
    let fp32_dir = root.join("models/PPOCR_v6_tiny");
    let fp16_dir = root.join("models/PPOCR_v6_tiny_fp16");

    // 1. Detector FP16 (.mnn & .onnx) vs corresponding FP32 models
    let (det_shape, det_in) = make_seeded_tensor(&[1, 3, 64, 64], 42);
    let (det32_mnn_shape, det32_mnn_out) =
        run_rusto_nn(&fp32_dir.join("det.mnn"), &det_shape, &det_in);
    let (det16_mnn_shape, det16_mnn_out) =
        run_rusto_nn(&fp16_dir.join("det.mnn"), &det_shape, &det_in);
    assert_eq!(det16_mnn_shape, det32_mnn_shape);
    let (max_mnn, mean_mnn) =
        diff_stats(&det16_mnn_out, &det32_mnn_out, "tiny_fp16/det.mnn vs fp32 mnn");
    println!("[tiny_fp16/det.mnn vs fp32 mnn] max_abs={max_mnn:.6} mean_abs={mean_mnn:.6}");
    assert!(
        max_mnn < 1e-2 && mean_mnn < 1e-3,
        "tiny_fp16/det.mnn parity: max={max_mnn}, mean={mean_mnn}"
    );

    #[cfg(feature = "onnx")]
    {
        let (det32_onnx_shape, det32_onnx_out) =
            run_rusto_nn(&fp32_dir.join("det.onnx"), &det_shape, &det_in);
        let (det16_onnx_shape, det16_onnx_out) =
            run_rusto_nn(&fp16_dir.join("det.onnx"), &det_shape, &det_in);
        assert_eq!(det16_onnx_shape, det32_onnx_shape);
        let (max_onnx, mean_onnx) =
            diff_stats(&det16_onnx_out, &det32_onnx_out, "tiny_fp16/det.onnx vs fp32 onnx");
        println!("[tiny_fp16/det.onnx vs fp32 onnx] max_abs={max_onnx:.6} mean_abs={mean_onnx:.6}");
        assert!(
            max_onnx < 1e-2 && mean_onnx < 1e-3,
            "tiny_fp16/det.onnx parity: max={max_onnx}, mean={mean_onnx}"
        );
    }

    // 2. Recognizer FP16 (.mnn & .onnx) vs corresponding FP32 models
    let (rec_shape, rec_in) = make_seeded_tensor(&[1, 3, 48, 160], 42);
    let (rec32_mnn_shape, rec32_mnn_out) =
        run_rusto_nn(&fp32_dir.join("rec.mnn"), &rec_shape, &rec_in);
    let (rec16_mnn_shape, rec16_mnn_out) =
        run_rusto_nn(&fp16_dir.join("rec.mnn"), &rec_shape, &rec_in);
    assert_eq!(rec16_mnn_shape, rec32_mnn_shape);
    assert_softmax_rows(&rec16_mnn_shape, &rec16_mnn_out, "tiny_fp16/rec.mnn");
    let (rmax_mnn, rmean_mnn) =
        diff_stats(&rec16_mnn_out, &rec32_mnn_out, "tiny_fp16/rec.mnn vs fp32 mnn");
    println!("[tiny_fp16/rec.mnn vs fp32 mnn] max_abs={rmax_mnn:.6} mean_abs={rmean_mnn:.6}");
    assert!(
        rmean_mnn < 1e-3 && rmax_mnn < 0.25,
        "tiny_fp16/rec.mnn parity: max={rmax_mnn}, mean={rmean_mnn}"
    );

    #[cfg(feature = "onnx")]
    {
        let (rec32_onnx_shape, rec32_onnx_out) =
            run_rusto_nn(&fp32_dir.join("rec.onnx"), &rec_shape, &rec_in);
        let (rec16_onnx_shape, rec16_onnx_out) =
            run_rusto_nn(&fp16_dir.join("rec.onnx"), &rec_shape, &rec_in);
        assert_eq!(rec16_onnx_shape, rec32_onnx_shape);
        assert_softmax_rows(&rec16_onnx_shape, &rec16_onnx_out, "tiny_fp16/rec.onnx");
        let (rmax_onnx, rmean_onnx) =
            diff_stats(&rec16_onnx_out, &rec32_onnx_out, "tiny_fp16/rec.onnx vs fp32 onnx");
        println!("[tiny_fp16/rec.onnx vs fp32 onnx] max_abs={rmax_onnx:.6} mean_abs={rmean_onnx:.6}");
        assert!(
            rmean_onnx < 1e-3 && rmax_onnx < 0.1,
            "tiny_fp16/rec.onnx parity: max={rmax_onnx}, mean={rmean_onnx}"
        );
    }
}
