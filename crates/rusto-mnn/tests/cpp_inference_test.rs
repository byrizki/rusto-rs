use std::path::PathBuf;

use ndarray::Array4;
use rusto_mnn::MnnCppSession;

fn get_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_mnn_cpp_det_inference() {
    let repo_root = get_repo_root();
    let det_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    assert!(det_path.exists(), "det.mnn exists");

    let mut session = MnnCppSession::from_file(&det_path, 4).expect("Load MNN C++ session");
    let input = Array4::<f32>::zeros((1, 3, 640, 640)).into_dyn();

    let output = session.run(input).expect("Run MNN C++ det forward");
    assert_eq!(output.shape(), &[1, 1, 640, 640]);
}

#[test]
fn test_mnn_cpp_rec_inference() {
    let repo_root = get_repo_root();
    let rec_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    assert!(rec_path.exists(), "rec.mnn exists");

    let mut session = MnnCppSession::from_file(&rec_path, 4).expect("Load MNN C++ session");
    let input = Array4::<f32>::zeros((1, 3, 48, 160)).into_dyn();

    let output = session.run(input).expect("Run MNN C++ rec forward");
    assert_eq!(output.shape(), &[1, 20, 6906]);
}

#[test]
fn test_mnn_cpp_indra_rec_inference() {
    let repo_root = get_repo_root();
    let rec_path = repo_root.join("models/indra/rec.mnn");
    assert!(rec_path.exists(), "indra rec.mnn exists");

    let mut session = MnnCppSession::from_file(&rec_path, 4).expect("Load MNN C++ session");
    assert_eq!(session.input_channels(), 1, "Indra rec model must have 1 input channel");
    let x = Array4::<f32>::zeros((1, 1, 96, 160));
    let widths = vec![160];

    let (ids, confs) = session
        .run_linenet_rec(x, widths)
        .expect("Run MNN C++ LineNet rec forward");

    assert_eq!(ids.len(), 1);
    assert_eq!(confs.len(), 1);
    assert!(!ids[0].is_empty());
}
