use std::path::PathBuf;
use rusto_rten::RtenSession;

#[test]
fn test_rusto_rten_load_and_run() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let det_path = repo_root.join("models/PPOCR_v6_tiny/det.rten");

    if !det_path.exists() {
        eprintln!("Skipping test_rusto_rten_load_and_run: det.rten not found");
        return;
    }

    let mut session = RtenSession::from_file(&det_path, 1).expect("Failed to load det.rten");
    assert_eq!(session.input_channels(), 3);

    let dummy_input = ndarray::Array4::<f32>::zeros((1, 3, 64, 64)).into_dyn();
    let out = session.run(&dummy_input).expect("Failed to run inference");
    assert!(!out.is_empty());
}

