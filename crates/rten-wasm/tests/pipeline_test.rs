use std::fs;
use std::path::Path;
use image::GenericImageView;
use rusto_rten_wasm::RtenOcrPipeline;

#[test]
fn test_ktp_inference_pipeline() {
    let ktp_path = Path::new("../../ktp.jpeg");
    let models_dir = Path::new("../../models/PPOCR_v6_tiny");

    if !ktp_path.exists() || !models_dir.exists() {
        println!("Skipping test: ktp.jpeg or models not found");
        return;
    }

    let det_data = fs::read(models_dir.join("det.rten")).expect("read det.rten");
    let rec_data = fs::read(models_dir.join("rec.rten")).expect("read rec.rten");
    let dict_text = fs::read_to_string(models_dir.join("dict.txt")).expect("read dict.txt");

    let pipeline = RtenOcrPipeline::new(det_data, rec_data, &dict_text)
        .expect("pipeline init");

    let img = image::open(ktp_path).expect("open ktp.jpeg");
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8().into_raw();

    let json_res = pipeline
        .detect_and_recognize(&rgba, w, h, 0.3, 0.5, 1.6, true)
        .expect("detect_and_recognize");

    println!("Recognized JSON:\n{}", json_res);
    assert!(json_res.contains("PROVINSI") || json_res.contains("JAKARTA"));
    assert!(json_res.contains("3217141102960006"));
    assert!(json_res.contains("MUHAMAD RIZKI"));
}
