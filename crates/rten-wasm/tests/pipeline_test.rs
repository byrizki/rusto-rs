use std::fs;
use std::path::Path;
use image::GenericImageView;
use rusto_rten_wasm::RtenOcrPipeline;

#[test]
fn test_idcard_inference_pipeline() {
    let img_path = Path::new("../../models/test_images/idcard.jpg");
    let models_dir = Path::new("../../models/PPOCR_v6_tiny");

    if !img_path.exists() || !models_dir.exists() {
        println!("Skipping test: idcard.jpg or models not found");
        return;
    }

    let det_data = fs::read(models_dir.join("det.rten")).expect("read det.rten");
    let rec_data = fs::read(models_dir.join("rec.rten")).expect("read rec.rten");
    let dict_text = fs::read_to_string(models_dir.join("dict.txt")).expect("read dict.txt");

    let pipeline = RtenOcrPipeline::new(det_data, rec_data, &dict_text)
        .expect("pipeline init");

    let img = image::open(img_path).expect("open idcard.jpg");
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8().into_raw();

    let json_res = pipeline
        .detect_and_recognize(&rgba, w, h, 0.3, 0.5, 1.8, true, None, None)
        .expect("detect_and_recognize");

    println!("Recognized JSON for idcard.jpg:\n{}", json_res);
    assert!(json_res.contains("PASSPORT") || json_res.contains("PASSEPORT"));
    assert!(json_res.contains("882802433"));
    assert!(json_res.contains("DOE"));
    assert!(json_res.contains("JOHN"));
}

#[test]
fn test_sample_images() {
    let models_dir = Path::new("../../docs/public/models/ppocrv6-small");
    if !models_dir.exists() {
        return;
    }
    let det_data = fs::read(models_dir.join("det.rten")).expect("read det.rten");
    let rec_data = fs::read(models_dir.join("rec.rten")).expect("read rec.rten");
    let dict_text = fs::read_to_string(models_dir.join("dict.txt")).expect("read dict.txt");

    let pipeline = RtenOcrPipeline::new(det_data, rec_data, &dict_text).expect("pipeline init");

    for sample_name in &["example1.png", "idcard.jpg", "invoice1.jpg"] {
        let sample_path = Path::new("../../docs/public/samples").join(sample_name);
        println!("Testing {}", sample_name);
        let img = image::open(&sample_path).expect("open sample");
        let (w, h) = img.dimensions();
        let rgba = img.to_rgba8().into_raw();
        let json_res = pipeline.detect_and_recognize(&rgba, w, h, 0.3, 0.5, 1.6, true, None, None).expect("detect_and_recognize");
        println!("{}: {} bytes output", sample_name, json_res.len());
    }
}

