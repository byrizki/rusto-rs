use std::fs;
use std::path::Path;
use image::GenericImageView;
use rusto_rten_wasm::RtenOcrPipeline;

#[test]
fn test_idcard_inference_pipeline() {
    let img_path = Path::new("../../models/test_images/idcard.jpg");
    let models_dir = Path::new("../../models/PPOCR_v6_tiny");

    let det_file = models_dir.join("det.rten");
    let rec_file = models_dir.join("rec.rten");
    let dict_file = models_dir.join("dict.txt");

    if !img_path.exists() || !det_file.exists() || !rec_file.exists() || !dict_file.exists() {
        println!("Skipping test: idcard.jpg or model files not found");
        return;
    }

    let det_data = fs::read(&det_file).expect("read det.rten");
    let rec_data = fs::read(&rec_file).expect("read rec.rten");
    let dict_text = fs::read_to_string(&dict_file).expect("read dict.txt");

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
    let det_file = models_dir.join("det.rten");
    let rec_file = models_dir.join("rec.rten");
    let dict_file = models_dir.join("dict.txt");

    if !det_file.exists() || !rec_file.exists() || !dict_file.exists() {
        println!("Skipping test: sample models not found");
        return;
    }
    let det_data = fs::read(&det_file).expect("read det.rten");
    let rec_data = fs::read(&rec_file).expect("read rec.rten");
    let dict_text = fs::read_to_string(&dict_file).expect("read dict.txt");

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

#[test]
fn test_indra_inference_pipeline() {
    let indra_dir = Path::new("../../models/indra");
    let det_file = indra_dir.join("det.rten");
    let rec_file = indra_dir.join("rec.rten");
    let meta_file = indra_dir.join("meta.json");
    if !det_file.exists() || !rec_file.exists() || !meta_file.exists() {
        return;
    }
    let det_data = fs::read(&det_file).expect("read det.rten");
    let rec_data = fs::read(&rec_file).expect("read rec.rten");
    let meta_text = fs::read_to_string(&meta_file).expect("read meta.json");

    let pipeline = RtenOcrPipeline::new(det_data, rec_data, &meta_text).expect("init indra pipeline");
    let img_path = Path::new("../../models/test_images/idcard.jpg");
    let img = image::open(img_path).expect("open idcard.jpg");
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8().into_raw();

    let json_res = pipeline
        .detect_and_recognize(&rgba, w, h, 0.4, 0.4, 1.8, false, None, None)
        .expect("detect_and_recognize with indra");

    println!("Indra pipeline recognized JSON:\n{}", json_res);
    assert!(json_res.contains("UNITED KINGDOM"));
    assert!(json_res.contains("PASSPORT") || json_res.contains("PASSEPORT"));
    assert!(json_res.contains("882802433"));
    assert!(json_res.contains("DOE") || json_res.contains("D O E"));
    assert!(json_res.contains("JOHN"));
}

#[test]
fn test_indra_order_model() {
    let order_file = Path::new("../../models/indra/indra_order.rten");
    if !order_file.exists() {
        return;
    }
    let order_data = fs::read(&order_file).expect("read order.rten");
    let model = rusto_rten_wasm::RtenModel::new(order_data).expect("load order.rten");
    let feats = vec![0.0f32; 32];
    let out = model.run_order(2, feats).expect("run_order");
    assert_eq!(out.shape(), vec![1, 2, 2]);
    assert_eq!(out.data().len(), 4);
}



