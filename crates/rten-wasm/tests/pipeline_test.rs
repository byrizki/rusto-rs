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
    assert!(json_res.contains("JOHN") || json_res.contains("J0HN"));
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

#[test]
fn test_batched_recognition_parity() {
    let sample_path = Path::new("../../docs/public/samples/example1.png");
    if !sample_path.exists() {
        return;
    }
    let img = image::open(sample_path).expect("open example1.png");
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8().into_raw();

    for (dir, dict_name) in [
        ("../../docs/public/models/indraocr", "meta.json"),
        ("../../docs/public/models/ppocrv6-tiny", "dict.txt"),
    ] {
        let model_dir = Path::new(dir);
        let det_file = model_dir.join("det.rten");
        let rec_file = model_dir.join("rec.rten");
        let dict_file = model_dir.join(dict_name);
        if !det_file.exists() || !rec_file.exists() || !dict_file.exists() {
            continue;
        }

        let pipeline = RtenOcrPipeline::new(
            fs::read(&det_file).expect("read det"),
            fs::read(&rec_file).expect("read rec"),
            &fs::read_to_string(&dict_file).expect("read dict"),
        )
        .expect("init pipeline");

        let single_json = pipeline
            .detect_text(
                &rgba,
                w,
                h,
                4,
                None,
                Some(r#"{"output":"lines","recognition":{"batchSize":1}}"#.to_string()),
            )
            .expect("detect_text batchSize=1");

        let batched_json = pipeline
            .detect_text(
                &rgba,
                w,
                h,
                4,
                None,
                Some(r#"{"output":"lines","recognition":{"batchSize":8}}"#.to_string()),
            )
            .expect("detect_text batchSize=8");

        let single_res: Vec<rusto_core::types::TextResult> =
            serde_json::from_str(&single_json).expect("parse single");
        let batched_res: Vec<rusto_core::types::TextResult> =
            serde_json::from_str(&batched_json).expect("parse batched");

        assert_eq!(single_res.len(), batched_res.len(), "line count mismatch for {dir}");
        for (idx, (s, b)) in single_res.iter().zip(batched_res.iter()).enumerate() {
            assert_eq!(s.text, b.text, "text mismatch at line {idx} for {dir}");
            assert!(
                (s.score - b.score).abs() <= 0.002,
                "score mismatch at line {idx} for {dir}: {} vs {}",
                s.score,
                b.score
            );
        }
    }
}



