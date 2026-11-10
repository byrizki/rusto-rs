use std::fs;
use std::path::Path;
use image::GenericImageView;
use rusto_wasm::{RtenModel, RtenOcrPipeline};

fn get_model_filenames() -> (&'static str, &'static str) {
    if cfg!(feature = "rusto-rten") {
        ("det.rten", "rec.rten")
    } else {
        ("det.mnn", "rec.mnn")
    }
}

#[test]
fn test_idcard_inference_pipeline() {
    let img_path = Path::new("../../models/test_images/idcard.jpg");
    let models_dir = Path::new("../../models/PPOCR_v6_tiny");

    let (det_name, rec_name) = get_model_filenames();
    let det_file = models_dir.join(det_name);
    let rec_file = models_dir.join(rec_name);
    let dict_file = models_dir.join("dict.txt");

    if !img_path.exists() || !det_file.exists() || !rec_file.exists() || !dict_file.exists() {
        println!("Skipping test: idcard.jpg or model files not found");
        return;
    }

    let det_data = fs::read(&det_file).expect("read det model");
    let rec_data = fs::read(&rec_file).expect("read rec model");
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
    let (det_name, rec_name) = get_model_filenames();
    let det_file = models_dir.join(det_name);
    let rec_file = models_dir.join(rec_name);
    let dict_file = models_dir.join("dict.txt");

    if !det_file.exists() || !rec_file.exists() || !dict_file.exists() {
        println!("Skipping test: sample models not found");
        return;
    }
    let det_data = fs::read(&det_file).expect("read det");
    let rec_data = fs::read(&rec_file).expect("read rec");
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
    let (det_name, rec_name) = get_model_filenames();
    let det_file = indra_dir.join(det_name);
    let rec_file = indra_dir.join(rec_name);
    let meta_file = indra_dir.join("meta.json");
    if !det_file.exists() || !rec_file.exists() || !meta_file.exists() {
        return;
    }
    let det_data = fs::read(&det_file).expect("read det");
    let rec_data = fs::read(&rec_file).expect("read rec");
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
    assert!(json_res.contains("DOE") || json_res.contains("D O E") || json_res.contains("DO E"));
    assert!(json_res.contains("JOHN"));
}

#[test]
fn test_indra_layout_model() {
    let candidates = [
        Path::new("../../models/indra/order.rten"),
        Path::new("../../models/indra/indra_order.rten"),
        Path::new("../../models/indra/order.onnx"),
        Path::new("../../models/indra/indra_order.onnx"),
    ];
    let layout_file = match candidates.iter().find(|p| p.exists()) {
        Some(p) => p,
        None => return,
    };
    let layout_data = fs::read(layout_file).expect("read layout model");
    let model = RtenModel::new(layout_data.clone()).expect("load layout model");
    let feats = vec![0.0f32; 32];
    let out = model.run_order(2, feats).expect("run_order");
    assert_eq!(out.shape(), vec![1, 2, 2]);
    assert_eq!(out.data().len(), 4);

    let indra_dir = Path::new("../../models/indra");
    let (det_name, rec_name) = get_model_filenames();
    let det_file = indra_dir.join(det_name);
    let rec_file = indra_dir.join(rec_name);
    let meta_file = indra_dir.join("meta.json");
    if det_file.exists() && rec_file.exists() && meta_file.exists() {
        let det_data = fs::read(&det_file).expect("read det");
        let rec_data = fs::read(&rec_file).expect("read rec");
        let meta_text = fs::read_to_string(&meta_file).expect("read meta.json");

        let pipeline = RtenOcrPipeline::new_with_models(
            det_data,
            rec_data,
            &meta_text,
            Some(layout_data),
        )
        .expect("init pipeline with layout model");

        let img_path = Path::new("../../models/test_images/idcard.jpg");
        if img_path.exists() {
            let img = image::open(img_path).expect("open idcard.jpg");
            let (w, h) = img.dimensions();
            let rgba = img.to_rgba8().into_raw();
            let opt = r#"{"output":"lines","layoutReading":true,"layoutModel":"indra"}"#;
            let res = pipeline
                .detect_text(&rgba, w, h, 4, None, Some(opt.to_string()))
                .expect("detect_text with layoutReading");
            assert!(res.contains("PASSPORT") || res.contains("PASSEPORT") || res.contains("UNITED KINGDOM"));
        }
    }
}

#[test]
fn test_batched_recognition_parity() {
    for sample_name in ["example1.png", "invoice1.jpg"] {
        let sample_path = Path::new("../../docs/public/samples").join(sample_name);
        if !sample_path.exists() {
            continue;
        }
        let img = image::open(&sample_path).expect("open sample");
        let (w, h) = img.dimensions();
        let rgba = img.to_rgba8().into_raw();

        let (det_name, rec_name) = get_model_filenames();
        for (dir, dict_name) in [
            ("../../docs/public/models/indraocr", "meta.json"),
            ("../../docs/public/models/ppocrv6-tiny", "dict.txt"),
        ] {
            let model_dir = Path::new(dir);
            let det_file = model_dir.join(det_name);
            let rec_file = model_dir.join(rec_name);
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
                    Some(r#"{"output":"lines","recognition":{"batchSize":12}}"#.to_string()),
                )
                .expect("detect_text batchSize=12");

            let single_res: Vec<rusto_core::types::TextResult> =
                nanoserde::DeJson::deserialize_json(&single_json).expect("parse single");
            let batched_res: Vec<rusto_core::types::TextResult> =
                nanoserde::DeJson::deserialize_json(&batched_json).expect("parse batched");

            assert_eq!(
                single_res.len(),
                batched_res.len(),
                "line count mismatch for {dir} on {sample_name}"
            );
            for (idx, (s, b)) in single_res.iter().zip(batched_res.iter()).enumerate() {
                assert_eq!(
                    s.text, b.text,
                    "text mismatch at line {idx} for {dir} on {sample_name}"
                );
                assert!(
                    (s.score - b.score).abs() <= 0.002,
                    "score mismatch at line {idx} for {dir} on {sample_name}: {} vs {}",
                    s.score,
                    b.score
                );
            }
        }
    }
}



