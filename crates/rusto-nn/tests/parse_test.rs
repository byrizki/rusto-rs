use std::path::Path;
use rusto_nn::Model;
use rusto_nn::schema::OpType;

#[test]
fn test_parse_ppocrv6_det_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    if !model_path.exists() {
        eprintln!("Skipping test_parse_ppocrv6_det_mnn: {} not found", model_path.display());
        return;
    }

    let model = Model::from_file(&model_path).expect("Failed to parse PP-OCRv6 det.mnn");
    assert_eq!(model.input_names(), &["x"]);
    assert!(model.op_count() > 50);

    let net = model.net();
    let ops = net.oplists().expect("Oplists exists");
    assert_eq!(ops.get(0).type_(), OpType::Input);

    let mut conv_count = 0;
    for i in 0..ops.len() {
        if ops.get(i).type_() == OpType::Convolution {
            conv_count += 1;
        }
    }
    assert!(conv_count > 10);
}

#[test]
fn test_parse_ppocrv6_rec_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    if !model_path.exists() {
        eprintln!("Skipping test_parse_ppocrv6_rec_mnn: {} not found", model_path.display());
        return;
    }

    let model = Model::from_file(&model_path).expect("Failed to parse PP-OCRv6 rec.mnn");
    assert_eq!(model.input_names(), &["x"]);
    assert!(model.op_count() > 50);

    let net = model.net();
    let ops = net.oplists().expect("Oplists exists");
    assert_eq!(ops.get(0).type_(), OpType::Input);

    let mut dw_count = 0;
    for i in 0..ops.len() {
        if ops.get(i).type_() == OpType::ConvolutionDepthwise {
            dw_count += 1;
        }
    }
    assert!(dw_count > 5);
}

#[test]
fn test_parse_indra_det_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/det.mnn");
    if !model_path.exists() {
        eprintln!("Skipping test_parse_indra_det_mnn: {} not found", model_path.display());
        return;
    }

    let model = Model::from_file(&model_path).expect("Failed to parse Indra det.mnn");
    assert_eq!(model.input_names(), &["x"]);
    assert_eq!(model.output_names(), &["out"]);
    assert!(model.op_count() > 30);
}

#[test]
fn test_parse_indra_rec_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/rec.mnn");
    if !model_path.exists() {
        eprintln!("Skipping test_parse_indra_rec_mnn: {} not found", model_path.display());
        return;
    }

    let model = Model::from_file(&model_path).expect("Failed to parse Indra rec.mnn");
    assert_eq!(model.input_names(), &["x", "widths"]);
    assert!(model.output_names().contains(&"ids".to_string()));
    assert!(model.output_names().contains(&"conf".to_string()));
    assert!(model.op_count() > 100);
}

#[test]
fn test_parse_ppocrv6_cls_and_orient_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    
    let cls_path = repo_root.join("models/PPOCR_v6_tiny/cls.mnn");
    if cls_path.exists() {
        let model = Model::from_file(&cls_path).expect("Failed to parse cls.mnn");
        assert_eq!(model.input_names(), &["x"]);
    }

    let orient_path = repo_root.join("models/PPOCR_v6_tiny/orient.mnn");
    if orient_path.exists() {
        let model = Model::from_file(&orient_path).expect("Failed to parse orient.mnn");
        assert_eq!(model.input_names(), &["x"]);
    }
}
