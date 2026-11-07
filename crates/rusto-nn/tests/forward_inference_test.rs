use std::path::Path;
use rusto_nn::{Model, Session, Tensor, TensorLayout};

#[test]
fn test_forward_ppocrv6_det_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/det.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load det.mnn");
    let session = Session::new(&model).expect("Create session for det.mnn");

    // Feed dummy 64x64 image [1, 3, 64, 64]
    let input = Tensor::zeros(&[1, 3, 64, 64], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run det.mnn forward pass");

    println!("det.mnn output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 1);
    assert_eq!(output.shape()[2], 64);
    assert_eq!(output.shape()[3], 64);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in det output: {}", val);
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_ppocrv6_det_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/det.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load det.onnx");
    let session = Session::new(&model).expect("Create session for det.onnx");

    let input = Tensor::zeros(&[1, 3, 64, 64], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run det.onnx forward pass");

    println!("det.onnx output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 1);
    assert_eq!(output.shape()[2], 64);
    assert_eq!(output.shape()[3], 64);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in det onnx output: {}", val);
    }
}

#[cfg(feature = "rten")]
#[test]
fn test_forward_ppocrv6_det_rten() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/det.rten");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load det.rten");
    let session = Session::new(&model).expect("Create session for det.rten");

    let input = Tensor::zeros(&[1, 3, 64, 64], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run det.rten forward pass");

    println!("det.rten output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 1);
    assert_eq!(output.shape()[2], 64);
    assert_eq!(output.shape()[3], 64);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in det rten output: {}", val);
    }
}

#[cfg(feature = "rten")]
#[test]
fn test_ppocrv6_det_rten_node_parity() {
    use rten::{ModelOptions, Value};
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor as RtenTensor;
    use std::fs;

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/det.rten");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model_data = fs::read(&model_path).expect("read det.rten");
    let rten_model = ModelOptions::with_all_ops().load(model_data).expect("rten load");
    let nn_model = Model::from_file(&model_path).expect("nn load");

    let h = 672;
    let w = 448;
    let mut input_data = vec![0.0f32; 1 * 3 * h * w];
    for (i, val) in input_data.iter_mut().enumerate() {
        *val = ((i % 100) as f32 / 100.0) * 2.0 - 1.0;
    }

    let nn_input = Tensor::from_floats(&[1, 3, h, w], input_data.clone(), TensorLayout::NCHW);
    let rten_input = RtenTensor::<f32>::from_data(&[1, 3, h, w], input_data);

    let graph = nn_model.graph();
    let tensor_names = &graph.tensor_names;
    let mut ctx = rusto_nn::graph::GraphContext::new(tensor_names.len(), tensor_names);

    // Bind input
    let in_name = &nn_model.input_names()[0];
    let in_idx = graph.tensor_names.iter().position(|n| n == in_name).unwrap();
    ctx.set_tensor(in_idx, nn_input);

    let rten_in_id = rten_model.input_ids()[0];

    for (i, node) in graph.nodes.iter().enumerate() {
        rusto_nn::graph::execute_node_with_liveness(node, i, &mut ctx, None)
            .unwrap_or_else(|e| panic!("Op [{:>3}] {} ({}): {:?}", i, node.op.kind_name(), node.name, e));

        for &out_idx in &node.out_indices {
            let out_name = &graph.tensor_names[out_idx];
            if let Some(rten_nid) = rten_model.find_node(out_name) {
                // Try to evaluate this node in rten
                if let Ok(outputs) = rten_model.run(vec![(rten_in_id, rten_input.view().into())], &[rten_nid], None) {
                    if let Some(Value::FloatTensor(rten_t)) = outputs.into_iter().next() {
                        let nn_t = ctx.get_tensor(out_idx).unwrap();
                        let nn_shape = nn_t.shape();
                        let rten_shape = rten_t.shape().to_vec();
                        let nn_floats = nn_t.to_f32_vec();
                        let rten_floats = rten_t.to_vec();

                        let mut max_diff: f32 = 0.0;
                        for (&a, &b) in nn_floats.iter().zip(rten_floats.iter()) {
                            let diff: f32 = (a - b).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }

                        if nn_shape != rten_shape || max_diff > 0.01 {
                            println!(
                                "DIVERGENCE at op [{:>3}] '{}' (op={}): nn_shape={:?}, rten_shape={:?}, max_diff={:.6}",
                                i, node.name, node.op.kind_name(), nn_shape, rten_shape, max_diff
                            );
                            if let rusto_nn::graph::ir::OpKind::Interp(p) = &node.op {
                                println!("InterpParams: {:?}", p);
                            }
                            println!("in_indices: {:?}", node.in_indices);
                            for &in_idx in &node.in_indices {
                                if let Ok(t) = ctx.get_tensor(in_idx) {
                                    println!("Input tensor {}: shape={:?}, dtype={:?}, floats={:?}", in_idx, t.shape(), t.dtype(), t.as_floats());
                                }
                            }
                            println!("NN first 10 floats: {:?}", &nn_floats[..10.min(nn_floats.len())]);
                            println!("RTEN first 10 floats: {:?}", &rten_floats[..10.min(rten_floats.len())]);
                            assert!(max_diff <= 0.01, "Diverged at node {}", node.name);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "rten")]
#[test]
fn test_forward_ppocrv6_rec_rten() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/rec.rten");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load rec.rten");
    let session = Session::new(&model).expect("Create session for rec.rten");

    let input = Tensor::zeros(&[1, 3, 48, 160], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run rec.rten forward pass");

    println!("rec.rten output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert!(output.shape().len() >= 3);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in rec rten output: {}", val);
    }
}

#[test]
fn test_forward_ppocrv6_rec_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/rec.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load rec.mnn");
    let session = Session::new(&model).expect("Create session for rec.mnn");

    // Feed dummy crop [1, 3, 48, 160]
    let input = Tensor::zeros(&[1, 3, 48, 160], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run rec.mnn forward pass");

    println!("rec.mnn output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert!(output.shape().len() >= 3);

    let floats = output.as_floats().expect("Float output");
    let c = output.shape()[2];
    let sum_t0 = floats[0..c].iter().sum::<f32>();
    assert!((sum_t0 - 1.0).abs() < 1e-4, "Expected softmax sum over classes to be ~1.0, got: {}", sum_t0);
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in rec output: {}", val);
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_ppocrv6_rec_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/rec.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load rec.onnx");
    let session = Session::new(&model).expect("Create session for rec.onnx");

    let input = Tensor::zeros(&[1, 3, 48, 160], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run rec.onnx forward pass");

    println!("rec.onnx output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert!(output.shape().len() >= 3);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in rec onnx output: {}", val);
    }
}

#[test]
fn test_forward_indra_det_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/det.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load indra det.mnn");
    println!("indra det.mnn input_channels: {}", model.input_channels());
    let session = Session::new(&model).expect("Create session for indra det.mnn");

    // Feed dummy 64x64 image [1, 1, 64, 64] (Indra det is 1-channel LineNet)
    let input = Tensor::zeros(&[1, 1, 64, 64], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run indra det.mnn forward pass");

    println!("indra det.mnn output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 3);
    assert_eq!(output.shape()[2], 32);
    assert_eq!(output.shape()[3], 32);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in indra det output: {}", val);
    }
}

#[cfg(feature = "rten")]
#[test]
fn test_forward_indra_det_rten() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/det.rten");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load indra det.rten");
    let session = Session::new(&model).expect("Create session for indra det.rten");

    let input = Tensor::zeros(&[1, 1, 64, 64], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run indra det.rten forward pass");

    println!("indra det.rten output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 3);
    assert_eq!(output.shape()[2], 32);
    assert_eq!(output.shape()[3], 32);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in indra det rten output: {}", val);
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_indra_det_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/det.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load indra det.onnx");
    println!("indra det.onnx outputs: {:?}", model.output_names());
    let session = Session::new(&model).expect("Create session for indra det.onnx");

    let input = Tensor::zeros(&[1, 1, 64, 64], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run indra det.onnx forward pass");

    println!("indra det.onnx output shape: {:?}", output.shape());
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 2);
    assert_eq!(output.shape()[2], 32);
    assert_eq!(output.shape()[3], 32);

    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite(), "Found non-finite value in indra det onnx output: {}", val);
    }
}

#[test]
fn test_forward_indra_rec_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/rec.mnn");
    let onnx_path = repo_root.join("models/indra/rec.onnx");
    assert!(model_path.exists(), "indra rec.mnn exists");
    assert!(onnx_path.exists(), "indra rec.onnx exists");

    let model_onnx = Model::from_file(&onnx_path).expect("Load indra rec.onnx");
    for inp in model_onnx.input_names() {
        println!("ONNX input name: {}", inp);
    }
    println!("--- ONNX outputs mapping ---");
    for (ni, node) in model_onnx.graph().nodes.iter().enumerate() {
        if node.out_indices.contains(&252) {
            println!("ONNX tensor 252 produced by node {}: '{}', op={}, in={:?}", ni, node.name, node.op.kind_name(), node.in_indices);
        }
    }
    for name in &["ids", "conf", "lengths", "topi", "topv", "blank"] {
        if let Some(t_idx) = model_onnx.graph().tensor_names.iter().position(|n| n == *name) {
            println!("ONNX Output tensor '{}': idx={}", name, t_idx);
            for (ni, node) in model_onnx.graph().nodes.iter().enumerate() {
                if node.out_indices.contains(&t_idx) {
                    println!("  produced by node {}: '{}', op={}, in={:?}", ni, node.name, node.op.kind_name(), node.in_indices);
                }
            }
        }
    }

    let model = Model::from_file(&model_path).expect("Load indra rec.mnn");
    println!("--- ONNX lengths subgraph ---");
    let mut curr = vec![4]; // tensor 4 is lengths
    for _ in 0..6 {
        let mut next = Vec::new();
        for &t in &curr {
            for (ni, n) in model_onnx.graph().nodes.iter().enumerate() {
                if n.out_indices.contains(&t) {
                    println!("ONNX node {}: '{}', op={}, in={:?}, out={:?}", ni, n.name, n.op.kind_name(), n.in_indices, n.out_indices);
                    next.extend_from_slice(&n.in_indices);
                }
            }
        }
        curr = next;
    }
    println!("--- MNN lengths subgraph ---");
    let mut curr = vec![111]; // tensor 111 is lengths
    for _ in 0..6 {
        let mut next = Vec::new();
        for &t in &curr {
            for (ni, n) in model.graph().nodes.iter().enumerate() {
                if n.out_indices.contains(&t) {
                    println!("MNN node {}: '{}', op={}, in={:?}, out={:?}", ni, n.name, n.op.kind_name(), n.in_indices, n.out_indices);
                    next.extend_from_slice(&n.in_indices);
                }
            }
        }
        curr = next;
    }
    for name in &["ids", "conf", "lengths", "topi", "topv", "blank"] {
        if let Some(t_idx) = model.graph().tensor_names.iter().position(|n| n == *name) {
            println!("Output tensor '{}': idx={}", name, t_idx);
            for (ni, node) in model.graph().nodes.iter().enumerate() {
                if node.out_indices.contains(&t_idx) {
                    let extra = match &node.op {
                        rusto_nn::graph::ir::OpKind::ArgMax { axis, out_max_val } => format!("axis={}, out_max_val={}", axis, out_max_val),
                        _ => String::new(),
                    };
                    println!("  produced by node {}: '{}', op={}, extra={}, in={:?}", ni, node.name, node.op.kind_name(), extra, node.in_indices);
                }
            }
        } else {
            println!("Output tensor '{}': NOT FOUND by name", name);
        }
    }
    let session = Session::new(&model).expect("Create session for indra rec.mnn");

    // Inputs: x [1, 1, 96, 160] (Indra rec expects height=96 grayscale), widths [1]
    let widths_f = Tensor::from_floats(&[1], vec![160.0], TensorLayout::NCHW);
    let widths_i = Tensor::from_ints(&[1], vec![160]);

    let session_onnx = Session::new(&model_onnx).expect("Create session for indra rec.onnx");
    let mut x_rand = vec![0.0f32; 1 * 1 * 96 * 160];
    for (i, v) in x_rand.iter_mut().enumerate() {
        *v = ((i % 100) as f32) / 50.0 - 1.0;
    }
    let x_t = Tensor::from_floats(&[1, 1, 96, 160], x_rand, TensorLayout::NCHW);
    let out_onnx = session_onnx.run(&[("x", &x_t), ("widths", &widths_f)]).expect("Run onnx");
    println!("ONNX with pseudo-random: ids={:?}, lengths={:?}, conf={:?}", out_onnx["ids"].to_i32_vec(), out_onnx["lengths"].to_i32_vec(), &out_onnx["conf"].to_f32_vec()[..5]);

    let out_mnn = session.run(&[("x", &x_t), ("widths", &widths_i)]).expect("Run mnn");
    println!("MNN with pseudo-random: ids={:?}, lengths={:?}, conf={:?}", out_mnn["ids"].to_i32_vec(), out_mnn["lengths"].to_i32_vec(), &out_mnn["conf"].to_f32_vec()[..5]);

    // Inspect tensors
    for &tid in &[287, 288, 289, 290, 291, 292, 303] {
        if let Ok(t) = session.get_tensor(tid) {
            let f = t.as_f32_cow();
            println!("Tensor {:>3} shape={:?}, floats={:?}", tid, t.shape(), &f[..f.len().min(10)]);
        }
    }

    let outputs = out_mnn;

    assert!(outputs.contains_key("ids"), "Missing 'ids' in output: {:?}", outputs.keys());
    let ids = &outputs["ids"];
    println!("indra rec.mnn 'ids' shape: {:?}", ids.shape());

    if outputs.contains_key("conf") {
        let conf = &outputs["conf"];
        println!("indra rec.mnn 'conf' shape: {:?}", conf.shape());
    }
}

#[cfg(feature = "rten")]
#[test]
fn test_forward_indra_rec_rten() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/rec.rten");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());
    let model = Model::from_file(&model_path).expect("Load indra rec.rten");
    for (i, node) in model.graph().nodes.iter().enumerate() {
        if node.op.kind_name() == "ArgMax" {
            let axis = match &node.op {
                rusto_nn::graph::ir::OpKind::ArgMax { axis, .. } => *axis,
                _ => 0,
            };
            println!("Node {}: ArgMax axis={}", i, axis);
        }
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_indra_rec_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/indra/rec.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load indra rec.onnx");
    let session = Session::new(&model).expect("Create session for indra rec.onnx");

    let x = Tensor::zeros(&[1, 1, 96, 160], TensorLayout::NCHW);
    let widths = Tensor::from_floats(&[1], vec![160.0], TensorLayout::NCHW);

    let inps: Vec<(&str, &Tensor)> = if model.input_names().len() == 1 {
        vec![(model.input_names()[0].as_str(), &x)]
    } else {
        vec![("x", &x), ("widths", &widths)]
    };

    let outputs = session.run(&inps).expect("Run indra rec.onnx forward pass");
    assert!(!outputs.is_empty());
    println!("indra rec.onnx outputs: {:?}", outputs.keys());
}

#[test]
fn test_forward_ppocrv6_cls_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/cls.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load cls.mnn");
    let session = Session::new(&model).expect("Create session for cls.mnn");

    let input = Tensor::zeros(&[1, 3, 48, 192], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run cls.mnn forward pass");

    println!("cls.mnn output shape: {:?}", output.shape());
    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite());
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_ppocrv6_cls_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/cls.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load cls.onnx");
    let session = Session::new(&model).expect("Create session for cls.onnx");

    let input = Tensor::zeros(&[1, 3, 48, 192], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run cls.onnx forward pass");

    println!("cls.onnx output shape: {:?}", output.shape());
    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite());
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_ppocrv6_orient_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/orient.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load orient.onnx");
    let session = Session::new(&model).expect("Create session for orient.onnx");

    let input = Tensor::zeros(&[1, 3, 80, 80], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run orient.onnx forward pass");

    println!("orient.onnx output shape: {:?}", output.shape());
    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite());
    }
}

#[test]
fn test_forward_ppocrv6_orient_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/orient.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load orient.mnn");
    let session = Session::new(&model).expect("Create session for orient.mnn");

    let input = Tensor::zeros(&[1, 3, 80, 80], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run orient.mnn forward pass");

    println!("orient.mnn output shape: {:?}", output.shape());
    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite());
    }
}

#[test]
fn test_forward_uvdoc_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/uvdoc.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load uvdoc.mnn");
    let session = Session::new(&model).expect("Create session for uvdoc.mnn");

    let input = Tensor::zeros(&[1, 3, 512, 512], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run uvdoc.mnn forward pass");

    println!("uvdoc.mnn output shape: {:?}", output.shape());
    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite());
    }
}

#[test]
fn test_forward_layout_cdla_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/layout_cdla.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load layout_cdla.mnn");
    let session = Session::new(&model).expect("Create session for layout_cdla.mnn");

    let in_name = model.input_names()[0].clone();
    let input = Tensor::zeros(&[1, 3, 800, 608], TensorLayout::NCHW);
    let outputs = session.run(&[(&in_name, &input)]).expect("Run layout_cdla.mnn forward pass");

    println!("layout_cdla.mnn outputs count: {}", outputs.len());
    for (name, out) in &outputs {
        println!("  output '{}': shape {:?}", name, out.shape());
    }
    assert!(!outputs.is_empty());
}

#[test]
fn test_forward_layout_publaynet_mnn() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/layout_publaynet.mnn");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load layout_publaynet.mnn");
    let session = Session::new(&model).expect("Create session for layout_publaynet.mnn");

    let in_name = model.input_names()[0].clone();
    let input = Tensor::zeros(&[1, 3, 800, 608], TensorLayout::NCHW);
    let outputs = session.run(&[(&in_name, &input)]).expect("Run layout_publaynet.mnn forward pass");

    println!("layout_publaynet.mnn outputs count: {}", outputs.len());
    for (name, out) in &outputs {
        println!("  output '{}': shape {:?}", name, out.shape());
    }
    assert!(!outputs.is_empty());
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_uvdoc_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/uvdoc.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load uvdoc.onnx");
    let session = Session::new(&model).expect("Create session for uvdoc.onnx");

    let input = Tensor::zeros(&[1, 3, 512, 512], TensorLayout::NCHW);
    let output = session.run_single(&input).expect("Run uvdoc.onnx forward pass");

    assert_eq!(output.shape(), &[1, 3, 512, 512]);
    let floats = output.as_floats().expect("Float output");
    assert!(!floats.is_empty());
    for &val in floats {
        assert!(val.is_finite());
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_layout_cdla_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/layout_cdla.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load layout_cdla.onnx");
    let session = Session::new(&model).expect("Create session for layout_cdla.onnx");

    let in_name = model.input_names()[0].clone();
    let input = Tensor::zeros(&[1, 3, 800, 608], TensorLayout::NCHW);
    let outputs = session.run(&[(&in_name, &input)]).expect("Run layout_cdla.onnx forward pass");

    assert!(!outputs.is_empty());
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_layout_publaynet_onnx() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_path = repo_root.join("models/PPOCR_v6_tiny/layout_publaynet.onnx");
    assert!(model_path.exists(), "Model exists: {}", model_path.display());

    let model = Model::from_file(&model_path).expect("Load layout_publaynet.onnx");
    let session = Session::new(&model).expect("Create session for layout_publaynet.onnx");

    let in_name = model.input_names()[0].clone();
    let input = Tensor::zeros(&[1, 3, 800, 608], TensorLayout::NCHW);
    let outputs = session.run(&[(&in_name, &input)]).expect("Run layout_publaynet.onnx forward pass");

    assert!(!outputs.is_empty());
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_indra_models() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let model_dir = repo_root.join("models/indra");
    if !model_dir.exists() {
        return;
    }

    // 1. orient (.onnx & .rten)
    for ext in &["onnx", "rten"] {
        let p = model_dir.join(format!("orient.{}", ext));
        if p.exists() {
            let model = Model::from_file(&p).expect("Load orient");
            let session = Session::new(&model).expect("Session orient");
            let x = Tensor::zeros(&[1, 1, 96, 96], TensorLayout::NCHW);
            let out = session.run(&[("x", &x)]).expect("Run orient");
            assert_eq!(out["p"].shape(), &[1, 5]);
        }
    }

    // 2. script (.onnx & .rten)
    for ext in &["onnx", "rten"] {
        let p = model_dir.join(format!("script.{}", ext));
        if p.exists() {
            let model = Model::from_file(&p).expect("Load script");
            let session = Session::new(&model).expect("Session script");
            let x = Tensor::zeros(&[1, 1, 96, 160], TensorLayout::NCHW);
            let widths = Tensor::from_floats(&[1], vec![160.0], TensorLayout::NCHW);
            let out = session.run(&[("x", &x), ("widths", &widths)]).expect("Run script");
            assert!(out.contains_key("p"));
        }
    }

    // 3. rec_cjk, rec_rtl, rec_world (.onnx & .rten)
    for model_name in &["rec_cjk", "rec_rtl", "rec_world"] {
        for ext in &["onnx", "rten"] {
            let p = model_dir.join(format!("{}.{}", model_name, ext));
            if p.exists() {
                let model = Model::from_file(&p).expect("Load rec variant");
                let session = Session::new(&model).expect("Session rec variant");
                let x = Tensor::zeros(&[1, 1, 96, 160], TensorLayout::NCHW);
                let widths = Tensor::from_floats(&[1], vec![160.0], TensorLayout::NCHW);
                let out = session.run(&[("x", &x), ("widths", &widths)]).expect("Run rec variant");
                assert!(out.contains_key("ids"));
                assert!(out.contains_key("conf"));
                assert!(out.contains_key("lengths"));
            }
        }
    }

    // 4. mathcls (.onnx & .rten)
    for ext in &["onnx", "rten"] {
        let p = model_dir.join(format!("mathcls.{}", ext));
        if p.exists() {
            let model = Model::from_file(&p).expect("Load mathcls");
            let session = Session::new(&model).expect("Session mathcls");
            let x = Tensor::zeros(&[1, 1, 96, 160], TensorLayout::NCHW);
            let widths = Tensor::from_floats(&[1], vec![160.0], TensorLayout::NCHW);
            let out = session.run(&[("x", &x), ("widths", &widths)]).expect("Run mathcls");
            assert!(out.contains_key("p"));
        }
    }

    // 5. indra_order (.onnx & .rten)
    for ext in &["onnx", "rten"] {
        let p = model_dir.join(format!("indra_order.{}", ext));
        if p.exists() {
            let model = Model::from_file(&p).expect("Load indra_order");
            let session = Session::new(&model).expect("Session indra_order");
            let n_boxes = 4;
            let feats = Tensor::zeros(&[1, n_boxes, 16], TensorLayout::NCHW);
            let mask = Tensor::from_ints(&[1, n_boxes], vec![1; n_boxes]);
            let out = session.run(&[("feats", &feats), ("mask", &mask)]).expect("Run indra_order");
            assert!(out.contains_key("a"));
            assert!(out.contains_key("cls"));
            assert!(out.contains_key("same"));
            assert!(out.contains_key("tab"));
        }
    }

    // 6. math_enc, math_dec_init, math_dec_step (.onnx & .rten)
    for ext in &["onnx", "rten"] {
        let enc_p = model_dir.join(format!("math_enc.{}", ext));
        let init_p = model_dir.join(format!("math_dec_init.{}", ext));
        let step_p = model_dir.join(format!("math_dec_step.{}", ext));

        if enc_p.exists() && init_p.exists() && step_p.exists() {
            // Encoder
            let enc_model = Model::from_file(&enc_p).expect("Load math_enc");
            let enc_session = Session::new(&enc_model).expect("Session math_enc");
            let x = Tensor::zeros(&[1, 1, 64, 128], TensorLayout::NCHW);
            let widths = Tensor::from_floats(&[1], vec![128.0], TensorLayout::NCHW);
            let enc_out = enc_session.run(&[("x", &x), ("widths", &widths)]).expect("Run math_enc");
            assert!(enc_out.contains_key("memory"));
            assert!(enc_out.contains_key("mask"));

            // Decoder Init
            let init_model = Model::from_file(&init_p).expect("Load math_dec_init");
            let init_session = Session::new(&init_model).expect("Session math_dec_init");
            let init_out = init_session.run(&[("memory", &enc_out["memory"])]).expect("Run math_dec_init");
            assert!(init_out.contains_key("cross_k"));
            assert!(init_out.contains_key("cross_v"));

            // Decoder Step
            let step_model = Model::from_file(&step_p).expect("Load math_dec_step");
            let step_session = Session::new(&step_model).expect("Session math_dec_step");
            let token = Tensor::from_ints(&[1], vec![1]);
            let pos = Tensor::from_ints(&[], vec![0]);
            let self_k = Tensor::zeros(&[3, 1, 8, 0, 24], TensorLayout::NCHW);
            let self_v = Tensor::zeros(&[3, 1, 8, 0, 24], TensorLayout::NCHW);
            let step_out = step_session.run(&[
                ("token", &token),
                ("pos", &pos),
                ("self_k", &self_k),
                ("self_v", &self_v),
                ("cross_k", &init_out["cross_k"]),
                ("cross_v", &init_out["cross_v"]),
                ("mask", &enc_out["mask"]),
            ]).expect("Run math_dec_step");
            assert!(step_out.contains_key("logits"));
            assert!(step_out.contains_key("new_k"));
            assert!(step_out.contains_key("new_v"));
            assert_eq!(step_out["logits"].shape(), &[1, 652]);
        }
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_ppocrv4_models() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let v4_models = [
        ("PPOCR_v4_mobile/det.onnx", vec![1, 3, 64, 64]),
        ("PPOCR_v4_mobile/cls.onnx", vec![1, 3, 48, 192]),
        ("PPOCR_v4_mobile/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v4_mobile/rec_en.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v4_japan/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v4_chinese_cht/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v4_kannada/rec.onnx", vec![1, 3, 48, 320]),
    ];

    for (rel_path, in_shape) in &v4_models {
        let p = repo_root.join("models").join(rel_path);
        if !p.exists() {
            continue;
        }
        let model = Model::from_file(&p).unwrap_or_else(|e| panic!("Load {}: {:?}", rel_path, e));
        let session = Session::new(&model).unwrap_or_else(|e| panic!("Session {}: {:?}", rel_path, e));
        let in_name = model.input_names()[0].clone();
        let input = Tensor::zeros(in_shape, TensorLayout::NCHW);
        let out = session.run(&[(&in_name, &input)]).unwrap_or_else(|e| panic!("Run {}: {:?}", rel_path, e));
        assert!(!out.is_empty(), "Model {} returned no outputs", rel_path);
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_forward_ppocrv5_models() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let v5_models = [
        ("PPOCR_v5_mobile/det.onnx", vec![1, 3, 64, 64]),
        ("PPOCR_v5_mobile/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v5_mobile/rec_en.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v5_arabic/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v5_devanagari/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v5_korean/rec.onnx", vec![1, 3, 48, 320]),
        ("PPOCR_v5_latin/rec.onnx", vec![1, 3, 48, 320]),
    ];

    for (rel_path, in_shape) in &v5_models {
        let p = repo_root.join("models").join(rel_path);
        if !p.exists() {
            continue;
        }
        let model = Model::from_file(&p).unwrap_or_else(|e| panic!("Load {}: {:?}", rel_path, e));
        let session = Session::new(&model).unwrap_or_else(|e| panic!("Session {}: {:?}", rel_path, e));
        let in_name = model.input_names()[0].clone();
        let input = Tensor::zeros(in_shape, TensorLayout::NCHW);
        let out = session.run(&[(&in_name, &input)]).unwrap_or_else(|e| panic!("Run {}: {:?}", rel_path, e));
        assert!(!out.is_empty(), "Model {} returned no outputs", rel_path);
    }
}
