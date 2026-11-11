use std::collections::{HashMap, HashSet};
use smallvec::SmallVec;

use crate::error::MnnError;
use crate::graph::ir::{
    BinaryParams, FusedActivation, Graph, GridSampleParams, InterpParams, LayerNormParams, MatMulParams, Node, OpKind,
    PoolParams, PrepackedConv, PrepackedDeconv, ResizeType, UnaryParams,
};
use crate::kernels::binary::BinaryOpType;
use crate::kernels::pool::PoolType;
use crate::kernels::reduction::ReductionOp;
use crate::kernels::unary::UnaryOpType;
use crate::schema::PadMode;
use crate::tensor::{DataType, Tensor, TensorLayout};

fn extract_tensor_proto(tp: &rten_onnx::onnx::TensorProto) -> Tensor {
    let shape: Vec<usize> = tp.dims.iter().map(|&d| d.max(0) as usize).collect();
    let num_elems: usize = shape.iter().product::<usize>().max(1);

    if !tp.float_data.is_empty() {
        return Tensor::from_floats(&shape, tp.float_data.clone(), TensorLayout::NCHW);
    }
    if !tp.int32_data.is_empty() {
        return Tensor::from_ints(&shape, tp.int32_data.clone());
    }
    if !tp.int64_data.is_empty() {
        let ints: Vec<i32> = tp.int64_data.iter().map(|&x| x as i32).collect();
        return Tensor::from_ints(&shape, ints);
    }

    if let Some(raw) = &tp.raw_data {
        let bytes = raw.borrow();
        let dtype = tp.data_type.map(|d| d.0).unwrap_or(1);
        if dtype == 1 {
            // FLOAT
            let mut floats = Vec::with_capacity(num_elems);
            for chunk in bytes.chunks_exact(4) {
                floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Tensor::from_floats(&shape, floats, TensorLayout::NCHW);
        } else if dtype == 6 {
            // INT32
            let mut ints = Vec::with_capacity(num_elems);
            for chunk in bytes.chunks_exact(4) {
                ints.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Tensor::from_ints(&shape, ints);
        } else if dtype == 7 {
            // INT64 -> convert to INT32
            let mut ints = Vec::with_capacity(num_elems);
            for chunk in bytes.chunks_exact(8) {
                let v = i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                ints.push(v as i32);
            }
            return Tensor::from_ints(&shape, ints);
        } else if dtype == 9 || dtype == 2 {
            // BOOL or UINT8
            let floats: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
            return Tensor::from_floats(&shape, floats, TensorLayout::NCHW);
        } else if dtype == 3 {
            // INT8
            let floats: Vec<f32> = bytes.iter().map(|&b| (b as i8) as f32).collect();
            return Tensor::from_floats(&shape, floats, TensorLayout::NCHW);
        }
    }

    Tensor::zeros(&shape, TensorLayout::NCHW)
}

pub fn load_onnx(slice: &[u8]) -> Result<Graph, MnnError> {
    use rten_onnx::onnx::ModelProto;

    let model = ModelProto::parse_buf(slice)
        .map_err(|e| MnnError::InvalidModel(format!("Failed to parse ONNX ModelProto: {:?}", e)))?;

    let graph_proto = model
        .graph
        .ok_or_else(|| MnnError::InvalidModel("ONNX ModelProto missing graph".into()))?;

    // Collect initializers from graph initializers and Constant nodes
    let mut initializers: HashMap<String, Tensor> = HashMap::with_capacity(graph_proto.initializer.len());
    for init in &graph_proto.initializer {
        if let Some(name) = &init.name {
            initializers.insert(name.clone(), extract_tensor_proto(init));
        }
    }
    for onnx_node in &graph_proto.node {
        if onnx_node.op_type.as_deref() == Some("Constant") {
            if let Some(out_name) = onnx_node.output.first() {
                for attr in &onnx_node.attribute {
                    if attr.name.as_deref() == Some("value") {
                        if let Some(tp) = &attr.t {
                            initializers.insert(out_name.clone(), extract_tensor_proto(tp));
                        }
                    } else if attr.name.as_deref() == Some("value_float") {
                        if let Some(f) = attr.f {
                            initializers.insert(out_name.clone(), Tensor::from_floats(&[1], vec![f], TensorLayout::NCHW));
                        }
                    } else if attr.name.as_deref() == Some("value_int") {
                        if let Some(i) = attr.i {
                            initializers.insert(out_name.clone(), Tensor::from_ints(&[1], vec![i as i32]));
                        }
                    } else if attr.name.as_deref() == Some("value_floats") {
                        if !attr.floats.is_empty() {
                            initializers.insert(out_name.clone(), Tensor::from_floats(&[attr.floats.len()], attr.floats.clone(), TensorLayout::NCHW));
                        }
                    } else if attr.name.as_deref() == Some("value_ints") {
                        if !attr.ints.is_empty() {
                            let ints32: Vec<i32> = attr.ints.iter().map(|&x| x as i32).collect();
                            initializers.insert(out_name.clone(), Tensor::from_ints(&[ints32.len()], ints32));
                        }
                    }
                }
            }
        }
    }

    // Determine model inputs (inputs that are not static initializers)
    let mut input_names: Vec<String> = Vec::new();
    let mut input_channels = 3usize;
    for inp in &graph_proto.input {
        if let Some(name) = &inp.name {
            if !initializers.contains_key(name) {
                input_names.push(name.clone());
                // Try to infer channels from shape
                if let Some(t_proto) = &inp.r#type {
                    if let Some(t_tensor) = &t_proto.tensor_type {
                        if let Some(shape_proto) = &t_tensor.shape {
                            if shape_proto.dim.len() >= 2 {
                                if let Some(dim_val) = shape_proto.dim[1].dim_value {
                                    if dim_val > 0 {
                                        input_channels = dim_val as usize;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut output_names: Vec<String> = Vec::with_capacity(graph_proto.output.len());
    for out in &graph_proto.output {
        if let Some(name) = &out.name {
            output_names.push(name.clone());
        }
    }

    // Allocate tensor IDs: mapping from tensor string name to unique index
    let mut tensor_name_to_id: HashMap<String, usize> = HashMap::new();
    let mut tensor_names: Vec<String> = Vec::new();

    let mut get_or_register_tensor = |name: &str| -> usize {
        if let Some(&id) = tensor_name_to_id.get(name) {
            id
        } else {
            let id = tensor_names.len();
            tensor_names.push(name.to_string());
            tensor_name_to_id.insert(name.to_string(), id);
            id
        }
    };

    // Pre-register inputs and outputs
    for name in &input_names {
        get_or_register_tensor(name);
    }
    for name in &output_names {
        get_or_register_tensor(name);
    }

    // Register initializers as Const nodes
    let mut nodes: Vec<Node> = Vec::new();
    let mut emitted_constants: HashSet<String> = HashSet::new();

    // Map each ONNX node
    for onnx_node in &graph_proto.node {
        let op_type = onnx_node.op_type.as_deref().unwrap_or("");
        if op_type == "Constant" {
            continue;
        }
        let node_name = onnx_node.name.as_deref().unwrap_or(op_type).to_string();

        // Helper to get attributes
        let find_attr = |name: &str| onnx_node.attribute.iter().find(|a| a.name.as_deref() == Some(name));

        // For inputs that are initializers: if this op doesn't absorb them (like Conv absorbs weights),
        // we emit Const nodes for them so they are populated in GraphContext.
        let is_absorbed_op = op_type == "Conv"
            || op_type == "ConvTranspose"
            || op_type == "BatchNormalization"
            || op_type == "Clip";

        let mut in_indices: SmallVec<[usize; 4]> = SmallVec::new();
        for (idx, in_name) in onnx_node.input.iter().enumerate() {
            if in_name.is_empty() {
                continue;
            }
            if is_absorbed_op && idx >= 1 {
                // Weights/bias/BN params are absorbed directly into the op, skip index binding
                continue;
            }

            if let Some(const_tensor) = initializers.get(in_name) {
                if !emitted_constants.contains(in_name) {
                    let const_id = get_or_register_tensor(in_name);
                    nodes.push(Node {
                        name: in_name.clone(),
                        in_indices: SmallVec::new(),
                        out_indices: smallvec::smallvec![const_id],
                        op: OpKind::Const(const_tensor.clone().into_shared()),
                    });
                    emitted_constants.insert(in_name.clone());
                }
            }

            let id = get_or_register_tensor(in_name);
            in_indices.push(id);
        }

        let mut out_indices: SmallVec<[usize; 2]> = SmallVec::new();
        for out_name in &onnx_node.output {
            if !out_name.is_empty() {
                out_indices.push(get_or_register_tensor(out_name));
            }
        }

        let op_kind = match op_type {
            "Conv" => {
                let weight_name = onnx_node.input.get(1).map(|s| s.as_str()).unwrap_or("");
                let bias_name = onnx_node.input.get(2).map(|s| s.as_str());

                let weight_tensor = initializers.get(weight_name);
                let bias_tensor = bias_name.and_then(|b| initializers.get(b));

                let weight_vec = weight_tensor.map(|t| t.to_f32_vec()).unwrap_or_default();
                let bias_vec = bias_tensor.map(|t| t.to_f32_vec());

                let w_shape = weight_tensor.map(|t| t.shape()).unwrap_or(&[]);
                let out_channels = if !w_shape.is_empty() { w_shape[0] } else { 1 };
                let in_channels_per_group = if w_shape.len() > 1 { w_shape[1] } else { 1 };
                let kh = if w_shape.len() > 2 { w_shape[2] } else { 1 };
                let kw = if w_shape.len() > 3 { w_shape[3] } else { 1 };

                let group = find_attr("group").and_then(|a| a.i).unwrap_or(1) as usize;
                let in_channels = in_channels_per_group * group;
                let is_depthwise = group == in_channels && in_channels == out_channels && in_channels > 1;

                let strides = find_attr("strides").map(|a| &a.ints[..]).unwrap_or(&[]);
                let stride_h = strides.first().copied().unwrap_or(1).max(1) as usize;
                let stride_w = strides.get(1).copied().unwrap_or(stride_h as i64).max(1) as usize;

                let dilations = find_attr("dilations").map(|a| &a.ints[..]).unwrap_or(&[]);
                let dilate_h = dilations.first().copied().unwrap_or(1).max(1) as usize;
                let dilate_w = dilations.get(1).copied().unwrap_or(dilate_h as i64).max(1) as usize;

                let pads = find_attr("pads").map(|a| &a.ints[..]).unwrap_or(&[]);
                let (pad_y, pad_x) = if pads.len() >= 4 {
                    (pads[0].max(0) as usize, pads[1].max(0) as usize)
                } else if pads.len() >= 2 {
                    (pads[0].max(0) as usize, pads[1].max(0) as usize)
                } else {
                    (0, 0)
                };

                let auto_pad = find_attr("auto_pad").and_then(|a| a.s.as_deref()).unwrap_or("NOTSET");
                let pad_mode = match auto_pad {
                    "SAME_UPPER" | "SAME_LOWER" => PadMode::SAME,
                    "VALID" => PadMode::VALID,
                    _ => PadMode::CAFFE,
                };

                let pads_vec = if !pads.is_empty() {
                    Some(pads.iter().map(|&x| x as i32).collect())
                } else {
                    None
                };

                OpKind::Conv(PrepackedConv {
                    weight: weight_vec,
                    bias: bias_vec,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h,
                    stride_w,
                    dilate_h,
                    dilate_w,
                    pad_mode,
                    pad_x,
                    pad_y,
                    pads: pads_vec,
                    group,
                    out_channels,
                    activation: FusedActivation::None,
                    is_depthwise,
                })
            }

            "ConvTranspose" => {
                let weight_name = onnx_node.input.get(1).map(|s| s.as_str()).unwrap_or("");
                let bias_name = onnx_node.input.get(2).map(|s| s.as_str());

                let weight_tensor = initializers.get(weight_name);
                let bias_tensor = bias_name.and_then(|b| initializers.get(b));

                let weight_vec = weight_tensor.map(|t| t.to_f32_vec()).unwrap_or_default();
                let bias_vec = bias_tensor.map(|t| t.to_f32_vec());

                let w_shape = weight_tensor.map(|t| t.shape()).unwrap_or(&[]);
                let group = find_attr("group").and_then(|a| a.i).unwrap_or(1) as usize;
                let out_channels = if w_shape.len() > 1 { w_shape[1] * group } else { 1 };
                let kh = if w_shape.len() > 2 { w_shape[2] } else { 1 };
                let kw = if w_shape.len() > 3 { w_shape[3] } else { 1 };

                let strides = find_attr("strides").map(|a| &a.ints[..]).unwrap_or(&[]);
                let stride_h = strides.first().copied().unwrap_or(1).max(1) as usize;
                let stride_w = strides.get(1).copied().unwrap_or(stride_h as i64).max(1) as usize;

                let dilations = find_attr("dilations").map(|a| &a.ints[..]).unwrap_or(&[]);
                let dilate_h = dilations.first().copied().unwrap_or(1).max(1) as usize;
                let dilate_w = dilations.get(1).copied().unwrap_or(dilate_h as i64).max(1) as usize;

                let pads = find_attr("pads").map(|a| &a.ints[..]).unwrap_or(&[]);
                let (pad_y, pad_x) = if pads.len() >= 4 {
                    (pads[0].max(0) as usize, pads[1].max(0) as usize)
                } else if pads.len() >= 2 {
                    (pads[0].max(0) as usize, pads[1].max(0) as usize)
                } else {
                    (0, 0)
                };

                let pads_vec = if !pads.is_empty() {
                    Some(pads.iter().map(|&x| x as i32).collect())
                } else {
                    None
                };

                OpKind::Deconv(PrepackedDeconv {
                    weight: weight_vec,
                    bias: bias_vec,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h,
                    stride_w,
                    dilate_h,
                    dilate_w,
                    pad_x,
                    pad_y,
                    pads: pads_vec,
                    group,
                    out_channels,
                    activation: FusedActivation::None,
                })
            }

            "BatchNormalization" => {
                let epsilon = find_attr("epsilon").and_then(|a| a.f).unwrap_or(1e-5);
                let scale_name = onnx_node.input.get(1).map(|s| s.as_str()).unwrap_or("");
                let b_name = onnx_node.input.get(2).map(|s| s.as_str()).unwrap_or("");
                let mean_name = onnx_node.input.get(3).map(|s| s.as_str()).unwrap_or("");
                let var_name = onnx_node.input.get(4).map(|s| s.as_str()).unwrap_or("");

                let scale_t = initializers.get(scale_name);
                let b_t = initializers.get(b_name);
                let mean_t = initializers.get(mean_name);
                let var_t = initializers.get(var_name);

                if let (Some(scale_t), Some(b_t), Some(mean_t), Some(var_t)) = (scale_t, b_t, mean_t, var_t) {
                    let s_vec = scale_t.to_f32_vec();
                    let b_vec = b_t.to_f32_vec();
                    let m_vec = mean_t.to_f32_vec();
                    let v_vec = var_t.to_f32_vec();

                    let num_channels = s_vec.len();
                    let mut eff_scale = Vec::with_capacity(num_channels);
                    let mut eff_bias = Vec::with_capacity(num_channels);

                    for c in 0..num_channels {
                        let s = s_vec.get(c).copied().unwrap_or(1.0);
                        let b = b_vec.get(c).copied().unwrap_or(0.0);
                        let m = m_vec.get(c).copied().unwrap_or(0.0);
                        let v = v_vec.get(c).copied().unwrap_or(0.0);

                        let inv_std = 1.0 / (v + epsilon).sqrt();
                        let eff_s = s * inv_std;
                        let eff_b = b - m * eff_s;

                        eff_scale.push(eff_s);
                        eff_bias.push(eff_b);
                    }

                    OpKind::Scale {
                        scale: eff_scale,
                        bias: Some(eff_bias),
                    }
                } else {
                    OpKind::NoOp
                }
            }

            "Relu" => OpKind::Relu { slope: 0.0 },

            "PRelu" => OpKind::PRelu { slopes: None },

            "LeakyRelu" => {
                let alpha = find_attr("alpha").and_then(|a| a.f).unwrap_or(0.01);
                OpKind::Relu { slope: alpha }
            }

            "Clip" => {
                let mut min_val = f32::NEG_INFINITY;
                let mut max_val = f32::INFINITY;

                if let Some(min_attr) = find_attr("min").and_then(|a| a.f) {
                    min_val = min_attr;
                } else if let Some(min_name) = onnx_node.input.get(1) {
                    if let Some(t) = initializers.get(min_name) {
                        min_val = t.to_f32_vec().first().copied().unwrap_or(min_val);
                    }
                }

                if let Some(max_attr) = find_attr("max").and_then(|a| a.f) {
                    max_val = max_attr;
                } else if let Some(max_name) = onnx_node.input.get(2) {
                    if let Some(t) = initializers.get(max_name) {
                        max_val = t.to_f32_vec().first().copied().unwrap_or(max_val);
                    }
                }

                if min_val == 0.0 && max_val == 6.0 {
                    OpKind::Relu6 { min_val: 0.0, max_val: 6.0 }
                } else {
                    OpKind::Unary(UnaryParams {
                        op_type: UnaryOpType::Identity,
                        activation: FusedActivation::Clip(min_val, max_val),
                        table: None,
                    })
                }
            }

            "Sigmoid" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sigmoid,
                activation: FusedActivation::None,
                table: None,
            }),

            "Softplus" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Softplus,
                activation: FusedActivation::None,
                table: None,
            }),

            "Tanh" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Tanh,
                activation: FusedActivation::None,
                table: None,
            }),

            "HardSigmoid" => {
                let alpha = find_attr("alpha").and_then(|a| a.f).unwrap_or(0.2);
                let beta = find_attr("beta").and_then(|a| a.f).unwrap_or(0.5);
                OpKind::Unary(UnaryParams {
                    op_type: UnaryOpType::HardSigmoid { alpha, beta },
                    activation: FusedActivation::None,
                    table: None,
                })
            },

            "HardSwish" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::HardSwish,
                activation: FusedActivation::None,
                table: None,
            }),

            "Gelu" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Gelu,
                activation: FusedActivation::None,
                table: None,
            }),

            "Abs" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Abs,
                activation: FusedActivation::None,
                table: None,
            }),

            "Not" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Not,
                activation: FusedActivation::None,
                table: None,
            }),

            "Erf" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Erf,
                activation: FusedActivation::None,
                table: None,
            }),

            "Neg" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Neg,
                activation: FusedActivation::None,
                table: None,
            }),

            "Floor" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Floor,
                activation: FusedActivation::None,
                table: None,
            }),

            "Ceil" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Ceil,
                activation: FusedActivation::None,
                table: None,
            }),

            "Round" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Round,
                activation: FusedActivation::None,
                table: None,
            }),

            "Sign" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sign,
                activation: FusedActivation::None,
                table: None,
            }),

            "Sin" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sin,
                activation: FusedActivation::None,
                table: None,
            }),

            "Cos" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Cos,
                activation: FusedActivation::None,
                table: None,
            }),

            "Exp" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Exp,
                activation: FusedActivation::None,
                table: None,
            }),

            "Log" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Log,
                activation: FusedActivation::None,
                table: None,
            }),

            "Sqrt" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sqrt,
                activation: FusedActivation::None,
                table: None,
            }),

            "Reciprocal" => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Reciprocal,
                activation: FusedActivation::None,
                table: None,
            }),

            "Add" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Add,
                activation: FusedActivation::None,
            }),

            "Sub" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Sub,
                activation: FusedActivation::None,
            }),

            "Mul" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Mul,
                activation: FusedActivation::None,
            }),

            "Div" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Div,
                activation: FusedActivation::None,
            }),

            "Pow" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Pow,
                activation: FusedActivation::None,
            }),

            "Min" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Min,
                activation: FusedActivation::None,
            }),

            "Max" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Max,
                activation: FusedActivation::None,
            }),

            "Equal" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Equal,
                activation: FusedActivation::None,
            }),

            "Greater" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Greater,
                activation: FusedActivation::None,
            }),

            "GreaterOrEqual" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::GreaterEqual,
                activation: FusedActivation::None,
            }),

            "Less" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Less,
                activation: FusedActivation::None,
            }),

            "LessOrEqual" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LessEqual,
                activation: FusedActivation::None,
            }),

            "Mod" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::FloorMod,
                activation: FusedActivation::None,
            }),

            "And" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LogicalAnd,
                activation: FusedActivation::None,
            }),

            "Or" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LogicalOr,
                activation: FusedActivation::None,
            }),

            "Xor" => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LogicalXor,
                activation: FusedActivation::None,
            }),

            "MaxPool" => {
                let k_shape = find_attr("kernel_shape").map(|a| &a.ints[..]).unwrap_or(&[]);
                let kh = k_shape.first().copied().unwrap_or(2) as usize;
                let kw = k_shape.get(1).copied().unwrap_or(kh as i64) as usize;

                let strides = find_attr("strides").map(|a| &a.ints[..]).unwrap_or(&[]);
                let sh = strides.first().copied().unwrap_or(1) as usize;
                let sw = strides.get(1).copied().unwrap_or(sh as i64) as usize;

                let pads = find_attr("pads").map(|a| &a.ints[..]).unwrap_or(&[]);
                let (py, px) = if pads.len() >= 4 {
                    (pads[0] as usize, pads[1] as usize)
                } else if pads.len() >= 2 {
                    (pads[0] as usize, pads[1] as usize)
                } else {
                    (0, 0)
                };

                let pads_vec = if !pads.is_empty() {
                    Some(pads.iter().map(|&x| x as i32).collect())
                } else {
                    None
                };

                let auto_pad = find_attr("auto_pad")
                    .and_then(|a| a.s.as_deref())
                    .unwrap_or("NOTSET");
                let pad_mode = match auto_pad {
                    "SAME_UPPER" | "SAME_LOWER" => PadMode::SAME,
                    "VALID" => PadMode::VALID,
                    _ => PadMode::CAFFE,
                };

                OpKind::Pool(PoolParams {
                    pool_type: PoolType::MaxPool,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h: sh,
                    stride_w: sw,
                    pad_mode,
                    pad_x: px,
                    pad_y: py,
                    pads: pads_vec,
                    is_global: false,
                })
            }

            "AveragePool" => {
                let k_shape = find_attr("kernel_shape").map(|a| &a.ints[..]).unwrap_or(&[]);
                let kh = k_shape.first().copied().unwrap_or(2) as usize;
                let kw = k_shape.get(1).copied().unwrap_or(kh as i64) as usize;

                let strides = find_attr("strides").map(|a| &a.ints[..]).unwrap_or(&[]);
                let sh = strides.first().copied().unwrap_or(1) as usize;
                let sw = strides.get(1).copied().unwrap_or(sh as i64) as usize;

                let pads = find_attr("pads").map(|a| &a.ints[..]).unwrap_or(&[]);
                let (py, px) = if pads.len() >= 4 {
                    (pads[0] as usize, pads[1] as usize)
                } else if pads.len() >= 2 {
                    (pads[0] as usize, pads[1] as usize)
                } else {
                    (0, 0)
                };

                let pads_vec = if !pads.is_empty() {
                    Some(pads.iter().map(|&x| x as i32).collect())
                } else {
                    None
                };

                let auto_pad = find_attr("auto_pad")
                    .and_then(|a| a.s.as_deref())
                    .unwrap_or("NOTSET");
                let pad_mode = match auto_pad {
                    "SAME_UPPER" | "SAME_LOWER" => PadMode::SAME,
                    "VALID" => PadMode::VALID,
                    _ => PadMode::CAFFE,
                };

                OpKind::Pool(PoolParams {
                    pool_type: PoolType::AvgPool,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h: sh,
                    stride_w: sw,
                    pad_mode,
                    pad_x: px,
                    pad_y: py,
                    pads: pads_vec,
                    is_global: false,
                })
            }

            "GlobalAveragePool" => OpKind::Pool(PoolParams {
                pool_type: PoolType::AvgPool,
                kernel_h: 0,
                kernel_w: 0,
                stride_h: 1,
                stride_w: 1,
                pad_mode: PadMode::CAFFE,
                pad_x: 0,
                pad_y: 0,
                pads: None,
                is_global: true,
            }),

            "Reshape" => OpKind::Reshape { dims: None },

            "Transpose" => {
                let perm = find_attr("perm")
                    .map(|a| a.ints.iter().map(|&x| x as usize).collect())
                    .unwrap_or_default();
                OpKind::Permute { dims: perm }
            }

            "Concat" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(1) as i32;
                OpKind::Concat { axis }
            }

            "Split" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(0) as i32;
                let split_points = find_attr("split")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                OpKind::Slice {
                    axis,
                    slice_points: split_points,
                }
            }

            "Slice" => {
                let starts = find_attr("starts").map(|a| a.ints.iter().map(|&x| x as i32).collect());
                let ends = find_attr("ends").map(|a| a.ints.iter().map(|&x| x as i32).collect());
                let axes = find_attr("axes").map(|a| a.ints.iter().map(|&x| x as i32).collect());
                let steps = find_attr("steps").map(|a| a.ints.iter().map(|&x| x as i32).collect());
                OpKind::StridedSlice {
                    begin_mask: 0,
                    end_mask: 0,
                    ellipsis_mask: 0,
                    new_axis_mask: 0,
                    shrink_axis_mask: 0,
                    starts,
                    ends,
                    axes,
                    strides: steps,
                }
            }

            "Squeeze" => {
                let axes = find_attr("axes")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                OpKind::Squeeze { axes }
            }

            "Unsqueeze" => {
                let axes = find_attr("axes")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                OpKind::Unsqueeze { axes }
            }

            "Flatten" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(1) as i32;
                OpKind::Flatten { axis, end_axis: 0 }
            }

            "Softmax" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(-1) as i32;
                OpKind::Softmax { axis }
            }

            "LogSoftmax" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(-1) as i32;
                OpKind::LogSoftmax { axis }
            }

            "ArgMax" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(0) as i32;
                OpKind::ArgMax { axis, out_max_val: false }
            }

            "TopK" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(-1) as i32;
                let largest = find_attr("largest").and_then(|a| a.i).map(|v| v != 0).unwrap_or(true);
                let sorted = find_attr("sorted").and_then(|a| a.i).map(|v| v != 0).unwrap_or(true);
                OpKind::TopKV2 {
                    k: 1,
                    axis,
                    largest,
                    sorted,
                }
            }

            "Pad" => {
                let pad_value = find_attr("value").and_then(|a| a.f).unwrap_or(0.0);
                OpKind::Pad { pad_value, is_interleaved: false }
            }

            "Shape" => OpKind::Shape,

            "Range" => OpKind::Range,

            "Gather" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(0) as i32;
                OpKind::Gather { axis }
            }

            "Cast" => OpKind::Cast {
                dst_type: DataType::Float,
            },

            "Where" => OpKind::Select,

            "MatMul" => OpKind::MatMul(MatMulParams {
                trans_a: false,
                trans_b: false,
                alpha: 1.0,
                beta: 0.0,
            }),

            "Gemm" => {
                let trans_a = find_attr("transA").and_then(|a| a.i).map(|v| v != 0).unwrap_or(false);
                let trans_b = find_attr("transB").and_then(|a| a.i).map(|v| v != 0).unwrap_or(false);
                let alpha = find_attr("alpha").and_then(|a| a.f).unwrap_or(1.0);
                let beta = find_attr("beta").and_then(|a| a.f).unwrap_or(1.0);
                OpKind::MatMul(MatMulParams {
                    trans_a,
                    trans_b,
                    alpha,
                    beta,
                })
            }

            "Resize" => {
                let mode_str = find_attr("mode").and_then(|a| a.s.as_deref()).unwrap_or("nearest");
                let resize_type = match mode_str {
                    "linear" | "bilinear" => ResizeType::Bilinear,
                    "cubic" => ResizeType::Cubic,
                    _ => ResizeType::Nearest,
                };
                let coord_mode = find_attr("coordinate_transformation_mode")
                    .and_then(|a| a.s.as_deref())
                    .unwrap_or("half_pixel");
                let align_corners = coord_mode == "align_corners";
                let half_pixel_centers = coord_mode == "half_pixel";

                OpKind::Interp(InterpParams {
                    resize_type,
                    width_scale: 0.0,
                    height_scale: 0.0,
                    output_w: 0,
                    output_h: 0,
                    align_corners,
                    half_pixel_centers,
                })
            }

            "DepthToSpace" => {
                let block_size = find_attr("blocksize").and_then(|a| a.i).unwrap_or(1) as usize;
                let mode_str = find_attr("mode").and_then(|a| a.s.as_deref()).unwrap_or("DCR");
                let mode = if mode_str == "CRD" { 1 } else { 0 };
                OpKind::DepthToSpace { block_size, mode }
            }

            "ReduceMean" => {
                let axes = find_attr("axes")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = find_attr("keepdims").and_then(|a| a.i).map(|v| v != 0).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Mean,
                    axes,
                    keep_dims,
                }
            }

            "ReduceSum" => {
                let axes = find_attr("axes")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = find_attr("keepdims").and_then(|a| a.i).map(|v| v != 0).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Sum,
                    axes,
                    keep_dims,
                }
            }

            "ReduceMax" => {
                let axes = find_attr("axes")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = find_attr("keepdims").and_then(|a| a.i).map(|v| v != 0).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Max,
                    axes,
                    keep_dims,
                }
            }

            "ReduceMin" => {
                let axes = find_attr("axes")
                    .map(|a| a.ints.iter().map(|&x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = find_attr("keepdims").and_then(|a| a.i).map(|v| v != 0).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Min,
                    axes,
                    keep_dims,
                }
            }

            "GridSample" => {
                let mode_str = find_attr("mode")
                    .and_then(|a| a.s.as_deref())
                    .unwrap_or("bilinear");
                let mode = match mode_str {
                    "bilinear" => 0,
                    "nearest" => 1,
                    _ => 0,
                };
                let pad_str = find_attr("padding_mode")
                    .and_then(|a| a.s.as_deref())
                    .unwrap_or("zeros");
                let padding_mode = match pad_str {
                    "zeros" => 0,
                    "border" | "clamp" => 1,
                    "reflection" => 2,
                    _ => 0,
                };
                let align_corners = find_attr("align_corners")
                    .and_then(|a| a.i)
                    .map(|v| v != 0)
                    .unwrap_or(false);
                OpKind::GridSample(GridSampleParams {
                    mode,
                    padding_mode,
                    align_corners,
                })
            }

            "ConstantOfShape" => {
                let val = if let Some(tp) = find_attr("value").and_then(|a| a.t.as_ref()) {
                    let val_t = extract_tensor_proto(tp);
                    val_t.to_f32_vec().first().copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                OpKind::ConstantOfShape { value: val }
            }

            "Expand" => OpKind::Expand,
            "ScatterND" => OpKind::ScatterND,

            "LayerNormalization" => {
                let axis = find_attr("axis").and_then(|a| a.i).unwrap_or(-1) as i32;
                let epsilon = find_attr("epsilon").and_then(|a| a.f).unwrap_or(1e-5);
                OpKind::LayerNorm(LayerNormParams {
                    axis,
                    epsilon,
                    use_rms_norm: false,
                    gamma: None,
                    beta: None,
                })
            }

            "Identity" => OpKind::ConvertTensor,

            _ => OpKind::NoOp,
        };

        nodes.push(Node {
            name: node_name,
            in_indices,
            out_indices,
            op: op_kind,
        });
    }

    Ok(Graph::new(
        nodes,
        tensor_names,
        input_names,
        output_names,
        input_channels,
    ))
}
