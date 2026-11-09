use std::collections::{HashMap, HashSet};
use smallvec::SmallVec;

use rten_model_file::header::{Header, HeaderError};
use rten_model_file::schema as sg;
use rten_model_file::schema::root_as_model;

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

fn extract_rten_constant(
    c: &sg::ConstantNode,
    slice: &[u8],
    tensor_data_offset: Option<u64>,
) -> Result<Tensor, MnnError> {
    let shape: Vec<usize> = c.shape().iter().map(|x| x as usize).collect();
    let num_elems: usize = shape.iter().product::<usize>().max(1);

    if let Some(data_offset) = c.data_offset() {
        let base = tensor_data_offset.ok_or_else(|| {
            MnnError::InvalidModel("External constant data offset without base offset".into())
        })?;
        let abs_offset = (base + data_offset) as usize;

        match c.dtype() {
            Some(sg::ConstantDataType::Float32) => {
                let byte_len = num_elems * 4;
                if abs_offset + byte_len > slice.len() {
                    return Err(MnnError::InvalidModel("Constant float buffer out of bounds".into()));
                }
                let bytes = &slice[abs_offset..abs_offset + byte_len];
                let mut floats = Vec::with_capacity(num_elems);
                for chunk in bytes.chunks_exact(4) {
                    floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(Tensor::from_floats(&shape, floats, TensorLayout::NCHW))
            }
            Some(sg::ConstantDataType::Int32) => {
                let byte_len = num_elems * 4;
                if abs_offset + byte_len > slice.len() {
                    return Err(MnnError::InvalidModel("Constant int32 buffer out of bounds".into()));
                }
                let bytes = &slice[abs_offset..abs_offset + byte_len];
                let mut ints = Vec::with_capacity(num_elems);
                for chunk in bytes.chunks_exact(4) {
                    ints.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(Tensor::from_ints(&shape, ints))
            }
            Some(sg::ConstantDataType::Int8) => {
                let byte_len = num_elems;
                if abs_offset + byte_len > slice.len() {
                    return Err(MnnError::InvalidModel("Constant int8 buffer out of bounds".into()));
                }
                let bytes = &slice[abs_offset..abs_offset + byte_len];
                let ints: Vec<i32> = bytes.iter().map(|&b| (b as i8) as i32).collect();
                Ok(Tensor::from_ints(&shape, ints))
            }
            Some(sg::ConstantDataType::UInt8) => {
                let byte_len = num_elems;
                if abs_offset + byte_len > slice.len() {
                    return Err(MnnError::InvalidModel("Constant uint8 buffer out of bounds".into()));
                }
                let bytes = &slice[abs_offset..abs_offset + byte_len];
                let ints: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();
                Ok(Tensor::from_ints(&shape, ints))
            }
            _ => {
                let byte_len = num_elems * 4;
                if abs_offset + byte_len > slice.len() {
                    return Err(MnnError::InvalidModel("Constant buffer out of bounds".into()));
                }
                let bytes = &slice[abs_offset..abs_offset + byte_len];
                let mut floats = Vec::with_capacity(num_elems);
                for chunk in bytes.chunks_exact(4) {
                    floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(Tensor::from_floats(&shape, floats, TensorLayout::NCHW))
            }
        }
    } else {
        // Inline constant data
        if let Some(f_data) = c.data_as_float_data() {
            let floats: Vec<f32> = f_data.data().iter().collect();
            Ok(Tensor::from_floats(&shape, floats, TensorLayout::NCHW))
        } else if let Some(i_data) = c.data_as_int_32_data() {
            let ints: Vec<i32> = i_data.data().iter().collect();
            Ok(Tensor::from_ints(&shape, ints))
        } else if let Some(i8_data) = c.data_as_int_8_data() {
            let ints: Vec<i32> = i8_data.data().iter().map(|b| b as i32).collect();
            Ok(Tensor::from_ints(&shape, ints))
        } else if let Some(u8_data) = c.data_as_uint_8_data() {
            let ints: Vec<i32> = u8_data.data().iter().map(|b| b as i32).collect();
            Ok(Tensor::from_ints(&shape, ints))
        } else {
            Ok(Tensor::zeros(&shape, TensorLayout::NCHW))
        }
    }
}

fn get_dim_pair<V: IntoIterator<Item = u32>>(vec: Option<V>, default: usize) -> (usize, usize) {
    if let Some(v) = vec {
        let mut it = v.into_iter();
        let first = it.next().map(|x| (x.max(1)) as usize).unwrap_or(default);
        let second = it.next().map(|x| (x.max(1)) as usize).unwrap_or(first);
        (first, second)
    } else {
        (default, default)
    }
}

fn get_pad_pair<V: IntoIterator<Item = u32>>(pads: Option<V>) -> (usize, usize) {
    if let Some(p) = pads {
        let vals: Vec<usize> = p.into_iter().map(|x| x as usize).collect();
        if vals.len() >= 4 {
            (vals[0], vals[1])
        } else if vals.len() >= 2 {
            (vals[0], vals[1])
        } else if vals.len() == 1 {
            (vals[0], vals[0])
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    }
}

pub fn load_rten(slice: &[u8]) -> Result<Graph, MnnError> {
    let header = match Header::from_buf(slice) {
        Ok(h) => Some(h),
        Err(HeaderError::InvalidMagic) => None,
        Err(e) => return Err(MnnError::InvalidModel(format!("Invalid RTen header: {:?}", e))),
    };

    let (model_data, tensor_data_offset) = if let Some(ref h) = header {
        let start = h.model_offset as usize;
        let end = start + h.model_len as usize;
        if end > slice.len() {
            return Err(MnnError::InvalidModel("RTen model offset exceeds buffer".into()));
        }
        (&slice[start..end], Some(h.tensor_data_offset))
    } else {
        (slice, None)
    };

    let model = root_as_model(model_data)
        .map_err(|e| MnnError::InvalidModel(format!("Failed to parse RTen model: {:?}", e)))?;

    let serialized_graph = model.graph();
    let rten_nodes = serialized_graph.nodes().ok_or_else(|| {
        MnnError::InvalidModel("RTen graph missing nodes".into())
    })?;
    let node_count = rten_nodes.len();

    // Collect tensor names for all node indices
    let mut tensor_names: Vec<String> = Vec::with_capacity(node_count);
    for (i, node) in rten_nodes.iter().enumerate() {
        let name = node.name().map(|s| s.to_string()).unwrap_or_else(|| format!("node_{}", i));
        tensor_names.push(name);
    }

    // Extract all ConstantNodes
    let mut constants: HashMap<usize, Tensor> = HashMap::new();
    for (i, node) in rten_nodes.iter().enumerate() {
        if let Some(c) = node.data_as_constant_node() {
            let tensor = extract_rten_constant(&c, slice, tensor_data_offset)?;
            constants.insert(i, tensor);
        }
    }

    // Parse graph inputs and outputs
    let input_node_ids: Vec<usize> = serialized_graph
        .inputs()
        .map(|v| v.iter().map(|x| x as usize).collect())
        .unwrap_or_default();

    let output_node_ids: Vec<usize> = serialized_graph
        .outputs()
        .map(|v| v.iter().map(|x| x as usize).collect())
        .unwrap_or_default();

    let input_names: Vec<String> = input_node_ids.iter().map(|&id| tensor_names[id].clone()).collect();
    let output_names: Vec<String> = output_node_ids.iter().map(|&id| tensor_names[id].clone()).collect();

    // Determine input channels
    let mut input_channels = 3usize;
    for &inp_id in &input_node_ids {
        if inp_id < rten_nodes.len() {
            if let Some(val_node) = rten_nodes.get(inp_id).data_as_value_node() {
                if let Some(shape) = val_node.shape() {
                    if shape.len() >= 2 {
                        let c_val = shape.get(1).value();
                        if c_val > 0 {
                            input_channels = c_val as usize;
                            break;
                        }
                    }
                }
            }
        }
    }

    let mut nodes: Vec<Node> = Vec::new();
    let mut absorbed_constants: HashSet<usize> = HashSet::new();

    // First pass: identify absorbed constants (weights/biases for Conv, ConvTranspose, BatchNorm)
    for node in rten_nodes.iter() {
        if let Some(op) = node.data_as_operator_node() {
            let op_inputs: Vec<i32> = op.inputs().map(|v| v.iter().collect()).unwrap_or_default();
            match op.type_() {
                sg::OperatorType::Conv | sg::OperatorType::ConvTranspose => {
                    if let Some(&w_idx) = op_inputs.get(1) {
                        if w_idx >= 0 && constants.contains_key(&(w_idx as usize)) {
                            absorbed_constants.insert(w_idx as usize);
                        }
                    }
                    if let Some(&b_idx) = op_inputs.get(2) {
                        if b_idx >= 0 && constants.contains_key(&(b_idx as usize)) {
                            absorbed_constants.insert(b_idx as usize);
                        }
                    }
                }
                sg::OperatorType::BatchNormalization => {
                    for &idx in &op_inputs[1..] {
                        if idx >= 0 && constants.contains_key(&(idx as usize)) {
                            absorbed_constants.insert(idx as usize);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Emit unabsorbed constants
    for (i, node) in rten_nodes.iter().enumerate() {
        if node.data_type() == sg::NodeKind::ConstantNode && !absorbed_constants.contains(&i) {
            if let Some(const_t) = constants.get(&i) {
                nodes.push(Node {
                    name: tensor_names[i].clone(),
                    in_indices: SmallVec::new(),
                    out_indices: smallvec::smallvec![i],
                    op: OpKind::Const(const_t.clone().into_shared()),
                });
            }
        }
    }

    // Second pass: map operator nodes
    for node in rten_nodes.iter() {
        let op = match node.data_as_operator_node() {
            Some(o) => o,
            None => continue,
        };

        let node_name = node.name().map(|s| s.to_string()).unwrap_or_else(|| format!("{:?}", op.type_()));
        let op_inputs: Vec<i32> = op.inputs().map(|v| v.iter().collect()).unwrap_or_default();
        let op_outputs: Vec<i32> = op.outputs().map(|v| v.iter().collect()).unwrap_or_default();

        let is_absorbed_op = op.type_() == sg::OperatorType::Conv
            || op.type_() == sg::OperatorType::ConvTranspose
            || op.type_() == sg::OperatorType::BatchNormalization;

        let mut in_indices: SmallVec<[usize; 4]> = SmallVec::new();
        for (idx, &in_idx) in op_inputs.iter().enumerate() {
            if in_idx < 0 {
                continue;
            }
            if is_absorbed_op && idx >= 1 {
                continue;
            }
            in_indices.push(in_idx as usize);
        }

        let mut out_indices: SmallVec<[usize; 2]> = SmallVec::new();
        for &out_idx in &op_outputs {
            if out_idx >= 0 {
                out_indices.push(out_idx as usize);
            }
        }

        let op_kind = match op.type_() {
            sg::OperatorType::Conv => {
                let attrs = op.attrs_as_conv_attrs();
                let w_idx = op_inputs.get(1).copied().unwrap_or(-1);
                let b_idx = op_inputs.get(2).copied().unwrap_or(-1);

                let weight_tensor = if w_idx >= 0 { constants.get(&(w_idx as usize)) } else { None };
                let bias_tensor = if b_idx >= 0 { constants.get(&(b_idx as usize)) } else { None };

                let weight_vec = weight_tensor.map(|t| t.to_f32_vec()).unwrap_or_default();
                let bias_vec = bias_tensor.map(|t| t.to_f32_vec());

                let w_shape = weight_tensor.map(|t| t.shape()).unwrap_or(&[]);
                let out_channels = if !w_shape.is_empty() { w_shape[0] } else { 1 };
                let in_channels_per_group = if w_shape.len() > 1 { w_shape[1] } else { 1 };
                let kh = if w_shape.len() > 2 { w_shape[2] } else { 1 };
                let kw = if w_shape.len() > 3 { w_shape[3] } else { 1 };

                let group = attrs.as_ref().map(|a| a.groups() as usize).unwrap_or(1).max(1);
                let in_channels = in_channels_per_group * group;
                let is_depthwise = group == in_channels && in_channels == out_channels && in_channels > 1;

                let strides = attrs.as_ref().and_then(|a| a.strides());
                let (stride_h, stride_w) = get_dim_pair(strides, 1);

                let dilations = attrs.as_ref().and_then(|a| a.dilations());
                let (dilate_h, dilate_w) = get_dim_pair(dilations, 1);

                let pads = attrs.as_ref().and_then(|a| a.pads());
                let (pad_y, pad_x) = get_pad_pair(pads);

                let auto_pad = attrs.as_ref().map(|a| a.auto_pad()).unwrap_or(sg::AutoPad::NotSet);
                let pad_mode = match auto_pad {
                    sg::AutoPad::Same => PadMode::SAME,
                    _ => PadMode::CAFFE,
                };
                let pads_vec = pads.map(|p| p.iter().map(|x| x as i32).collect());

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

            sg::OperatorType::ConvTranspose => {
                let attrs = op.attrs_as_conv_transpose_attrs();
                let w_idx = op_inputs.get(1).copied().unwrap_or(-1);
                let b_idx = op_inputs.get(2).copied().unwrap_or(-1);

                let weight_tensor = if w_idx >= 0 { constants.get(&(w_idx as usize)) } else { None };
                let bias_tensor = if b_idx >= 0 { constants.get(&(b_idx as usize)) } else { None };

                let weight_vec = weight_tensor.map(|t| t.to_f32_vec()).unwrap_or_default();
                let bias_vec = bias_tensor.map(|t| t.to_f32_vec());

                let w_shape = weight_tensor.map(|t| t.shape()).unwrap_or(&[]);
                let _in_channels = if !w_shape.is_empty() { w_shape[0] } else { 1 };
                let out_channels_per_group = if w_shape.len() > 1 { w_shape[1] } else { 1 };
                let kh = if w_shape.len() > 2 { w_shape[2] } else { 1 };
                let kw = if w_shape.len() > 3 { w_shape[3] } else { 1 };

                let group = attrs.as_ref().map(|a| a.groups() as usize).unwrap_or(1).max(1);
                let out_channels = out_channels_per_group * group;

                let strides = attrs.as_ref().and_then(|a| a.strides());
                let (stride_h, stride_w) = get_dim_pair(strides, 1);

                let dilations = attrs.as_ref().and_then(|a| a.dilations());
                let (dilate_h, dilate_w) = get_dim_pair(dilations, 1);

                let pads = attrs.as_ref().and_then(|a| a.pads());
                let (pad_y, pad_x) = get_pad_pair(pads);

                let pads_vec = pads.map(|p| p.iter().map(|x| x as i32).collect());

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

            sg::OperatorType::AveragePool => {
                let attrs = op.attrs_as_average_pool_attrs();
                let (kh, kw) = get_dim_pair(attrs.as_ref().map(|a| a.kernel_size()), 1);

                let strides = attrs.as_ref().and_then(|a| a.strides());
                let (stride_h, stride_w) = get_dim_pair(strides, 1);

                let pads = attrs.as_ref().and_then(|a| a.pads());
                let (pad_y, pad_x) = get_pad_pair(pads);

                let pads_vec = pads.map(|p| p.iter().map(|x| x as i32).collect());
                let pad_mode = match attrs.as_ref().map(|a| a.auto_pad()) {
                    Some(sg::AutoPad::Same) => PadMode::SAME,
                    _ => PadMode::CAFFE,
                };

                OpKind::Pool(PoolParams {
                    pool_type: PoolType::AvgPool,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h,
                    stride_w,
                    pad_mode,
                    pad_x,
                    pad_y,
                    pads: pads_vec,
                    is_global: false,
                })
            }

            sg::OperatorType::MaxPool => {
                let attrs = op.attrs_as_max_pool_attrs();
                let (kh, kw) = get_dim_pair(attrs.as_ref().map(|a| a.kernel_size()), 1);

                let strides = attrs.as_ref().and_then(|a| a.strides());
                let (stride_h, stride_w) = get_dim_pair(strides, 1);

                let pads = attrs.as_ref().and_then(|a| a.pads());
                let (pad_y, pad_x) = get_pad_pair(pads);

                let pads_vec = pads.map(|p| p.iter().map(|x| x as i32).collect());
                let pad_mode = match attrs.as_ref().map(|a| a.auto_pad()) {
                    Some(sg::AutoPad::Same) => PadMode::SAME,
                    _ => PadMode::CAFFE,
                };

                OpKind::Pool(PoolParams {
                    pool_type: PoolType::MaxPool,
                    kernel_h: kh,
                    kernel_w: kw,
                    stride_h,
                    stride_w,
                    pad_mode,
                    pad_x,
                    pad_y,
                    pads: pads_vec,
                    is_global: false,
                })
            }

            sg::OperatorType::GlobalAveragePool => OpKind::Pool(PoolParams {
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

            sg::OperatorType::GlobalMaxPool => OpKind::Pool(PoolParams {
                pool_type: PoolType::MaxPool,
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

            sg::OperatorType::BatchNormalization => {
                let attrs = op.attrs_as_batch_normalization_attrs();
                let epsilon = attrs.as_ref().map(|a| a.epsilon()).unwrap_or(1e-5);

                let s_idx = op_inputs.get(1).copied().unwrap_or(-1);
                let b_idx = op_inputs.get(2).copied().unwrap_or(-1);
                let m_idx = op_inputs.get(3).copied().unwrap_or(-1);
                let v_idx = op_inputs.get(4).copied().unwrap_or(-1);

                let scale_t = if s_idx >= 0 { constants.get(&(s_idx as usize)) } else { None };
                let b_t = if b_idx >= 0 { constants.get(&(b_idx as usize)) } else { None };
                let mean_t = if m_idx >= 0 { constants.get(&(m_idx as usize)) } else { None };
                let var_t = if v_idx >= 0 { constants.get(&(v_idx as usize)) } else { None };

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

            sg::OperatorType::Relu => OpKind::Relu { slope: 0.0 },

            sg::OperatorType::PRelu => OpKind::PRelu { slopes: None },

            sg::OperatorType::LeakyRelu => {
                let attrs = op.attrs_as_leaky_relu_attrs();
                let alpha = attrs.as_ref().map(|a| a.alpha()).unwrap_or(0.01);
                OpKind::Relu { slope: alpha }
            }

            sg::OperatorType::Clip => {
                let mut min_val = f32::NEG_INFINITY;
                let mut max_val = f32::INFINITY;

                if let Some(&min_id) = op_inputs.get(1) {
                    if min_id >= 0 {
                        if let Some(t) = constants.get(&(min_id as usize)) {
                            min_val = t.to_f32_vec().first().copied().unwrap_or(min_val);
                        }
                    }
                }
                if let Some(&max_id) = op_inputs.get(2) {
                    if max_id >= 0 {
                        if let Some(t) = constants.get(&(max_id as usize)) {
                            max_val = t.to_f32_vec().first().copied().unwrap_or(max_val);
                        }
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

            sg::OperatorType::Sigmoid => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sigmoid,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Softplus => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Softplus,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Tanh => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Tanh,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::HardSigmoid => {
                let attrs = op.attrs_as_hard_sigmoid_attrs();
                let alpha = attrs.map(|a| a.alpha()).unwrap_or(0.2);
                let beta = attrs.map(|a| a.beta()).unwrap_or(0.5);
                OpKind::Unary(UnaryParams {
                    op_type: UnaryOpType::HardSigmoid { alpha, beta },
                    activation: FusedActivation::None,
                    table: None,
                })
            },

            sg::OperatorType::HardSwish => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::HardSwish,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Gelu => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Gelu,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Abs => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Abs,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Not => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Not,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Erf => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Erf,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Neg => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Neg,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Floor => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Floor,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Ceil => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Ceil,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Round => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Round,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Sign => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sign,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Sin => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sin,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Cos => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Cos,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Tan => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Tan,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Exp => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Exp,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Log => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Log,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Sqrt => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Sqrt,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Reciprocal => OpKind::Unary(UnaryParams {
                op_type: UnaryOpType::Reciprocal,
                activation: FusedActivation::None,
                table: None,
            }),

            sg::OperatorType::Add => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Add,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Sub => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Sub,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Mul => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Mul,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Div => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Div,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Pow => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Pow,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Min => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Min,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Max => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Max,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Equal => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Equal,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Greater => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Greater,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::GreaterOrEqual => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::GreaterEqual,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Less => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::Less,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::LessOrEqual => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LessEqual,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Mod => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::FloorMod,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::And => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LogicalAnd,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Or => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LogicalOr,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Xor => OpKind::Binary(BinaryParams {
                op_type: BinaryOpType::LogicalXor,
                activation: FusedActivation::None,
            }),

            sg::OperatorType::Concat => {
                let attrs = op.attrs_as_concat_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(0);
                OpKind::Concat { axis }
            }

            sg::OperatorType::Split => {
                let attrs = op.attrs_as_split_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(0);
                OpKind::Slice {
                    axis,
                    slice_points: Vec::new(),
                }
            }

            sg::OperatorType::Reshape => OpKind::Reshape { dims: None },

            sg::OperatorType::Transpose => {
                let attrs = op.attrs_as_transpose_attrs();
                let perm = attrs
                    .as_ref()
                    .and_then(|a| a.perm())
                    .map(|p| p.iter().map(|x| x as usize).collect())
                    .unwrap_or_default();
                OpKind::Permute { dims: perm }
            }

            sg::OperatorType::Slice => OpKind::StridedSlice {
                begin_mask: 0,
                end_mask: 0,
                ellipsis_mask: 0,
                new_axis_mask: 0,
                shrink_axis_mask: 0,
                starts: None,
                ends: None,
                axes: None,
                strides: None,
            },

            sg::OperatorType::Squeeze => OpKind::Squeeze { axes: Vec::new() },

            sg::OperatorType::Unsqueeze => OpKind::Unsqueeze { axes: Vec::new() },

            sg::OperatorType::Flatten => {
                let attrs = op.attrs_as_flatten_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(1);
                OpKind::Flatten { axis, end_axis: 0 }
            }

            sg::OperatorType::Softmax => {
                let attrs = op.attrs_as_softmax_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(-1);
                OpKind::Softmax { axis }
            }

            sg::OperatorType::LogSoftmax => {
                let attrs = op.attrs_as_softmax_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(-1);
                OpKind::LogSoftmax { axis }
            }

            sg::OperatorType::ArgMax => {
                let attrs = op.attrs_as_arg_max_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(0);
                OpKind::ArgMax { axis, out_max_val: false }
            }

            sg::OperatorType::TopK => {
                let attrs = op.attrs_as_top_kattrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(-1);
                let largest = attrs.as_ref().map(|a| a.largest()).unwrap_or(true);
                let sorted = attrs.as_ref().map(|a| a.sorted()).unwrap_or(true);
                OpKind::TopKV2 {
                    k: 1,
                    axis,
                    largest,
                    sorted,
                }
            }

            sg::OperatorType::Pad => {
                let attrs = op.attrs_as_pad_attrs();
                let _mode = attrs.as_ref().map(|a| a.mode()).unwrap_or(sg::PadMode::Constant);
                OpKind::Pad { pad_value: 0.0, is_interleaved: false }
            }

            sg::OperatorType::Shape => OpKind::Shape,

            sg::OperatorType::Range => OpKind::Range,

            sg::OperatorType::Gather => {
                let attrs = op.attrs_as_gather_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(0);
                OpKind::Gather { axis }
            }

            sg::OperatorType::Cast => OpKind::Cast { dst_type: DataType::Float },

            sg::OperatorType::Where => OpKind::Select,

            sg::OperatorType::MatMul => OpKind::MatMul(MatMulParams {
                trans_a: false,
                trans_b: false,
                alpha: 1.0,
                beta: 0.0,
            }),

            sg::OperatorType::Gemm => {
                let attrs = op.attrs_as_gemm_attrs();
                let alpha = attrs.as_ref().map(|a| a.alpha()).unwrap_or(1.0);
                let beta = attrs.as_ref().map(|a| a.beta()).unwrap_or(1.0);
                let trans_a = attrs.as_ref().map(|a| a.transpose_a()).unwrap_or(false);
                let trans_b = attrs.as_ref().map(|a| a.transpose_b()).unwrap_or(false);
                OpKind::MatMul(MatMulParams {
                    trans_a,
                    trans_b,
                    alpha,
                    beta,
                })
            }

            sg::OperatorType::Resize => {
                let attrs = op.attrs_as_resize_attrs();
                let mode = attrs.as_ref().map(|a| a.mode()).unwrap_or(sg::ResizeMode::Nearest);
                let resize_type = match mode {
                    sg::ResizeMode::Linear => ResizeType::Bilinear,
                    _ => ResizeType::Nearest,
                };
                let coord_mode = attrs.as_ref().map(|a| a.coord_mode()).unwrap_or(sg::CoordTransformMode::HalfPixel);
                let align_corners = coord_mode == sg::CoordTransformMode::AlignCorners;
                let half_pixel_centers = coord_mode == sg::CoordTransformMode::HalfPixel;

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

            sg::OperatorType::Upsample => {
                let attrs = op.attrs_as_upsample_attrs();
                let mode = attrs.as_ref().map(|a| a.mode()).unwrap_or(sg::ResizeMode::Nearest);
                let resize_type = match mode {
                    sg::ResizeMode::Linear => ResizeType::Bilinear,
                    _ => ResizeType::Nearest,
                };
                OpKind::Interp(InterpParams {
                    resize_type,
                    width_scale: 0.0,
                    height_scale: 0.0,
                    output_w: 0,
                    output_h: 0,
                    align_corners: false,
                    half_pixel_centers: true,
                })
            }

            sg::OperatorType::DepthToSpace => {
                let attrs = op.attrs_as_depth_to_space_attrs();
                let block_size = attrs.as_ref().map(|a| a.block_size() as usize).unwrap_or(1);
                let mode = match attrs.as_ref().map(|a| a.mode()).unwrap_or(sg::DepthToSpaceMode::DCR) {
                    sg::DepthToSpaceMode::CRD => 1,
                    _ => 0,
                };
                OpKind::DepthToSpace { block_size, mode }
            }

            sg::OperatorType::ReduceMean => {
                let attrs = op.attrs_as_reduce_mean_attrs();
                let axes = attrs
                    .as_ref()
                    .and_then(|a| a.axes())
                    .map(|ax| ax.iter().map(|x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = attrs.as_ref().map(|a| a.keep_dims()).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Mean,
                    axes,
                    keep_dims,
                }
            }

            sg::OperatorType::ReduceSum => {
                let attrs = op.attrs_as_reduce_mean_attrs();
                let axes = attrs
                    .as_ref()
                    .and_then(|a| a.axes())
                    .map(|ax| ax.iter().map(|x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = attrs.as_ref().map(|a| a.keep_dims()).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Sum,
                    axes,
                    keep_dims,
                }
            }

            sg::OperatorType::ReduceMax => {
                let attrs = op.attrs_as_reduce_mean_attrs();
                let axes = attrs
                    .as_ref()
                    .and_then(|a| a.axes())
                    .map(|ax| ax.iter().map(|x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = attrs.as_ref().map(|a| a.keep_dims()).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Max,
                    axes,
                    keep_dims,
                }
            }

            sg::OperatorType::ReduceMin => {
                let attrs = op.attrs_as_reduce_mean_attrs();
                let axes = attrs
                    .as_ref()
                    .and_then(|a| a.axes())
                    .map(|ax| ax.iter().map(|x| x as i32).collect())
                    .unwrap_or_default();
                let keep_dims = attrs.as_ref().map(|a| a.keep_dims()).unwrap_or(true);
                OpKind::Reduction {
                    op: ReductionOp::Min,
                    axes,
                    keep_dims,
                }
            }

            sg::OperatorType::GridSample => {
                let attrs = op.attrs_as_grid_sample_attrs();
                let align_corners = attrs.as_ref().map(|a| a.align_corners()).unwrap_or(false);
                OpKind::GridSample(GridSampleParams {
                    mode: 0,
                    padding_mode: 0,
                    align_corners,
                })
            }

            sg::OperatorType::ConstantOfShape => {
                let attrs = op.attrs_as_constant_of_shape_attrs();
                let val = attrs
                    .as_ref()
                    .and_then(|a| {
                        a.value_as_float_scalar()
                            .map(|v| v.value())
                            .or_else(|| a.value_as_int_scalar().map(|v| v.value() as f32))
                    })
                    .unwrap_or(0.0);
                OpKind::ConstantOfShape { value: val }
            }

            sg::OperatorType::Expand => OpKind::Expand,
            sg::OperatorType::ScatterND => OpKind::ScatterND,

            sg::OperatorType::LayerNormalization => {
                let attrs = op.attrs_as_layer_normalization_attrs();
                let axis = attrs.as_ref().map(|a| a.axis()).unwrap_or(-1);
                let epsilon = attrs.as_ref().map(|a| a.epsilon()).unwrap_or(1e-5);
                OpKind::LayerNorm(LayerNormParams {
                    axis,
                    epsilon,
                    use_rms_norm: false,
                    gamma: None,
                    beta: None,
                })
            }

            sg::OperatorType::Identity => OpKind::ConvertTensor,

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
