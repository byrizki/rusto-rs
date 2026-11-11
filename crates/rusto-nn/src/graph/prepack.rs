use crate::kernels::activation::FusedActivation;
use crate::schema::{Blob, Convolution2D, DataType, IDSTQuan, Op, OpType, PadMode, ScaleStorageType};
use crate::tensor::{f16_to_f32, Tensor, TensorLayout};

/// Pre-extracted and pre-dequantized parameters for Convolution operations
#[derive(Clone)]
pub struct PrepackedConv {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub pad_mode: PadMode,
    pub pad_x: usize,
    pub pad_y: usize,
    pub pads: Option<Vec<i32>>,
    pub group: usize,
    pub out_channels: usize,
    pub activation: FusedActivation,
    pub is_depthwise: bool,
}

/// Pre-extracted and pre-dequantized parameters for Deconvolution operations
#[derive(Clone)]
pub struct PrepackedDeconv {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub pad_x: usize,
    pub pad_y: usize,
    pub pads: Option<Vec<i32>>,
    pub group: usize,
    pub out_channels: usize,
    pub activation: FusedActivation,
}

/// Pre-packed static operation data created once at model load time
#[derive(Clone)]
pub enum PrepackedOp {
    Conv(PrepackedConv),
    Deconv(PrepackedDeconv),
    Const(Tensor),
}

/// Extract float weights and bias from a Convolution2D FlatBuffers structure
pub fn extract_conv_weights_and_bias(
    conv: &Convolution2D,
    output_count: usize,
) -> (Vec<f32>, Option<Vec<f32>>) {
    if let Some(w) = conv.weight() {
        let weight_vec: Vec<f32> = w.safe_slice().to_vec();
        let bias_vec = conv.bias().map(|b| b.safe_slice().to_vec());
        return (weight_vec, bias_vec);
    }

    if let Some(sq) = conv.symmetricQuan() {
        if let Some(raw_weights) = sq.weight() {
            let scales: Vec<f32> = sq.scale().map(|s| s.iter().collect()).unwrap_or_default();
            let total_weights = raw_weights.len();
            let weights_per_channel = total_weights.checked_div(output_count).unwrap_or(1).max(1);

            let mut float_weights = Vec::with_capacity(total_weights);
            for (idx, &w_byte) in raw_weights.iter().enumerate() {
                let oc = idx / weights_per_channel;
                let scale = scales.get(oc).copied().unwrap_or(1.0);
                float_weights.push(w_byte as f32 * scale);
            }

            let bias_vec = if let Some(b) = conv.bias() {
                Some(b.iter().collect())
            } else if let Some(ib) = sq.bias() {
                let float_bias: Vec<f32> = ib
                    .iter()
                    .enumerate()
                    .map(|(oc, b_val)| {
                        let scale = scales.get(oc).copied().unwrap_or(1.0);
                        b_val as f32 * scale
                    })
                    .collect();
                Some(float_bias)
            } else {
                None
            };

            return (float_weights, bias_vec);
        }
    }

    if let Some(quan) = conv.quanParameter() {
        let float_weights = decode_idst_quan(&quan);
        let bias_vec = conv.bias().map(|b| b.safe_slice().to_vec());
        return (float_weights, bias_vec);
    }

    (Vec::new(), conv.bias().map(|b| b.safe_slice().to_vec()))
}

fn read_blob_dim(buf: &[u8], shape_int32: bool) -> Option<(Vec<usize>, usize)> {
    let u_size = *buf.first()? as usize;
    if u_size == 0 || u_size > 32 {
        return None;
    }
    let elem_bytes = if shape_int32 { 4 } else { 2 };
    let header_end = 1 + u_size * elem_bytes;
    if buf.len() < header_end {
        return None;
    }
    let mut shape = Vec::with_capacity(u_size);
    for chunk in buf[1..header_end].chunks_exact(elem_bytes) {
        let dim = if shape_int32 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as usize
        };
        shape.push(dim);
    }
    Some((shape, header_end))
}

fn split_buf_to_array(buf: &[u8], count: usize, bits: usize) -> Option<Vec<u8>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if bits == 0 || bits > 8 {
        return None;
    }
    let need_bytes = (bits * count).div_ceil(8);
    if buf.len() < need_bytes {
        return None;
    }
    let mask = (1u16 << bits) - 1;
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let byte_idx = bit_pos / 8;
        let bit_off = bit_pos % 8;
        let shift = 8isize - bits as isize - bit_off as isize;
        let val = if shift < 0 {
            let hi = (buf[byte_idx] as u16) << (-shift);
            let lo = (buf[byte_idx + 1] as u16) >> (8 + shift);
            (hi | lo) & mask
        } else {
            ((buf[byte_idx] as u16) >> shift) & mask
        };
        out.push(val as u8);
        bit_pos += bits;
    }
    Some(out)
}

#[inline]
fn bits_for_count(count: usize) -> usize {
    if count <= 1 {
        1
    } else {
        usize::BITS as usize - (count - 1).leading_zeros() as usize
    }
}

fn decode_idst_dense(buf: &[u8], shape_int32: bool) -> Option<Vec<i8>> {
    let (shape, mut pos) = read_blob_dim(buf, shape_int32)?;
    let data_cnt: usize = shape.iter().product();
    let sample_byte = *buf.get(pos)?;
    pos += 1;
    let sample_cnt = if sample_byte == 0 { 256 } else { sample_byte as usize };
    let sample_bytes = buf.get(pos..pos + sample_cnt)?;
    pos += sample_cnt;

    let mut samples: Vec<i8> = sample_bytes.iter().map(|&b| b as i8).collect();
    samples.sort_unstable();

    let idx_bits = bits_for_count(sample_cnt);
    let idxs = split_buf_to_array(buf.get(pos..)?, data_cnt, idx_bits)?;
    let mut weights = Vec::with_capacity(data_cnt);
    for idx in idxs {
        weights.push(*samples.get(idx as usize)?);
    }
    Some(weights)
}

fn decode_idst_sparse(buf: &[u8], shape_int32: bool, alphas: &[f32]) -> Option<Vec<i8>> {
    let (shape, mut pos) = read_blob_dim(buf, shape_int32)?;
    let out_channels = *shape.first()?;
    let total: usize = shape.iter().product();
    let nnz_bytes = buf.get(pos..pos + 4)?;
    let nnz = u32::from_le_bytes([nnz_bytes[0], nnz_bytes[1], nnz_bytes[2], nnz_bytes[3]]) as usize;
    pos += 4;

    let step_bits = *buf.get(pos)? as usize;
    pos += 1;
    let step_bytes = (step_bits * nnz).div_ceil(8);
    let steps = if nnz == 0 {
        Vec::new()
    } else {
        split_buf_to_array(buf.get(pos..pos + step_bytes)?, nnz, step_bits)?
    };
    pos += step_bytes;

    let map_byte = *buf.get(pos)?;
    pos += 1;
    let map_size = if map_byte == 0 { 256 } else { map_byte as usize };
    let raw_map = buf.get(pos..pos + map_size)?;
    pos += map_size;

    let mut uniq_vals: Vec<i8> = raw_map.iter().map(|&b| b as i8).collect();
    uniq_vals.sort_unstable();
    uniq_vals.dedup();

    let data_bits = bits_for_count(map_size);
    let val_idxs = if nnz == 0 {
        Vec::new()
    } else {
        split_buf_to_array(buf.get(pos..)?, nnz, data_bits)?
    };

    let mut weights = vec![0i8; total];
    if out_channels > 0 && alphas.len() == 2 * out_channels {
        let min_value = -(1i32 << (data_bits - 1));
        let area = total / out_channels;
        for oc in 0..out_channels {
            let min = alphas[2 * oc];
            let scale = alphas[2 * oc + 1];
            let zero_quant = if scale > 1e-6 {
                ((0.0f32 - min) / scale).round() as i32 + min_value
            } else {
                min_value
            };
            weights[oc * area..(oc + 1) * area].fill(zero_quant as i8);
        }
    }

    let mut w_pos = 0usize;
    for i in 0..nnz {
        w_pos += steps[i] as usize;
        if w_pos >= weights.len() {
            return None;
        }
        weights[w_pos] = *uniq_vals.get(val_idxs[i] as usize)?;
    }
    Some(weights)
}

fn extract_idst_alphas(quan: &IDSTQuan<'_>) -> Vec<f32> {
    if quan.scaleStorage() == ScaleStorageType::FP16 {
        quan.alphaFp16()
            .map(|a| a.iter().map(f16_to_f32).collect())
            .unwrap_or_default()
    } else {
        quan.alpha().map(|a| a.iter().collect()).unwrap_or_default()
    }
}

/// Decode `IDSTQuan` quantized or FP16 weights into FP32 weights, matching MNN's `ConvolutionCommon::load`.
pub fn decode_idst_quan(quan: &IDSTQuan<'_>) -> Vec<f32> {
    if let Some(indices) = quan.index() {
        let weight_size = quan.weightSize() as usize;
        let alphas = extract_idst_alphas(quan);
        if weight_size > 0 && alphas.len() == indices.len() {
            let mut weights = vec![0.0f32; weight_size];
            for (idx, alpha) in indices.iter().zip(alphas.iter()) {
                if (idx as usize) < weight_size {
                    weights[idx as usize] = *alpha;
                }
            }
            return weights;
        }
    }

    let Some(raw_i8) = quan.buffer() else {
        return Vec::new();
    };
    let raw_u8: &[u8] = unsafe { std::slice::from_raw_parts(raw_i8.as_ptr() as *const u8, raw_i8.len()) };

    if quan.type_() == 3 {
        return raw_u8
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
    }

    let mut alphas = extract_idst_alphas(quan);
    let q_weights: Vec<i8> = match quan.type_() {
        1 => decode_idst_dense(raw_u8, quan.shapeInt32()).unwrap_or_default(),
        2 => decode_idst_sparse(raw_u8, quan.shapeInt32(), &alphas).unwrap_or_default(),
        _ => raw_i8.to_vec(),
    };
    if q_weights.is_empty() {
        return Vec::new();
    }

    let old_type4 = quan.type_() == 4 && quan.aMin() == 0 && quan.quantScale().abs() < 1e-6;
    let asymmetric = quan.readType() != 0 || old_type4;

    if asymmetric {
        let oc_cnt = alphas.len() / 2;
        let clamp_min = if quan.aMin() == 0 { -128.0f32 } else { quan.aMin() as f32 };
        if clamp_min < 0.0 {
            for oc in 0..oc_cnt {
                alphas[2 * oc] -= clamp_min * alphas[2 * oc + 1];
            }
        }
    }

    if !quan.has_scaleInt() {
        let extra_factor = if old_type4 { 1.0f32 } else { quan.quantScale() };
        if extra_factor != 1.0 && extra_factor != 0.0 {
            for a in &mut alphas {
                *a *= extra_factor;
            }
        }
    }

    let total = q_weights.len();
    let mut float_weights = Vec::with_capacity(total);
    if asymmetric && alphas.len() >= 2 {
        let oc_cnt = alphas.len() / 2;
        let part_size = total.checked_div(oc_cnt).unwrap_or(1).max(1);
        for (idx, &q) in q_weights.iter().enumerate() {
            let oc = (idx / part_size).min(oc_cnt - 1);
            let min = alphas[2 * oc];
            let scale = alphas[2 * oc + 1];
            float_weights.push(q as f32 * scale + min);
        }
    } else if !alphas.is_empty() {
        let oc_cnt = alphas.len();
        let part_size = total.checked_div(oc_cnt).unwrap_or(1).max(1);
        for (idx, &q) in q_weights.iter().enumerate() {
            let oc = (idx / part_size).min(oc_cnt - 1);
            float_weights.push(q as f32 * alphas[oc]);
        }
    } else {
        let global_scale = if quan.quantScale() != 0.0 { quan.quantScale() } else { 1.0 };
        for &q in &q_weights {
            float_weights.push(q as f32 * global_scale);
        }
    }
    float_weights
}

/// Decode an MNN `Blob` into a `Tensor`, supporting FP16 (`DT_HALF`), FP32, and integer buffers.
pub fn blob_to_tensor(blob: &Blob<'_>) -> Tensor {
    let dims: Vec<usize> = blob
        .dims()
        .map(|d| d.iter().map(|v| v.max(0) as usize).collect())
        .unwrap_or_else(|| vec![1]);

    if blob.dataType() == DataType::DT_HALF {
        if let Some(u8s) = blob.uint8s() {
            let floats: Vec<f32> = u8s
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            return Tensor::from_floats(&dims, floats, TensorLayout::NCHW);
        }
        if let Some(i8s) = blob.int8s() {
            let floats: Vec<f32> = i8s
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0] as u8, c[1] as u8])))
                .collect();
            return Tensor::from_floats(&dims, floats, TensorLayout::NCHW);
        }
        if let Some(floats) = blob.float32s() {
            return Tensor::from_floats(&dims, floats.safe_slice().to_vec(), TensorLayout::NCHW);
        }
    }

    if let Some(floats) = blob.float32s() {
        let v: Vec<f32> = floats.safe_slice().to_vec();
        Tensor::from_floats(&dims, v, TensorLayout::NCHW)
    } else if let Some(ints) = blob.int32s() {
        let v: Vec<i32> = ints.safe_slice().to_vec();
        Tensor::from_ints(&dims, v)
    } else if let Some(i64s) = blob.int64s() {
        let v: Vec<i32> = i64s.iter().map(|x| x as i32).collect();
        Tensor::from_ints(&dims, v)
    } else if let Some(i8s) = blob.int8s() {
        let v: Vec<i32> = i8s.iter().map(|&x| x as i32).collect();
        Tensor::from_ints(&dims, v)
    } else if let Some(u8s) = blob.uint8s() {
        let v: Vec<i32> = u8s.iter().map(|&x| x as i32).collect();
        Tensor::from_ints(&dims, v)
    } else {
        Tensor::zeros(&dims, TensorLayout::NCHW)
    }
}

/// Pre-pack operation metadata and static parameters at model load time
pub fn prepack_op(op: &Op<'_>) -> Option<PrepackedOp> {
    match op.type_() {
        OpType::Convolution | OpType::ConvolutionDepthwise => {
            let conv = op.main_as_convolution_2d()?;
            let common = conv.common()?;
            let out_channels = common.outputCount() as usize;

            let (weight, bias) = extract_conv_weights_and_bias(&conv, out_channels);

            let mut act = FusedActivation::None;
            if common.relu() {
                act = FusedActivation::Relu;
            } else if common.relu6() {
                act = FusedActivation::Relu6;
            }

            let group = (common.group() as usize).max(1);
            let is_depthwise = op.type_() == OpType::ConvolutionDepthwise
                || (group > 1 && group == out_channels);

            let pads: Option<Vec<i32>> = common.pads().map(|p| p.safe_slice().to_vec());

            Some(PrepackedOp::Conv(PrepackedConv {
                weight,
                bias,
                kernel_h: common.kernelY() as usize,
                kernel_w: common.kernelX() as usize,
                stride_h: common.strideY() as usize,
                stride_w: common.strideX() as usize,
                dilate_h: common.dilateY() as usize,
                dilate_w: common.dilateX() as usize,
                pad_mode: common.padMode(),
                pad_x: common.padX() as usize,
                pad_y: common.padY() as usize,
                pads,
                group,
                out_channels,
                activation: act,
                is_depthwise,
            }))
        }

        OpType::Deconvolution => {
            let conv = op.main_as_convolution_2d()?;
            let common = conv.common()?;
            let out_channels = common.outputCount() as usize;

            let (weight, bias) = extract_conv_weights_and_bias(&conv, out_channels);

            let mut act = FusedActivation::None;
            if common.relu() {
                act = FusedActivation::Relu;
            } else if common.relu6() {
                act = FusedActivation::Relu6;
            }

            let pads: Option<Vec<i32>> = common.pads().map(|p| p.safe_slice().to_vec());

            Some(PrepackedOp::Deconv(PrepackedDeconv {
                weight,
                bias,
                kernel_h: common.kernelY() as usize,
                kernel_w: common.kernelX() as usize,
                stride_h: common.strideY() as usize,
                stride_w: common.strideX() as usize,
                dilate_h: common.dilateY() as usize,
                dilate_w: common.dilateX() as usize,
                pad_x: common.padX() as usize,
                pad_y: common.padY() as usize,
                pads,
                group: (common.group() as usize).max(1),
                out_channels,
                activation: act,
            }))
        }

        OpType::Const => {
            let blob = op.main_as_blob()?;
            Some(PrepackedOp::Const(blob_to_tensor(&blob)))
        }

        _ => None,
    }
}

