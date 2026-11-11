use crate::kernels::activation::FusedActivation;
use crate::schema::{Convolution2D, Op, OpType, PadMode};
use crate::tensor::{Tensor, TensorLayout};

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
        if let Some(raw_buf) = quan.buffer() {
            let alphas: Vec<f32> = quan.alpha().map(|a| a.iter().collect()).unwrap_or_default();
            let global_scale = quan.quantScale();
            let total_weights = raw_buf.len();
            let weights_per_channel = total_weights.checked_div(output_count).unwrap_or(1).max(1);

            let mut float_weights = Vec::with_capacity(total_weights);
            for (idx, &w_byte) in raw_buf.iter().enumerate() {
                let oc = idx / weights_per_channel;
                let scale = if !alphas.is_empty() {
                    alphas.get(oc).copied().unwrap_or(1.0)
                } else if global_scale != 0.0 {
                    global_scale
                } else {
                    1.0
                };
                float_weights.push(w_byte as f32 * scale);
            }

            let bias_vec = conv.bias().map(|b| b.safe_slice().to_vec());
            return (float_weights, bias_vec);
        }
    }

    (Vec::new(), conv.bias().map(|b| b.safe_slice().to_vec()))
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
            let dims: Vec<usize> = blob
                .dims()
                .map(|d| d.iter().map(|v| v.max(0) as usize).collect())
                .unwrap_or_else(|| vec![1]);

            let tensor = if let Some(floats) = blob.float32s() {
                let v: Vec<f32> = floats.safe_slice().to_vec();
                Tensor::from_floats(&dims, v, TensorLayout::NCHW)
            } else if let Some(ints) = blob.int32s() {
                let v: Vec<i32> = ints.safe_slice().to_vec();
                Tensor::from_ints(&dims, v)
            } else if let Some(i64s) = blob.int64s() {
                let v: Vec<i32> = i64s.iter().map(|x| x as i32).collect();
                Tensor::from_ints(&dims, v)
            } else if let Some(u8s) = blob.uint8s() {
                let v: Vec<i32> = u8s.iter().map(|&x| x as i32).collect();
                Tensor::from_ints(&dims, v)
            } else {
                Tensor::zeros(&dims, TensorLayout::NCHW)
            };

            Some(PrepackedOp::Const(tensor))
        }

        _ => None,
    }
}

