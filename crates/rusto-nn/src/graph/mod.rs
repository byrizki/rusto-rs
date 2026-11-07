use crate::error::MnnError;
use crate::kernels::activation::apply_activation_slice;
use crate::kernels::binary::{binary_op, try_binary_op_inplace_lhs};
use crate::kernels::classification::{argmax, log_softmax, softmax, topk};
use crate::kernels::conv::{conv2d, conv2d_output_dims, Conv2dConfig};
use crate::kernels::deconv::{deconv2d, deconv2d_output_dims, Deconv2dConfig};
use crate::kernels::depthwise::{depthwise_conv2d_nchw, depthwise_conv2d_output_dims, DepthwiseConv2dConfig};
use crate::kernels::interp::{interp2d, InterpConfig};
use crate::kernels::pool::{pool2d, pool2d_output_dims, Pool2dConfig};
use crate::kernels::reduction::reduce;
use crate::kernels::tensor_ops::{
    compute_reshape_dims, concat, gather, grid_sample_2d, pad_tensor, permute, slice_tf, strided_slice,
};
use crate::kernels::unary::unary_op;
use crate::schema::{BorderMode, Op, SampleMode};
use crate::tensor::{Tensor, TensorLayout};

pub mod ir;
pub mod loader;
pub mod optimize;
pub mod prepack;

pub use ir::*;
pub use loader::*;
pub use prepack::*;

fn compute_conv_padding(
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    dilate_h: usize,
    dilate_w: usize,
    pad_mode: PadMode,
    pad_x: usize,
    pad_y: usize,
    pads: Option<&[i32]>,
) -> (usize, usize, usize, usize) {
    match pad_mode {
        PadMode::SAME => {
            let out_h = in_h.div_ceil(stride_h);
            let out_w = in_w.div_ceil(stride_w);

            let k_eff_h = (kh - 1) * dilate_h + 1;
            let k_eff_w = (kw - 1) * dilate_w + 1;

            let pad_needed_h = ((out_h - 1) * stride_h + k_eff_h).saturating_sub(in_h);
            let pad_needed_w = ((out_w - 1) * stride_w + k_eff_w).saturating_sub(in_w);

            let pad_top = pad_needed_h / 2;
            let pad_left = pad_needed_w / 2;

            let right = (out_w - 1) * stride_w + (kw - 1) * dilate_w;
            let pad_right = if right + 1 >= in_w + pad_left {
                (right + 1) - (in_w + pad_left)
            } else {
                pad_needed_w - pad_left
            };

            let bottom = (out_h - 1) * stride_h + (kh - 1) * dilate_h;
            let pad_bottom = if bottom + 1 >= in_h + pad_top {
                (bottom + 1) - (in_h + pad_top)
            } else {
                pad_needed_h - pad_top
            };

            (pad_top, pad_left, pad_bottom, pad_right)
        }
        PadMode::VALID => (0, 0, 0, 0),
        _ => {
            if let Some(p) = pads {
                if p.len() >= 4 {
                    (p[0].max(0) as usize, p[1].max(0) as usize, p[2].max(0) as usize, p[3].max(0) as usize)
                } else if p.len() >= 2 {
                    if kw == 1 && kh > 1 {
                        (p[0].max(0) as usize, 0, p[1].max(0) as usize, 0)
                    } else if kh == 1 && kw > 1 {
                        (0, p[0].max(0) as usize, 0, p[1].max(0) as usize)
                    } else {
                        (p[0].max(0) as usize, p[1].max(0) as usize, p[0].max(0) as usize, p[1].max(0) as usize)
                    }
                } else {
                    if kw == 1 && kh > 1 && pad_x == pad_y {
                        (pad_y, 0, pad_y, 0)
                    } else if kh == 1 && kw > 1 && pad_x == pad_y {
                        (0, pad_x, 0, pad_x)
                    } else {
                        (pad_y, pad_x, pad_y, pad_x)
                    }
                }
            } else {
                if kw == 1 && kh > 1 && pad_x == pad_y {
                    (pad_y, 0, pad_y, 0)
                } else if kh == 1 && kw > 1 && pad_x == pad_y {
                    (0, pad_x, 0, pad_x)
                } else {
                    (pad_y, pad_x, pad_y, pad_x)
                }
            }
        }
    }
}

thread_local! {
    static TENSOR_BUFFER_POOL: core::cell::RefCell<Vec<Vec<f32>>> = core::cell::RefCell::new(Vec::with_capacity(64));
    static MATMUL_SCRATCH_A: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    static MATMUL_SCRATCH_B: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
}

#[inline(always)]
fn pop_pooled_buffer(len: usize) -> Option<Vec<f32>> {
    TENSOR_BUFFER_POOL.with(|cell| {
        let mut pool = cell.borrow_mut();
        let mut best_fit: Option<(usize, usize)> = None;
        let mut largest: Option<(usize, usize)> = None;

        for (i, buf) in pool.iter().enumerate() {
            let cap = buf.capacity();
            if cap >= len {
                match best_fit {
                    None => best_fit = Some((i, cap)),
                    Some((_, best_cap)) if cap < best_cap => best_fit = Some((i, cap)),
                    _ => {}
                }
            } else {
                match largest {
                    None => largest = Some((i, cap)),
                    Some((_, max_cap)) if cap > max_cap => largest = Some((i, cap)),
                    _ => {}
                }
            }
        }

        best_fit.or(largest).map(|(i, _)| pool.swap_remove(i))
    })
}

#[inline(always)]
pub fn alloc_tensor_buffer(len: usize) -> Vec<f32> {
    if let Some(mut buf) = pop_pooled_buffer(len) {
        buf.clear();
        buf.resize(len, 0.0);
        buf
    } else {
        vec![0.0f32; len]
    }
}

#[inline(always)]
pub fn alloc_tensor_buffer_overwrite(len: usize) -> Vec<f32> {
    if let Some(mut buf) = pop_pooled_buffer(len) {
        if buf.len() >= len {
            buf.truncate(len);
        } else {
            buf.resize(len, 0.0);
        }
        buf
    } else {
        vec![0.0f32; len]
    }
}

#[inline(always)]
pub fn recycle_tensor_buffer(buf: Vec<f32>) {
    if buf.capacity() == 0 {
        return;
    }
    TENSOR_BUFFER_POOL.with(|cell| {
        let mut pool = cell.borrow_mut();
        if pool.len() < 128 {
            pool.push(buf);
        }
    });
}

/// Execution context maintaining intermediate tensor buffers
pub struct GraphContext {
    tensors: Vec<Option<Tensor>>,
    tensor_names: Vec<String>,
}

impl GraphContext {
    pub fn new(total_tensors: usize, names: &[String]) -> Self {
        Self {
            tensors: (0..total_tensors).map(|_| None).collect(),
            tensor_names: names.to_vec(),
        }
    }

    pub fn reset(&mut self) {
        for slot in &mut self.tensors {
            if let Some(t) = slot.take() {
                if let Some(vec) = t.into_owned_floats() {
                    recycle_tensor_buffer(vec);
                }
            }
        }
    }

    #[inline]
    pub fn recycle_dead_inputs(&mut self, node: &Node, op_idx: usize, last_use: &[usize]) {
        for &in_tid in &node.in_indices {
            if in_tid < last_use.len()
                && last_use[in_tid] == op_idx
                && !node.out_indices.contains(&in_tid)
            {
                if let Some(t) = self.tensors[in_tid].take() {
                    if let Some(vec) = t.into_owned_floats() {
                        recycle_tensor_buffer(vec);
                    }
                }
            }
        }
    }

    pub fn set_tensor(&mut self, idx: usize, tensor: Tensor) {
        if idx < self.tensors.len() {
            self.tensors[idx] = Some(tensor);
        }
    }

    pub fn get_tensor(&self, idx: usize) -> Result<&Tensor, MnnError> {
        if idx >= self.tensors.len() {
            return Err(MnnError::ExecutionError(format!(
                "Tensor index {} out of bounds (max {})",
                idx, self.tensors.len()
            )));
        }
        self.tensors[idx].as_ref().ok_or_else(|| {
            let name = self.tensor_names.get(idx).map(|s| s.as_str()).unwrap_or("<unknown>");
            MnnError::ExecutionError(format!("Tensor index {} ({}) is uninitialized", idx, name))
        })
    }

    pub fn get_tensor_mut(&mut self, idx: usize) -> Result<&mut Tensor, MnnError> {
        if idx >= self.tensors.len() {
            return Err(MnnError::ExecutionError(format!(
                "Tensor index {} out of bounds (max {})",
                idx, self.tensors.len()
            )));
        }
        self.tensors[idx].as_mut().ok_or_else(|| {
            let name = self.tensor_names.get(idx).map(|s| s.as_str()).unwrap_or("<unknown>");
            MnnError::ExecutionError(format!("Tensor index {} ({}) is uninitialized", idx, name))
        })
    }

    pub fn take_tensor(&mut self, idx: usize) -> Result<Tensor, MnnError> {
        if idx >= self.tensors.len() {
            return Err(MnnError::ExecutionError(format!(
                "Tensor index {} out of bounds",
                idx
            )));
        }
        self.tensors[idx].take().ok_or_else(|| {
            MnnError::ExecutionError(format!("Tensor index {} is uninitialized", idx))
        })
    }

    #[inline]
    pub fn take_unique_dead_float_buffer(
        &mut self,
        tid: usize,
        op_idx: usize,
        last_use: Option<&[usize]>,
    ) -> Option<Vec<f32>> {
        let lu = last_use?;
        if tid >= lu.len() || lu[tid] != op_idx || tid >= self.tensors.len() {
            return None;
        }
        if !self.tensors[tid].as_ref()?.has_unique_float_buffer() {
            return None;
        }
        self.tensors[tid].take()?.into_owned_floats()
    }
}

/// Execute a single graph IR node against the execution context
pub fn execute_node(
    node: &Node,
    op_idx: usize,
    ctx: &mut GraphContext,
) -> Result<(), MnnError> {
    execute_node_with_liveness(node, op_idx, ctx, None)
}

/// Execute a single graph IR node with optional tensor liveness metadata for in-place reuse
pub fn execute_node_with_liveness(
    node: &Node,
    op_idx: usize,
    ctx: &mut GraphContext,
    last_use: Option<&[usize]>,
) -> Result<(), MnnError> {
    let in_indices = &node.in_indices;
    let out_indices = &node.out_indices;

    match &node.op {
        OpKind::Input => Ok(()),

        OpKind::Const(t) => {
            if !out_indices.is_empty() {
                ctx.set_tensor(out_indices[0], t.clone());
            }
            Ok(())
        }

        OpKind::ConvertTensor => {
            if !in_indices.is_empty() && !out_indices.is_empty() {
                let src = ctx.get_tensor_mut(in_indices[0])?.share();
                ctx.set_tensor(out_indices[0], src);
            }
            Ok(())
        }

        OpKind::Conv(p) => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let is_3d = in_shape.len() == 3;
            let batch = if in_shape.is_empty() { 1 } else { in_shape[0] };
            let in_channels = if in_shape.len() > 1 { in_shape[1] } else { 1 };
            let (in_h, in_w) = if is_3d {
                (in_shape[2], 1)
            } else {
                (
                    if in_shape.len() > 2 { in_shape[2] } else { 1 },
                    if in_shape.len() > 3 { in_shape[3] } else { 1 },
                )
            };

            let in_floats = in_tensor.as_f32_cow();

            let (pad_top, pad_left, pad_bottom, pad_right) = compute_conv_padding(
                in_h,
                in_w,
                p.kernel_h,
                p.kernel_w,
                p.stride_h,
                p.stride_w,
                p.dilate_h,
                p.dilate_w,
                p.pad_mode,
                p.pad_x,
                p.pad_y,
                p.pads.as_deref(),
            );

            if p.is_depthwise {
                let dw_cfg = DepthwiseConv2dConfig {
                    channels: in_channels,
                    kernel_h: p.kernel_h,
                    kernel_w: p.kernel_w,
                    stride_h: p.stride_h,
                    stride_w: p.stride_w,
                    pad_top,
                    pad_left,
                    pad_bottom,
                    pad_right,
                    dilate_h: p.dilate_h,
                    dilate_w: p.dilate_w,
                    activation: p.activation,
                };
                let (out_h, out_w) = depthwise_conv2d_output_dims(in_h, in_w, &dw_cfg);
                let mut out_data = alloc_tensor_buffer_overwrite(batch * in_channels * out_h * out_w);
                depthwise_conv2d_nchw(
                    &in_floats,
                    batch,
                    in_h,
                    in_w,
                    &p.weight,
                    p.bias.as_deref(),
                    &mut out_data,
                    &dw_cfg,
                );
                let out_shape = if is_3d {
                    vec![batch, in_channels, out_h]
                } else {
                    vec![batch, in_channels, out_h, out_w]
                };
                ctx.set_tensor(out_idx, Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW));
            } else {
                let cfg = Conv2dConfig {
                    in_channels,
                    out_channels: p.out_channels,
                    kernel_h: p.kernel_h,
                    kernel_w: p.kernel_w,
                    stride_h: p.stride_h,
                    stride_w: p.stride_w,
                    pad_top,
                    pad_left,
                    pad_bottom,
                    pad_right,
                    dilate_h: p.dilate_h,
                    dilate_w: p.dilate_w,
                    group: p.group,
                    activation: p.activation,
                };
                let (out_h, out_w) = conv2d_output_dims(in_h, in_w, &cfg);
                let mut out_data = if in_channels > 0 && p.kernel_h > 0 && p.kernel_w > 0 {
                    alloc_tensor_buffer_overwrite(batch * p.out_channels * out_h * out_w)
                } else {
                    alloc_tensor_buffer(batch * p.out_channels * out_h * out_w)
                };
                conv2d(
                    &in_floats,
                    batch,
                    in_h,
                    in_w,
                    &p.weight,
                    p.bias.as_deref(),
                    &mut out_data,
                    &cfg,
                );
                let out_shape = if is_3d {
                    vec![batch, p.out_channels, out_h]
                } else {
                    vec![batch, p.out_channels, out_h, out_w]
                };
                ctx.set_tensor(out_idx, Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW));
            }
            Ok(())
        }

        OpKind::Deconv(p) => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let is_3d = in_shape.len() == 3;
            let batch = if in_shape.is_empty() { 1 } else { in_shape[0] };
            let in_channels = if in_shape.len() > 1 { in_shape[1] } else { 1 };
            let (in_h, in_w) = if is_3d {
                (in_shape[2], 1)
            } else {
                (
                    if in_shape.len() > 2 { in_shape[2] } else { 1 },
                    if in_shape.len() > 3 { in_shape[3] } else { 1 },
                )
            };

            let in_floats = in_tensor.as_f32_cow();

            let (pad_top, pad_left, pad_bottom, pad_right) = compute_conv_padding(
                in_h,
                in_w,
                p.kernel_h,
                p.kernel_w,
                p.stride_h,
                p.stride_w,
                p.dilate_h,
                p.dilate_w,
                PadMode::CAFFE,
                p.pad_x,
                p.pad_y,
                p.pads.as_deref(),
            );

            let cfg = Deconv2dConfig {
                in_channels,
                out_channels: p.out_channels,
                kernel_h: p.kernel_h,
                kernel_w: p.kernel_w,
                stride_h: p.stride_h,
                stride_w: p.stride_w,
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                out_pad_h: 0,
                out_pad_w: 0,
                dilate_h: p.dilate_h,
                dilate_w: p.dilate_w,
                group: p.group,
                activation: p.activation,
            };

            let (out_h, out_w) = deconv2d_output_dims(in_h, in_w, &cfg);
            let mut out_data = alloc_tensor_buffer(batch * p.out_channels * out_h * out_w);
            deconv2d(
                &in_floats,
                batch,
                in_h,
                in_w,
                &p.weight,
                p.bias.as_deref(),
                &mut out_data,
                &cfg,
            );
            let out_tensor = Tensor::from_floats(
                &[batch, p.out_channels, out_h, out_w],
                out_data,
                TensorLayout::NCHW,
            );
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Pool(p) => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let batch = if in_shape.is_empty() { 1 } else { in_shape[0] };
            let channels = if in_shape.len() > 1 { in_shape[1] } else { 1 };
            let in_h = if in_shape.len() > 2 { in_shape[2] } else { 1 };
            let in_w = if in_shape.len() > 3 { in_shape[3] } else { 1 };

            let in_floats = in_tensor.as_f32_cow();

            let (pad_top, pad_left, pad_bottom, pad_right) = compute_conv_padding(
                in_h,
                in_w,
                p.kernel_h,
                p.kernel_w,
                p.stride_h,
                p.stride_w,
                1,
                1,
                p.pad_mode,
                p.pad_x,
                p.pad_y,
                p.pads.as_deref(),
            );

            let pool_cfg = Pool2dConfig {
                pool_type: p.pool_type,
                kernel_h: p.kernel_h,
                kernel_w: p.kernel_w,
                stride_h: p.stride_h.max(1),
                stride_w: p.stride_w.max(1),
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                is_global: p.is_global,
                ceil_mode: false,
                count_include_pad: false,
            };

            let (out_h, out_w) = pool2d_output_dims(in_h, in_w, &pool_cfg);
            let mut out_data = alloc_tensor_buffer(batch * channels * out_h * out_w);
            pool2d(&in_floats, batch, channels, in_h, in_w, &mut out_data, &pool_cfg);
            let out_tensor = Tensor::from_floats(&[batch, channels, out_h, out_w], out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Binary(p) => {
            if in_indices.len() < 2 || out_indices.is_empty() {
                return Ok(());
            }
            let lhs_tid = in_indices[0];
            let rhs_tid = in_indices[1];
            let out_idx = out_indices[0];

            let (lhs_shape, rhs_shape) = {
                let lhs = ctx.get_tensor(lhs_tid)?;
                let rhs = ctx.get_tensor(rhs_tid)?;
                (lhs.shape().to_vec(), rhs.shape().to_vec())
            };

            let ndim = lhs_shape.len().max(rhs_shape.len());
            let mut out_shape = vec![1usize; ndim];
            for d in 0..ndim {
                let l = if d + lhs_shape.len() >= ndim {
                    lhs_shape[d + lhs_shape.len() - ndim]
                } else {
                    1
                };
                let r = if d + rhs_shape.len() >= ndim {
                    rhs_shape[d + rhs_shape.len() - ndim]
                } else {
                    1
                };

                out_shape[d] = l.max(r);
            }

            if lhs_tid != rhs_tid && lhs_shape == out_shape {
                if let Some(mut lhs_buf) =
                    ctx.take_unique_dead_float_buffer(lhs_tid, op_idx, last_use)
                {
                    let rhs_floats = ctx.get_tensor(rhs_tid)?.as_f32_cow();
                    if try_binary_op_inplace_lhs(
                        &lhs_shape,
                        &mut lhs_buf,
                        &rhs_shape,
                        &rhs_floats,
                        &out_shape,
                        p.op_type,
                        p.activation,
                    ) {
                        drop(rhs_floats);
                        ctx.set_tensor(
                            out_idx,
                            Tensor::from_floats(&out_shape, lhs_buf, TensorLayout::NCHW),
                        );
                        return Ok(());
                    }
                    let out_len = out_shape.iter().product::<usize>();
                    let mut out_data = alloc_tensor_buffer_overwrite(out_len);
                    binary_op(
                        &lhs_shape,
                        &lhs_buf,
                        &rhs_shape,
                        &rhs_floats,
                        &out_shape,
                        &mut out_data,
                        p.op_type,
                        p.activation,
                    );
                    drop(rhs_floats);
                    recycle_tensor_buffer(lhs_buf);
                    ctx.set_tensor(
                        out_idx,
                        Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW),
                    );
                    return Ok(());
                }
            } else if lhs_tid != rhs_tid
                && rhs_shape == out_shape
                && matches!(
                    p.op_type,
                    crate::kernels::binary::BinaryOpType::Add
                        | crate::kernels::binary::BinaryOpType::Mul
                        | crate::kernels::binary::BinaryOpType::Min
                        | crate::kernels::binary::BinaryOpType::Max
                )
            {
                if let Some(mut rhs_buf) =
                    ctx.take_unique_dead_float_buffer(rhs_tid, op_idx, last_use)
                {
                    let lhs_floats = ctx.get_tensor(lhs_tid)?.as_f32_cow();
                    if try_binary_op_inplace_lhs(
                        &rhs_shape,
                        &mut rhs_buf,
                        &lhs_shape,
                        &lhs_floats,
                        &out_shape,
                        p.op_type,
                        p.activation,
                    ) {
                        drop(lhs_floats);
                        ctx.set_tensor(
                            out_idx,
                            Tensor::from_floats(&out_shape, rhs_buf, TensorLayout::NCHW),
                        );
                        return Ok(());
                    }
                    let out_len = out_shape.iter().product::<usize>();
                    let mut out_data = alloc_tensor_buffer_overwrite(out_len);
                    binary_op(
                        &lhs_shape,
                        &lhs_floats,
                        &rhs_shape,
                        &rhs_buf,
                        &out_shape,
                        &mut out_data,
                        p.op_type,
                        p.activation,
                    );
                    drop(lhs_floats);
                    recycle_tensor_buffer(rhs_buf);
                    ctx.set_tensor(
                        out_idx,
                        Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW),
                    );
                    return Ok(());
                }
            }

            let out_len = out_shape.iter().product::<usize>();
            let mut out_data = alloc_tensor_buffer_overwrite(out_len);
            {
                let lhs_floats = ctx.get_tensor(lhs_tid)?.as_f32_cow();
                let rhs_floats = ctx.get_tensor(rhs_tid)?.as_f32_cow();
                binary_op(
                    &lhs_shape,
                    &lhs_floats,
                    &rhs_shape,
                    &rhs_floats,
                    &out_shape,
                    &mut out_data,
                    p.op_type,
                    p.activation,
                );
            }

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Unary(p) => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_floats = in_tensor.as_f32_cow();
            let mut out_data = alloc_tensor_buffer_overwrite(in_floats.len());
            unary_op(&in_floats, &mut out_data, p.op_type);
            if p.activation != FusedActivation::None {
                apply_activation_slice(&mut out_data, p.activation);
            }

            let out_tensor = Tensor::from_floats(in_tensor.shape(), out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Relu { slope } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tid = in_indices[0];
            let out_idx = out_indices[0];
            let (in_shape, in_layout) = {
                let t = ctx.get_tensor(in_tid)?;
                (t.shape().to_vec(), t.layout())
            };

            let mut out_data = if let Some(buf) =
                ctx.take_unique_dead_float_buffer(in_tid, op_idx, last_use)
            {
                buf
            } else {
                let in_floats = ctx.get_tensor(in_tid)?.as_f32_cow();
                let mut buf = alloc_tensor_buffer_overwrite(in_floats.len());
                buf.copy_from_slice(&in_floats);
                buf
            };
            if *slope == 0.0 {
                apply_activation_slice(&mut out_data, FusedActivation::Relu);
            } else {
                for v in &mut out_data {
                    if *v < 0.0 {
                        *v *= *slope;
                    }
                }
            }
            let out_tensor = Tensor::from_floats(&in_shape, out_data, in_layout);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Relu6 { min_val, max_val } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tid = in_indices[0];
            let out_idx = out_indices[0];
            let (in_shape, in_layout) = {
                let t = ctx.get_tensor(in_tid)?;
                (t.shape().to_vec(), t.layout())
            };

            let mut out_data = if let Some(buf) =
                ctx.take_unique_dead_float_buffer(in_tid, op_idx, last_use)
            {
                buf
            } else {
                let in_floats = ctx.get_tensor(in_tid)?.as_f32_cow();
                let mut buf = alloc_tensor_buffer_overwrite(in_floats.len());
                buf.copy_from_slice(&in_floats);
                buf
            };
            if *min_val == 0.0 && *max_val == 6.0 {
                apply_activation_slice(&mut out_data, FusedActivation::Relu6);
            } else {
                for v in &mut out_data {
                    *v = v.max(*min_val).min(*max_val);
                }
            }
            let out_tensor = Tensor::from_floats(&in_shape, out_data, in_layout);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::PRelu { slopes } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_floats = in_tensor.as_f32_cow();
            let in_shape = in_tensor.shape();
            let mut out_data = alloc_tensor_buffer(in_floats.len());

            let slope_floats = slopes.clone().or_else(|| {
                if in_indices.len() > 1 {
                    ctx.get_tensor(in_indices[1]).ok().map(|t| t.to_f32_vec())
                } else {
                    None
                }
            });

            if let Some(slopes) = slope_floats {
                if slopes.len() == 1 {
                    let s = slopes[0];
                    for (i, &x) in in_floats.iter().enumerate() {
                        out_data[i] = if x >= 0.0 { x } else { x * s };
                    }
                } else if in_shape.len() >= 2 && slopes.len() == in_shape[1] {
                    let c_dim = in_shape[1];
                    let spatial: usize = in_shape[2..].iter().product();
                    let cs = c_dim * spatial;
                    for b in 0..in_shape[0] {
                        for c in 0..c_dim {
                            let s = slopes[c];
                            let off = b * cs + c * spatial;
                            for i in 0..spatial {
                                let x = in_floats[off + i];
                                out_data[off + i] = if x >= 0.0 { x } else { x * s };
                            }
                        }
                    }
                } else {
                    let n = slopes.len().max(1);
                    for (i, &x) in in_floats.iter().enumerate() {
                        let s = slopes[i % n];
                        out_data[i] = if x >= 0.0 { x } else { x * s };
                    }
                }
            } else {
                for (i, &x) in in_floats.iter().enumerate() {
                    out_data[i] = if x >= 0.0 { x } else { 0.0 };
                }
            }

            let out_tensor = Tensor::from_floats(in_shape, out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Reduction { op, axes, keep_dims } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let mut actual_axes = axes.clone();
            if actual_axes.is_empty() && in_indices.len() > 1 {
                if let Ok(axes_t) = ctx.get_tensor(in_indices[1]) {
                    actual_axes = axes_t.to_i32_vec();
                }
            }

            let in_floats = in_tensor.as_f32_cow();
            let mut out_shape = Vec::new();
            let mut out_data = Vec::new();
            reduce(
                &in_floats,
                in_tensor.shape(),
                &actual_axes,
                *keep_dims,
                *op,
                &mut out_shape,
                &mut out_data,
            );

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Interp(p) => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let batch = if in_shape.is_empty() { 1 } else { in_shape[0] };
            let channels = if in_shape.len() > 1 { in_shape[1] } else { 1 };
            let in_h = if in_shape.len() > 2 { in_shape[2] } else { 1 };
            let in_w = if in_shape.len() > 3 { in_shape[3] } else { 1 };

            let mut out_h = p.output_h.max(0) as usize;
            let mut out_w = p.output_w.max(0) as usize;

            if (out_h == 0 || out_w == 0) && in_indices.len() > 1 {
                for &inp_idx in in_indices.iter().skip(1) {
                    if let Ok(shape_t) = ctx.get_tensor(inp_idx) {
                        let floats = shape_t.to_f32_vec();
                        let ints = shape_t.to_i32_vec();
                        if shape_t.is_empty() {
                            continue;
                        }
                        if floats.len() >= 4 {
                            if floats[1] > 1.0 || shape_t.dtype() == crate::tensor::DataType::Int32 {
                                out_h = (ints[2].max(floats[2].round() as i32)).max(1) as usize;
                                out_w = (ints[3].max(floats[3].round() as i32)).max(1) as usize;
                                break;
                            } else {
                                let scale_h = floats[2];
                                let scale_w = floats[3];
                                out_h = ((in_h as f32) * scale_h).round() as usize;
                                out_w = ((in_w as f32) * scale_w).round() as usize;
                                break;
                            }
                        } else if floats.len() >= 2 {
                            if floats[0] > 10.0 || floats[1] > 10.0 || shape_t.dtype() == crate::tensor::DataType::Int32 {
                                out_h = (ints[0].max(floats[0].round() as i32)).max(1) as usize;
                                out_w = (ints[1].max(floats[1].round() as i32)).max(1) as usize;
                                break;
                            } else {
                                let scale_h = floats[0];
                                let scale_w = floats[1];
                                out_h = ((in_h as f32) * scale_h).round() as usize;
                                out_w = ((in_w as f32) * scale_w).round() as usize;
                                break;
                            }
                        }
                    }
                }
            }

            if out_h == 0 {
                let scale_h = if p.height_scale > 0.0 { p.height_scale } else { 1.0 };
                out_h = ((in_h as f32) * scale_h).round() as usize;
            }
            if out_w == 0 {
                let scale_w = if p.width_scale > 0.0 { p.width_scale } else { 1.0 };
                out_w = ((in_w as f32) * scale_w).round() as usize;
            }

            let cfg = InterpConfig {
                resize_type: p.resize_type,
                align_corners: p.align_corners,
                half_pixel_centers: p.half_pixel_centers,
            };

            let in_floats = in_tensor.as_f32_cow();
            let mut out_data = alloc_tensor_buffer(batch * channels * out_h * out_w);
            interp2d(
                &in_floats,
                batch,
                channels,
                in_h,
                in_w,
                out_h,
                out_w,
                &mut out_data,
                &cfg,
            );

            let out_tensor = Tensor::from_floats(&[batch, channels, out_h, out_w], out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Reshape { dims } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let target_dims: Vec<i32> = if in_indices.len() > 1 {
                let t = ctx.get_tensor(in_indices[1])?;
                t.to_i32_vec()
            } else if let Some(d) = dims {
                d.clone()
            } else {
                in_tensor.shape().iter().map(|&x| x as i32).collect()
            };

            let resolved_shape = compute_reshape_dims(in_tensor.shape(), &target_dims)
                .map_err(MnnError::ExecutionError)?;

            let mut out_t = ctx.get_tensor_mut(in_indices[0])?.share();
            out_t.set_shape(&resolved_shape);
            ctx.set_tensor(out_idx, out_t);
            Ok(())
        }

        OpKind::Permute { dims } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let default_dims;
            let effective_dims = if dims.is_empty() {
                default_dims = (0..in_shape.len()).rev().collect::<Vec<_>>();
                &default_dims
            } else {
                dims
            };

            let in_floats = in_tensor.as_f32_cow();
            let total_elems: usize = in_shape.iter().product();
            let mut out_data = alloc_tensor_buffer(total_elems.max(in_floats.len()));
            permute(&in_floats, in_shape, effective_dims, &mut out_data);

            let out_shape: Vec<usize> = effective_dims.iter().map(|&d| in_shape[d]).collect();
            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Concat { axis } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let out_idx = out_indices[0];

            let mut in_shapes: Vec<Vec<usize>> = Vec::new();
            for &idx in in_indices.iter() {
                in_shapes.push(ctx.get_tensor(idx)?.shape().to_vec());
            }

            let rank = in_shapes.first().map(|s| s.len()).unwrap_or(1);
            let mut actual_axis = if *axis < 0 {
                ((rank as i32 + *axis).max(0)) as usize
            } else {
                *axis as usize
            };

            let max_dim = in_shapes.iter().map(|s| s.len()).max().unwrap_or(1);
            let min_dim = in_shapes.iter().map(|s| s.len()).min().unwrap_or(1);
            if max_dim != min_dim {
                let all_vector_like = in_shapes.iter().all(|s| s.iter().filter(|&&x| x > 1).count() <= 1);
                if all_vector_like {
                    for s in &mut in_shapes {
                        let total: usize = s.iter().product();
                        *s = vec![total];
                    }
                    actual_axis = 0;
                }
            }

            let mut flat_buffers = Vec::new();
            for &idx in in_indices.iter() {
                let t = ctx.get_tensor(idx)?;
                flat_buffers.push(t.as_f32_cow());
            }

            let mut in_data_shapes = Vec::new();
            for (i, _) in in_indices.iter().enumerate() {
                in_data_shapes.push((flat_buffers[i].as_ref(), in_shapes[i].as_slice()));
            }

            let mut out_shape = in_data_shapes[0].1.to_vec();
            let mut axis_sum = 0;
            for (_, s) in &in_data_shapes {
                axis_sum += s[actual_axis];
            }
            out_shape[actual_axis] = axis_sum;

            let out_len = out_shape.iter().product::<usize>();
            let mut out_data = alloc_tensor_buffer(out_len);
            concat(&in_data_shapes, actual_axis, &mut out_data);

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Slice { axis, slice_points } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let (in_shape, in_floats) = {
                let in_t = ctx.get_tensor(in_indices[0])?;
                (in_t.shape().to_vec(), in_t.to_f32_vec())
            };
            let ndim = in_shape.len();
            let raw_axis = *axis;
            let actual_axis = if raw_axis < 0 {
                (raw_axis + ndim as i32) as usize
            } else {
                raw_axis as usize
            };

            let num_outputs = out_indices.len();
            let mut split_sizes: Vec<usize> = Vec::new();

            if in_indices.len() > 1 {
                let p_vec = ctx.get_tensor(in_indices[1])?.to_i32_vec();
                split_sizes = p_vec.iter().map(|&x| if x < 0 { usize::MAX } else { x as usize }).collect();
            } else if !slice_points.is_empty() {
                split_sizes = slice_points.iter().map(|&x| if x < 0 { usize::MAX } else { x as usize }).collect();
            }

            if split_sizes.is_empty() {
                let split_dim = in_shape[actual_axis] / num_outputs;
                split_sizes = vec![split_dim; num_outputs];
            } else {
                let sum_known: usize = split_sizes.iter().filter(|&&s| s != usize::MAX).sum();
                for s in &mut split_sizes {
                    if *s == usize::MAX {
                        *s = in_shape[actual_axis].saturating_sub(sum_known);
                    }
                }
            }

            let outside: usize = in_shape[..actual_axis].iter().product();
            let inside: usize = in_shape[actual_axis + 1..].iter().product();
            let axis_len = in_shape[actual_axis];

            let mut offset = 0;
            for (i, &out_idx) in out_indices.iter().enumerate() {
                let cur_size = split_sizes.get(i).copied().unwrap_or(0);
                let mut out_shape = in_shape.clone();
                out_shape[actual_axis] = cur_size;

                let out_len = outside * cur_size * inside;
                let mut out_data = vec![0.0f32; out_len];

                for o in 0..outside {
                    for a in 0..cur_size {
                        let src_a = offset + a;
                        let src_idx = (o * axis_len + src_a) * inside;
                        let dst_idx = (o * cur_size + a) * inside;
                        out_data[dst_idx..dst_idx + inside]
                            .copy_from_slice(&in_floats[src_idx..src_idx + inside]);
                    }
                }

                offset += cur_size;
                let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
                ctx.set_tensor(out_idx, out_tensor);
            }
            Ok(())
        }

        OpKind::SliceTf => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let ndim = in_shape.len();

            let mut begins: Vec<usize> = if in_indices.len() > 1 {
                ctx.get_tensor(in_indices[1])?
                    .to_i32_vec()
                    .iter()
                    .map(|&b| b.max(0) as usize)
                    .collect()
            } else {
                vec![0; ndim]
            };
            while begins.len() < ndim {
                begins.push(0);
            }

            let mut sizes: Vec<usize> = if in_indices.len() > 2 {
                ctx.get_tensor(in_indices[2])?
                    .to_i32_vec()
                    .iter()
                    .enumerate()
                    .map(|(d, &s)| {
                        if s < 0 {
                            in_shape.get(d).copied().unwrap_or(1).saturating_sub(begins[d])
                        } else {
                            s as usize
                        }
                    })
                    .collect()
            } else {
                in_shape.to_vec()
            };
            while sizes.len() < ndim {
                sizes.push(1);
            }

            let in_floats = in_tensor.as_f32_cow();
            let out_len = sizes.iter().product::<usize>();
            let mut out_data = alloc_tensor_buffer(out_len);

            slice_tf(&in_floats, in_shape, &begins, &sizes, &mut out_data);

            let out_tensor = Tensor::from_floats(&sizes, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::StridedSlice {
            begin_mask: _,
            end_mask: _,
            ellipsis_mask: _,
            new_axis_mask: _,
            shrink_axis_mask: _,
            starts,
            ends,
            axes,
            strides,
        } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let ndim = in_shape.len();

            let starts_vec = if in_indices.len() > 1 {
                ctx.get_tensor(in_indices[1])?.to_i32_vec()
            } else if let Some(ref s) = starts {
                s.clone()
            } else {
                vec![0; ndim]
            };
            let ends_vec = if in_indices.len() > 2 {
                ctx.get_tensor(in_indices[2])?.to_i32_vec()
            } else if let Some(ref e) = ends {
                e.clone()
            } else {
                in_shape.iter().map(|&x| x as i32).collect()
            };
            let axes_vec = if in_indices.len() > 3 {
                ctx.get_tensor(in_indices[3])?.to_i32_vec()
            } else if let Some(ref a) = axes {
                a.clone()
            } else {
                (0..starts_vec.len() as i32).collect()
            };
            let strides_vec = if in_indices.len() > 4 {
                ctx.get_tensor(in_indices[4])?.to_i32_vec()
            } else if let Some(ref st) = strides {
                st.clone()
            } else {
                vec![1; starts_vec.len()]
            };

            let mut begins = vec![0i32; ndim];
            let mut ends = in_shape.iter().map(|&x| x as i32).collect::<Vec<_>>();
            let mut strides = vec![1i32; ndim];

            for (i, &ax) in axes_vec.iter().enumerate() {
                let actual_ax = if ax < 0 {
                    (ndim as i32 + ax) as usize
                } else {
                    ax as usize
                };
                if actual_ax < ndim {
                    if i < starts_vec.len() {
                        begins[actual_ax] = starts_vec[i];
                    }
                    if i < ends_vec.len() {
                        ends[actual_ax] = ends_vec[i];
                    }
                    if i < strides_vec.len() {
                        strides[actual_ax] = strides_vec[i];
                    }
                }
            }

            let in_floats = in_tensor.as_f32_cow();
            let mut out_shape = Vec::new();
            let mut out_data = Vec::new();

            strided_slice(
                &in_floats,
                in_tensor.shape(),
                &begins,
                &ends,
                &strides,
                &mut out_shape,
                &mut out_data,
            );

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Squeeze { axes } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let mut out_shape = Vec::new();

            let target_axes: Vec<i32> = if !axes.is_empty() {
                axes.clone()
            } else if in_indices.len() > 1 {
                if let Ok(ax_t) = ctx.get_tensor(in_indices[1]) {
                    ax_t.to_i32_vec()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            if target_axes.is_empty() {
                for &dim in in_shape {
                    if dim != 1 {
                        out_shape.push(dim);
                    }
                }
            } else {
                let ndim = in_shape.len() as i32;
                let norm_axes: Vec<usize> = target_axes
                    .iter()
                    .map(|&ax| if ax < 0 { (ax + ndim) as usize } else { ax as usize })
                    .collect();

                for (d, &dim) in in_shape.iter().enumerate() {
                    if !norm_axes.contains(&d) || dim != 1 {
                        out_shape.push(dim);
                    }
                }
            }

            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let mut out_t = ctx.get_tensor_mut(in_indices[0])?.share();
            out_t.set_shape(&out_shape);
            ctx.set_tensor(out_idx, out_t);
            Ok(())
        }

        OpKind::Unsqueeze { axes } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let target_axes: Vec<i32> = if !axes.is_empty() {
                axes.clone()
            } else if in_indices.len() > 1 {
                if let Ok(ax_t) = ctx.get_tensor(in_indices[1]) {
                    ax_t.to_i32_vec()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let in_shape = in_tensor.shape();
            let target_ndim = in_shape.len() + target_axes.len();
            let mut out_shape = Vec::with_capacity(target_ndim);

            let mut norm_axes: Vec<usize> = target_axes
                .iter()
                .map(|&ax| {
                    if ax < 0 {
                        (ax + target_ndim as i32) as usize
                    } else {
                        ax as usize
                    }
                })
                .collect();
            norm_axes.sort_unstable();

            let mut in_d = 0;
            for d in 0..target_ndim {
                if norm_axes.contains(&d) {
                    out_shape.push(1);
                } else if in_d < in_shape.len() {
                    out_shape.push(in_shape[in_d]);
                    in_d += 1;
                }
            }

            let mut out_t = ctx.get_tensor_mut(in_indices[0])?.share();
            out_t.set_shape(&out_shape);
            ctx.set_tensor(out_idx, out_t);
            Ok(())
        }

        OpKind::Scale { scale, bias } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let (scale_cow, bias_cow) = if in_indices.len() > 1 {
                let s_t = ctx.get_tensor(in_indices[1])?;
                let b_t = if in_indices.len() > 2 {
                    Some(ctx.get_tensor(in_indices[2])?.as_f32_cow())
                } else {
                    bias.as_ref().map(|b| std::borrow::Cow::Borrowed(b.as_slice()))
                };
                (s_t.as_f32_cow(), b_t)
            } else {
                (
                    std::borrow::Cow::Borrowed(scale.as_slice()),
                    bias.as_ref().map(|b| std::borrow::Cow::Borrowed(b.as_slice())),
                )
            };

            let in_shape = in_tensor.shape();
            let channels = if in_shape.len() > 1 { in_shape[1].max(1) } else { 1 };
            let spatial: usize = in_shape.iter().skip(2).product::<usize>().max(1);
            let in_floats = in_tensor.as_f32_cow();
            let mut out_data = alloc_tensor_buffer_overwrite(in_floats.len());

            let full_planes = in_floats.len() / spatial;
            for bc in 0..full_planes {
                let c = bc % channels;
                let s = scale_cow.get(c).copied().unwrap_or(1.0);
                let b = bias_cow.as_ref().and_then(|bs| bs.get(c).copied()).unwrap_or(0.0);
                let start = bc * spatial;
                let in_plane = &in_floats[start..start + spatial];
                let out_plane = &mut out_data[start..start + spatial];
                for i in 0..spatial {
                    out_plane[i] = in_plane[i] * s + b;
                }
            }
            for i in (full_planes * spatial)..in_floats.len() {
                let c = (i / spatial) % channels;
                let s = scale_cow.get(c).copied().unwrap_or(1.0);
                let b = bias_cow.as_ref().and_then(|bs| bs.get(c).copied()).unwrap_or(0.0);
                out_data[i] = in_floats[i] * s + b;
            }

            let out_tensor = Tensor::from_floats(in_shape, out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::DepthToSpace { block_size, mode } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            if in_shape.len() == 4 {
                let b = in_shape[0];
                let c = in_shape[1];
                let h = in_shape[2];
                let w = in_shape[3];
                let bs = *block_size;

                let out_c = c / (bs * bs);
                let out_h = h * bs;
                let out_w = w * bs;

                let in_floats = in_tensor.as_f32_cow();
                let mut out_data = alloc_tensor_buffer(b * out_c * out_h * out_w);

                for batch_idx in 0..b {
                    for oc in 0..out_c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let ih = oh / bs;
                                let iw = ow / bs;
                                let rh = oh % bs;
                                let rw = ow % bs;

                                let ic = if *mode == 1 {
                                    (oc * bs + rh) * bs + rw
                                } else {
                                    oc + (rh * bs + rw) * out_c
                                };

                                let in_offset = ((batch_idx * c + ic) * h + ih) * w + iw;
                                let out_offset = ((batch_idx * out_c + oc) * out_h + oh) * out_w + ow;

                                out_data[out_offset] = in_floats[in_offset];
                            }
                        }
                    }
                }

                let out_tensor = Tensor::from_floats(&[b, out_c, out_h, out_w], out_data, TensorLayout::NCHW);
                ctx.set_tensor(out_idx, out_tensor);
            }
            Ok(())
        }

        OpKind::Flatten { axis, end_axis } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let dim = in_shape.len() as i32;
            let mut ax = *axis;
            if ax < 0 {
                ax += dim;
            }
            let mut end_ax = *end_axis;
            if end_ax < 0 {
                end_ax += dim;
            }
            let actual_axis = (ax as usize).min(in_shape.len());

            let new_shape = if end_ax == 0 {
                let d0: usize = in_shape[..actual_axis].iter().product::<usize>().max(1);
                let d1: usize = in_shape[actual_axis..].iter().product::<usize>().max(1);
                vec![d0, d1]
            } else {
                let end_idx = (end_ax as usize).min(in_shape.len() - 1);
                let mut ns = Vec::new();
                for i in 0..actual_axis {
                    ns.push(in_shape[i]);
                }
                let middle: usize = in_shape[actual_axis..=end_idx].iter().product::<usize>().max(1);
                ns.push(middle);
                for i in (end_idx + 1)..in_shape.len() {
                    ns.push(in_shape[i]);
                }
                ns
            };
            let mut out_t = ctx.get_tensor_mut(in_indices[0])?.share();
            out_t.set_shape(&new_shape);
            ctx.set_tensor(out_idx, out_t);
            Ok(())
        }

        OpKind::Softmax { axis } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let in_floats = in_tensor.as_f32_cow();
            let mut out_data = alloc_tensor_buffer_overwrite(in_floats.len());

            let raw_axis = *axis;
            let actual_axis = if raw_axis < 0 {
                (raw_axis + in_shape.len() as i32) as usize
            } else {
                raw_axis as usize
            };

            softmax(&in_floats, in_shape, actual_axis as isize, &mut out_data);

            let out_tensor = Tensor::from_floats(in_shape, out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::LogSoftmax { axis } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let in_floats = in_tensor.as_f32_cow();
            let mut out_data = alloc_tensor_buffer_overwrite(in_floats.len());

            let ndim = in_shape.len();
            let raw_axis = *axis;
            let actual_axis = if raw_axis < 0 {
                (ndim as i32 + raw_axis).max(0) as usize
            } else {
                raw_axis as usize
            };

            log_softmax(&in_floats, in_shape, actual_axis as isize, &mut out_data);

            let out_tensor = Tensor::from_floats(in_shape, out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::ArgMax { axis, out_max_val: _ } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let in_floats = in_tensor.as_f32_cow();

            let norm_axis = if *axis < 0 {
                (in_shape.len() as isize + *axis as isize) as usize
            } else {
                *axis as usize
            };

            let mut out_shape = in_shape.to_vec();
            if norm_axis < out_shape.len() {
                out_shape.remove(norm_axis);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            let total_len: usize = out_shape.iter().product();
            let mut out_indices_vec = vec![0i32; total_len];

            argmax(
                &in_floats,
                in_shape,
                *axis as isize,
                &mut out_indices_vec,
            );

            let out_tensor = Tensor::from_ints(&out_shape, out_indices_vec);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::TopKV2 { k, axis: _, largest: _, sorted: _ } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;

            let mut k_val = *k;
            if in_indices.len() > 1 {
                if let Ok(k_t) = ctx.get_tensor(in_indices[1]) {
                    let ints = k_t.to_i32_vec();
                    if !ints.is_empty() && ints[0] > 0 {
                        k_val = ints[0] as usize;
                    }
                }
            }

            let in_shape = in_tensor.shape();
            let in_floats = in_tensor.as_f32_cow();

            let k_dim = in_shape.last().copied().unwrap_or(1);
            let outer_size: usize = in_shape[..in_shape.len().saturating_sub(1)].iter().product::<usize>().max(1);

            let mut out_values = vec![0.0f32; outer_size * k_val];
            let mut out_indices_data = vec![0i32; outer_size * k_val];

            topk(
                &in_floats,
                outer_size,
                k_dim,
                k_val,
                &mut out_values,
                &mut out_indices_data,
            );

            let mut out_shape = in_shape[..in_shape.len().saturating_sub(1)].to_vec();
            out_shape.push(k_val);

            if !out_indices.is_empty() {
                let val_tensor = Tensor::from_floats(&out_shape, out_values, TensorLayout::NCHW);
                ctx.set_tensor(out_indices[0], val_tensor);
            }
            if out_indices.len() > 1 {
                let idx_tensor = Tensor::from_ints(&out_shape, out_indices_data);
                ctx.set_tensor(out_indices[1], idx_tensor);
            }
            Ok(())
        }

        OpKind::Pad { pad_value, is_interleaved } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let in_shape = in_tensor.shape();
            let ndim = in_shape.len();
            let mut paddings = vec![0usize; ndim * 2];
            if in_indices.len() > 1 {
                let pad_t = ctx.get_tensor(in_indices[1])?;
                let ints = pad_t.to_i32_vec();
                let p_shape = pad_t.shape();
                if *is_interleaved || (p_shape.len() == 2 && p_shape[1] == 2) {
                    // MNN layout: [ndim, 2] or interleaved [before_0, after_0, before_1, after_1, ...]
                    for (i, &v) in ints.iter().take(ndim * 2).enumerate() {
                        paddings[i] = v.max(0) as usize;
                    }
                } else if ints.len() >= ndim * 2 {
                    // ONNX / RTen layout: [2 * ndim] -> [begin_0, ..., begin_n-1, end_0, ..., end_n-1]
                    for d in 0..ndim {
                        paddings[2 * d] = ints[d].max(0) as usize;
                        paddings[2 * d + 1] = ints[ndim + d].max(0) as usize;
                    }
                } else {
                    for (i, &v) in ints.iter().take(ndim * 2).enumerate() {
                        paddings[i] = v.max(0) as usize;
                    }
                }
            }

            let mut out_shape = Vec::with_capacity(ndim);
            for d in 0..ndim {
                out_shape.push(in_shape[d] + paddings[2 * d] + paddings[2 * d + 1]);
            }
            let out_len: usize = out_shape.iter().product();
            let in_floats = in_tensor.as_f32_cow();
            let mut out_data = alloc_tensor_buffer(out_len);

            pad_tensor(
                &in_floats,
                in_shape,
                &paddings,
                crate::schema::PadValueMode::CONSTANT,
                *pad_value,
                &mut out_data,
            );

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Shape => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let shape_ints: Vec<i32> = in_tensor.shape().iter().map(|&x| x as i32).collect();
            let out_tensor = Tensor::from_ints(&[shape_ints.len()], shape_ints);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Rank => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let out_idx = out_indices[0];

            let rank = in_tensor.shape().len() as i32;
            let out_tensor = Tensor::from_ints(&[1], vec![rank]);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Range => {
            if in_indices.len() < 3 || out_indices.is_empty() {
                return Ok(());
            }
            let start = ctx.get_tensor(in_indices[0])?.to_f32_vec().first().copied().unwrap_or(0.0);
            let limit = ctx.get_tensor(in_indices[1])?.to_f32_vec().first().copied().unwrap_or(0.0);
            let delta = ctx.get_tensor(in_indices[2])?.to_f32_vec().first().copied().unwrap_or(1.0);
            let out_idx = out_indices[0];

            let mut values = Vec::new();
            let mut cur = start;
            if delta > 0.0 {
                while cur < limit {
                    values.push(cur);
                    cur += delta;
                }
            } else if delta < 0.0 {
                while cur > limit {
                    values.push(cur);
                    cur += delta;
                }
            }

            let len = values.len();
            let out_tensor = Tensor::from_floats(&[len], values, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::ConstantOfShape { value } => {
            if out_indices.is_empty() {
                return Ok(());
            }
            let out_idx = out_indices[0];
            let out_shape = if !in_indices.is_empty() {
                let shape_tensor = ctx.get_tensor(in_indices[0])?;
                shape_tensor.to_i32_vec().iter().map(|&x| x.max(0) as usize).collect::<Vec<_>>()
            } else {
                vec![1]
            };
            let out_len: usize = out_shape.iter().product();
            let out_tensor = Tensor::from_floats(&out_shape, vec![*value; out_len], TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Expand => {
            if in_indices.len() < 2 || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let shape_tensor = ctx.get_tensor(in_indices[1])?;
            let out_idx = out_indices[0];

            let target_shape: Vec<usize> = shape_tensor
                .to_i32_vec()
                .iter()
                .map(|&x| x.max(1) as usize)
                .collect();

            let in_floats = in_tensor.as_f32_cow();
            let max_ndim = target_shape.len().max(in_tensor.shape().len());
            let mut resolved_out = vec![1usize; max_ndim];
            let out_offset = max_ndim - target_shape.len();
            for (i, &dim) in target_shape.iter().enumerate() {
                resolved_out[out_offset + i] = dim;
            }
            let in_offset = max_ndim - in_tensor.shape().len();
            for (i, &dim) in in_tensor.shape().iter().enumerate() {
                resolved_out[in_offset + i] = resolved_out[in_offset + i].max(dim);
            }
            let out_len: usize = resolved_out.iter().product();

            let mut out_data = alloc_tensor_buffer(out_len);
            let final_shape = crate::kernels::tensor_ops::expand(
                &in_floats,
                in_tensor.shape(),
                &target_shape,
                &mut out_data,
            );

            let out_tensor = Tensor::from_floats(&final_shape, out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::ScatterND => {
            if in_indices.len() < 3 || out_indices.is_empty() {
                return Ok(());
            }
            let data_tensor = ctx.get_tensor(in_indices[0])?;
            let indices_tensor = ctx.get_tensor(in_indices[1])?;
            let updates_tensor = ctx.get_tensor(in_indices[2])?;
            let out_idx = out_indices[0];

            let data_floats = data_tensor.as_f32_cow();
            let updates_floats = updates_tensor.as_f32_cow();
            let indices = indices_tensor.to_i32_vec();

            let mut out_data = alloc_tensor_buffer(data_floats.len());
            crate::kernels::tensor_ops::scatter_nd(
                &data_floats,
                data_tensor.shape(),
                &indices,
                indices_tensor.shape(),
                &updates_floats,
                &mut out_data,
            );

            let out_tensor = Tensor::from_floats(data_tensor.shape(), out_data, data_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::LayerNorm(p) => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let in_shape = in_tensor.shape();
            let in_floats = in_tensor.as_f32_cow();
            let out_idx = out_indices[0];

            let ndim = in_shape.len();
            let actual_axis = if p.axis < 0 {
                ((ndim as i32 + p.axis).max(0) as usize).min(ndim)
            } else {
                (p.axis as usize).min(ndim)
            };

            let outer_count: usize = in_shape[..actual_axis].iter().product::<usize>().max(1);
            let inner_count: usize = in_shape[actual_axis..].iter().product::<usize>().max(1);

            let gamma = p.gamma.clone().or_else(|| {
                if in_indices.len() > 1 {
                    ctx.get_tensor(in_indices[1]).ok().map(|t| t.to_f32_vec())
                } else {
                    None
                }
            });

            let beta = p.beta.clone().or_else(|| {
                if in_indices.len() > 2 {
                    ctx.get_tensor(in_indices[2]).ok().map(|t| t.to_f32_vec())
                } else {
                    None
                }
            });

            let mut out_data = alloc_tensor_buffer(in_floats.len());

            for outer in 0..outer_count {
                let off = outer * inner_count;
                let slice = &in_floats[off..off + inner_count];
                let out_slice = &mut out_data[off..off + inner_count];

                let (mean, var) = if p.use_rms_norm {
                    let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
                    (0.0f32, sum_sq / inner_count as f32)
                } else {
                    let sum: f32 = slice.iter().sum();
                    let m = sum / inner_count as f32;
                    let var: f32 = slice.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / inner_count as f32;
                    (m, var)
                };

                let inv_std = 1.0 / (var + p.epsilon).sqrt();

                for i in 0..inner_count {
                    let norm = (slice[i] - mean) * inv_std;
                    let scaled = if let Some(ref g) = gamma {
                        norm * g[i % g.len()]
                    } else {
                        norm
                    };
                    let shifted = if let Some(ref b) = beta {
                        scaled + b[i % b.len()]
                    } else {
                        scaled
                    };
                    out_slice[i] = shifted;
                }
            }

            let out_tensor = Tensor::from_floats(in_shape, out_data, in_tensor.layout());
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Select => {
            if in_indices.len() < 3 || out_indices.is_empty() {
                return Ok(());
            }
            let cond_tensor = ctx.get_tensor(in_indices[0])?;
            let true_tensor = ctx.get_tensor(in_indices[1])?;
            let false_tensor = ctx.get_tensor(in_indices[2])?;
            let out_idx = out_indices[0];

            let cond_shape = cond_tensor.shape();
            let t_shape = true_tensor.shape();
            let f_shape = false_tensor.shape();

            let ndim = cond_shape.len().max(t_shape.len()).max(f_shape.len());
            let mut out_shape = vec![1usize; ndim];
            for d in 0..ndim {
                let c = if d + cond_shape.len() >= ndim { cond_shape[d + cond_shape.len() - ndim] } else { 1 };
                let t = if d + t_shape.len() >= ndim { t_shape[d + t_shape.len() - ndim] } else { 1 };
                let f = if d + f_shape.len() >= ndim { f_shape[d + f_shape.len() - ndim] } else { 1 };
                out_shape[d] = c.max(t).max(f);
            }

            let cond_floats = cond_tensor.as_f32_cow();
            let true_floats = true_tensor.as_f32_cow();
            let false_floats = false_tensor.as_f32_cow();

            let total_len: usize = out_shape.iter().product();
            if total_len == 0 {
                let out_tensor = Tensor::from_floats(&out_shape, Vec::new(), TensorLayout::NCHW);
                ctx.set_tensor(out_idx, out_tensor);
                return Ok(());
            }
            let mut out_data = alloc_tensor_buffer(total_len);

            let t_is_scalar = true_floats.len() == 1;
            let f_is_scalar = false_floats.len() == 1;
            let t_matches = true_floats.len() == total_len;
            let f_matches = false_floats.len() == total_len;

            if (t_matches || t_is_scalar) && (f_matches || f_is_scalar) {
                let cond_pad = ndim - cond_shape.len();
                let leading_ones = out_shape[..cond_pad].iter().all(|&d| d == 1);
                if leading_ones {
                    let mut prefix_end = cond_shape.len();
                    while prefix_end > 0 && cond_shape[prefix_end - 1] == 1 {
                        prefix_end -= 1;
                    }
                    let prefix_matches = (0..prefix_end).all(|d| cond_shape[d] == out_shape[cond_pad + d]);
                    if prefix_matches {
                        let outer = cond_floats.len();
                        let inner = total_len / outer.max(1);
                        if outer * inner == total_len {
                            if inner == 1 {
                                match (t_is_scalar, f_is_scalar) {
                                    (false, false) => {
                                        for i in 0..total_len {
                                            out_data[i] = if cond_floats[i] != 0.0 { true_floats[i] } else { false_floats[i] };
                                        }
                                    }
                                    (true, false) => {
                                        let tv = true_floats[0];
                                        for i in 0..total_len {
                                            out_data[i] = if cond_floats[i] != 0.0 { tv } else { false_floats[i] };
                                        }
                                    }
                                    (false, true) => {
                                        let fv = false_floats[0];
                                        for i in 0..total_len {
                                            out_data[i] = if cond_floats[i] != 0.0 { true_floats[i] } else { fv };
                                        }
                                    }
                                    (true, true) => {
                                        let tv = true_floats[0];
                                        let fv = false_floats[0];
                                        for i in 0..total_len {
                                            out_data[i] = if cond_floats[i] != 0.0 { tv } else { fv };
                                        }
                                    }
                                }
                            } else {
                                for o in 0..outer {
                                    let dst = &mut out_data[o * inner..(o + 1) * inner];
                                    if cond_floats[o] != 0.0 {
                                        if t_is_scalar {
                                            dst.fill(true_floats[0]);
                                        } else {
                                            dst.copy_from_slice(&true_floats[o * inner..(o + 1) * inner]);
                                        }
                                    } else if f_is_scalar {
                                        dst.fill(false_floats[0]);
                                    } else {
                                        dst.copy_from_slice(&false_floats[o * inner..(o + 1) * inner]);
                                    }
                                }
                            }
                            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
                            ctx.set_tensor(out_idx, out_tensor);
                            return Ok(());
                        }
                    }
                }
            }

            let mut coord = vec![0usize; ndim];
            for i in 0..total_len {
                let mut c_idx = 0;
                let mut c_stride = 1;
                for d in (0..cond_shape.len()).rev() {
                    let out_d = ndim - cond_shape.len() + d;
                    let pos = if cond_shape[d] == 1 { 0 } else { coord[out_d] };
                    c_idx += pos * c_stride;
                    c_stride *= cond_shape[d];
                }

                let mut t_idx = 0;
                let mut t_stride = 1;
                for d in (0..t_shape.len()).rev() {
                    let out_d = ndim - t_shape.len() + d;
                    let pos = if t_shape[d] == 1 { 0 } else { coord[out_d] };
                    t_idx += pos * t_stride;
                    t_stride *= t_shape[d];
                }

                let mut f_idx = 0;
                let mut f_stride = 1;
                for d in (0..f_shape.len()).rev() {
                    let out_d = ndim - f_shape.len() + d;
                    let pos = if f_shape[d] == 1 { 0 } else { coord[out_d] };
                    f_idx += pos * f_stride;
                    f_stride *= f_shape[d];
                }

                let condition = cond_floats.get(c_idx).copied().unwrap_or(0.0) != 0.0;
                out_data[i] = if condition {
                    true_floats.get(t_idx).copied().unwrap_or(0.0)
                } else {
                    false_floats.get(f_idx).copied().unwrap_or(0.0)
                };

                for d in (0..ndim).rev() {
                    coord[d] += 1;
                    if coord[d] < out_shape[d] {
                        break;
                    }
                    coord[d] = 0;
                }
            }
            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Gather { axis } => {
            if in_indices.len() < 2 || out_indices.is_empty() {
                return Ok(());
            }
            let in_tensor = ctx.get_tensor(in_indices[0])?;
            let indices_tensor = ctx.get_tensor(in_indices[1])?;
            let out_idx = out_indices[0];

            let in_rank = in_tensor.shape().len();
            let raw_axis = if in_indices.len() > 2 {
                let ax_vec = ctx.get_tensor(in_indices[2])?.to_i32_vec();
                if !ax_vec.is_empty() { ax_vec[0] } else { *axis }
            } else {
                *axis
            };
            let actual_axis = if raw_axis < 0 {
                ((in_rank as i32 + raw_axis).max(0)) as usize
            } else {
                raw_axis as usize
            };

            let in_floats = in_tensor.as_f32_cow();
            let indices = indices_tensor.to_i32_vec();
            let mut out_shape = Vec::new();
            let mut out_data = Vec::new();

            gather(
                &in_floats,
                in_tensor.shape(),
                &indices,
                indices_tensor.shape(),
                actual_axis,
                &mut out_shape,
                &mut out_data,
            );

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::Cast { dst_type: _ } => {
            if in_indices.is_empty() || out_indices.is_empty() {
                return Ok(());
            }
            let out_idx = out_indices[0];
            let out_t = ctx.get_tensor_mut(in_indices[0])?.share();
            ctx.set_tensor(out_idx, out_t);
            Ok(())
        }

        OpKind::GridSample(p) => {
            if in_indices.len() < 2 || out_indices.is_empty() {
                return Ok(());
            }
            let (in_shape, in_floats) = {
                let in_t = ctx.get_tensor(in_indices[0])?;
                (in_t.shape().to_vec(), in_t.to_f32_vec())
            };
            let (grid_shape, grid_floats) = {
                let grid_t = ctx.get_tensor(in_indices[1])?;
                (grid_t.shape().to_vec(), grid_t.to_f32_vec())
            };
            let out_idx = out_indices[0];

            let sample_mode = match p.mode {
                0 => SampleMode::BILINEAR,
                1 => SampleMode::NEAREST,
                _ => SampleMode::BILINEAR,
            };
            let border_mode = match p.padding_mode {
                0 => BorderMode::ZEROS,
                1 => BorderMode::CLAMP,
                2 => BorderMode::REFLECTION,
                _ => BorderMode::ZEROS,
            };

            if in_shape.len() == 4 && grid_shape.len() == 4 {
                let out_shape = vec![in_shape[0], in_shape[1], grid_shape[1], grid_shape[2]];
                let out_len: usize = out_shape.iter().product();
                let mut out_data = vec![0.0f32; out_len];

                grid_sample_2d(
                    &in_floats,
                    &in_shape,
                    &grid_floats,
                    &grid_shape,
                    sample_mode,
                    border_mode,
                    p.align_corners,
                    &mut out_data,
                );

                let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
                ctx.set_tensor(out_idx, out_tensor);
            }
            Ok(())
        }

        OpKind::MatMul(p) => {
            if in_indices.len() < 2 || out_indices.is_empty() {
                return Ok(());
            }
            let a = ctx.get_tensor(in_indices[0])?;
            let b = ctx.get_tensor(in_indices[1])?;
            let out_idx = out_indices[0];

            let a_shape = a.shape();
            let b_shape = b.shape();
            let a_floats = a.as_f32_cow();
            let b_floats = b.as_f32_cow();

            let m = if p.trans_a {
                if a_shape.len() > 1 { a_shape[a_shape.len() - 1] } else { 1 }
            } else {
                if a_shape.len() > 1 { a_shape[a_shape.len() - 2] } else { a_shape[0] }
            };
            let k = if p.trans_a {
                if a_shape.len() > 1 { a_shape[a_shape.len() - 2] } else { a_shape[0] }
            } else {
                if a_shape.len() > 1 { a_shape[a_shape.len() - 1] } else { 1 }
            };
            let n = if p.trans_b {
                if b_shape.len() > 1 { b_shape[b_shape.len() - 2] } else { b_shape[0] }
            } else {
                if b_shape.len() > 1 { b_shape[b_shape.len() - 1] } else { 1 }
            };

            let a_batch_dims = if a_shape.len() > 2 {
                &a_shape[..a_shape.len() - 2]
            } else {
                &[]
            };
            let b_batch_dims = if b_shape.len() > 2 {
                &b_shape[..b_shape.len() - 2]
            } else {
                &[]
            };
            let a_batches: usize = a_batch_dims.iter().product::<usize>().max(1);
            let b_batches: usize = b_batch_dims.iter().product::<usize>().max(1);
            let batch_count = a_batches.max(b_batches);
            let out_batch_dims = if a_batch_dims.len() >= b_batch_dims.len() {
                a_batch_dims
            } else {
                b_batch_dims
            };

            let mut out_shape = out_batch_dims.to_vec();
            out_shape.push(m);
            out_shape.push(n);

            let mut out_data = if p.beta == 0.0 && k > 0 {
                alloc_tensor_buffer_overwrite(batch_count * m * n)
            } else {
                alloc_tensor_buffer(batch_count * m * n)
            };

            if b_batches == 1 && !p.trans_a {
                MATMUL_SCRATCH_B.with(|scratch_cell| {
                    let mut scratch = scratch_cell.borrow_mut();
                    let b_buf: &[f32] = if p.trans_b {
                        let kn = k * n;
                        if scratch.len() < kn {
                            scratch.resize(kn, 0.0);
                        }
                        for r in 0..k {
                            for c in 0..n {
                                scratch[r * n + c] = b_floats[c * k + r];
                            }
                        }
                        &scratch[..kn]
                    } else {
                        &b_floats[..k * n]
                    };

                    crate::kernels::gemm::gemm(
                        batch_count * m,
                        n,
                        k,
                        &a_floats[..batch_count * m * k],
                        b_buf,
                        &mut out_data,
                        p.alpha,
                        p.beta,
                        None,
                        FusedActivation::None,
                    );
                });
            } else {
                MATMUL_SCRATCH_A.with(|a_scratch_cell| {
                    MATMUL_SCRATCH_B.with(|b_scratch_cell| {
                        let mut a_scratch = a_scratch_cell.borrow_mut();
                        let mut b_scratch = b_scratch_cell.borrow_mut();
                        if p.trans_a && a_scratch.len() < m * k {
                            a_scratch.resize(m * k, 0.0);
                        }
                        if p.trans_b && b_scratch.len() < k * n {
                            b_scratch.resize(k * n, 0.0);
                        }

                        for b_i in 0..batch_count {
                            let a_bi = if a_batches == 1 { 0 } else { b_i % a_batches };
                            let b_bi = if b_batches == 1 { 0 } else { b_i % b_batches };
                            let a_raw = &a_floats[a_bi * m * k..(a_bi + 1) * m * k];
                            let b_raw = &b_floats[b_bi * k * n..(b_bi + 1) * k * n];

                            let a_slice: &[f32] = if p.trans_a {
                                for r in 0..m {
                                    for c in 0..k {
                                        a_scratch[r * k + c] = a_raw[c * m + r];
                                    }
                                }
                                &a_scratch[..m * k]
                            } else {
                                a_raw
                            };

                            let b_slice: &[f32] = if p.trans_b {
                                for r in 0..k {
                                    for c in 0..n {
                                        b_scratch[r * n + c] = b_raw[c * k + r];
                                    }
                                }
                                &b_scratch[..k * n]
                            } else {
                                b_raw
                            };

                            let out_slice = &mut out_data[b_i * m * n..(b_i + 1) * m * n];
                            crate::kernels::gemm::gemm(
                                m,
                                n,
                                k,
                                a_slice,
                                b_slice,
                                out_slice,
                                p.alpha,
                                p.beta,
                                None,
                                FusedActivation::None,
                            );
                        }
                    });
                });
            }

            let out_tensor = Tensor::from_floats(&out_shape, out_data, TensorLayout::NCHW);
            ctx.set_tensor(out_idx, out_tensor);
            Ok(())
        }

        OpKind::NoOp => Ok(()),
    }
}

/// Backward compatibility stub for execute_op
pub fn execute_op(
    _op: &Op<'_>,
    _op_idx: usize,
    _prepacked_ops: &[Option<PrepackedOp>],
    _ctx: &mut GraphContext,
) -> Result<(), MnnError> {
    Ok(())
}
