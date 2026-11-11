#[cfg(feature = "parallel")]
use crate::par::prelude::*;

/// Outputs from dynamic float-to-integer quantization.
pub struct DynamicQuantResult {
    pub quant: Vec<i32>,
    pub scale: f32,
    pub zp_i32: i32,
    pub zp_f32: f32,
}

/// Configuration for integer 2D convolution (`ConvInteger` / `ConvInt8`).
#[derive(Clone, Copy, Debug)]
pub struct ConvIntConfig {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub group: usize,
}

/// Compute output height and width for an integer 2D convolution.
#[inline]
pub fn conv_int_output_dims(in_h: usize, in_w: usize, cfg: &ConvIntConfig) -> (usize, usize) {
    let eff_kh = (cfg.kernel_h - 1) * cfg.dilate_h + 1;
    let eff_kw = (cfg.kernel_w - 1) * cfg.dilate_w + 1;
    let out_h = (in_h + cfg.pad_top + cfg.pad_bottom).saturating_sub(eff_kh) / cfg.stride_h + 1;
    let out_w = (in_w + cfg.pad_left + cfg.pad_right).saturating_sub(eff_kw) / cfg.stride_w + 1;
    (out_h, out_w)
}

/// Dynamically quantize FP32 values into signed `int8` (`MNN DynamicQuant`) or unsigned `uint8` (`ONNX/RTen DynamicQuantizeLinear`).
pub fn dynamic_quantize_f32(input: &[f32], signed: bool) -> DynamicQuantResult {
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;
    for &v in input {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    let range = max_val - min_val;

    if signed {
        if range == 0.0 {
            return DynamicQuantResult {
                quant: vec![0; input.len()],
                scale: 0.0,
                zp_i32: 0,
                zp_f32: 0.0,
            };
        }
        let inv_scale = 255.0 / range;
        let scale = range / 255.0;
        let zp_f32 = (-(min_val * 255.0) / range).round() - 128.0;
        let quant = input
            .iter()
            .map(|&v| ((v * inv_scale + zp_f32).round().clamp(-128.0, 127.0)) as i32)
            .collect();
        DynamicQuantResult {
            quant,
            scale,
            zp_i32: zp_f32 as i32,
            zp_f32,
        }
    } else {
        if range == 0.0 {
            return DynamicQuantResult {
                quant: vec![0; input.len()],
                scale: 1.0,
                zp_i32: 0,
                zp_f32: 0.0,
            };
        }
        let scale = range / 255.0;
        let zp_f32 = (-min_val / scale).clamp(0.0, 255.0).round_ties_even();
        let quant = input
            .iter()
            .map(|&v| ((v / scale).round_ties_even() + zp_f32).clamp(0.0, 255.0) as i32)
            .collect();
        DynamicQuantResult {
            quant,
            scale,
            zp_i32: zp_f32 as i32,
            zp_f32,
        }
    }
}

#[inline]
fn gemm_i32_rows(m_start: usize, m_end: usize, n: usize, k: usize, a: &[i32], b: &[i32], c_slice: &mut [i32]) {
    let mut m = m_start;
    while m + 4 <= m_end {
        let local_m = m - m_start;
        let a0 = &a[m * k..(m + 1) * k];
        let a1 = &a[(m + 1) * k..(m + 2) * k];
        let a2 = &a[(m + 2) * k..(m + 3) * k];
        let a3 = &a[(m + 3) * k..(m + 4) * k];
        let (c0_slice, rest) = c_slice[local_m * n..(local_m + 4) * n].split_at_mut(n);
        let (c1_slice, rest) = rest.split_at_mut(n);
        let (c2_slice, c3_slice) = rest.split_at_mut(n);

        let mut j = 0usize;
        while j + 8 <= n {
            let mut s0 = [0i32; 8];
            let mut s1 = [0i32; 8];
            let mut s2 = [0i32; 8];
            let mut s3 = [0i32; 8];
            for kk in 0..k {
                let va0 = a0[kk];
                let va1 = a1[kk];
                let va2 = a2[kk];
                let va3 = a3[kk];
                let brow = &b[kk * n + j..kk * n + j + 8];
                for lane in 0..8 {
                    let vb = brow[lane];
                    s0[lane] = s0[lane].wrapping_add(va0.wrapping_mul(vb));
                    s1[lane] = s1[lane].wrapping_add(va1.wrapping_mul(vb));
                    s2[lane] = s2[lane].wrapping_add(va2.wrapping_mul(vb));
                    s3[lane] = s3[lane].wrapping_add(va3.wrapping_mul(vb));
                }
            }
            c0_slice[j..j + 8].copy_from_slice(&s0);
            c1_slice[j..j + 8].copy_from_slice(&s1);
            c2_slice[j..j + 8].copy_from_slice(&s2);
            c3_slice[j..j + 8].copy_from_slice(&s3);
            j += 8;
        }
        for jj in j..n {
            let mut s0 = 0i32;
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            let mut s3 = 0i32;
            for kk in 0..k {
                let vb = b[kk * n + jj];
                s0 = s0.wrapping_add(a0[kk].wrapping_mul(vb));
                s1 = s1.wrapping_add(a1[kk].wrapping_mul(vb));
                s2 = s2.wrapping_add(a2[kk].wrapping_mul(vb));
                s3 = s3.wrapping_add(a3[kk].wrapping_mul(vb));
            }
            c0_slice[jj] = s0;
            c1_slice[jj] = s1;
            c2_slice[jj] = s2;
            c3_slice[jj] = s3;
        }
        m += 4;
    }

    while m < m_end {
        let local_m = m - m_start;
        let a0 = &a[m * k..(m + 1) * k];
        let c0 = &mut c_slice[local_m * n..(local_m + 1) * n];
        let mut j = 0usize;
        while j + 8 <= n {
            let mut s0 = [0i32; 8];
            for kk in 0..k {
                let va = a0[kk];
                let brow = &b[kk * n + j..kk * n + j + 8];
                for lane in 0..8 {
                    s0[lane] = s0[lane].wrapping_add(va.wrapping_mul(brow[lane]));
                }
            }
            c0[j..j + 8].copy_from_slice(&s0);
            j += 8;
        }
        for jj in j..n {
            let mut s0 = 0i32;
            for kk in 0..k {
                s0 = s0.wrapping_add(a0[kk].wrapping_mul(b[kk * n + jj]));
            }
            c0[jj] = s0;
        }
        m += 1;
    }
}

/// Exact integer matrix multiplication `C[m, n] = A[m, k] * B[k, n]`.
pub fn gemm_i32_nn(m: usize, n: usize, k: usize, a: &[i32], b: &[i32], c: &mut [i32]) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c[..m * n].fill(0);
        return;
    }

    #[cfg(feature = "parallel")]
    if crate::par::current_thread_index().is_none() && m >= 16 && m * n * k >= 65_536 {
        let block_rows = 16usize;
        c[..m * n]
            .par_chunks_mut(block_rows * n)
            .enumerate()
            .for_each(|(blk_idx, c_chunk)| {
                let m_start = blk_idx * block_rows;
                let rows = c_chunk.len() / n;
                gemm_i32_rows(m_start, m_start + rows, n, k, a, b, c_chunk);
            });
        return;
    }

    gemm_i32_rows(0, m, n, k, a, b, &mut c[..m * n]);
}

/// Zero-padded integer im2col for shifted inputs (`(padded_x - x_zp) == 0` outside bounds).
fn im2col_i32(
    input: &[i32],
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    cfg: &ConvIntConfig,
    col: &mut [i32],
) {
    let out_hw = out_h * out_w;
    let in_hw = in_h * in_w;
    let kh = cfg.kernel_h;
    let kw = cfg.kernel_w;
    let pad_t = cfg.pad_top as isize;
    let pad_l = cfg.pad_left as isize;
    let sh = cfg.stride_h as isize;
    let sw = cfg.stride_w as isize;
    let dh = cfg.dilate_h as isize;
    let dw = cfg.dilate_w as isize;

    let mut col_row = 0usize;
    for c in 0..channels {
        let src_c = &input[c * in_hw..(c + 1) * in_hw];
        for ky in 0..kh {
            for kx in 0..kw {
                let dst_row = &mut col[col_row * out_hw..(col_row + 1) * out_hw];
                col_row += 1;
                for oy in 0..out_h {
                    let iy = oy as isize * sh - pad_t + ky as isize * dh;
                    let dst_line = &mut dst_row[oy * out_w..(oy + 1) * out_w];
                    if iy < 0 || iy >= in_h as isize {
                        dst_line.fill(0);
                        continue;
                    }
                    let src_line = &src_c[iy as usize * in_w..(iy as usize + 1) * in_w];
                    if cfg.stride_w == 1 && cfg.dilate_w == 1 {
                        let start_ix = -pad_l + kx as isize;
                        for (ox, dst) in dst_line.iter_mut().enumerate() {
                            let ix = start_ix + ox as isize;
                            *dst = if ix >= 0 && (ix as usize) < in_w {
                                src_line[ix as usize]
                            } else {
                                0
                            };
                        }
                    } else {
                        for (ox, dst) in dst_line.iter_mut().enumerate() {
                            let ix = ox as isize * sw - pad_l + kx as isize * dw;
                            *dst = if ix >= 0 && (ix as usize) < in_w {
                                src_line[ix as usize]
                            } else {
                                0
                            };
                        }
                    }
                }
            }
        }
    }
}

/// Exact integer convolution accumulation `sum((x - x_zp) * (w - w_zp))` in `i32`.
/// Outside bounds, `x` is treated as padded with `x_zp` so `(x - x_zp) == 0`.
pub fn conv_integer_accumulate_i32(
    input: &[i32],
    in_shape: [usize; 4],
    weight: &[i32],
    w_shape: [usize; 4],
    x_zp: i32,
    w_zp: &[i32],
    cfg: &ConvIntConfig,
) -> (Vec<i32>, usize, usize) {
    let [batch, in_c, in_h, in_w] = in_shape;
    let [out_c, ic_per_group, kh, kw] = w_shape;
    let group = cfg.group.max(1);
    let oc_per_group = out_c / group;
    let (out_h, out_w) = conv_int_output_dims(in_h, in_w, cfg);
    let out_hw = out_h * out_w;

    let x_shifted: Vec<i32> = if x_zp == 0 {
        input.to_vec()
    } else {
        input.iter().map(|&v| v.wrapping_sub(x_zp)).collect()
    };

    let k_per_group = ic_per_group * kh * kw;
    let mut w_shifted = Vec::with_capacity(weight.len());
    for oc in 0..out_c {
        let wz = w_zp.get(oc).copied().or_else(|| w_zp.first().copied()).unwrap_or(0);
        let w_slice = &weight[oc * k_per_group..(oc + 1) * k_per_group];
        if wz == 0 {
            w_shifted.extend_from_slice(w_slice);
        } else {
            w_shifted.extend(w_slice.iter().map(|&v| v.wrapping_sub(wz)));
        }
    }

    let mut out = vec![0i32; batch * out_c * out_hw];
    let is_1x1_direct = kh == 1
        && kw == 1
        && cfg.stride_h == 1
        && cfg.stride_w == 1
        && cfg.pad_top == 0
        && cfg.pad_left == 0
        && cfg.pad_bottom == 0
        && cfg.pad_right == 0;

    let mut col = if is_1x1_direct {
        Vec::new()
    } else {
        vec![0i32; k_per_group * out_hw]
    };

    let in_batch_stride = in_c * in_h * in_w;
    let in_group_stride = ic_per_group * in_h * in_w;
    let out_batch_stride = out_c * out_hw;
    let out_group_stride = oc_per_group * out_hw;
    let w_group_stride = oc_per_group * k_per_group;

    for b in 0..batch {
        let x_b = &x_shifted[b * in_batch_stride..(b + 1) * in_batch_stride];
        let o_b = &mut out[b * out_batch_stride..(b + 1) * out_batch_stride];
        for g in 0..group {
            let x_g = &x_b[g * in_group_stride..(g + 1) * in_group_stride];
            let w_g = &w_shifted[g * w_group_stride..(g + 1) * w_group_stride];
            let o_g = &mut o_b[g * out_group_stride..(g + 1) * out_group_stride];
            if is_1x1_direct {
                gemm_i32_nn(oc_per_group, out_hw, k_per_group, w_g, x_g, o_g);
            } else {
                im2col_i32(x_g, ic_per_group, in_h, in_w, out_h, out_w, cfg, &mut col);
                gemm_i32_nn(oc_per_group, out_hw, k_per_group, w_g, &col, o_g);
            }
        }
    }

    (out, out_h, out_w)
}

fn subtract_broadcast_i32(data: &[i32], shape: &[usize], zp: Option<&[i32]>, zp_shape: &[usize]) -> Vec<i32> {
    let Some(zp_vals) = zp else {
        return data.to_vec();
    };
    if zp_vals.is_empty() {
        return data.to_vec();
    }
    if zp_vals.len() == 1 {
        let z = zp_vals[0];
        return if z == 0 {
            data.to_vec()
        } else {
            data.iter().map(|&v| v.wrapping_sub(z)).collect()
        };
    }

    let ndim = shape.len();
    let mut padded_zp_shape = vec![1usize; ndim];
    let offset = ndim.saturating_sub(zp_shape.len());
    for (i, &d) in zp_shape.iter().enumerate() {
        if i + offset < ndim {
            padded_zp_shape[i + offset] = d;
        }
    }
    let mut zp_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for d in (0..ndim).rev() {
        zp_strides[d] = if padded_zp_shape[d] == 1 { 0 } else { stride };
        stride *= padded_zp_shape[d];
    }

    let mut out = Vec::with_capacity(data.len());
    let mut coord = vec![0usize; ndim];
    for &val in data {
        let mut zp_idx = 0usize;
        for d in 0..ndim {
            zp_idx += coord[d] * zp_strides[d];
        }
        out.push(val.wrapping_sub(zp_vals[zp_idx]));
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    out
}

/// Exact integer `MatMulInteger` over 1D/2D/ND tensors with optional zero points.
pub fn matmul_integer_i32(
    a: &[i32],
    a_shape: &[usize],
    b: &[i32],
    b_shape: &[usize],
    a_zp: Option<&[i32]>,
    a_zp_shape: &[usize],
    b_zp: Option<&[i32]>,
    b_zp_shape: &[usize],
) -> (Vec<i32>, Vec<usize>) {
    let squeeze_a = a_shape.len() == 1;
    let squeeze_b = b_shape.len() == 1;
    let a_eff_shape: Vec<usize> = if squeeze_a { vec![1, a_shape[0]] } else { a_shape.to_vec() };
    let b_eff_shape: Vec<usize> = if squeeze_b { vec![b_shape[0], 1] } else { b_shape.to_vec() };

    let m = a_eff_shape[a_eff_shape.len() - 2];
    let k = a_eff_shape[a_eff_shape.len() - 1];
    let n = b_eff_shape[b_eff_shape.len() - 1];

    // Per ONNX MatMulInteger spec, 1D a_zero_point on >=2D A maps to [M, 1], not [1, M].
    let eff_a_zp_shape: Vec<usize> = if a_zp_shape.len() == 1 && !squeeze_a && a_zp_shape[0] == m {
        vec![m, 1]
    } else {
        a_zp_shape.to_vec()
    };

    let a_shifted = subtract_broadcast_i32(a, &a_eff_shape, a_zp, &eff_a_zp_shape);
    let b_shifted = subtract_broadcast_i32(b, &b_eff_shape, b_zp, b_zp_shape);

    let a_batch_dims = &a_eff_shape[..a_eff_shape.len() - 2];
    let b_batch_dims = &b_eff_shape[..b_eff_shape.len() - 2];
    let b_batches: usize = b_batch_dims.iter().product::<usize>().max(1);
    let a_batches: usize = a_batch_dims.iter().product::<usize>().max(1);

    let out_batch_ndim = a_batch_dims.len().max(b_batch_dims.len());
    let mut out_batch_dims = Vec::with_capacity(out_batch_ndim);
    for d in 0..out_batch_ndim {
        let ad = a_batch_dims
            .len()
            .checked_sub(out_batch_ndim - d)
            .map(|i| a_batch_dims[i])
            .unwrap_or(1);
        let bd = b_batch_dims
            .len()
            .checked_sub(out_batch_ndim - d)
            .map(|i| b_batch_dims[i])
            .unwrap_or(1);
        out_batch_dims.push(ad.max(bd));
    }
    let out_batches: usize = out_batch_dims.iter().product::<usize>().max(1);

    let mut out_data = vec![0i32; out_batches * m * n];
    if b_batches == 1 && a_batches == out_batches {
        gemm_i32_nn(out_batches * m, n, k, &a_shifted, &b_shifted, &mut out_data);
    } else {
        let mk = m * k;
        let kn = k * n;
        let mn = m * n;
        let mut a_b_pad = vec![1usize; out_batch_ndim];
        let mut b_b_pad = vec![1usize; out_batch_ndim];
        a_b_pad[out_batch_ndim - a_batch_dims.len()..].copy_from_slice(a_batch_dims);
        b_b_pad[out_batch_ndim - b_batch_dims.len()..].copy_from_slice(b_batch_dims);

        for bi in 0..out_batches {
            let mut rem = bi;
            let mut a_bi = 0usize;
            let mut b_bi = 0usize;
            let mut a_st = 1usize;
            let mut b_st = 1usize;
            for d in (0..out_batch_ndim).rev() {
                let od = out_batch_dims[d];
                let coord = rem % od;
                rem /= od;
                if a_b_pad[d] > 1 {
                    a_bi += coord * a_st;
                }
                if b_b_pad[d] > 1 {
                    b_bi += coord * b_st;
                }
                a_st *= a_b_pad[d];
                b_st *= b_b_pad[d];
            }
            gemm_i32_nn(
                m,
                n,
                k,
                &a_shifted[a_bi * mk..(a_bi + 1) * mk],
                &b_shifted[b_bi * kn..(b_bi + 1) * kn],
                &mut out_data[bi * mn..(bi + 1) * mn],
            );
        }
    }

    let mut out_shape = out_batch_dims;
    if !squeeze_a {
        out_shape.push(m);
    }
    if !squeeze_b {
        out_shape.push(n);
    }
    (out_data, out_shape)
}

/// Apply MNN `ConvInt8` / `DepthwiseConvInt8` scaling and optional requantization after `i32` accumulation.
pub fn conv_int8_postprocess(
    acc: &[i32],
    batch: usize,
    out_channels: usize,
    out_hw: usize,
    bias: &[i32],
    scale: &[f32],
    output_float: bool,
    output_zero_point: i32,
    clamp_min: i8,
    clamp_max: i8,
) -> (Option<Vec<f32>>, Option<Vec<i32>>) {
    let total = acc.len();
    if output_float {
        let mut out_f = vec![0.0f32; total];
        for b in 0..batch {
            for oc in 0..out_channels {
                let b_val = bias.get(oc).copied().unwrap_or(0);
                let s_val = scale.get(oc).copied().or_else(|| scale.first().copied()).unwrap_or(1.0);
                let off = (b * out_channels + oc) * out_hw;
                for i in 0..out_hw {
                    out_f[off + i] = acc[off + i].wrapping_add(b_val) as f32 * s_val;
                }
            }
        }
        (Some(out_f), None)
    } else {
        let mut out_i = vec![0i32; total];
        let min_f = clamp_min as f32;
        let max_f = clamp_max as f32;
        let out_zp = output_zero_point as f32;
        for b in 0..batch {
            for oc in 0..out_channels {
                let b_val = bias.get(oc).copied().unwrap_or(0);
                let s_val = scale.get(oc).copied().or_else(|| scale.first().copied()).unwrap_or(1.0);
                let off = (b * out_channels + oc) * out_hw;
                for i in 0..out_hw {
                    let scaled = acc[off + i].wrapping_add(b_val) as f32 * s_val;
                    out_i[off + i] = (scaled.round() + out_zp).clamp(min_f, max_f) as i32;
                }
            }
        }
        (None, Some(out_i))
    }
}

/// Dequantize NCHW integer values using MNN `Int8ToFloat` semantics: `(x - zp) * scale`.
pub fn int8_to_float_nchw(input: &[i32], shape: &[usize], scale: &[f32], zero_points: &[f32]) -> Vec<f32> {
    if scale.len() <= 1 && zero_points.len() <= 1 {
        let s = scale.first().copied().unwrap_or(1.0);
        let z = zero_points.first().copied().unwrap_or(0.0);
        return input.iter().map(|&v| (v as f32 - z) * s).collect();
    }
    let batch = shape.first().copied().unwrap_or(1).max(1);
    let channels = shape.get(1).copied().unwrap_or(1).max(1);
    let spatial = input.len() / (batch * channels).max(1);
    let mut out = vec![0.0f32; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            let s = scale.get(c).copied().or_else(|| scale.first().copied()).unwrap_or(1.0);
            let z = zero_points
                .get(c)
                .copied()
                .or_else(|| zero_points.first().copied())
                .unwrap_or(0.0);
            let off = (b * channels + c) * spatial;
            for i in 0..spatial {
                out[off + i] = (input[off + i] as f32 - z) * s;
            }
        }
    }
    out
}

/// Quantize NCHW FP32 values using MNN `FloatToInt8` semantics: `round(x * scale + zp).clamp(min, max)`.
pub fn float_to_int8_nchw(
    input: &[f32],
    shape: &[usize],
    scale: &[f32],
    zero_points: &[f32],
    clamp_min: i8,
    clamp_max: i8,
) -> Vec<i32> {
    let min_f = clamp_min as f32;
    let max_f = clamp_max as f32;
    if scale.len() <= 1 && zero_points.len() <= 1 {
        let s = scale.first().copied().unwrap_or(1.0);
        let z = zero_points.first().copied().unwrap_or(0.0);
        return input
            .iter()
            .map(|&v| (v * s + z).round().clamp(min_f, max_f) as i32)
            .collect();
    }
    let batch = shape.first().copied().unwrap_or(1).max(1);
    let channels = shape.get(1).copied().unwrap_or(1).max(1);
    let spatial = input.len() / (batch * channels).max(1);
    let mut out = vec![0i32; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            let s = scale.get(c).copied().or_else(|| scale.first().copied()).unwrap_or(1.0);
            let z = zero_points
                .get(c)
                .copied()
                .or_else(|| zero_points.first().copied())
                .unwrap_or(0.0);
            let off = (b * channels + c) * spatial;
            for i in 0..spatial {
                out[off + i] = (input[off + i] * s + z).round().clamp(min_f, max_f) as i32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_quantize_onnx_matches_ort_reference() {
        let mixed = [-2.0f32, -0.3, 0.0, 1.7, 2.5];
        let res = dynamic_quantize_f32(&mixed, false);
        assert!((res.scale - 4.5 / 255.0).abs() < 1e-7);
        assert_eq!(res.zp_i32, 113);
        assert_eq!(res.quant, vec![0, 96, 113, 209, 255]);

        let neg = [-3.0f32, -2.5, -1.0, -0.5];
        let res_neg = dynamic_quantize_f32(&neg, false);
        assert_eq!(res_neg.zp_i32, 255);
        assert_eq!(res_neg.quant, vec![0, 43, 170, 213]);

        let zeros = [0.0f32; 4];
        let res_z = dynamic_quantize_f32(&zeros, false);
        assert_eq!(res_z.scale, 1.0);
        assert_eq!(res_z.zp_i32, 0);
        assert_eq!(res_z.quant, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_dynamic_quantize_mnn_signed() {
        let vals = [-2.0f32, 0.0, 2.5];
        let res = dynamic_quantize_f32(&vals, true);
        assert!((res.scale - 4.5 / 255.0).abs() < 1e-7);
        assert_eq!(res.zp_i32, -15);
        assert_eq!(res.quant[0], -128);
        assert_eq!(res.quant[1], -15);
        assert_eq!(res.quant[2], 127);
    }

    #[test]
    fn test_conv_integer_padding_uses_x_zp() {
        let x = [10i32, 20, 30, 40];
        let w = [2i32; 9];
        let cfg = ConvIntConfig {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            dilate_h: 1,
            dilate_w: 1,
            pad_top: 1,
            pad_left: 1,
            pad_bottom: 1,
            pad_right: 1,
            group: 1,
        };
        let (out, oh, ow) = conv_integer_accumulate_i32(&x, [1, 1, 2, 2], &w, [1, 1, 3, 3], 5, &[-1], &cfg);
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(out, vec![240, 240, 240, 240]);
    }

    #[test]
    fn test_matmul_integer_1d_and_2d_zero_points() {
        // A: [2, 3], B: [3, 4], a_zp: [2] per-row, b_zp: [4] per-col
        let a = [10, 20, 30, 40, 50, 60];
        let b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let a_zp = [10, 40];
        let b_zp = [1, 2, 3, 4];
        let (out, shape) = matmul_integer_i32(&a, &[2, 3], &b, &[3, 4], Some(&a_zp), &[2], Some(&b_zp), &[4]);
        assert_eq!(shape, vec![2, 4]);
        // Row 0 shifted: [0, 10, 20]; Row 1 shifted: [0, 10, 20]; B shifted: col j = [0, 4, 8]
        // Every entry = 10*4 + 20*8 = 200
        assert_eq!(out, vec![200; 8]);
    }
}

