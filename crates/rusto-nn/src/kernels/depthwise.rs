#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

/// Configuration parameters for depthwise 2D convolution
#[derive(Debug, Clone)]
pub struct DepthwiseConv2dConfig {
    pub channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub activation: FusedActivation,
}

impl Default for DepthwiseConv2dConfig {
    fn default() -> Self {
        Self {
            channels: 1,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_top: 0,
            pad_left: 0,
            pad_bottom: 0,
            pad_right: 0,
            dilate_h: 1,
            dilate_w: 1,
            activation: FusedActivation::None,
        }
    }
}

/// Calculate depthwise 2D convolution output spatial dimensions
pub fn depthwise_conv2d_output_dims(
    in_h: usize,
    in_w: usize,
    config: &DepthwiseConv2dConfig,
) -> (usize, usize) {
    let effective_kh = (config.kernel_h - 1) * config.dilate_h + 1;
    let effective_kw = (config.kernel_w - 1) * config.dilate_w + 1;

    let out_h = (in_h + config.pad_top + config.pad_bottom).saturating_sub(effective_kh) / config.stride_h + 1;
    let out_w = (in_w + config.pad_left + config.pad_right).saturating_sub(effective_kw) / config.stride_w + 1;

    (out_h, out_w)
}

/// Fallback single-pixel evaluation for depthwise conv
#[inline(always)]
fn eval_depthwise_pixel(
    in_channel: &[f32],
    w_channel: &[f32],
    b_val: f32,
    in_h: usize,
    in_w: usize,
    oh: usize,
    ow: usize,
    config: &DepthwiseConv2dConfig,
) -> f32 {
    let mut sum = 0.0f32;
    for kh in 0..config.kernel_h {
        let ih_unclamped = oh as isize * config.stride_h as isize
            + kh as isize * config.dilate_h as isize
            - config.pad_top as isize;

        if ih_unclamped >= 0 && ih_unclamped < in_h as isize {
            let ih = ih_unclamped as usize;
            let in_row = &in_channel[ih * in_w..(ih + 1) * in_w];
            let w_row = &w_channel[kh * config.kernel_w..(kh + 1) * config.kernel_w];

            for kw in 0..config.kernel_w {
                let iw_unclamped = ow as isize * config.stride_w as isize
                    + kw as isize * config.dilate_w as isize
                    - config.pad_left as isize;

                if iw_unclamped >= 0 && iw_unclamped < in_w as isize {
                    sum += in_row[iw_unclamped as usize] * w_row[kw];
                }
            }
        }
    }
    sum + b_val
}

/// Optimized AVX2 kernel for 3x3 depthwise convolution with stride 1 and dilation 1
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_3x3_s1_avx2(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    out_channel: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) {
    use core::arch::x86_64::*;

    let b_vec = _mm256_set1_ps(bias);
    let zero_vec = _mm256_setzero_ps();
    let six_vec = _mm256_set1_ps(6.0);
    let three_vec = _mm256_set1_ps(3.0);
    let inv6_vec = _mm256_set1_ps(1.0 / 6.0);

    let w00 = _mm256_set1_ps(w[0]);
    let w01 = _mm256_set1_ps(w[1]);
    let w02 = _mm256_set1_ps(w[2]);
    let w10 = _mm256_set1_ps(w[3]);
    let w11 = _mm256_set1_ps(w[4]);
    let w12 = _mm256_set1_ps(w[5]);
    let w20 = _mm256_set1_ps(w[6]);
    let w21 = _mm256_set1_ps(w[7]);
    let w22 = _mm256_set1_ps(w[8]);

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let act = config.activation;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let r0 = in_channel.as_ptr().add(ih_base as usize * in_w);
            let r1 = in_channel.as_ptr().add((ih_base as usize + 1) * in_w);
            let r2 = in_channel.as_ptr().add((ih_base as usize + 2) * in_w);
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left;
            let v_limit = if in_w + pad_left >= 10 {
                (in_w + pad_left - 9).min(out_w)
            } else {
                0
            };

            for ow in 0..ow_start.min(out_w) {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 8 <= v_limit {
                let iw = ow - pad_left;

                let mut acc = b_vec;
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw)), w00, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw + 1)), w01, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw + 2)), w02, acc);

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw)), w10, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw + 1)), w11, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw + 2)), w12, acc);

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw)), w20, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw + 1)), w21, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw + 2)), w22, acc);

                match act {
                    FusedActivation::Relu => {
                        acc = _mm256_max_ps(zero_vec, acc);
                    }
                    FusedActivation::Relu6 => {
                        acc = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, acc));
                    }
                    FusedActivation::HardSwish => {
                        let p = _mm256_add_ps(acc, three_vec);
                        let cl = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, p));
                        acc = _mm256_mul_ps(acc, _mm256_mul_ps(cl, inv6_vec));
                    }
                    _ => {}
                }
                _mm256_storeu_ps(out_row.add(ow), acc);
                ow += 8;
            }

            while ow < out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                out_channel[oh * out_w + ow] = apply_activation_scalar(val, act);
            }
        }
    }
}

/// Optimized AVX2 kernel for 5x5 depthwise convolution with stride 1 and dilation 1
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_5x5_s1_avx2(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    out_channel: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) {
    use core::arch::x86_64::*;

    let b_vec = _mm256_set1_ps(bias);
    let zero_vec = _mm256_setzero_ps();
    let six_vec = _mm256_set1_ps(6.0);
    let three_vec = _mm256_set1_ps(3.0);
    let inv6_vec = _mm256_set1_ps(1.0 / 6.0);

    let mut w_vec = [_mm256_setzero_ps(); 25];
    for i in 0..25 {
        w_vec[i] = _mm256_set1_ps(w[i]);
    }

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let act = config.activation;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 4 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_channel.as_ptr().add(ih_u * in_w);
            let r1 = in_channel.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_channel.as_ptr().add((ih_u + 2) * in_w);
            let r3 = in_channel.as_ptr().add((ih_u + 3) * in_w);
            let r4 = in_channel.as_ptr().add((ih_u + 4) * in_w);
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left;
            let v_limit = if in_w + pad_left >= 12 {
                (in_w + pad_left - 11).min(out_w)
            } else {
                0
            };

            for ow in 0..ow_start.min(out_w) {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 8 <= v_limit {
                let iw = ow - pad_left;

                let mut acc = b_vec;
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw)), w_vec[0], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw + 1)), w_vec[1], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw + 2)), w_vec[2], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw + 3)), w_vec[3], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(iw + 4)), w_vec[4], acc);

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw)), w_vec[5], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw + 1)), w_vec[6], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw + 2)), w_vec[7], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw + 3)), w_vec[8], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(iw + 4)), w_vec[9], acc);

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw)), w_vec[10], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw + 1)), w_vec[11], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw + 2)), w_vec[12], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw + 3)), w_vec[13], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(iw + 4)), w_vec[14], acc);

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(iw)), w_vec[15], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(iw + 1)), w_vec[16], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(iw + 2)), w_vec[17], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(iw + 3)), w_vec[18], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(iw + 4)), w_vec[19], acc);

                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r4.add(iw)), w_vec[20], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r4.add(iw + 1)), w_vec[21], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r4.add(iw + 2)), w_vec[22], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r4.add(iw + 3)), w_vec[23], acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r4.add(iw + 4)), w_vec[24], acc);

                match act {
                    FusedActivation::Relu => {
                        acc = _mm256_max_ps(zero_vec, acc);
                    }
                    FusedActivation::Relu6 => {
                        acc = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, acc));
                    }
                    FusedActivation::HardSwish => {
                        let p = _mm256_add_ps(acc, three_vec);
                        let cl = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, p));
                        acc = _mm256_mul_ps(acc, _mm256_mul_ps(cl, inv6_vec));
                    }
                    _ => {}
                }
                _mm256_storeu_ps(out_row.add(ow), acc);
                ow += 8;
            }

            while ow < out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                out_channel[oh * out_w + ow] = apply_activation_scalar(val, act);
            }
        }
    }
}

/// Optimized AVX2 kernel for 3x3 depthwise convolution with stride_w = 2 (stride_h = 1 or 2)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_3x3_sw2_avx2(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    out_channel: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) {
    use core::arch::x86_64::*;

    let b_vec = _mm_set1_ps(bias);
    let zero_vec = _mm_setzero_ps();
    let six_vec = _mm_set1_ps(6.0);
    let three_vec = _mm_set1_ps(3.0);
    let inv6_vec = _mm_set1_ps(1.0 / 6.0);

    let w00 = _mm_set1_ps(w[0]);
    let w01 = _mm_set1_ps(w[1]);
    let w02 = _mm_set1_ps(w[2]);
    let w10 = _mm_set1_ps(w[3]);
    let w11 = _mm_set1_ps(w[4]);
    let w12 = _mm_set1_ps(w[5]);
    let w20 = _mm_set1_ps(w[6]);
    let w21 = _mm_set1_ps(w[7]);
    let w22 = _mm_set1_ps(w[8]);

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let stride_h = config.stride_h;
    let act = config.activation;

    for oh in 0..out_h {
        let ih_base = oh as isize * stride_h as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let r0 = in_channel.as_ptr().add(ih_base as usize * in_w);
            let r1 = in_channel.as_ptr().add((ih_base as usize + 1) * in_w);
            let r2 = in_channel.as_ptr().add((ih_base as usize + 2) * in_w);
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.div_ceil(2);
            let v_limit = if in_w + pad_left >= 9 {
                ((in_w + pad_left - 9) / 2 + 1).min(out_w)
            } else {
                0
            };

            for ow in 0..ow_start.min(out_w) {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow * 2 - pad_left;

                let mut acc = b_vec;

                // Row 0
                let x0_kw0 = _mm_set_ps(*r0.add(iw + 6), *r0.add(iw + 4), *r0.add(iw + 2), *r0.add(iw));
                let x0_kw1 = _mm_set_ps(*r0.add(iw + 7), *r0.add(iw + 5), *r0.add(iw + 3), *r0.add(iw + 1));
                let x0_kw2 = _mm_set_ps(*r0.add(iw + 8), *r0.add(iw + 6), *r0.add(iw + 4), *r0.add(iw + 2));
                acc = _mm_fmadd_ps(x0_kw0, w00, acc);
                acc = _mm_fmadd_ps(x0_kw1, w01, acc);
                acc = _mm_fmadd_ps(x0_kw2, w02, acc);

                // Row 1
                let x1_kw0 = _mm_set_ps(*r1.add(iw + 6), *r1.add(iw + 4), *r1.add(iw + 2), *r1.add(iw));
                let x1_kw1 = _mm_set_ps(*r1.add(iw + 7), *r1.add(iw + 5), *r1.add(iw + 3), *r1.add(iw + 1));
                let x1_kw2 = _mm_set_ps(*r1.add(iw + 8), *r1.add(iw + 6), *r1.add(iw + 4), *r1.add(iw + 2));
                acc = _mm_fmadd_ps(x1_kw0, w10, acc);
                acc = _mm_fmadd_ps(x1_kw1, w11, acc);
                acc = _mm_fmadd_ps(x1_kw2, w12, acc);

                // Row 2
                let x2_kw0 = _mm_set_ps(*r2.add(iw + 6), *r2.add(iw + 4), *r2.add(iw + 2), *r2.add(iw));
                let x2_kw1 = _mm_set_ps(*r2.add(iw + 7), *r2.add(iw + 5), *r2.add(iw + 3), *r2.add(iw + 1));
                let x2_kw2 = _mm_set_ps(*r2.add(iw + 8), *r2.add(iw + 6), *r2.add(iw + 4), *r2.add(iw + 2));
                acc = _mm_fmadd_ps(x2_kw0, w20, acc);
                acc = _mm_fmadd_ps(x2_kw1, w21, acc);
                acc = _mm_fmadd_ps(x2_kw2, w22, acc);

                match act {
                    FusedActivation::Relu => {
                        acc = _mm_max_ps(zero_vec, acc);
                    }
                    FusedActivation::Relu6 => {
                        acc = _mm_min_ps(six_vec, _mm_max_ps(zero_vec, acc));
                    }
                    FusedActivation::HardSwish => {
                        let p = _mm_add_ps(acc, three_vec);
                        let cl = _mm_min_ps(six_vec, _mm_max_ps(zero_vec, p));
                        acc = _mm_mul_ps(acc, _mm_mul_ps(cl, inv6_vec));
                    }
                    _ => {}
                }
                _mm_storeu_ps(out_row.add(ow), acc);
                ow += 4;
            }

            while ow < out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                out_channel[oh * out_w + ow] = apply_activation_scalar(val, act);
            }
        }
    }
}

/// Optimized AVX2 kernel for 5x1 depthwise convolution with stride 1 and dilation 1
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_5x1_s1_avx2(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    out_channel: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) {
    use core::arch::x86_64::*;

    let b_vec = _mm256_set1_ps(bias);
    let zero_vec = _mm256_setzero_ps();
    let six_vec = _mm256_set1_ps(6.0);
    let three_vec = _mm256_set1_ps(3.0);
    let inv6_vec = _mm256_set1_ps(1.0 / 6.0);

    let w0 = _mm256_set1_ps(w[0]);
    let w1 = _mm256_set1_ps(w[1]);
    let w2 = _mm256_set1_ps(w[2]);
    let w3 = _mm256_set1_ps(w[3]);
    let w4 = _mm256_set1_ps(w[4]);

    let pad_top = config.pad_top;
    let act = config.activation;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 4 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_channel.as_ptr().add(ih_u * in_w);
            let r1 = in_channel.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_channel.as_ptr().add((ih_u + 2) * in_w);
            let r3 = in_channel.as_ptr().add((ih_u + 3) * in_w);
            let r4 = in_channel.as_ptr().add((ih_u + 4) * in_w);
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let chunks = out_w / 8;
            for c_idx in 0..chunks {
                let ow = c_idx * 8;
                let mut acc = b_vec;
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(ow)), w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(ow)), w1, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(ow)), w2, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(ow)), w3, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(r4.add(ow)), w4, acc);

                match act {
                    FusedActivation::Relu => {
                        acc = _mm256_max_ps(zero_vec, acc);
                    }
                    FusedActivation::Relu6 => {
                        acc = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, acc));
                    }
                    FusedActivation::HardSwish => {
                        let p = _mm256_add_ps(acc, three_vec);
                        let cl = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, p));
                        acc = _mm256_mul_ps(acc, _mm256_mul_ps(cl, inv6_vec));
                    }
                    _ => {}
                }
                _mm256_storeu_ps(out_row.add(ow), acc);
            }

            for ow in (chunks * 8)..out_w {
                let sum = bias
                    + *r0.add(ow) * w[0]
                    + *r1.add(ow) * w[1]
                    + *r2.add(ow) * w[2]
                    + *r3.add(ow) * w[3]
                    + *r4.add(ow) * w[4];
                *out_row.add(ow) = apply_activation_scalar(sum, act);
            }
        } else {
            for ow in 0..out_w {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                out_channel[oh * out_w + ow] = apply_activation_scalar(val, act);
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum DepthwiseKernelKind {
    #[allow(dead_code)]
    Avx2_3x3S1,
    #[allow(dead_code)]
    Avx2_5x5S1,
    #[allow(dead_code)]
    Avx2_3x3Sw2,
    #[allow(dead_code)]
    Avx2_5x1S1,
    #[allow(dead_code)]
    Neon_3x3S1,
    Fallback,
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn depthwise_3x3_s1_neon(
    in_channel: &[f32],
    w_channel: &[f32],
    b_val: f32,
    out_channel: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) {
    let b_vec = vdupq_n_f32(b_val);
    let zero_vec = vdupq_n_f32(0.0);
    let six_vec = vdupq_n_f32(6.0);
    let three_vec = vdupq_n_f32(3.0);
    let inv_six_vec = vdupq_n_f32(1.0 / 6.0);

    let w00 = vdupq_n_f32(w_channel[0]);
    let w01 = vdupq_n_f32(w_channel[1]);
    let w02 = vdupq_n_f32(w_channel[2]);
    let w10 = vdupq_n_f32(w_channel[3]);
    let w11 = vdupq_n_f32(w_channel[4]);
    let w12 = vdupq_n_f32(w_channel[5]);
    let w20 = vdupq_n_f32(w_channel[6]);
    let w21 = vdupq_n_f32(w_channel[7]);
    let w22 = vdupq_n_f32(w_channel[8]);

    for oh in 0..out_h {
        let r0 = oh as isize - config.pad_top as isize;
        let r1 = r0 + 1;
        let r2 = r0 + 2;

        let r0_valid = r0 >= 0 && (r0 as usize) < in_h;
        let r1_valid = r1 >= 0 && (r1 as usize) < in_h;
        let r2_valid = r2 >= 0 && (r2 as usize) < in_h;

        let row0_ptr = if r0_valid { in_channel.as_ptr().add(r0 as usize * in_w) } else { core::ptr::null() };
        let row1_ptr = if r1_valid { in_channel.as_ptr().add(r1 as usize * in_w) } else { core::ptr::null() };
        let row2_ptr = if r2_valid { in_channel.as_ptr().add(r2 as usize * in_w) } else { core::ptr::null() };

        let out_row_ptr = out_channel.as_mut_ptr().add(oh * out_w);

        let mut ow = 0;
        let safe_w_end = if out_w >= (config.pad_left + 2) {
            (out_w - 2).saturating_sub(config.pad_right)
        } else {
            0
        };

        while ow < config.pad_left && ow < out_w {
            let val = eval_depthwise_pixel(in_channel, w_channel, b_val, in_h, in_w, oh, ow, config);
            *out_row_ptr.add(ow) = apply_activation_scalar(val, config.activation);
            ow += 1;
        }

        while ow + 4 <= safe_w_end {
            let in_col = ow - config.pad_left;
            let mut acc = b_vec;

            if !row0_ptr.is_null() {
                let v00 = vld1q_f32(row0_ptr.add(in_col));
                let v01 = vld1q_f32(row0_ptr.add(in_col + 1));
                let v02 = vld1q_f32(row0_ptr.add(in_col + 2));
                acc = vfmaq_f32(acc, v00, w00);
                acc = vfmaq_f32(acc, v01, w01);
                acc = vfmaq_f32(acc, v02, w02);
            }
            if !row1_ptr.is_null() {
                let v10 = vld1q_f32(row1_ptr.add(in_col));
                let v11 = vld1q_f32(row1_ptr.add(in_col + 1));
                let v12 = vld1q_f32(row1_ptr.add(in_col + 2));
                acc = vfmaq_f32(acc, v10, w10);
                acc = vfmaq_f32(acc, v11, w11);
                acc = vfmaq_f32(acc, v12, w12);
            }
            if !row2_ptr.is_null() {
                let v20 = vld1q_f32(row2_ptr.add(in_col));
                let v21 = vld1q_f32(row2_ptr.add(in_col + 1));
                let v22 = vld1q_f32(row2_ptr.add(in_col + 2));
                acc = vfmaq_f32(acc, v20, w20);
                acc = vfmaq_f32(acc, v21, w21);
                acc = vfmaq_f32(acc, v22, w22);
            }

            match config.activation {
                FusedActivation::Relu => {
                    acc = vmaxq_f32(zero_vec, acc);
                }
                FusedActivation::Relu6 => {
                    acc = vminq_f32(six_vec, vmaxq_f32(zero_vec, acc));
                }
                FusedActivation::HardSwish => {
                    let val_p3 = vaddq_f32(acc, three_vec);
                    let clamped = vmaxq_f32(zero_vec, vminq_f32(six_vec, val_p3));
                    acc = vmulq_f32(acc, vmulq_f32(clamped, inv_six_vec));
                }
                _ => {}
            }

            vst1q_f32(out_row_ptr.add(ow), acc);
            ow += 4;
        }

        while ow < out_w {
            let val = eval_depthwise_pixel(in_channel, w_channel, b_val, in_h, in_w, oh, ow, config);
            *out_row_ptr.add(ow) = apply_activation_scalar(val, config.activation);
            ow += 1;
        }
    }
}

/// Process a single depthwise channel
#[inline(always)]
#[allow(unused_variables)]
fn process_depthwise_channel(
    kernel_kind: DepthwiseKernelKind,
    in_channel: &[f32],
    w_channel: &[f32],
    b_val: f32,
    out_channel: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) {
    #[cfg(target_arch = "x86_64")]
    {
        match kernel_kind {
            DepthwiseKernelKind::Avx2_3x3S1 => {
                unsafe {
                    depthwise_3x3_s1_avx2(
                        in_channel,
                        w_channel,
                        b_val,
                        out_channel,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Avx2_5x5S1 => {
                unsafe {
                    depthwise_5x5_s1_avx2(
                        in_channel,
                        w_channel,
                        b_val,
                        out_channel,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Avx2_3x3Sw2 => {
                unsafe {
                    depthwise_3x3_sw2_avx2(
                        in_channel,
                        w_channel,
                        b_val,
                        out_channel,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Avx2_5x1S1 => {
                unsafe {
                    depthwise_5x1_s1_avx2(
                        in_channel,
                        w_channel,
                        b_val,
                        out_channel,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        config,
                    );
                }
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let DepthwiseKernelKind::Neon_3x3S1 = kernel_kind {
            unsafe {
                depthwise_3x3_s1_neon(
                    in_channel,
                    w_channel,
                    b_val,
                    out_channel,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    config,
                );
            }
            return;
        }
    }

    let act = config.activation;
    for oh in 0..out_h {
        for ow in 0..out_w {
            let val = eval_depthwise_pixel(in_channel, w_channel, b_val, in_h, in_w, oh, ow, config);
            out_channel[oh * out_w + ow] = apply_activation_scalar(val, act);
        }
    }
}

/// Execute Depthwise 2D Convolution on planar NCHW tensors with AVX2 and Rayon parallelization
pub fn depthwise_conv2d_nchw(
    input: &[f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    config: &DepthwiseConv2dConfig,
) {
    let (out_h, out_w) = depthwise_conv2d_output_dims(in_h, in_w, config);
    let hw_out = out_h * out_w;
    let hw_in = in_h * in_w;
    let kernel_size = config.kernel_h * config.kernel_w;
    let channels = config.channels;

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    let kernel_kind = {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2 && config.dilate_h == 1 && config.dilate_w == 1 {
                if config.kernel_h == 3 && config.kernel_w == 3 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Avx2_3x3S1
                } else if config.kernel_h == 5 && config.kernel_w == 5 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Avx2_5x5S1
                } else if config.kernel_h == 3 && config.kernel_w == 3 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                    DepthwiseKernelKind::Avx2_3x3Sw2
                } else if config.kernel_h == 5 && config.kernel_w == 1 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Avx2_5x1S1
                } else {
                    DepthwiseKernelKind::Fallback
                }
            } else {
                DepthwiseKernelKind::Fallback
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if config.dilate_h == 1 && config.dilate_w == 1 && config.kernel_h == 3 && config.kernel_w == 3 && config.stride_h == 1 && config.stride_w == 1 {
                DepthwiseKernelKind::Neon_3x3S1
            } else {
                DepthwiseKernelKind::Fallback
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            DepthwiseKernelKind::Fallback
        }
    };

    for b in 0..batch {
        let in_batch = &input[b * channels * hw_in..(b + 1) * channels * hw_in];
        let out_batch = &mut output[b * channels * hw_out..(b + 1) * channels * hw_out];

        #[cfg(feature = "rayon")]
        {
            if channels >= 2 {
                out_batch
                    .par_chunks_exact_mut(hw_out)
                    .enumerate()
                    .for_each(|(c, out_channel)| {
                        let in_channel = &in_batch[c * hw_in..(c + 1) * hw_in];
                        let w_channel = &weight[c * kernel_size..(c + 1) * kernel_size];
                        let b_val = bias.map(|b_slice| b_slice[c]).unwrap_or(0.0);
                        process_depthwise_channel(
                            kernel_kind,
                            in_channel,
                            w_channel,
                            b_val,
                            out_channel,
                            in_h,
                            in_w,
                            out_h,
                            out_w,
                            config,
                        );
                    });
                continue;
            }
        }

        for c in 0..channels {
            let in_channel = &in_batch[c * hw_in..(c + 1) * hw_in];
            let out_channel = &mut out_batch[c * hw_out..(c + 1) * hw_out];
            let w_channel = &weight[c * kernel_size..(c + 1) * kernel_size];
            let b_val = bias.map(|b_slice| b_slice[c]).unwrap_or(0.0);
            process_depthwise_channel(
                kernel_kind,
                in_channel,
                w_channel,
                b_val,
                out_channel,
                in_h,
                in_w,
                out_h,
                out_w,
                config,
            );
        }
    }
}
