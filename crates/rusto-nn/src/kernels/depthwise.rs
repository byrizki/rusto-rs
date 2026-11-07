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

#[inline(always)]
fn eval_depthwise_1d_point(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    in_len: usize,
    k: usize,
    pad_before: usize,
    out_idx: usize,
) -> f32 {
    let mut sum = bias;
    let base = out_idx as isize - pad_before as isize;
    for t in 0..k {
        let idx = base + t as isize;
        if idx >= 0 && (idx as usize) < in_len {
            sum += in_channel[idx as usize] * w[t];
        }
    }
    sum
}

#[inline(always)]
fn extract_1d_s1_params(
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &DepthwiseConv2dConfig,
) -> Option<(usize, usize, usize, usize)> {
    if in_h == 1
        && out_h == 1
        && config.kernel_h == 1
        && config.stride_w == 1
        && config.pad_top == 0
        && config.pad_bottom == 0
        && config.kernel_w >= 1
    {
        Some((in_w, out_w, config.kernel_w, config.pad_left))
    } else if in_w == 1
        && out_w == 1
        && config.kernel_w == 1
        && config.stride_h == 1
        && config.pad_left == 0
        && config.pad_right == 0
        && config.kernel_h >= 1
    {
        Some((in_h, out_h, config.kernel_h, config.pad_top))
    } else {
        None
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_1d_s1_avx2(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    out_channel: &mut [f32],
    in_len: usize,
    out_len: usize,
    k: usize,
    pad_before: usize,
    act: FusedActivation,
) {
    use core::arch::x86_64::*;

    let b_vec = _mm256_set1_ps(bias);
    let zero_vec = _mm256_setzero_ps();
    let six_vec = _mm256_set1_ps(6.0);
    let three_vec = _mm256_set1_ps(3.0);
    let inv6_vec = _mm256_set1_ps(1.0 / 6.0);

    let in_ptr = in_channel.as_ptr();
    let out_ptr = out_channel.as_mut_ptr();

    let start = pad_before.min(out_len);
    for idx in 0..start {
        let val = eval_depthwise_1d_point(in_channel, w, bias, in_len, k, pad_before, idx);
        *out_ptr.add(idx) = apply_activation_scalar(val, act);
    }

    let mut idx = start;
    while idx + 8 <= out_len && (idx - pad_before) + k + 7 <= in_len {
        let base = idx - pad_before;
        let mut acc = b_vec;
        for t in 0..k {
            let wt = _mm256_set1_ps(*w.get_unchecked(t));
            let vin = _mm256_loadu_ps(in_ptr.add(base + t));
            acc = _mm256_fmadd_ps(vin, wt, acc);
        }

        match act {
            FusedActivation::None => {}
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
            _ => {
                let mut tmp = [0.0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                for v in &mut tmp {
                    *v = apply_activation_scalar(*v, act);
                }
                acc = _mm256_loadu_ps(tmp.as_ptr());
            }
        }

        _mm256_storeu_ps(out_ptr.add(idx), acc);
        idx += 8;
    }

    while idx < out_len {
        let val = eval_depthwise_1d_point(in_channel, w, bias, in_len, k, pad_before, idx);
        *out_ptr.add(idx) = apply_activation_scalar(val, act);
        idx += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_5x5_sw2_avx2(
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

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let stride_h = config.stride_h;
    let act = config.activation;

    for oh in 0..out_h {
        let ih_base = oh as isize * stride_h as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 4 < in_h as isize {
            let ih_u = ih_base as usize;
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);
            let ow_start = pad_left.div_ceil(2).min(out_w);

            for ow in 0..ow_start {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 4 <= out_w && (ow * 2 - pad_left) + 12 <= in_w {
                let iw = ow * 2 - pad_left;
                let mut acc = b_vec;

                for kh in 0..5 {
                    let r_ptr = in_channel.as_ptr().add((ih_u + kh) * in_w + iw);
                    let wk = w.as_ptr().add(kh * 5);
                    let v0 = _mm_loadu_ps(r_ptr);
                    let v1 = _mm_loadu_ps(r_ptr.add(4));
                    let v2 = _mm_loadu_ps(r_ptr.add(8));

                    let even01 = _mm_shuffle_ps(v0, v1, 0b10_00_10_00);
                    let odd01 = _mm_shuffle_ps(v0, v1, 0b11_01_11_01);
                    let even2 = _mm_shuffle_ps(v2, v2, 0b10_00_10_00);
                    let odd2 = _mm_shuffle_ps(v2, v2, 0b11_01_11_01);

                    let c2 = _mm_castsi128_ps(_mm_alignr_epi8(
                        _mm_castps_si128(even2),
                        _mm_castps_si128(even01),
                        4,
                    ));
                    let c3 = _mm_castsi128_ps(_mm_alignr_epi8(
                        _mm_castps_si128(odd2),
                        _mm_castps_si128(odd01),
                        4,
                    ));
                    let c4 = _mm_castsi128_ps(_mm_alignr_epi8(
                        _mm_castps_si128(even2),
                        _mm_castps_si128(even01),
                        8,
                    ));

                    acc = _mm_fmadd_ps(even01, _mm_set1_ps(*wk), acc);
                    acc = _mm_fmadd_ps(odd01, _mm_set1_ps(*wk.add(1)), acc);
                    acc = _mm_fmadd_ps(c2, _mm_set1_ps(*wk.add(2)), acc);
                    acc = _mm_fmadd_ps(c3, _mm_set1_ps(*wk.add(3)), acc);
                    acc = _mm_fmadd_ps(c4, _mm_set1_ps(*wk.add(4)), acc);
                }

                match act {
                    FusedActivation::None => {}
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
                    _ => {
                        let mut tmp = [0.0f32; 4];
                        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
                        for v in &mut tmp {
                            *v = apply_activation_scalar(*v, act);
                        }
                        acc = _mm_loadu_ps(tmp.as_ptr());
                    }
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

#[derive(Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum DepthwiseKernelKind {
    #[allow(dead_code)]
    Avx2_1dS1,
    #[allow(dead_code)]
    Avx2_3x3S1,
    #[allow(dead_code)]
    Avx2_5x5S1,
    #[allow(dead_code)]
    Avx2_3x3Sw2,
    #[allow(dead_code)]
    Avx2_5x5Sw2,
    #[allow(dead_code)]
    Avx2_5x1S1,
    #[allow(dead_code)]
    Neon_3x3S1,
    #[allow(dead_code)]
    Wasm_1dS1,
    #[allow(dead_code)]
    Wasm_3x3S1,
    #[allow(dead_code)]
    Wasm_5x5S1,
    #[allow(dead_code)]
    Wasm_3x3Sw2,
    #[allow(dead_code)]
    Wasm_5x5Sw2,
    #[allow(dead_code)]
    Wasm_5x1S1,
    Fallback,
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn apply_dw_act_wasm(
    acc: core::arch::wasm32::v128,
    act: FusedActivation,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    match act {
        FusedActivation::None => acc,
        FusedActivation::Relu => f32x4_pmax(acc, f32x4_splat(0.0)),
        FusedActivation::Relu6 => f32x4_pmin(f32x4_pmax(acc, f32x4_splat(0.0)), f32x4_splat(6.0)),
        FusedActivation::HardSwish => {
            let zero = f32x4_splat(0.0);
            let six = f32x4_splat(6.0);
            let p = f32x4_add(acc, f32x4_splat(3.0));
            let cl = f32x4_pmin(f32x4_pmax(p, zero), six);
            f32x4_mul(acc, f32x4_mul(cl, f32x4_splat(1.0 / 6.0)))
        }
        _ => {
            let mut arr = [0.0f32; 4];
            v128_store(arr.as_mut_ptr() as *mut v128, acc);
            for x in &mut arr {
                *x = apply_activation_scalar(*x, act);
            }
            v128_load(arr.as_ptr() as *const v128)
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn depthwise_3x3_s1_wasm(
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
    use core::arch::wasm32::*;

    let b_vec = f32x4_splat(bias);
    let w00 = f32x4_splat(w[0]);
    let w01 = f32x4_splat(w[1]);
    let w02 = f32x4_splat(w[2]);
    let w10 = f32x4_splat(w[3]);
    let w11 = f32x4_splat(w[4]);
    let w12 = f32x4_splat(w[5]);
    let w20 = f32x4_splat(w[6]);
    let w21 = f32x4_splat(w[7]);
    let w22 = f32x4_splat(w[8]);

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let act = config.activation;
    let v_limit = (in_w + pad_left).saturating_sub(2).min(out_w);

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_channel.as_ptr().add(ih_u * in_w);
            let r1 = in_channel.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_channel.as_ptr().add((ih_u + 2) * in_w);
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.min(out_w);
            for ow in 0..ow_start {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow - pad_left;
                let mut acc = b_vec;

                acc = f32x4_add(acc, f32x4_mul(v128_load(r0.add(iw) as *const v128), w00));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r0.add(iw + 1) as *const v128), w01));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r0.add(iw + 2) as *const v128), w02));

                acc = f32x4_add(acc, f32x4_mul(v128_load(r1.add(iw) as *const v128), w10));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r1.add(iw + 1) as *const v128), w11));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r1.add(iw + 2) as *const v128), w12));

                acc = f32x4_add(acc, f32x4_mul(v128_load(r2.add(iw) as *const v128), w20));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r2.add(iw + 1) as *const v128), w21));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r2.add(iw + 2) as *const v128), w22));

                v128_store(out_row.add(ow) as *mut v128, apply_dw_act_wasm(acc, act));
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn depthwise_5x5_s1_wasm(
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
    use core::arch::wasm32::*;

    let b_vec = f32x4_splat(bias);
    let mut w_vec = [f32x4_splat(0.0); 25];
    for i in 0..25 {
        w_vec[i] = f32x4_splat(w[i]);
    }

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let act = config.activation;
    let v_limit = (in_w + pad_left).saturating_sub(4).min(out_w);

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 4 < in_h as isize {
            let ih_u = ih_base as usize;
            let rows = [
                in_channel.as_ptr().add(ih_u * in_w),
                in_channel.as_ptr().add((ih_u + 1) * in_w),
                in_channel.as_ptr().add((ih_u + 2) * in_w),
                in_channel.as_ptr().add((ih_u + 3) * in_w),
                in_channel.as_ptr().add((ih_u + 4) * in_w),
            ];
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.min(out_w);
            for ow in 0..ow_start {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow - pad_left;
                let mut acc = b_vec;
                for (kh, &r_ptr) in rows.iter().enumerate() {
                    let wb = kh * 5;
                    acc = f32x4_add(acc, f32x4_mul(v128_load(r_ptr.add(iw) as *const v128), w_vec[wb]));
                    acc = f32x4_add(acc, f32x4_mul(v128_load(r_ptr.add(iw + 1) as *const v128), w_vec[wb + 1]));
                    acc = f32x4_add(acc, f32x4_mul(v128_load(r_ptr.add(iw + 2) as *const v128), w_vec[wb + 2]));
                    acc = f32x4_add(acc, f32x4_mul(v128_load(r_ptr.add(iw + 3) as *const v128), w_vec[wb + 3]));
                    acc = f32x4_add(acc, f32x4_mul(v128_load(r_ptr.add(iw + 4) as *const v128), w_vec[wb + 4]));
                }
                v128_store(out_row.add(ow) as *mut v128, apply_dw_act_wasm(acc, act));
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn depthwise_3x3_sw2_wasm(
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
    use core::arch::wasm32::*;

    let b_vec = f32x4_splat(bias);
    let w00 = f32x4_splat(w[0]);
    let w01 = f32x4_splat(w[1]);
    let w02 = f32x4_splat(w[2]);
    let w10 = f32x4_splat(w[3]);
    let w11 = f32x4_splat(w[4]);
    let w12 = f32x4_splat(w[5]);
    let w20 = f32x4_splat(w[6]);
    let w21 = f32x4_splat(w[7]);
    let w22 = f32x4_splat(w[8]);

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let stride_h = config.stride_h;
    let act = config.activation;
    let in_w_plus_pad = in_w + pad_left;

    for oh in 0..out_h {
        let ih_base = oh as isize * stride_h as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_channel.as_ptr().add(ih_u * in_w);
            let r1 = in_channel.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_channel.as_ptr().add((ih_u + 2) * in_w);
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.div_ceil(2).min(out_w);
            for ow in 0..ow_start {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 4 <= out_w && ow * 2 + 8 < in_w_plus_pad {
                let iw = ow * 2 - pad_left;
                let mut acc = b_vec;

                macro_rules! accum_sw2_row {
                    ($r:expr, $wk0:expr, $wk1:expr, $wk2:expr) => {{
                        let v0 = v128_load($r.add(iw) as *const v128);
                        let v1 = v128_load($r.add(iw + 4) as *const v128);
                        let c0 = i32x4_shuffle::<0, 2, 4, 6>(v0, v1);
                        let c1 = i32x4_shuffle::<1, 3, 5, 7>(v0, v1);
                        let c2 = f32x4_replace_lane::<3>(i32x4_shuffle::<2, 4, 6, 6>(v0, v1), *$r.add(iw + 8));
                        acc = f32x4_add(acc, f32x4_mul(c0, $wk0));
                        acc = f32x4_add(acc, f32x4_mul(c1, $wk1));
                        acc = f32x4_add(acc, f32x4_mul(c2, $wk2));
                    }};
                }

                accum_sw2_row!(r0, w00, w01, w02);
                accum_sw2_row!(r1, w10, w11, w12);
                accum_sw2_row!(r2, w20, w21, w22);

                v128_store(out_row.add(ow) as *mut v128, apply_dw_act_wasm(acc, act));
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn depthwise_5x1_s1_wasm(
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
    use core::arch::wasm32::*;

    let b_vec = f32x4_splat(bias);
    let w0 = f32x4_splat(w[0]);
    let w1 = f32x4_splat(w[1]);
    let w2 = f32x4_splat(w[2]);
    let w3 = f32x4_splat(w[3]);
    let w4 = f32x4_splat(w[4]);

    let pad_top = config.pad_top;
    let act = config.activation;
    let chunks = out_w / 4;

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

            for c_idx in 0..chunks {
                let ow = c_idx * 4;
                let mut acc = b_vec;
                acc = f32x4_add(acc, f32x4_mul(v128_load(r0.add(ow) as *const v128), w0));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r1.add(ow) as *const v128), w1));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r2.add(ow) as *const v128), w2));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r3.add(ow) as *const v128), w3));
                acc = f32x4_add(acc, f32x4_mul(v128_load(r4.add(ow) as *const v128), w4));
                v128_store(out_row.add(ow) as *mut v128, apply_dw_act_wasm(acc, act));
            }

            for ow in (chunks * 4)..out_w {
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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn depthwise_1d_s1_wasm(
    in_channel: &[f32],
    w: &[f32],
    bias: f32,
    out_channel: &mut [f32],
    in_len: usize,
    out_len: usize,
    k: usize,
    pad_before: usize,
    act: FusedActivation,
) {
    use core::arch::wasm32::*;

    let b_vec = f32x4_splat(bias);
    let in_ptr = in_channel.as_ptr();
    let out_ptr = out_channel.as_mut_ptr();

    let start = pad_before.min(out_len);
    for idx in 0..start {
        let val = eval_depthwise_1d_point(in_channel, w, bias, in_len, k, pad_before, idx);
        *out_ptr.add(idx) = apply_activation_scalar(val, act);
    }

    let mut idx = start;
    while idx + 8 <= out_len && (idx - pad_before) + k + 7 <= in_len {
        let base = idx - pad_before;
        let mut acc0 = b_vec;
        let mut acc1 = b_vec;
        for t in 0..k {
            let wt = f32x4_splat(*w.get_unchecked(t));
            let p = in_ptr.add(base + t);
            acc0 = f32x4_add(acc0, f32x4_mul(v128_load(p as *const v128), wt));
            acc1 = f32x4_add(acc1, f32x4_mul(v128_load(p.add(4) as *const v128), wt));
        }
        v128_store(out_ptr.add(idx) as *mut v128, apply_dw_act_wasm(acc0, act));
        v128_store(out_ptr.add(idx + 4) as *mut v128, apply_dw_act_wasm(acc1, act));
        idx += 8;
    }

    while idx + 4 <= out_len && (idx - pad_before) + k + 3 <= in_len {
        let base = idx - pad_before;
        let mut acc = b_vec;
        for t in 0..k {
            let wt = f32x4_splat(*w.get_unchecked(t));
            let v = v128_load(in_ptr.add(base + t) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(v, wt));
        }
        v128_store(out_ptr.add(idx) as *mut v128, apply_dw_act_wasm(acc, act));
        idx += 4;
    }

    while idx < out_len {
        let val = eval_depthwise_1d_point(in_channel, w, bias, in_len, k, pad_before, idx);
        *out_ptr.add(idx) = apply_activation_scalar(val, act);
        idx += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn depthwise_5x5_sw2_wasm(
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
    use core::arch::wasm32::*;

    let b_vec = f32x4_splat(bias);
    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let stride_h = config.stride_h;
    let act = config.activation;

    for oh in 0..out_h {
        let ih_base = oh as isize * stride_h as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 4 < in_h as isize {
            let ih_u = ih_base as usize;
            let out_row = out_channel.as_mut_ptr().add(oh * out_w);
            let ow_start = pad_left.div_ceil(2).min(out_w);

            for ow in 0..ow_start {
                let val = eval_depthwise_pixel(in_channel, w, bias, in_h, in_w, oh, ow, config);
                *out_row.add(ow) = apply_activation_scalar(val, act);
            }

            let mut ow = ow_start;
            while ow + 4 <= out_w && (ow * 2 - pad_left) + 12 <= in_w {
                let iw = ow * 2 - pad_left;
                let mut acc = b_vec;

                for kh in 0..5 {
                    let r_ptr = in_channel.as_ptr().add((ih_u + kh) * in_w + iw);
                    let wk = w.as_ptr().add(kh * 5);
                    let v0 = v128_load(r_ptr as *const v128);
                    let v1 = v128_load(r_ptr.add(4) as *const v128);
                    let v2 = v128_load(r_ptr.add(8) as *const v128);

                    let c0 = i32x4_shuffle::<0, 2, 4, 6>(v0, v1);
                    let c1 = i32x4_shuffle::<1, 3, 5, 7>(v0, v1);
                    let c2 = i32x4_shuffle::<1, 2, 3, 4>(c0, v2);
                    let c3 = i32x4_shuffle::<1, 2, 3, 5>(c1, v2);
                    let c4 = i32x4_shuffle::<2, 3, 4, 6>(c0, v2);

                    acc = f32x4_add(acc, f32x4_mul(c0, f32x4_splat(*wk)));
                    acc = f32x4_add(acc, f32x4_mul(c1, f32x4_splat(*wk.add(1))));
                    acc = f32x4_add(acc, f32x4_mul(c2, f32x4_splat(*wk.add(2))));
                    acc = f32x4_add(acc, f32x4_mul(c3, f32x4_splat(*wk.add(3))));
                    acc = f32x4_add(acc, f32x4_mul(c4, f32x4_splat(*wk.add(4))));
                }

                v128_store(out_row.add(ow) as *mut v128, apply_dw_act_wasm(acc, act));
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
            DepthwiseKernelKind::Avx2_1dS1 => {
                if let Some((in_len, out_len, k, pad_before)) =
                    extract_1d_s1_params(in_h, in_w, out_h, out_w, config)
                {
                    unsafe {
                        depthwise_1d_s1_avx2(
                            in_channel,
                            w_channel,
                            b_val,
                            out_channel,
                            in_len,
                            out_len,
                            k,
                            pad_before,
                            config.activation,
                        );
                    }
                    return;
                }
            }
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
            DepthwiseKernelKind::Avx2_5x5Sw2 => {
                unsafe {
                    depthwise_5x5_sw2_avx2(
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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        match kernel_kind {
            DepthwiseKernelKind::Wasm_1dS1 => {
                if let Some((in_len, out_len, k, pad_before)) =
                    extract_1d_s1_params(in_h, in_w, out_h, out_w, config)
                {
                    unsafe {
                        depthwise_1d_s1_wasm(
                            in_channel,
                            w_channel,
                            b_val,
                            out_channel,
                            in_len,
                            out_len,
                            k,
                            pad_before,
                            config.activation,
                        );
                    }
                    return;
                }
            }
            DepthwiseKernelKind::Wasm_3x3S1 => {
                unsafe {
                    depthwise_3x3_s1_wasm(
                        in_channel, w_channel, b_val, out_channel, in_h, in_w, out_h, out_w, config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Wasm_5x5S1 => {
                unsafe {
                    depthwise_5x5_s1_wasm(
                        in_channel, w_channel, b_val, out_channel, in_h, in_w, out_h, out_w, config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Wasm_3x3Sw2 => {
                unsafe {
                    depthwise_3x3_sw2_wasm(
                        in_channel, w_channel, b_val, out_channel, in_h, in_w, out_h, out_w, config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Wasm_5x5Sw2 => {
                unsafe {
                    depthwise_5x5_sw2_wasm(
                        in_channel, w_channel, b_val, out_channel, in_h, in_w, out_h, out_w, config,
                    );
                }
                return;
            }
            DepthwiseKernelKind::Wasm_5x1S1 => {
                unsafe {
                    depthwise_5x1_s1_wasm(
                        in_channel, w_channel, b_val, out_channel, in_h, in_w, out_h, out_w, config,
                    );
                }
                return;
            }
            _ => {}
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
                if extract_1d_s1_params(in_h, in_w, out_h, out_w, config).is_some() {
                    DepthwiseKernelKind::Avx2_1dS1
                } else if config.kernel_h == 3 && config.kernel_w == 3 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Avx2_3x3S1
                } else if config.kernel_h == 5 && config.kernel_w == 5 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Avx2_5x5S1
                } else if config.kernel_h == 3 && config.kernel_w == 3 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                    DepthwiseKernelKind::Avx2_3x3Sw2
                } else if config.kernel_h == 5 && config.kernel_w == 5 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                    DepthwiseKernelKind::Avx2_5x5Sw2
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
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if config.dilate_h == 1 && config.dilate_w == 1 {
                if extract_1d_s1_params(in_h, in_w, out_h, out_w, config).is_some() {
                    DepthwiseKernelKind::Wasm_1dS1
                } else if config.kernel_h == 3 && config.kernel_w == 3 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Wasm_3x3S1
                } else if config.kernel_h == 5 && config.kernel_w == 5 && config.stride_h == 1 && config.stride_w == 1 {
                    DepthwiseKernelKind::Wasm_5x5S1
                } else if config.kernel_h == 3 && config.kernel_w == 3 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                    DepthwiseKernelKind::Wasm_3x3Sw2
                } else if config.kernel_h == 5 && config.kernel_w == 5 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                    DepthwiseKernelKind::Wasm_5x5Sw2
                } else if config.kernel_h == 5 && config.kernel_w == 1 && config.stride_h == 1 && config.stride_w == 1 && config.pad_left == 0 && config.pad_right == 0 && in_w >= out_w {
                    DepthwiseKernelKind::Wasm_5x1S1
                } else {
                    DepthwiseKernelKind::Fallback
                }
            } else {
                DepthwiseKernelKind::Fallback
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run_reference(
        input: &[f32],
        batch: usize,
        in_h: usize,
        in_w: usize,
        weight: &[f32],
        bias: &[f32],
        cfg: &DepthwiseConv2dConfig,
    ) -> Vec<f32> {
        let (out_h, out_w) = depthwise_conv2d_output_dims(in_h, in_w, cfg);
        let hw_in = in_h * in_w;
        let hw_out = out_h * out_w;
        let ksz = cfg.kernel_h * cfg.kernel_w;
        let mut out = vec![0.0f32; batch * cfg.channels * hw_out];
        for b in 0..batch {
            for c in 0..cfg.channels {
                let in_c = &input[(b * cfg.channels + c) * hw_in..(b * cfg.channels + c + 1) * hw_in];
                let w_c = &weight[c * ksz..(c + 1) * ksz];
                let out_c = &mut out[(b * cfg.channels + c) * hw_out..(b * cfg.channels + c + 1) * hw_out];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let v = eval_depthwise_pixel(in_c, w_c, bias[c], in_h, in_w, oh, ow, cfg);
                        out_c[oh * out_w + ow] = apply_activation_scalar(v, cfg.activation);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_depthwise_1d_horizontal_and_vertical() {
        for &len in &[5usize, 9, 17, 79, 80, 133] {
            let channels = 4;
            let batch = 2;
            let input: Vec<f32> = (0..batch * channels * len)
                .map(|i| ((i * 31) % 97) as f32 * 0.05 - 2.3)
                .collect();
            let weight: Vec<f32> = (0..channels * 9)
                .map(|i| ((i * 19) % 41) as f32 * 0.04 - 0.8)
                .collect();
            let bias = vec![0.15, -0.25, 0.5, -1.1];

            for act in [FusedActivation::None, FusedActivation::Relu, FusedActivation::HardSwish] {
                let cfg_h = DepthwiseConv2dConfig {
                    channels,
                    kernel_h: 1,
                    kernel_w: 9,
                    stride_h: 1,
                    stride_w: 1,
                    pad_top: 0,
                    pad_left: 4,
                    pad_bottom: 0,
                    pad_right: 4,
                    dilate_h: 1,
                    dilate_w: 1,
                    activation: act,
                };
                let expected_h = run_reference(&input, batch, 1, len, &weight, &bias, &cfg_h);
                let mut actual_h = vec![0.0f32; expected_h.len()];
                depthwise_conv2d_nchw(&input, batch, 1, len, &weight, Some(&bias), &mut actual_h, &cfg_h);
                for (i, (&e, &a)) in expected_h.iter().zip(&actual_h).enumerate() {
                    assert!((e - a).abs() < 1e-4, "1x9 mismatch at {i} (len={len}): {e} vs {a}");
                }

                let cfg_v = DepthwiseConv2dConfig {
                    channels,
                    kernel_h: 9,
                    kernel_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                    pad_top: 4,
                    pad_left: 0,
                    pad_bottom: 4,
                    pad_right: 0,
                    dilate_h: 1,
                    dilate_w: 1,
                    activation: act,
                };
                let expected_v = run_reference(&input, batch, len, 1, &weight, &bias, &cfg_v);
                let mut actual_v = vec![0.0f32; expected_v.len()];
                depthwise_conv2d_nchw(&input, batch, len, 1, &weight, Some(&bias), &mut actual_v, &cfg_v);
                for (i, (&e, &a)) in expected_v.iter().zip(&actual_v).enumerate() {
                    assert!((e - a).abs() < 1e-4, "9x1 mismatch at {i} (len={len}): {e} vs {a}");
                }
            }
        }
    }

    #[test]
    fn test_depthwise_5x5_sw2() {
        for &(in_h, in_w, sh) in &[(17usize, 31usize, 2usize), (24, 40, 2), (12, 35, 1)] {
            let channels = 3;
            let batch = 2;
            let input: Vec<f32> = (0..batch * channels * in_h * in_w)
                .map(|i| ((i * 23) % 89) as f32 * 0.06 - 2.5)
                .collect();
            let weight: Vec<f32> = (0..channels * 25)
                .map(|i| ((i * 13) % 53) as f32 * 0.03 - 0.7)
                .collect();
            let bias = vec![0.2, -0.4, 0.9];

            for act in [FusedActivation::None, FusedActivation::HardSwish] {
                let cfg = DepthwiseConv2dConfig {
                    channels,
                    kernel_h: 5,
                    kernel_w: 5,
                    stride_h: sh,
                    stride_w: 2,
                    pad_top: 2,
                    pad_left: 2,
                    pad_bottom: 2,
                    pad_right: 2,
                    dilate_h: 1,
                    dilate_w: 1,
                    activation: act,
                };
                let expected = run_reference(&input, batch, in_h, in_w, &weight, &bias, &cfg);
                let mut actual = vec![0.0f32; expected.len()];
                depthwise_conv2d_nchw(&input, batch, in_h, in_w, &weight, Some(&bias), &mut actual, &cfg);
                for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
                    assert!((e - a).abs() < 1e-4, "5x5 sw2 mismatch at {i}: {e} vs {a}");
                }
            }
        }
    }
}
