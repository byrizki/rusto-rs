#[cfg(feature = "parallel")]
use crate::par::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

/// Configuration parameters for 2D transposed convolution (deconvolution)
#[derive(Debug, Clone)]
pub struct Deconv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub out_pad_h: usize,
    pub out_pad_w: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub group: usize,
    pub activation: FusedActivation,
}

impl Default for Deconv2dConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_top: 0,
            pad_left: 0,
            pad_bottom: 0,
            pad_right: 0,
            out_pad_h: 0,
            out_pad_w: 0,
            dilate_h: 1,
            dilate_w: 1,
            group: 1,
            activation: FusedActivation::None,
        }
    }
}

/// Calculate 2D transposed convolution output dimensions
pub fn deconv2d_output_dims(
    in_h: usize,
    in_w: usize,
    config: &Deconv2dConfig,
) -> (usize, usize) {
    let out_h = (in_h - 1) * config.stride_h
        + config.dilate_h * (config.kernel_h - 1)
        + 1
        + config.out_pad_h;
    let out_h = out_h.saturating_sub(config.pad_top + config.pad_bottom);

    let out_w = (in_w - 1) * config.stride_w
        + config.dilate_w * (config.kernel_w - 1)
        + 1
        + config.out_pad_w;
    let out_w = out_w.saturating_sub(config.pad_left + config.pad_right);

    (out_h, out_w)
}

/// Process a single output channel of deconvolution
fn process_deconv_channel(
    in_batch: &[f32],
    g: usize,
    oc: usize,
    in_c_per_group: usize,
    out_c_per_group: usize,
    hw_in: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    weight: &[f32],
    b_val: f32,
    out_channel: &mut [f32],
    config: &Deconv2dConfig,
) {
    let k_size = config.kernel_h * config.kernel_w;
    out_channel.fill(0.0);

    for ic in 0..in_c_per_group {
        let in_c_idx = g * in_c_per_group + ic;
        let in_channel = &in_batch[in_c_idx * hw_in..(in_c_idx + 1) * hw_in];

        // Weight layout: [in_c, out_c_per_group, kh, kw]
        let w_idx = (in_c_idx * out_c_per_group + oc) * k_size;
        let kernel = &weight[w_idx..w_idx + k_size];

        for ih in 0..in_h {
            for iw in 0..in_w {
                let in_val = in_channel[ih * in_w + iw];
                if in_val == 0.0 {
                    continue;
                }

                for kh in 0..config.kernel_h {
                    let oh_unclamped = ih as isize * config.stride_h as isize
                        + kh as isize * config.dilate_h as isize
                        - config.pad_top as isize;

                    if oh_unclamped >= 0 && oh_unclamped < out_h as isize {
                        let oh = oh_unclamped as usize;
                        let out_row = &mut out_channel[oh * out_w..(oh + 1) * out_w];
                        let k_row = &kernel[kh * config.kernel_w..(kh + 1) * config.kernel_w];

                        for kw in 0..config.kernel_w {
                            let ow_unclamped = iw as isize * config.stride_w as isize
                                + kw as isize * config.dilate_w as isize
                                - config.pad_left as isize;

                            if ow_unclamped >= 0 && ow_unclamped < out_w as isize {
                                out_row[ow_unclamped as usize] += in_val * k_row[kw];
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply bias and activation
    for v in out_channel.iter_mut() {
        *v = apply_activation_scalar(*v + b_val, config.activation);
    }
}

fn deconv2d_2x2_s2(
    in_batch: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out_batch: &mut [f32],
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    act: FusedActivation,
) {
    let out_h = in_h * 2;
    let out_w = in_w * 2;
    let hw_in = in_h * in_w;
    let hw_out = out_h * out_w;

    #[cfg(feature = "parallel")]
    if crate::par::current_thread_index().is_none() {
        if out_c == 1 {
            let b_val = bias.map(|b| b[0]).unwrap_or(0.0);
            out_batch
                .par_chunks_exact_mut(2 * out_w)
                .enumerate()
                .for_each(|(ih, out_2rows)| {
                    let (out_row0, out_row1) = out_2rows.split_at_mut(out_w);
                    out_row0.fill(b_val);
                    out_row1.fill(b_val);

                    for ic in 0..in_c {
                        let in_channel = &in_batch[ic * hw_in..(ic + 1) * hw_in];
                        let w_idx = ic * 4;
                        let k00 = weight[w_idx];
                        let k01 = weight[w_idx + 1];
                        let k10 = weight[w_idx + 2];
                        let k11 = weight[w_idx + 3];

                        let in_row = &in_channel[ih * in_w..(ih + 1) * in_w];
                        for iw in 0..in_w {
                            let val = in_row[iw];
                            out_row0[2 * iw] += val * k00;
                            out_row0[2 * iw + 1] += val * k01;
                            out_row1[2 * iw] += val * k10;
                            out_row1[2 * iw + 1] += val * k11;
                        }
                    }

                    if act != FusedActivation::None {
                        for v in out_row0.iter_mut().chain(out_row1.iter_mut()) {
                            *v = apply_activation_scalar(*v, act);
                        }
                    }
                });
            return;
        }

        out_batch
            .par_chunks_exact_mut(hw_out)
            .enumerate()
            .for_each(|(oc, out_channel)| {
                let b_val = bias.map(|b| b[oc]).unwrap_or(0.0);
                out_channel.fill(b_val);

                for ic in 0..in_c {
                    let in_channel = &in_batch[ic * hw_in..(ic + 1) * hw_in];
                    let w_idx = (ic * out_c + oc) * 4;
                    let k00 = weight[w_idx];
                    let k01 = weight[w_idx + 1];
                    let k10 = weight[w_idx + 2];
                    let k11 = weight[w_idx + 3];

                    for ih in 0..in_h {
                        let in_row = &in_channel[ih * in_w..(ih + 1) * in_w];
                        let (out_row0, out_row1) = out_channel[(2 * ih) * out_w..(2 * ih + 2) * out_w].split_at_mut(out_w);

                        for iw in 0..in_w {
                            let val = in_row[iw];
                            out_row0[2 * iw] += val * k00;
                            out_row0[2 * iw + 1] += val * k01;
                            out_row1[2 * iw] += val * k10;
                            out_row1[2 * iw + 1] += val * k11;
                        }
                    }
                }

                if act != FusedActivation::None {
                    for v in out_channel.iter_mut() {
                        *v = apply_activation_scalar(*v, act);
                    }
                }
            });
        return;
    }

    for oc in 0..out_c {
        let b_val = bias.map(|b| b[oc]).unwrap_or(0.0);
        let out_channel = &mut out_batch[oc * hw_out..(oc + 1) * hw_out];
        out_channel.fill(b_val);

        for ic in 0..in_c {
            let in_channel = &in_batch[ic * hw_in..(ic + 1) * hw_in];
            let w_idx = (ic * out_c + oc) * 4;
            let k00 = weight[w_idx];
            let k01 = weight[w_idx + 1];
            let k10 = weight[w_idx + 2];
            let k11 = weight[w_idx + 3];

            for ih in 0..in_h {
                let in_row = &in_channel[ih * in_w..(ih + 1) * in_w];
                let (out_row0, out_row1) = out_channel[(2 * ih) * out_w..(2 * ih + 2) * out_w].split_at_mut(out_w);

                for iw in 0..in_w {
                    let val = in_row[iw];
                    out_row0[2 * iw] += val * k00;
                    out_row0[2 * iw + 1] += val * k01;
                    out_row1[2 * iw] += val * k10;
                    out_row1[2 * iw + 1] += val * k11;
                }
            }
        }

        if act != FusedActivation::None {
            for v in out_channel.iter_mut() {
                *v = apply_activation_scalar(*v, act);
            }
        }
    }
}

/// Execute 2D Transposed Convolution: Input [N, C_in, H_in, W_in] -> Output [N, C_out, H_out, W_out]
pub fn deconv2d(
    input: &[f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    config: &Deconv2dConfig,
) {
    let (out_h, out_w) = deconv2d_output_dims(in_h, in_w, config);
    let hw_out = out_h * out_w;
    let hw_in = in_h * in_w;

    let group = config.group;
    let in_c_per_group = config.in_channels / group;
    let out_c_per_group = config.out_channels / group;

    let total_out_elems = batch * config.out_channels * hw_out;
    assert!(output.len() >= total_out_elems);

    // Fast path: 2x2 Transposed Convolution with stride 2 and pad 0 (common for 2x upsampling)
    if config.kernel_h == 2
        && config.kernel_w == 2
        && config.stride_h == 2
        && config.stride_w == 2
        && config.pad_top == 0
        && config.pad_left == 0
        && config.pad_bottom == 0
        && config.pad_right == 0
        && config.dilate_h == 1
        && config.dilate_w == 1
        && config.group == 1
        && config.out_pad_h == 0
        && config.out_pad_w == 0
    {
        for b in 0..batch {
            let in_batch = &input[b * config.in_channels * hw_in..(b + 1) * config.in_channels * hw_in];
            let out_batch = &mut output[b * config.out_channels * hw_out..(b + 1) * config.out_channels * hw_out];
            deconv2d_2x2_s2(
                in_batch,
                weight,
                bias,
                out_batch,
                config.in_channels,
                config.out_channels,
                in_h,
                in_w,
                config.activation,
            );
        }
        return;
    }

    for b in 0..batch {
        let in_batch = &input[b * config.in_channels * hw_in..(b + 1) * config.in_channels * hw_in];
        let out_batch = &mut output[b * config.out_channels * hw_out..(b + 1) * config.out_channels * hw_out];

        #[cfg(feature = "parallel")]
        {
            if crate::par::current_thread_index().is_none() && config.out_channels >= 2 {
                out_batch
                    .par_chunks_exact_mut(hw_out)
                    .enumerate()
                    .for_each(|(global_oc, out_channel)| {
                        let g = global_oc / out_c_per_group;
                        let oc = global_oc % out_c_per_group;
                        let b_val = bias.map(|b_slice| b_slice[global_oc]).unwrap_or(0.0);
                        process_deconv_channel(
                            in_batch,
                            g,
                            oc,
                            in_c_per_group,
                            out_c_per_group,
                            hw_in,
                            in_h,
                            in_w,
                            out_h,
                            out_w,
                            weight,
                            b_val,
                            out_channel,
                            config,
                        );
                    });
                continue;
            }
        }

        for global_oc in 0..config.out_channels {
            let g = global_oc / out_c_per_group;
            let oc = global_oc % out_c_per_group;
            let out_channel = &mut out_batch[global_oc * hw_out..(global_oc + 1) * hw_out];
            let b_val = bias.map(|b_slice| b_slice[global_oc]).unwrap_or(0.0);
            process_deconv_channel(
                in_batch,
                g,
                oc,
                in_c_per_group,
                out_c_per_group,
                hw_in,
                in_h,
                in_w,
                out_h,
                out_w,
                weight,
                b_val,
                out_channel,
                config,
            );
        }
    }
}

