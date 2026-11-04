#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::cell::RefCell;

use crate::kernels::activation::FusedActivation;
use crate::kernels::gemm::gemm;

thread_local! {
    static COL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Configuration parameters for 2D convolution
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
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
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub group: usize,
    pub activation: FusedActivation,
}

impl Default for Conv2dConfig {
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
            dilate_h: 1,
            dilate_w: 1,
            group: 1,
            activation: FusedActivation::None,
        }
    }
}

/// Calculate 2D convolution output spatial dimensions
pub fn conv2d_output_dims(
    in_h: usize,
    in_w: usize,
    config: &Conv2dConfig,
) -> (usize, usize) {
    let effective_kh = (config.kernel_h - 1) * config.dilate_h + 1;
    let effective_kw = (config.kernel_w - 1) * config.dilate_w + 1;

    let out_h = (in_h + config.pad_top + config.pad_bottom).saturating_sub(effective_kh) / config.stride_h + 1;
    let out_w = (in_w + config.pad_left + config.pad_right).saturating_sub(effective_kw) / config.stride_w + 1;

    (out_h, out_w)
}

/// Execute 2D Convolution: Input [N, C_in, H_in, W_in] -> Output [N, C_out, H_out, W_out]
pub fn conv2d(
    input: &[f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    config: &Conv2dConfig,
) {
    let (out_h, out_w) = conv2d_output_dims(in_h, in_w, config);
    let hw_out = out_h * out_w;
    let hw_in = in_h * in_w;

    let expected_out_len = batch * config.out_channels * hw_out;
    assert!(output.len() >= expected_out_len, "Output buffer size too small");

    // Fast path: 1x1 Pointwise Convolution (no im2col, direct continuous GEMM)
    if config.kernel_h == 1
        && config.kernel_w == 1
        && config.stride_h == 1
        && config.stride_w == 1
        && config.pad_top == 0
        && config.pad_left == 0
        && config.pad_bottom == 0
        && config.pad_right == 0
        && config.dilate_h == 1
        && config.dilate_w == 1
        && config.group == 1
    {
        let m = config.out_channels;
        let k = config.in_channels;
        let n = hw_out;

        for b in 0..batch {
            let in_b = &input[b * k * n..(b + 1) * k * n];
            let out_b = &mut output[b * m * n..(b + 1) * m * n];
            gemm(m, n, k, weight, in_b, out_b, 1.0, 0.0, bias, config.activation);
        }
        return;
    }

    // General Convolution via Im2Col + GEMM
    let group = config.group;
    let in_c_per_group = config.in_channels / group;
    let out_c_per_group = config.out_channels / group;
    let k_gemm = in_c_per_group * config.kernel_h * config.kernel_w;
    let m_gemm = out_c_per_group;
    let n_gemm = hw_out;
    let needed_col_len = k_gemm * n_gemm;

    COL_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < needed_col_len {
            buf.resize(needed_col_len, 0.0);
        }
        let col_buf = &mut buf[..needed_col_len];

        for b in 0..batch {
            let in_batch = &input[b * config.in_channels * hw_in..(b + 1) * config.in_channels * hw_in];
            let out_batch = &mut output[b * config.out_channels * hw_out..(b + 1) * config.out_channels * hw_out];

            for g in 0..group {
                let w_offset = g * out_c_per_group * k_gemm;
                let weight_group = &weight[w_offset..w_offset + out_c_per_group * k_gemm];

                let bias_group = bias.map(|b_slice| {
                    let start = g * out_c_per_group;
                    &b_slice[start..start + out_c_per_group]
                });

                // Im2Col into col_buf
                im2col(
                    in_batch,
                    g * in_c_per_group,
                    in_c_per_group,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    config,
                    col_buf,
                );

                let out_offset = g * out_c_per_group * hw_out;
                let out_group = &mut out_batch[out_offset..out_offset + out_c_per_group * hw_out];

                gemm(
                    m_gemm,
                    n_gemm,
                    k_gemm,
                    weight_group,
                    col_buf,
                    out_group,
                    1.0,
                    0.0,
                    bias_group,
                    config.activation,
                );
            }
        }
    });
}

/// Im2Col spatial layout extraction into column matrix
fn im2col(
    in_batch: &[f32],
    c_start: usize,
    num_channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Conv2dConfig,
    col_buf: &mut [f32],
) {
    let hw_out = out_h * out_w;
    let hw_in = in_h * in_w;
    let k_spatial = config.kernel_h * config.kernel_w;

    let fill_row = |row: usize, col_row: &mut [f32]| {
        let c = row / k_spatial;
        let rem = row % k_spatial;
        let kh = rem / config.kernel_w;
        let kw = rem % config.kernel_w;

        let channel_idx = c_start + c;
        let in_channel = &in_batch[channel_idx * hw_in..(channel_idx + 1) * hw_in];

        let mut col_idx = 0;
        for oh in 0..out_h {
            let ih_unclamped = oh as isize * config.stride_h as isize
                + kh as isize * config.dilate_h as isize
                - config.pad_top as isize;

            if ih_unclamped < 0 || ih_unclamped >= in_h as isize {
                col_row[col_idx..col_idx + out_w].fill(0.0);
                col_idx += out_w;
            } else {
                let ih = ih_unclamped as usize;
                let in_h_row = &in_channel[ih * in_w..(ih + 1) * in_w];

                if config.stride_w == 1 && config.dilate_w == 1 {
                    let shift = config.pad_left as isize - kw as isize;
                    let ow_start = (shift.max(0) as usize).min(out_w);
                    let ow_end = ((shift + in_w as isize).max(0) as usize).min(out_w);

                    if ow_start > 0 {
                        col_row[col_idx..col_idx + ow_start].fill(0.0);
                    }
                    if ow_end > ow_start {
                        let iw_start = (ow_start as isize - shift) as usize;
                        let iw_end = (ow_end as isize - shift) as usize;
                        col_row[col_idx + ow_start..col_idx + ow_end]
                            .copy_from_slice(&in_h_row[iw_start..iw_end]);
                    }
                    if ow_end < out_w {
                        col_row[col_idx + ow_end..col_idx + out_w].fill(0.0);
                    }
                    col_idx += out_w;
                } else if config.stride_w == 2 && config.dilate_w == 1 {
                    let shift = config.pad_left as isize - kw as isize;
                    let ow_start = if shift <= 0 {
                        0
                    } else {
                        (shift as usize).div_ceil(2).min(out_w)
                    };
                    let limit = shift + in_w as isize;
                    let ow_end = if limit <= 0 {
                        0
                    } else {
                        (limit as usize).div_ceil(2).min(out_w)
                    };

                    if ow_start > 0 {
                        col_row[col_idx..col_idx + ow_start].fill(0.0);
                    }
                    for ow in ow_start..ow_end {
                        let iw = (ow as isize * 2 - shift) as usize;
                        col_row[col_idx + ow] = in_h_row[iw];
                    }
                    if ow_end < out_w {
                        col_row[col_idx + ow_end..col_idx + out_w].fill(0.0);
                    }
                    col_idx += out_w;
                } else {
                    for ow in 0..out_w {
                        let iw_unclamped = ow as isize * config.stride_w as isize
                            + kw as isize * config.dilate_w as isize
                            - config.pad_left as isize;

                        if iw_unclamped >= 0 && iw_unclamped < in_w as isize {
                            col_row[col_idx] = in_h_row[iw_unclamped as usize];
                        } else {
                            col_row[col_idx] = 0.0;
                        }
                        col_idx += 1;
                    }
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        if num_channels * k_spatial >= 4 {
            col_buf
                .par_chunks_exact_mut(hw_out)
                .enumerate()
                .for_each(|(row, col_row)| {
                    fill_row(row, col_row);
                });
            return;
        }
    }

    let total_rows = num_channels * k_spatial;
    for row in 0..total_rows {
        let col_row = &mut col_buf[row * hw_out..(row + 1) * hw_out];
        fill_row(row, col_row);
    }
}
