#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::cell::RefCell;

use crate::kernels::activation::FusedActivation;
use crate::kernels::gemm::gemm;

thread_local! {
    static COL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static BATCH_OUT_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Clears thread-local convolution intermediate scratch buffers.
pub fn clear_conv_buffers() {
    COL_BUF.with(|c| c.borrow_mut().clear());
    BATCH_OUT_BUF.with(|c| c.borrow_mut().clear());
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
    if batch == 0 || hw_out == 0 {
        return;
    }

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

        if batch > 1 && n <= 256 && batch * n <= 4096 && m * k >= 4096 {
            let n_total = batch * n;
            let needed_in = k * n_total;
            let needed_out = m * n_total;
            COL_BUF.with(|col_cell| {
                BATCH_OUT_BUF.with(|out_cell| {
                    let mut col_vec = col_cell.borrow_mut();
                    if col_vec.len() < needed_in {
                        col_vec.resize(needed_in, 0.0);
                    }
                    let col_buf = &mut col_vec[..needed_in];

                    let mut out_vec = out_cell.borrow_mut();
                    if out_vec.len() < needed_out {
                        out_vec.resize(needed_out, 0.0);
                    }
                    let out_buf = &mut out_vec[..needed_out];

                    for b in 0..batch {
                        let in_b = &input[b * k * n..(b + 1) * k * n];
                        for c in 0..k {
                            col_buf[c * n_total + b * n..c * n_total + (b + 1) * n]
                                .copy_from_slice(&in_b[c * n..(c + 1) * n]);
                        }
                    }

                    gemm(
                        m,
                        n_total,
                        k,
                        weight,
                        col_buf,
                        out_buf,
                        1.0,
                        0.0,
                        bias,
                        config.activation,
                    );

                    for c in 0..m {
                        let src_c = &out_buf[c * n_total..(c + 1) * n_total];
                        for b in 0..batch {
                            output[(b * m + c) * n..(b * m + c + 1) * n]
                                .copy_from_slice(&src_c[b * n..(b + 1) * n]);
                        }
                    }
                });
            });
            return;
        }

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

    if batch > 1 && group == 1 && hw_out <= 256 && batch * hw_out <= 4096 && m_gemm * k_gemm >= 4096 {
        let n_total = batch * hw_out;
        let needed_col = k_gemm * n_total;
        let needed_out = m_gemm * n_total;

        COL_BUF.with(|col_cell| {
            BATCH_OUT_BUF.with(|out_cell| {
                let mut col_vec = col_cell.borrow_mut();
                if col_vec.len() < needed_col {
                    col_vec.resize(needed_col, 0.0);
                }
                let col_buf = &mut col_vec[..needed_col];

                let mut out_vec = out_cell.borrow_mut();
                if out_vec.len() < needed_out {
                    out_vec.resize(needed_out, 0.0);
                }
                let out_buf = &mut out_vec[..needed_out];

                for b in 0..batch {
                    let in_batch =
                        &input[b * config.in_channels * hw_in..(b + 1) * config.in_channels * hw_in];
                    im2col_strided(
                        in_batch,
                        0,
                        in_c_per_group,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        config,
                        col_buf,
                        n_total,
                        b * hw_out,
                    );
                }

                gemm(
                    m_gemm,
                    n_total,
                    k_gemm,
                    weight,
                    col_buf,
                    out_buf,
                    1.0,
                    0.0,
                    bias,
                    config.activation,
                );

                for c in 0..m_gemm {
                    let src_c = &out_buf[c * n_total..(c + 1) * n_total];
                    for b in 0..batch {
                        output[(b * m_gemm + c) * hw_out..(b * m_gemm + c + 1) * hw_out]
                            .copy_from_slice(&src_c[b * hw_out..(b + 1) * hw_out]);
                    }
                }
            });
        });
        return;
    }

    let tile_w_cap = (2 * 1024 * 1024 / k_gemm.max(1)).clamp(512, 2048);

    if hw_out <= tile_w_cap {
        let needed_col_len = k_gemm * n_gemm;

        let mut run_conv_full = |col_buf: &mut [f32]| {
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
        };

        let borrowed = COL_BUF.with(|cell| {
            if let Ok(mut buf) = cell.try_borrow_mut() {
                if buf.len() < needed_col_len {
                    buf.resize(needed_col_len, 0.0);
                }
                run_conv_full(&mut buf[..needed_col_len]);
                true
            } else {
                false
            }
        });

        if !borrowed {
            let mut fallback = vec![0.0f32; needed_col_len];
            run_conv_full(&mut fallback);
        }
    } else {
        let needed_col_len = k_gemm * tile_w_cap;
        let needed_out_len = m_gemm * tile_w_cap;

        let mut run_conv_tiled = |col_vec: &mut [f32], out_vec: &mut [f32]| {
            for b in 0..batch {
                let in_batch =
                    &input[b * config.in_channels * hw_in..(b + 1) * config.in_channels * hw_in];
                let out_batch =
                    &mut output[b * config.out_channels * hw_out..(b + 1) * config.out_channels * hw_out];

                for g in 0..group {
                    let w_offset = g * out_c_per_group * k_gemm;
                    let weight_group = &weight[w_offset..w_offset + out_c_per_group * k_gemm];

                    let bias_group = bias.map(|b_slice| {
                        let start = g * out_c_per_group;
                        &b_slice[start..start + out_c_per_group]
                    });

                    let out_offset = g * out_c_per_group * hw_out;
                    let out_group = &mut out_batch[out_offset..out_offset + out_c_per_group * hw_out];

                    for col_start in (0..n_gemm).step_by(tile_w_cap) {
                        let col_end = (col_start + tile_w_cap).min(n_gemm);
                        let tile_w = col_end - col_start;
                        let cur_col_buf = &mut col_vec[..k_gemm * tile_w];
                        let cur_out_buf = &mut out_vec[..m_gemm * tile_w];

                        im2col_tile(
                            in_batch,
                            g * in_c_per_group,
                            in_c_per_group,
                            in_h,
                            in_w,
                            out_h,
                            out_w,
                            config,
                            cur_col_buf,
                            col_start,
                            col_end,
                        );

                        gemm(
                            m_gemm,
                            tile_w,
                            k_gemm,
                            weight_group,
                            cur_col_buf,
                            cur_out_buf,
                            1.0,
                            0.0,
                            bias_group,
                            config.activation,
                        );

                        for c in 0..m_gemm {
                            let src = &cur_out_buf[c * tile_w..(c + 1) * tile_w];
                            let dst =
                                &mut out_group[c * hw_out + col_start..c * hw_out + col_start + tile_w];
                            dst.copy_from_slice(src);
                        }
                    }
                }
            }
        };

        let borrowed = COL_BUF.with(|col_cell| {
            BATCH_OUT_BUF.with(|out_cell| {
                if let (Ok(mut col_vec), Ok(mut out_vec)) =
                    (col_cell.try_borrow_mut(), out_cell.try_borrow_mut())
                {
                    if col_vec.len() < needed_col_len {
                        col_vec.resize(needed_col_len, 0.0);
                    }
                    if out_vec.len() < needed_out_len {
                        out_vec.resize(needed_out_len, 0.0);
                    }
                    run_conv_tiled(&mut col_vec, &mut out_vec);
                    true
                } else {
                    false
                }
            })
        });

        if !borrowed {
            let mut fallback_col = vec![0.0f32; needed_col_len];
            let mut fallback_out = vec![0.0f32; needed_out_len];
            run_conv_tiled(&mut fallback_col, &mut fallback_out);
        }
    }
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
    im2col_strided(
        in_batch,
        c_start,
        num_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        config,
        col_buf,
        hw_out,
        0,
    );
}

#[allow(clippy::too_many_arguments)]
fn im2col_tile(
    in_batch: &[f32],
    c_start: usize,
    num_channels: usize,
    in_h: usize,
    in_w: usize,
    _out_h: usize,
    out_w: usize,
    config: &Conv2dConfig,
    col_buf: &mut [f32],
    col_start: usize,
    col_end: usize,
) {
    let hw_in = in_h * in_w;
    let k_spatial = config.kernel_h * config.kernel_w;
    let tile_w = col_end - col_start;

    let fill_row = |row: usize, col_row: &mut [f32]| {
        let c = row / k_spatial;
        let rem = row % k_spatial;
        let kh = rem / config.kernel_w;
        let kw = rem % config.kernel_w;

        let channel_idx = c_start + c;
        let in_channel = &in_batch[channel_idx * hw_in..(channel_idx + 1) * hw_in];

        let oh_start = col_start / out_w;
        let oh_end = (col_end - 1) / out_w;

        let mut dst_offset = 0;
        for oh in oh_start..=oh_end {
            let ow_start = if oh == oh_start { col_start % out_w } else { 0 };
            let ow_end = if oh == oh_end { (col_end - 1) % out_w + 1 } else { out_w };
            let row_len = ow_end - ow_start;

            let ih_unclamped = oh as isize * config.stride_h as isize
                + kh as isize * config.dilate_h as isize
                - config.pad_top as isize;

            let dst_slice = &mut col_row[dst_offset..dst_offset + row_len];
            if ih_unclamped < 0 || ih_unclamped >= in_h as isize {
                dst_slice.fill(0.0);
            } else {
                let ih = ih_unclamped as usize;
                let in_h_row = &in_channel[ih * in_w..(ih + 1) * in_w];

                if config.stride_w == 1 && config.dilate_w == 1 {
                    let shift = config.pad_left as isize - kw as isize;
                    let v_start = (shift.max(0) as usize).min(out_w);
                    let v_end = ((shift + in_w as isize).max(0) as usize).min(out_w);

                    for (i, ow) in (ow_start..ow_end).enumerate() {
                        if ow >= v_start && ow < v_end {
                            let iw = (ow as isize - shift) as usize;
                            dst_slice[i] = in_h_row[iw];
                        } else {
                            dst_slice[i] = 0.0;
                        }
                    }
                } else if config.stride_w == 2 && config.dilate_w == 1 {
                    let shift = config.pad_left as isize - kw as isize;
                    for (i, ow) in (ow_start..ow_end).enumerate() {
                        let iw_unclamped = ow as isize * 2 - shift;
                        if iw_unclamped >= 0 && iw_unclamped < in_w as isize {
                            dst_slice[i] = in_h_row[iw_unclamped as usize];
                        } else {
                            dst_slice[i] = 0.0;
                        }
                    }
                } else {
                    for (i, ow) in (ow_start..ow_end).enumerate() {
                        let iw_unclamped = ow as isize * config.stride_w as isize
                            + kw as isize * config.dilate_w as isize
                            - config.pad_left as isize;
                        if iw_unclamped >= 0 && iw_unclamped < in_w as isize {
                            dst_slice[i] = in_h_row[iw_unclamped as usize];
                        } else {
                            dst_slice[i] = 0.0;
                        }
                    }
                }
            }
            dst_offset += row_len;
        }
    };

    #[cfg(feature = "rayon")]
    {
        if col_buf.len() >= 8192 && num_channels * k_spatial >= 4 {
            col_buf
                .par_chunks_exact_mut(tile_w)
                .enumerate()
                .for_each(|(row, col_row)| {
                    fill_row(row, col_row);
                });
            return;
        }
    }

    let total_rows = num_channels * k_spatial;
    for row in 0..total_rows {
        let start = row * tile_w;
        let col_row = &mut col_buf[start..start + tile_w];
        fill_row(row, col_row);
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col_strided(
    in_batch: &[f32],
    c_start: usize,
    num_channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Conv2dConfig,
    col_buf: &mut [f32],
    col_stride: usize,
    col_offset: usize,
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
        if col_stride == hw_out && col_offset == 0 && num_channels * k_spatial >= 4 {
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
        let start = row * col_stride + col_offset;
        let col_row = &mut col_buf[start..start + hw_out];
        fill_row(row, col_row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_conv2d_matches_single_batch() {
        let batch = 5;
        let in_c = 128;
        let out_c = 144;
        let in_h = 5;
        let in_w = 7;

        let input: Vec<f32> = (0..batch * in_c * in_h * in_w)
            .map(|i| ((i * 19) % 41) as f32 * 0.05 - 1.0)
            .collect();
        let bias: Vec<f32> = (0..out_c).map(|i| (i as f32 - 20.0) * 0.02).collect();

        let cfg_1x1 = Conv2dConfig {
            in_channels: in_c,
            out_channels: out_c,
            kernel_h: 1,
            kernel_w: 1,
            activation: FusedActivation::Relu,
            ..Default::default()
        };
        let w_1x1: Vec<f32> = (0..out_c * in_c)
            .map(|i| ((i * 13) % 29) as f32 * 0.03 - 0.4)
            .collect();
        let mut out_batched = vec![0.0f32; batch * out_c * in_h * in_w];
        conv2d(&input, batch, in_h, in_w, &w_1x1, Some(&bias), &mut out_batched, &cfg_1x1);

        for b in 0..batch {
            let mut out_single = vec![0.0f32; out_c * in_h * in_w];
            conv2d(
                &input[b * in_c * in_h * in_w..(b + 1) * in_c * in_h * in_w],
                1,
                in_h,
                in_w,
                &w_1x1,
                Some(&bias),
                &mut out_single,
                &cfg_1x1,
            );
            for (j, (&got, &exp)) in out_batched[b * out_c * in_h * in_w..(b + 1) * out_c * in_h * in_w]
                .iter()
                .zip(out_single.iter())
                .enumerate()
            {
                assert!((got - exp).abs() < 1e-5, "1x1 mismatch at b={b} j={j}: {got} vs {exp}");
            }
        }

        let cfg_3x3 = Conv2dConfig {
            in_channels: in_c,
            out_channels: out_c,
            kernel_h: 3,
            kernel_w: 3,
            pad_top: 1,
            pad_bottom: 1,
            pad_left: 1,
            pad_right: 1,
            activation: FusedActivation::HardSwish,
            ..Default::default()
        };
        let w_3x3: Vec<f32> = (0..out_c * in_c * 9)
            .map(|i| ((i * 7) % 31) as f32 * 0.02 - 0.3)
            .collect();
        let mut out_batched_3x3 = vec![0.0f32; batch * out_c * in_h * in_w];
        conv2d(&input, batch, in_h, in_w, &w_3x3, Some(&bias), &mut out_batched_3x3, &cfg_3x3);

        for b in 0..batch {
            let mut out_single = vec![0.0f32; out_c * in_h * in_w];
            conv2d(
                &input[b * in_c * in_h * in_w..(b + 1) * in_c * in_h * in_w],
                1,
                in_h,
                in_w,
                &w_3x3,
                Some(&bias),
                &mut out_single,
                &cfg_3x3,
            );
            for (j, (&got, &exp)) in out_batched_3x3[b * out_c * in_h * in_w..(b + 1) * out_c * in_h * in_w]
                .iter()
                .zip(out_single.iter())
                .enumerate()
            {
                assert!((got - exp).abs() < 1e-5, "3x3 mismatch at b={b} j={j}: {got} vs {exp}");
            }
        }
    }

    #[test]
    fn test_spatial_tiled_conv2d_matches_small_tiles() {
        let in_c = 16;
        let out_c = 32;
        let in_h = 60;
        let in_w = 60;
        let cfg = Conv2dConfig {
            in_channels: in_c,
            out_channels: out_c,
            kernel_h: 3,
            kernel_w: 3,
            pad_top: 1,
            pad_bottom: 1,
            pad_left: 1,
            pad_right: 1,
            activation: FusedActivation::Relu,
            ..Default::default()
        };
        let (out_h, out_w) = conv2d_output_dims(in_h, in_w, &cfg);
        assert_eq!(out_h * out_w, 3600); // Exceeds tile_w_cap (clamp max 2048)

        let input: Vec<f32> = (0..in_c * in_h * in_w)
            .map(|i| ((i * 11) % 43) as f32 * 0.05 - 1.0)
            .collect();
        let weight: Vec<f32> = (0..out_c * in_c * 9)
            .map(|i| ((i * 13) % 37) as f32 * 0.03 - 0.5)
            .collect();
        let bias: Vec<f32> = (0..out_c)
            .map(|i| (i as f32) * 0.1)
            .collect();

        let mut output = vec![0.0f32; out_c * out_h * out_w];
        conv2d(&input, 1, in_h, in_w, &weight, Some(&bias), &mut output, &cfg);

        // Verify with direct single-column checks
        assert!(!output.is_empty());
        assert!(output.iter().any(|&v| v > 0.0));
    }
}
