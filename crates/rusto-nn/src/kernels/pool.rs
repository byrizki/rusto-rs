#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolType {
    MaxPool,
    AvgPool,
}

#[derive(Debug, Clone)]
pub struct Pool2dConfig {
    pub pool_type: PoolType,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub ceil_mode: bool,
    pub count_include_pad: bool,
    pub is_global: bool,
}

impl Default for Pool2dConfig {
    fn default() -> Self {
        Self {
            pool_type: PoolType::MaxPool,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_top: 0,
            pad_left: 0,
            pad_bottom: 0,
            pad_right: 0,
            ceil_mode: false,
            count_include_pad: true,
            is_global: false,
        }
    }
}

pub fn pool2d_output_dims(
    in_h: usize,
    in_w: usize,
    config: &Pool2dConfig,
) -> (usize, usize) {
    if config.is_global {
        return (1, 1);
    }

    let h_total = in_h + config.pad_top + config.pad_bottom;
    let w_total = in_w + config.pad_left + config.pad_right;

    let out_h = if config.ceil_mode {
        if h_total >= config.kernel_h {
            (h_total - config.kernel_h).div_ceil(config.stride_h) + 1
        } else {
            1
        }
    } else {
        if h_total >= config.kernel_h {
            (h_total - config.kernel_h) / config.stride_h + 1
        } else {
            1
        }
    };

    let out_w = if config.ceil_mode {
        if w_total >= config.kernel_w {
            (w_total - config.kernel_w).div_ceil(config.stride_w) + 1
        } else {
            1
        }
    } else {
        if w_total >= config.kernel_w {
            (w_total - config.kernel_w) / config.stride_w + 1
        } else {
            1
        }
    };

    (out_h, out_w)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn maxpool_3x3_s1_avx2(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_w: usize,
    out_h: usize,
    out_w: usize,
) {
    use core::arch::x86_64::*;

    for oh in 0..out_h {
        let r0 = in_ch.as_ptr().add(oh * in_w);
        let r1 = in_ch.as_ptr().add((oh + 1) * in_w);
        let r2 = in_ch.as_ptr().add((oh + 2) * in_w);
        let out_row = out_ch.as_mut_ptr().add(oh * out_w);

        let v_chunks = out_w / 8;
        for c_idx in 0..v_chunks {
            let ow = c_idx * 8;

            let r0_0 = _mm256_loadu_ps(r0.add(ow));
            let r0_1 = _mm256_loadu_ps(r0.add(ow + 1));
            let r0_2 = _mm256_loadu_ps(r0.add(ow + 2));
            let m0 = _mm256_max_ps(r0_0, _mm256_max_ps(r0_1, r0_2));

            let r1_0 = _mm256_loadu_ps(r1.add(ow));
            let r1_1 = _mm256_loadu_ps(r1.add(ow + 1));
            let r1_2 = _mm256_loadu_ps(r1.add(ow + 2));
            let m1 = _mm256_max_ps(r1_0, _mm256_max_ps(r1_1, r1_2));

            let r2_0 = _mm256_loadu_ps(r2.add(ow));
            let r2_1 = _mm256_loadu_ps(r2.add(ow + 1));
            let r2_2 = _mm256_loadu_ps(r2.add(ow + 2));
            let m2 = _mm256_max_ps(r2_0, _mm256_max_ps(r2_1, r2_2));

            let final_max = _mm256_max_ps(m0, _mm256_max_ps(m1, m2));
            _mm256_storeu_ps(out_row.add(ow), final_max);
        }

        for ow in (v_chunks * 8)..out_w {
            let mut max_val = *r0.add(ow);
            for r in [r0, r1, r2] {
                for kw in 0..3 {
                    let v = *r.add(ow + kw);
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            *out_row.add(ow) = max_val;
        }
    }
}

fn process_pool_channel(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    kh: usize,
    kw: usize,
    config: &Pool2dConfig,
) {
    if config.is_global {
        let total = (in_h * in_w) as f32;
        match config.pool_type {
            PoolType::MaxPool => {
                out_ch[0] = in_ch.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            }
            PoolType::AvgPool => {
                out_ch[0] = in_ch.iter().sum::<f32>() / total;
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && config.pool_type == PoolType::MaxPool
            && kh == 3
            && kw == 3
            && config.stride_h == 1
            && config.stride_w == 1
            && config.pad_top == 0
            && config.pad_left == 0
            && in_h >= out_h + 2
            && in_w >= out_w + 2
        {
            unsafe {
                maxpool_3x3_s1_avx2(in_ch, out_ch, in_w, out_h, out_w);
            }
            return;
        }
    }

    for oh in 0..out_h {
        let h_start = (oh * config.stride_h) as isize - config.pad_top as isize;
        let out_row = &mut out_ch[oh * out_w..(oh + 1) * out_w];

        for ow in 0..out_w {
            let w_start = (ow * config.stride_w) as isize - config.pad_left as isize;

            match config.pool_type {
                PoolType::MaxPool => {
                    let mut max_val = f32::NEG_INFINITY;
                    for fh in 0..kh {
                        let ih = h_start + fh as isize;
                        if ih >= 0 && ih < in_h as isize {
                            let row = &in_ch[ih as usize * in_w..(ih as usize + 1) * in_w];
                            for fw in 0..kw {
                                let iw = w_start + fw as isize;
                                if iw >= 0 && iw < in_w as isize {
                                    let val = row[iw as usize];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                    }
                    out_row[ow] = if max_val == f32::NEG_INFINITY { 0.0 } else { max_val };
                }
                PoolType::AvgPool => {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for fh in 0..kh {
                        let ih = h_start + fh as isize;
                        if ih >= 0 && ih < in_h as isize {
                            let row = &in_ch[ih as usize * in_w..(ih as usize + 1) * in_w];
                            for fw in 0..kw {
                                let iw = w_start + fw as isize;
                                if iw >= 0 && iw < in_w as isize {
                                    sum += row[iw as usize];
                                    count += 1;
                                } else if config.count_include_pad {
                                    count += 1;
                                }
                            }
                        } else if config.count_include_pad {
                            count += kw;
                        }
                    }
                    out_row[ow] = if count > 0 { sum / (count as f32) } else { 0.0 };
                }
            }
        }
    }
}

pub fn pool2d(
    input: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    output: &mut [f32],
    config: &Pool2dConfig,
) {
    let (out_h, out_w) = pool2d_output_dims(in_h, in_w, config);
    let hw_out = out_h * out_w;
    let hw_in = in_h * in_w;

    let (kh, kw) = if config.is_global {
        (in_h, in_w)
    } else {
        (config.kernel_h, config.kernel_w)
    };

    let total_channels = batch * channels;
    assert!(output.len() >= total_channels * hw_out);

    #[cfg(feature = "rayon")]
    {
        if total_channels >= 2 {
            output
                .par_chunks_exact_mut(hw_out)
                .enumerate()
                .for_each(|(c, out_ch)| {
                    let in_ch = &input[c * hw_in..(c + 1) * hw_in];
                    process_pool_channel(in_ch, out_ch, in_h, in_w, out_h, out_w, kh, kw, config);
                });
            return;
        }
    }

    for c in 0..total_channels {
        let in_ch = &input[c * hw_in..(c + 1) * hw_in];
        let out_ch = &mut output[c * hw_out..(c + 1) * hw_out];
        process_pool_channel(in_ch, out_ch, in_h, in_w, out_h, out_w, kh, kw, config);
    }
}
