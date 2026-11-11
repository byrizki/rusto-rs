#[cfg(feature = "parallel")]
use crate::par::prelude::*;

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
    } else if h_total >= config.kernel_h {
        (h_total - config.kernel_h) / config.stride_h + 1
    } else {
        1
    };

    let out_w = if config.ceil_mode {
        if w_total >= config.kernel_w {
            (w_total - config.kernel_w).div_ceil(config.stride_w) + 1
        } else {
            1
        }
    } else if w_total >= config.kernel_w {
        (w_total - config.kernel_w) / config.stride_w + 1
    } else {
        1
    };

    (out_h, out_w)
}

#[inline(always)]
fn eval_maxpool_pixel(
    in_ch: &[f32],
    in_h: usize,
    in_w: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    config: &Pool2dConfig,
) -> f32 {
    let h_start = (oh * config.stride_h) as isize - config.pad_top as isize;
    let w_start = (ow * config.stride_w) as isize - config.pad_left as isize;
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
    if max_val == f32::NEG_INFINITY {
        0.0
    } else {
        max_val
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn maxpool_3x3_s1_avx2(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Pool2dConfig,
) {
    use core::arch::x86_64::*;

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_ch.as_ptr().add(ih_u * in_w);
            let r1 = in_ch.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_ch.as_ptr().add((ih_u + 2) * in_w);
            let out_row = out_ch.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.min(out_w);
            let v_limit = (in_w + pad_left).saturating_sub(2).min(out_w);

            for ow in 0..ow_start {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }

            let mut ow = ow_start;
            while ow + 8 <= v_limit {
                let iw = ow - pad_left;

                let m0 = _mm256_max_ps(
                    _mm256_loadu_ps(r0.add(iw)),
                    _mm256_max_ps(_mm256_loadu_ps(r0.add(iw + 1)), _mm256_loadu_ps(r0.add(iw + 2))),
                );
                let m1 = _mm256_max_ps(
                    _mm256_loadu_ps(r1.add(iw)),
                    _mm256_max_ps(_mm256_loadu_ps(r1.add(iw + 1)), _mm256_loadu_ps(r1.add(iw + 2))),
                );
                let m2 = _mm256_max_ps(
                    _mm256_loadu_ps(r2.add(iw)),
                    _mm256_max_ps(_mm256_loadu_ps(r2.add(iw + 1)), _mm256_loadu_ps(r2.add(iw + 2))),
                );

                _mm256_storeu_ps(out_row.add(ow), _mm256_max_ps(m0, _mm256_max_ps(m1, m2)));
                ow += 8;
            }

            while ow < out_w {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                out_ch[oh * out_w + ow] = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn maxpool_3x3_sw2_avx2(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Pool2dConfig,
) {
    use core::arch::x86_64::*;

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let stride_h = config.stride_h;

    for oh in 0..out_h {
        let ih_base = (oh * stride_h) as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_ch.as_ptr().add(ih_u * in_w);
            let r1 = in_ch.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_ch.as_ptr().add((ih_u + 2) * in_w);
            let out_row = out_ch.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.div_ceil(2).min(out_w);
            let v_limit = if in_w + pad_left >= 9 {
                ((in_w + pad_left - 9) / 2 + 1).min(out_w)
            } else {
                0
            };

            for ow in 0..ow_start {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow * 2 - pad_left;

                macro_rules! row_max4 {
                    ($r:expr) => {{
                        let v0 = _mm_loadu_ps($r.add(iw));
                        let v1 = _mm_loadu_ps($r.add(iw + 4));
                        let c0 = _mm_shuffle_ps(v0, v1, 0x88);
                        let c1 = _mm_shuffle_ps(v0, v1, 0xDD);
                        let c2 = _mm_set_ps(*$r.add(iw + 8), *$r.add(iw + 6), *$r.add(iw + 4), *$r.add(iw + 2));
                        _mm_max_ps(c0, _mm_max_ps(c1, c2))
                    }};
                }

                let m0 = row_max4!(r0);
                let m1 = row_max4!(r1);
                let m2 = row_max4!(r2);
                _mm_storeu_ps(out_row.add(ow), _mm_max_ps(m0, _mm_max_ps(m1, m2)));
                ow += 4;
            }

            while ow < out_w {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                out_ch[oh * out_w + ow] = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn maxpool_2x2_s2_avx2(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_w: usize,
    out_h: usize,
    out_w: usize,
) {
    use core::arch::x86_64::*;

    let chunks = out_w / 4;
    for oh in 0..out_h {
        let r0 = in_ch.as_ptr().add((oh * 2) * in_w);
        let r1 = in_ch.as_ptr().add((oh * 2 + 1) * in_w);
        let out_row = out_ch.as_mut_ptr().add(oh * out_w);

        for c_idx in 0..chunks {
            let ow = c_idx * 4;
            let iw = ow * 2;
            let v00 = _mm_loadu_ps(r0.add(iw));
            let v01 = _mm_loadu_ps(r0.add(iw + 4));
            let m0 = _mm_max_ps(_mm_shuffle_ps(v00, v01, 0x88), _mm_shuffle_ps(v00, v01, 0xDD));

            let v10 = _mm_loadu_ps(r1.add(iw));
            let v11 = _mm_loadu_ps(r1.add(iw + 4));
            let m1 = _mm_max_ps(_mm_shuffle_ps(v10, v11, 0x88), _mm_shuffle_ps(v10, v11, 0xDD));

            _mm_storeu_ps(out_row.add(ow), _mm_max_ps(m0, m1));
        }

        for ow in (chunks * 4)..out_w {
            let iw = ow * 2;
            let m0 = (*r0.add(iw)).max(*r0.add(iw + 1));
            let m1 = (*r1.add(iw)).max(*r1.add(iw + 1));
            *out_row.add(ow) = m0.max(m1);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn maxpool_2x2_s1_avx2(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Pool2dConfig,
) {
    use core::arch::x86_64::*;

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 1 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_ch.as_ptr().add(ih_u * in_w);
            let r1 = in_ch.as_ptr().add((ih_u + 1) * in_w);
            let out_row = out_ch.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.min(out_w);
            let v_limit = (in_w + pad_left).saturating_sub(1).min(out_w);

            for ow in 0..ow_start {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 2, 2, config);
            }

            let mut ow = ow_start;
            while ow + 8 <= v_limit {
                let iw = ow - pad_left;
                let m0 = _mm256_max_ps(_mm256_loadu_ps(r0.add(iw)), _mm256_loadu_ps(r0.add(iw + 1)));
                let m1 = _mm256_max_ps(_mm256_loadu_ps(r1.add(iw)), _mm256_loadu_ps(r1.add(iw + 1)));
                _mm256_storeu_ps(out_row.add(ow), _mm256_max_ps(m0, m1));
                ow += 8;
            }

            while ow < out_w {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 2, 2, config);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                out_ch[oh * out_w + ow] = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 2, 2, config);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn maxpool_2x2_s1_wasm(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Pool2dConfig,
) {
    use core::arch::wasm32::*;

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 1 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_ch.as_ptr().add(ih_u * in_w);
            let r1 = in_ch.as_ptr().add((ih_u + 1) * in_w);
            let out_row = out_ch.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.min(out_w);
            let v_limit = (in_w + pad_left).saturating_sub(1).min(out_w);

            for ow in 0..ow_start {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 2, 2, config);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow - pad_left;
                let m0 = f32x4_pmax(
                    v128_load(r0.add(iw) as *const v128),
                    v128_load(r0.add(iw + 1) as *const v128),
                );
                let m1 = f32x4_pmax(
                    v128_load(r1.add(iw) as *const v128),
                    v128_load(r1.add(iw + 1) as *const v128),
                );
                v128_store(out_row.add(ow) as *mut v128, f32x4_pmax(m0, m1));
                ow += 4;
            }

            while ow < out_w {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 2, 2, config);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                out_ch[oh * out_w + ow] = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 2, 2, config);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn maxpool_3x3_s1_wasm(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Pool2dConfig,
) {
    use core::arch::wasm32::*;

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;

    for oh in 0..out_h {
        let ih_base = oh as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_ch.as_ptr().add(ih_u * in_w);
            let r1 = in_ch.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_ch.as_ptr().add((ih_u + 2) * in_w);
            let out_row = out_ch.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.min(out_w);
            let v_limit = (in_w + pad_left).saturating_sub(2).min(out_w);

            for ow in 0..ow_start {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow - pad_left;

                let m0 = f32x4_pmax(
                    v128_load(r0.add(iw) as *const v128),
                    f32x4_pmax(
                        v128_load(r0.add(iw + 1) as *const v128),
                        v128_load(r0.add(iw + 2) as *const v128),
                    ),
                );
                let m1 = f32x4_pmax(
                    v128_load(r1.add(iw) as *const v128),
                    f32x4_pmax(
                        v128_load(r1.add(iw + 1) as *const v128),
                        v128_load(r1.add(iw + 2) as *const v128),
                    ),
                );
                let m2 = f32x4_pmax(
                    v128_load(r2.add(iw) as *const v128),
                    f32x4_pmax(
                        v128_load(r2.add(iw + 1) as *const v128),
                        v128_load(r2.add(iw + 2) as *const v128),
                    ),
                );

                v128_store(out_row.add(ow) as *mut v128, f32x4_pmax(m0, f32x4_pmax(m1, m2)));
                ow += 4;
            }

            while ow < out_w {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                out_ch[oh * out_w + ow] = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn maxpool_3x3_sw2_wasm(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    config: &Pool2dConfig,
) {
    use core::arch::wasm32::*;

    let pad_top = config.pad_top;
    let pad_left = config.pad_left;
    let stride_h = config.stride_h;

    for oh in 0..out_h {
        let ih_base = (oh * stride_h) as isize - pad_top as isize;
        if ih_base >= 0 && ih_base + 2 < in_h as isize {
            let ih_u = ih_base as usize;
            let r0 = in_ch.as_ptr().add(ih_u * in_w);
            let r1 = in_ch.as_ptr().add((ih_u + 1) * in_w);
            let r2 = in_ch.as_ptr().add((ih_u + 2) * in_w);
            let out_row = out_ch.as_mut_ptr().add(oh * out_w);

            let ow_start = pad_left.div_ceil(2).min(out_w);
            let v_limit = if in_w + pad_left >= 9 {
                ((in_w + pad_left - 9) / 2 + 1).min(out_w)
            } else {
                0
            };

            for ow in 0..ow_start {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }

            let mut ow = ow_start;
            while ow + 4 <= v_limit {
                let iw = ow * 2 - pad_left;

                macro_rules! row_max4_wasm {
                    ($r:expr) => {{
                        let v0 = v128_load($r.add(iw) as *const v128);
                        let v1 = v128_load($r.add(iw + 4) as *const v128);
                        let c0 = i32x4_shuffle::<0, 2, 4, 6>(v0, v1);
                        let c1 = i32x4_shuffle::<1, 3, 5, 7>(v0, v1);
                        let c2 = f32x4_replace_lane::<3>(i32x4_shuffle::<2, 4, 6, 6>(v0, v1), *$r.add(iw + 8));
                        f32x4_pmax(c0, f32x4_pmax(c1, c2))
                    }};
                }

                let m0 = row_max4_wasm!(r0);
                let m1 = row_max4_wasm!(r1);
                let m2 = row_max4_wasm!(r2);
                v128_store(out_row.add(ow) as *mut v128, f32x4_pmax(m0, f32x4_pmax(m1, m2)));
                ow += 4;
            }

            while ow < out_w {
                *out_row.add(ow) = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
                ow += 1;
            }
        } else {
            for ow in 0..out_w {
                out_ch[oh * out_w + ow] = eval_maxpool_pixel(in_ch, in_h, in_w, oh, ow, 3, 3, config);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn maxpool_2x2_s2_wasm(
    in_ch: &[f32],
    out_ch: &mut [f32],
    in_w: usize,
    out_h: usize,
    out_w: usize,
) {
    use core::arch::wasm32::*;

    let chunks = out_w / 4;
    for oh in 0..out_h {
        let r0 = in_ch.as_ptr().add((oh * 2) * in_w);
        let r1 = in_ch.as_ptr().add((oh * 2 + 1) * in_w);
        let out_row = out_ch.as_mut_ptr().add(oh * out_w);

        for c_idx in 0..chunks {
            let ow = c_idx * 4;
            let iw = ow * 2;
            let v00 = v128_load(r0.add(iw) as *const v128);
            let v01 = v128_load(r0.add(iw + 4) as *const v128);
            let m0 = f32x4_pmax(i32x4_shuffle::<0, 2, 4, 6>(v00, v01), i32x4_shuffle::<1, 3, 5, 7>(v00, v01));

            let v10 = v128_load(r1.add(iw) as *const v128);
            let v11 = v128_load(r1.add(iw + 4) as *const v128);
            let m1 = f32x4_pmax(i32x4_shuffle::<0, 2, 4, 6>(v10, v11), i32x4_shuffle::<1, 3, 5, 7>(v10, v11));

            v128_store(out_row.add(ow) as *mut v128, f32x4_pmax(m0, m1));
        }

        for ow in (chunks * 4)..out_w {
            let iw = ow * 2;
            let m0 = (*r0.add(iw)).max(*r0.add(iw + 1));
            let m1 = (*r1.add(iw)).max(*r1.add(iw + 1));
            *out_row.add(ow) = m0.max(m1);
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

    if config.pool_type == PoolType::MaxPool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            if kh == 3 && kw == 3 && config.stride_h == 1 && config.stride_w == 1 {
                unsafe { maxpool_3x3_s1_avx2(in_ch, out_ch, in_h, in_w, out_h, out_w, config) };
                return;
            }
            if kh == 3 && kw == 3 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                unsafe { maxpool_3x3_sw2_avx2(in_ch, out_ch, in_h, in_w, out_h, out_w, config) };
                return;
            }
            if kh == 2 && kw == 2 && config.stride_h == 1 && config.stride_w == 1 {
                unsafe { maxpool_2x2_s1_avx2(in_ch, out_ch, in_h, in_w, out_h, out_w, config) };
                return;
            }
            if kh == 2
                && kw == 2
                && config.stride_h == 2
                && config.stride_w == 2
                && config.pad_top == 0
                && config.pad_left == 0
                && in_h >= out_h * 2
                && in_w >= out_w * 2
            {
                unsafe { maxpool_2x2_s2_avx2(in_ch, out_ch, in_w, out_h, out_w) };
                return;
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if kh == 3 && kw == 3 && config.stride_h == 1 && config.stride_w == 1 {
                unsafe { maxpool_3x3_s1_wasm(in_ch, out_ch, in_h, in_w, out_h, out_w, config) };
                return;
            }
            if kh == 3 && kw == 3 && config.stride_w == 2 && (config.stride_h == 1 || config.stride_h == 2) {
                unsafe { maxpool_3x3_sw2_wasm(in_ch, out_ch, in_h, in_w, out_h, out_w, config) };
                return;
            }
            if kh == 2 && kw == 2 && config.stride_h == 1 && config.stride_w == 1 {
                unsafe { maxpool_2x2_s1_wasm(in_ch, out_ch, in_h, in_w, out_h, out_w, config) };
                return;
            }
            if kh == 2
                && kw == 2
                && config.stride_h == 2
                && config.stride_w == 2
                && config.pad_top == 0
                && config.pad_left == 0
                && in_h >= out_h * 2
                && in_w >= out_w * 2
            {
                unsafe { maxpool_2x2_s2_wasm(in_ch, out_ch, in_w, out_h, out_w) };
                return;
            }
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

    #[cfg(feature = "parallel")]
    {
        if crate::par::current_thread_index().is_none()
            && total_channels * hw_out >= 8192
            && total_channels >= 4
        {
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
