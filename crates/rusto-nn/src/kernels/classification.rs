#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let v_hi = _mm256_extractf128_ps(v, 1);
    let v_lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(v_hi, v_lo);
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 0x55));
    _mm_cvtss_f32(max32)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let v_hi = _mm256_extractf128_ps(v, 1);
    let v_lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(v_hi, v_lo);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn exp_approx_avx2(x: __m256) -> __m256 {
    // Clamping to avoid overflow / underflow
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654f32));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.722839f32));

    let log2_e = _mm256_set1_ps(1.4426950408889634f32);
    let y = _mm256_mul_ps(x, log2_e);
    let k = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm256_sub_ps(y, k);

    // Minimax polynomial for 2^f on [-0.5, 0.5]
    let c1 = _mm256_set1_ps(0.69314718056f32);
    let c2 = _mm256_set1_ps(0.24022650696f32);
    let c3 = _mm256_set1_ps(0.05550410866f32);
    let c4 = _mm256_set1_ps(0.00961812911f32);
    let c5 = _mm256_set1_ps(0.00133335581f32);

    let mut p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f32));

    let k_i = _mm256_cvtps_epi32(k);
    let exp_bias = _mm256_add_epi32(k_i, _mm256_set1_epi32(127));
    let pow2_k = _mm256_castsi256_ps(_mm256_slli_epi32(exp_bias, 23));

    _mm256_mul_ps(p, pow2_k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_row_avx2(in_row: &[f32], out_row: &mut [f32]) {
    let len = in_row.len();
    let chunks = len / 8;
    let in_ptr = in_row.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    // 1. Find max value
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(in_ptr.add(i * 8));
        max_vec = _mm256_max_ps(max_vec, v);
    }
    let mut max_val = hmax_avx2(max_vec);
    for i in (chunks * 8)..len {
        let v = *in_ptr.add(i);
        if v > max_val {
            max_val = v;
        }
    }
    let max_broadcast = _mm256_set1_ps(max_val);

    // 2. Compute exp(x - max) and accumulate sum
    let mut sum_vec = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(in_ptr.add(i * 8));
        let diff = _mm256_sub_ps(v, max_broadcast);
        let e = exp_approx_avx2(diff);
        _mm256_storeu_ps(out_ptr.add(i * 8), e);
        sum_vec = _mm256_add_ps(sum_vec, e);
    }
    let mut sum_exp = hsum_avx2(sum_vec);
    for i in (chunks * 8)..len {
        let diff = *in_ptr.add(i) - max_val;
        let e = diff.exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
    }

    // 3. Normalize by sum_exp
    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let inv_vec = _mm256_set1_ps(inv_sum);
    for i in 0..chunks {
        let e = _mm256_loadu_ps(out_ptr.add(i * 8));
        _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_mul_ps(e, inv_vec));
    }
    for i in (chunks * 8)..len {
        *out_ptr.add(i) *= inv_sum;
    }
}

pub fn softmax_row_scalar(in_row: &[f32], out_row: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in in_row {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum_exp = 0.0f32;
    for i in 0..in_row.len() {
        let exp_val = (in_row[i] - max_val).exp();
        out_row[i] = exp_val;
        sum_exp += exp_val;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-12);
    for i in 0..out_row.len() {
        out_row[i] *= inv_sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn exp_approx_avx512(x: __m512) -> __m512 {
    let x = _mm512_max_ps(x, _mm512_set1_ps(-87.33654f32));
    let x = _mm512_min_ps(x, _mm512_set1_ps(88.722839f32));

    let log2_e = _mm512_set1_ps(1.4426950408889634f32);
    let y = _mm512_mul_ps(x, log2_e);
    let k = _mm512_roundscale_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm512_sub_ps(y, k);

    let c1 = _mm512_set1_ps(0.69314718056f32);
    let c2 = _mm512_set1_ps(0.24022650696f32);
    let c3 = _mm512_set1_ps(0.05550410866f32);
    let c4 = _mm512_set1_ps(0.00961812911f32);
    let c5 = _mm512_set1_ps(0.00133335581f32);

    let mut p = _mm512_fmadd_ps(c5, f, c4);
    p = _mm512_fmadd_ps(p, f, c3);
    p = _mm512_fmadd_ps(p, f, c2);
    p = _mm512_fmadd_ps(p, f, c1);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f32));

    let k_i = _mm512_cvtps_epi32(k);
    let exp_bias = _mm512_add_epi32(k_i, _mm512_set1_epi32(127));
    let pow2_k = _mm512_castsi512_ps(_mm512_slli_epi32(exp_bias, 23));

    _mm512_mul_ps(p, pow2_k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_row_avx512(in_row: &[f32], out_row: &mut [f32]) {
    let len = in_row.len();
    let chunks = len / 16;
    let in_ptr = in_row.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    // 1. Find max value
    let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm512_loadu_ps(in_ptr.add(i * 16));
        max_vec = _mm512_max_ps(max_vec, v);
    }
    let mut max_val = _mm512_reduce_max_ps(max_vec);
    for i in (chunks * 16)..len {
        let v = *in_ptr.add(i);
        if v > max_val {
            max_val = v;
        }
    }
    let max_broadcast = _mm512_set1_ps(max_val);

    // 2. Compute exp(x - max) and accumulate sum
    let mut sum_vec = _mm512_setzero_ps();
    for i in 0..chunks {
        let v = _mm512_loadu_ps(in_ptr.add(i * 16));
        let diff = _mm512_sub_ps(v, max_broadcast);
        let e = exp_approx_avx512(diff);
        _mm512_storeu_ps(out_ptr.add(i * 16), e);
        sum_vec = _mm512_add_ps(sum_vec, e);
    }
    let mut sum_exp = _mm512_reduce_add_ps(sum_vec);
    for i in (chunks * 16)..len {
        let diff = *in_ptr.add(i) - max_val;
        let e = diff.exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
    }

    // 3. Normalize by sum_exp
    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let inv_vec = _mm512_set1_ps(inv_sum);
    for i in 0..chunks {
        let e = _mm512_loadu_ps(out_ptr.add(i * 16));
        _mm512_storeu_ps(out_ptr.add(i * 16), _mm512_mul_ps(e, inv_vec));
    }
    for i in (chunks * 16)..len {
        *out_ptr.add(i) *= inv_sum;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn exp_approx_wasm(x: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    let min_x = f32x4_splat(-87.33654f32);
    let max_x = f32x4_splat(88.722839f32);
    let underflow_mask = f32x4_ge(x, min_x);
    let xc = f32x4_pmin(f32x4_pmax(x, min_x), max_x);

    let log2_e = f32x4_splat(1.4426950408889634f32);
    let k = f32x4_nearest(f32x4_mul(xc, log2_e));
    let f = f32x4_sub(f32x4_mul(xc, log2_e), k);

    let c1 = f32x4_splat(0.69314718056f32);
    let c2 = f32x4_splat(0.24022650696f32);
    let c3 = f32x4_splat(0.05550410866f32);
    let c4 = f32x4_splat(0.00961812911f32);
    let c5 = f32x4_splat(0.00133335581f32);

    let mut p = f32x4_add(f32x4_mul(c5, f), c4);
    p = f32x4_add(f32x4_mul(p, f), c3);
    p = f32x4_add(f32x4_mul(p, f), c2);
    p = f32x4_add(f32x4_mul(p, f), c1);
    p = f32x4_add(f32x4_mul(p, f), f32x4_splat(1.0f32));

    let k_i = i32x4_trunc_sat_f32x4(k);
    let pow2_k = i32x4_shl(i32x4_add(k_i, i32x4_splat(127)), 23);

    v128_and(f32x4_mul(p, pow2_k), underflow_mask)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub unsafe fn softmax_row_wasm(in_row: &[f32], out_row: &mut [f32]) {
    use core::arch::wasm32::*;

    let len = in_row.len();
    let in_ptr = in_row.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let mut i = 0;
    let mut max_v0 = f32x4_splat(f32::NEG_INFINITY);
    let mut max_v1 = f32x4_splat(f32::NEG_INFINITY);
    let mut max_v2 = f32x4_splat(f32::NEG_INFINITY);
    let mut max_v3 = f32x4_splat(f32::NEG_INFINITY);

    while i + 16 <= len {
        max_v0 = f32x4_pmax(max_v0, v128_load(in_ptr.add(i) as *const v128));
        max_v1 = f32x4_pmax(max_v1, v128_load(in_ptr.add(i + 4) as *const v128));
        max_v2 = f32x4_pmax(max_v2, v128_load(in_ptr.add(i + 8) as *const v128));
        max_v3 = f32x4_pmax(max_v3, v128_load(in_ptr.add(i + 12) as *const v128));
        i += 16;
    }
    let mut max_vec = f32x4_pmax(f32x4_pmax(max_v0, max_v1), f32x4_pmax(max_v2, max_v3));
    while i + 4 <= len {
        max_vec = f32x4_pmax(max_vec, v128_load(in_ptr.add(i) as *const v128));
        i += 4;
    }
    let mut max_val = f32x4_extract_lane::<0>(max_vec)
        .max(f32x4_extract_lane::<1>(max_vec))
        .max(f32x4_extract_lane::<2>(max_vec))
        .max(f32x4_extract_lane::<3>(max_vec));
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }
    let max_v = f32x4_splat(max_val);

    i = 0;
    let mut sum_v0 = f32x4_splat(0.0);
    let mut sum_v1 = f32x4_splat(0.0);
    let mut sum_v2 = f32x4_splat(0.0);
    let mut sum_v3 = f32x4_splat(0.0);

    while i + 16 <= len {
        let e0 = exp_approx_wasm(f32x4_sub(v128_load(in_ptr.add(i) as *const v128), max_v));
        let e1 = exp_approx_wasm(f32x4_sub(v128_load(in_ptr.add(i + 4) as *const v128), max_v));
        let e2 = exp_approx_wasm(f32x4_sub(v128_load(in_ptr.add(i + 8) as *const v128), max_v));
        let e3 = exp_approx_wasm(f32x4_sub(v128_load(in_ptr.add(i + 12) as *const v128), max_v));

        v128_store(out_ptr.add(i) as *mut v128, e0);
        v128_store(out_ptr.add(i + 4) as *mut v128, e1);
        v128_store(out_ptr.add(i + 8) as *mut v128, e2);
        v128_store(out_ptr.add(i + 12) as *mut v128, e3);

        sum_v0 = f32x4_add(sum_v0, e0);
        sum_v1 = f32x4_add(sum_v1, e1);
        sum_v2 = f32x4_add(sum_v2, e2);
        sum_v3 = f32x4_add(sum_v3, e3);

        i += 16;
    }
    let mut sum_vec = f32x4_add(f32x4_add(sum_v0, sum_v1), f32x4_add(sum_v2, sum_v3));
    while i + 4 <= len {
        let e = exp_approx_wasm(f32x4_sub(v128_load(in_ptr.add(i) as *const v128), max_v));
        v128_store(out_ptr.add(i) as *mut v128, e);
        sum_vec = f32x4_add(sum_vec, e);
        i += 4;
    }
    let mut sum_exp = (f32x4_extract_lane::<0>(sum_vec) + f32x4_extract_lane::<1>(sum_vec))
        + (f32x4_extract_lane::<2>(sum_vec) + f32x4_extract_lane::<3>(sum_vec));
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let inv_v = f32x4_splat(inv_sum);

    i = 0;
    while i + 16 <= len {
        let e0 = v128_load(out_ptr.add(i) as *const v128);
        let e1 = v128_load(out_ptr.add(i + 4) as *const v128);
        let e2 = v128_load(out_ptr.add(i + 8) as *const v128);
        let e3 = v128_load(out_ptr.add(i + 12) as *const v128);

        v128_store(out_ptr.add(i) as *mut v128, f32x4_mul(e0, inv_v));
        v128_store(out_ptr.add(i + 4) as *mut v128, f32x4_mul(e1, inv_v));
        v128_store(out_ptr.add(i + 8) as *mut v128, f32x4_mul(e2, inv_v));
        v128_store(out_ptr.add(i + 12) as *mut v128, f32x4_mul(e3, inv_v));

        i += 16;
    }
    while i + 4 <= len {
        let e = v128_load(out_ptr.add(i) as *const v128);
        v128_store(out_ptr.add(i) as *mut v128, f32x4_mul(e, inv_v));
        i += 4;
    }
    while i < len {
        *out_ptr.add(i) *= inv_sum;
        i += 1;
    }
}

pub fn softmax_row(in_row: &[f32], out_row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                softmax_row_avx512(in_row, out_row);
                return;
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                softmax_row_avx2(in_row, out_row);
                return;
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            softmax_row_wasm(in_row, out_row);
            return;
        }
    }
    #[allow(unreachable_code)]
    softmax_row_scalar(in_row, out_row);
}

/// Numerically stable Softmax along specified axis with parallel row processing
pub fn softmax(
    input: &[f32],
    shape: &[usize],
    axis: isize,
    output: &mut [f32],
) {
    let ndim = shape.len();
    let norm_axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    assert!(norm_axis < ndim);

    let total_len = shape.iter().product::<usize>();
    assert!(output.len() >= total_len);

    let outer_size: usize = shape[..norm_axis].iter().product();
    let axis_size = shape[norm_axis];
    let inner_size: usize = shape[norm_axis + 1..].iter().product();

    if inner_size == 1 {
        #[cfg(feature = "rayon")]
        {
            if outer_size > 1 {
                output
                    .par_chunks_mut(axis_size)
                    .zip(input.par_chunks(axis_size))
                    .for_each(|(out_row, in_row)| {
                        softmax_row(in_row, out_row);
                    });
                return;
            }
        }

        for i in 0..outer_size {
            let in_row = &input[i * axis_size..(i + 1) * axis_size];
            let out_row = &mut output[i * axis_size..(i + 1) * axis_size];
            softmax_row(in_row, out_row);
        }
        return;
    }

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = (outer * axis_size + a) * inner_size + inner;
                if input[idx] > max_val {
                    max_val = input[idx];
                }
            }

            let mut sum_exp = 0.0f32;
            for a in 0..axis_size {
                let idx = (outer * axis_size + a) * inner_size + inner;
                let exp_val = (input[idx] - max_val).exp();
                output[idx] = exp_val;
                sum_exp += exp_val;
            }

            let inv_sum = 1.0 / sum_exp.max(1e-12);
            for a in 0..axis_size {
                let idx = (outer * axis_size + a) * inner_size + inner;
                output[idx] *= inv_sum;
            }
        }
    }
}

pub fn log_softmax_row_scalar(in_row: &[f32], out_row: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &v in in_row {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum_exp = 0.0f32;
    for &v in in_row {
        sum_exp += (v - max_val).exp();
    }

    let log_sum_exp = max_val + sum_exp.max(1e-12).ln();
    for i in 0..in_row.len() {
        out_row[i] = in_row[i] - log_sum_exp;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log_softmax_row_avx2(in_row: &[f32], out_row: &mut [f32]) {
    let len = in_row.len();
    let chunks = len / 8;
    let in_ptr = in_row.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(in_ptr.add(i * 8)));
    }
    let mut max_val = hmax_avx2(max_vec);
    for i in (chunks * 8)..len {
        max_val = max_val.max(*in_ptr.add(i));
    }
    let max_v = _mm256_set1_ps(max_val);

    let mut sum_vec = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(in_ptr.add(i * 8));
        sum_vec = _mm256_add_ps(sum_vec, exp_approx_avx2(_mm256_sub_ps(v, max_v)));
    }
    let mut sum_exp = hsum_avx2(sum_vec);
    for i in (chunks * 8)..len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
    }

    let log_sum_exp = max_val + sum_exp.max(1e-12).ln();
    let lse_v = _mm256_set1_ps(log_sum_exp);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(in_ptr.add(i * 8));
        _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_sub_ps(v, lse_v));
    }
    for i in (chunks * 8)..len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_sum_exp;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn log_softmax_row_wasm(in_row: &[f32], out_row: &mut [f32]) {
    use core::arch::wasm32::*;

    let len = in_row.len();
    let chunks = len / 4;
    let in_ptr = in_row.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let mut max_vec = f32x4_splat(f32::NEG_INFINITY);
    for i in 0..chunks {
        max_vec = f32x4_pmax(max_vec, v128_load(in_ptr.add(i * 4) as *const v128));
    }
    let mut max_val = f32x4_extract_lane::<0>(max_vec)
        .max(f32x4_extract_lane::<1>(max_vec))
        .max(f32x4_extract_lane::<2>(max_vec))
        .max(f32x4_extract_lane::<3>(max_vec));
    for i in (chunks * 4)..len {
        max_val = max_val.max(*in_ptr.add(i));
    }
    let max_v = f32x4_splat(max_val);

    let mut sum_vec = f32x4_splat(0.0);
    for i in 0..chunks {
        let v = v128_load(in_ptr.add(i * 4) as *const v128);
        sum_vec = f32x4_add(sum_vec, exp_approx_wasm(f32x4_sub(v, max_v)));
    }
    let mut sum_exp = (f32x4_extract_lane::<0>(sum_vec) + f32x4_extract_lane::<1>(sum_vec))
        + (f32x4_extract_lane::<2>(sum_vec) + f32x4_extract_lane::<3>(sum_vec));
    for i in (chunks * 4)..len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
    }

    let log_sum_exp = max_val + sum_exp.max(1e-12).ln();
    let lse_v = f32x4_splat(log_sum_exp);
    for i in 0..chunks {
        let v = v128_load(in_ptr.add(i * 4) as *const v128);
        v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_sub(v, lse_v));
    }
    for i in (chunks * 4)..len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_sum_exp;
    }
}

pub fn log_softmax_row(in_row: &[f32], out_row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                log_softmax_row_avx2(in_row, out_row);
                return;
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            log_softmax_row_wasm(in_row, out_row);
            return;
        }
    }
    #[allow(unreachable_code)]
    log_softmax_row_scalar(in_row, out_row);
}

pub fn log_softmax(
    input: &[f32],
    shape: &[usize],
    axis: isize,
    output: &mut [f32],
) {
    let ndim = shape.len();
    let norm_axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    assert!(norm_axis < ndim);

    let total_len = shape.iter().product::<usize>();
    assert!(output.len() >= total_len);

    let outer_size: usize = shape[..norm_axis].iter().product();
    let axis_size = shape[norm_axis];
    let inner_size: usize = shape[norm_axis + 1..].iter().product();

    if inner_size == 1 {
        #[cfg(feature = "rayon")]
        {
            if outer_size > 1 {
                output
                    .par_chunks_mut(axis_size)
                    .zip(input.par_chunks(axis_size))
                    .for_each(|(out_row, in_row)| {
                        log_softmax_row(in_row, out_row);
                    });
                return;
            }
        }

        for i in 0..outer_size {
            let in_row = &input[i * axis_size..(i + 1) * axis_size];
            let out_row = &mut output[i * axis_size..(i + 1) * axis_size];
            log_softmax_row(in_row, out_row);
        }
        return;
    }

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = (outer * axis_size + a) * inner_size + inner;
                if input[idx] > max_val {
                    max_val = input[idx];
                }
            }

            let mut sum_exp = 0.0f32;
            for a in 0..axis_size {
                let idx = (outer * axis_size + a) * inner_size + inner;
                sum_exp += (input[idx] - max_val).exp();
            }

            let log_sum_exp = max_val + sum_exp.max(1e-12).ln();
            for a in 0..axis_size {
                let idx = (outer * axis_size + a) * inner_size + inner;
                output[idx] = input[idx] - log_sum_exp;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_row_avx2(row: &[f32]) -> (i32, f32) {
    use core::arch::x86_64::*;

    let len = row.len();
    let chunks = len / 8;
    let ptr = row.as_ptr();

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut best_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut cur_idx = best_idx;
    let step = _mm256_set1_epi32(8);

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(v, max_vec);
        max_vec = _mm256_blendv_ps(max_vec, v, gt);
        best_idx = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(best_idx),
            _mm256_castsi256_ps(cur_idx),
            gt,
        ));
        cur_idx = _mm256_add_epi32(cur_idx, step);
    }

    let mut vals = [0.0f32; 8];
    let mut idxs = [0i32; 8];
    _mm256_storeu_ps(vals.as_mut_ptr(), max_vec);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, best_idx);

    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0i32;
    if chunks > 0 {
        for l in 0..8 {
            if vals[l] > max_val || (vals[l] == max_val && idxs[l] < max_idx) {
                max_val = vals[l];
                max_idx = idxs[l];
            }
        }
    }

    for a in (chunks * 8)..len {
        let v = *ptr.add(a);
        if v > max_val {
            max_val = v;
            max_idx = a as i32;
        }
    }

    (max_idx, max_val)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn argmax_row_wasm(row: &[f32]) -> (i32, f32) {
    use core::arch::wasm32::*;

    let len = row.len();
    let chunks = len / 4;
    let ptr = row.as_ptr();

    let mut max_vec = f32x4_splat(f32::NEG_INFINITY);
    let mut best_idx = i32x4(0, 1, 2, 3);
    let mut cur_idx = best_idx;
    let step = i32x4_splat(4);

    for i in 0..chunks {
        let v = v128_load(ptr.add(i * 4) as *const v128);
        let gt = f32x4_gt(v, max_vec);
        max_vec = v128_bitselect(v, max_vec, gt);
        best_idx = v128_bitselect(cur_idx, best_idx, gt);
        cur_idx = i32x4_add(cur_idx, step);
    }

    let mut vals = [0.0f32; 4];
    let mut idxs = [0i32; 4];
    v128_store(vals.as_mut_ptr() as *mut v128, max_vec);
    v128_store(idxs.as_mut_ptr() as *mut v128, best_idx);

    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0i32;
    if chunks > 0 {
        for l in 0..4 {
            if vals[l] > max_val || (vals[l] == max_val && idxs[l] < max_idx) {
                max_val = vals[l];
                max_idx = idxs[l];
            }
        }
    }

    for a in (chunks * 4)..len {
        let v = *ptr.add(a);
        if v > max_val {
            max_val = v;
            max_idx = a as i32;
        }
    }

    (max_idx, max_val)
}

#[inline]
pub fn argmax_row(row: &[f32]) -> (i32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { argmax_row_avx2(row) };
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { argmax_row_wasm(row) };
    }
    #[allow(unreachable_code)]
    {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0i32;
        for (a, &v) in row.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = a as i32;
            }
        }
        (max_idx, max_val)
    }
}

/// Fast greedy CTC decode shortcut directly from logits or probabilities
pub fn ctc_greedy_argmax_tokens(
    probs_or_logits: &[f32],
    timesteps: usize,
    vocab_size: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut tokens = Vec::with_capacity(timesteps);
    let mut confs = Vec::with_capacity(timesteps);

    for t in 0..timesteps {
        let row = &probs_or_logits[t * vocab_size..(t + 1) * vocab_size];
        let (best_idx, best_val) = argmax_row(row);
        tokens.push(best_idx as usize);
        confs.push(best_val);
    }

    (tokens, confs)
}

/// ArgMax along specified axis
pub fn argmax(
    input: &[f32],
    shape: &[usize],
    axis: isize,
    out_indices: &mut [i32],
) {
    let ndim = shape.len();
    let norm_axis = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    assert!(norm_axis < ndim);

    let outer_size: usize = shape[..norm_axis].iter().product();
    let axis_size = shape[norm_axis];
    let inner_size: usize = shape[norm_axis + 1..].iter().product();

    assert!(out_indices.len() >= outer_size * inner_size);

    if inner_size == 1 {
        for outer in 0..outer_size {
            let row = &input[outer * axis_size..(outer + 1) * axis_size];
            out_indices[outer] = argmax_row(row).0;
        }
        return;
    }

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0i32;

            for a in 0..axis_size {
                let in_idx = (outer * axis_size + a) * inner_size + inner;
                if input[in_idx] > max_val {
                    max_val = input[in_idx];
                    max_idx = a as i32;
                }
            }

            let out_idx = outer * inner_size + inner;
            out_indices[out_idx] = max_idx;
        }
    }
}

/// TopK along last dimension
pub fn topk(
    input: &[f32],
    outer_size: usize,
    k_dim: usize,
    k: usize,
    out_values: &mut [f32],
    out_indices: &mut [i32],
) {
    if k == 0 || k_dim == 0 || outer_size == 0 {
        return;
    }
    assert!(out_values.len() >= outer_size * k);
    assert!(out_indices.len() >= outer_size * k);

    let k_eff = k.min(k_dim);

    if k_eff == 1 {
        for outer in 0..outer_size {
            let slice = &input[outer * k_dim..(outer + 1) * k_dim];
            let mut best_val = slice[0];
            let mut best_idx = 0i32;
            for (i, &v) in slice.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i as i32;
                }
            }
            out_values[outer * k] = best_val;
            out_indices[outer * k] = best_idx;
        }
        return;
    }

    if k_eff <= 8 {
        for outer in 0..outer_size {
            let slice = &input[outer * k_dim..(outer + 1) * k_dim];
            let val_slice = &mut out_values[outer * k..(outer + 1) * k];
            let idx_slice = &mut out_indices[outer * k..(outer + 1) * k];

            let mut top_v = [f32::NEG_INFINITY; 8];
            let mut top_i = [0i32; 8];

            for (i, &v) in slice.iter().enumerate().take(k_eff) {
                let mut pos = i;
                while pos > 0 && v > top_v[pos - 1] {
                    top_v[pos] = top_v[pos - 1];
                    top_i[pos] = top_i[pos - 1];
                    pos -= 1;
                }
                top_v[pos] = v;
                top_i[pos] = i as i32;
            }

            let mut min_top = top_v[k_eff - 1];
            for (i, &v) in slice.iter().enumerate().skip(k_eff) {
                if v > min_top {
                    let mut pos = k_eff - 1;
                    while pos > 0 && v > top_v[pos - 1] {
                        top_v[pos] = top_v[pos - 1];
                        top_i[pos] = top_i[pos - 1];
                        pos -= 1;
                    }
                    top_v[pos] = v;
                    top_i[pos] = i as i32;
                    min_top = top_v[k_eff - 1];
                }
            }

            val_slice[..k_eff].copy_from_slice(&top_v[..k_eff]);
            idx_slice[..k_eff].copy_from_slice(&top_i[..k_eff]);
        }
        return;
    }

    let mut pairs: Vec<(f32, i32)> = Vec::with_capacity(k_dim);
    for outer in 0..outer_size {
        let slice = &input[outer * k_dim..(outer + 1) * k_dim];
        let val_slice = &mut out_values[outer * k..(outer + 1) * k];
        let idx_slice = &mut out_indices[outer * k..(outer + 1) * k];

        pairs.clear();
        pairs.extend(slice.iter().enumerate().map(|(i, &v)| (v, i as i32)));

        pairs.select_nth_unstable_by(k_eff - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        pairs[..k_eff].sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..k_eff {
            val_slice[i] = pairs[i].0;
            idx_slice[i] = pairs[i].1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_vectorized_vs_scalar() {
        let size = 6906;
        let mut input = vec![0.0f32; size];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 23) as f32 - 11.0) * 0.25;
        }

        let mut out_scalar = vec![0.0f32; size];
        softmax_row_scalar(&input, &mut out_scalar);

        let mut out_fast = vec![0.0f32; size];
        softmax_row(&input, &mut out_fast);

        for i in 0..size {
            let diff = (out_scalar[i] - out_fast[i]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at {}: scalar={}, fast={}, diff={}",
                i, out_scalar[i], out_fast[i], diff
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                let mut out_512 = vec![0.0f32; size];
                unsafe { softmax_row_avx512(&input, &mut out_512) };
                for i in 0..size {
                    let diff = (out_scalar[i] - out_512[i]).abs();
                    assert!(
                        diff < 1e-4,
                        "AVX512 mismatch at {}: scalar={}, avx512={}, diff={}",
                        i, out_scalar[i], out_512[i], diff
                    );
                }
            }

            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let mut out_256 = vec![0.0f32; size];
                unsafe { softmax_row_avx2(&input, &mut out_256) };
                for i in 0..size {
                    let diff = (out_scalar[i] - out_256[i]).abs();
                    assert!(
                        diff < 1e-4,
                        "AVX2 mismatch at {}: scalar={}, avx2={}, diff={}",
                        i, out_scalar[i], out_256[i], diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_topk_fast_paths() {
        let outer = 4;
        let k_dim = 257;
        let mut input = vec![0.0f32; outer * k_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i * 37) % 101) as f32 * 0.13 - 6.5 + (i as f32) * 1e-5;
        }

        for k in [1usize, 5, 8, 12] {
            let mut vals = vec![0.0f32; outer * k];
            let mut idxs = vec![0i32; outer * k];
            topk(&input, outer, k_dim, k, &mut vals, &mut idxs);

            for o in 0..outer {
                let row = &input[o * k_dim..(o + 1) * k_dim];
                let mut expected: Vec<(f32, i32)> =
                    row.iter().enumerate().map(|(i, &v)| (v, i as i32)).collect();
                expected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for j in 0..k {
                    assert_eq!(idxs[o * k + j], expected[j].1);
                    assert!((vals[o * k + j] - expected[j].0).abs() < 1e-6);
                }
            }
        }
    }
}

