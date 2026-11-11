#[cfg(feature = "parallel")]
use crate::par::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOpType {
    Identity,
    Abs,
    Neg,
    Floor,
    Ceil,
    Square,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Reciprocal,
    Sigmoid,
    Tanh,
    HardSwish,
    HardSigmoid { alpha: f32, beta: f32 },
    Gelu,
    GeluStandard,
    Silu,
    Round,
    Sign,
    Erf,
    Bnll,
    Softplus,
    Not,
}

#[inline(always)]
pub fn eval_unary(x: f32, op: UnaryOpType) -> f32 {
    match op {
        UnaryOpType::Identity => x,
        UnaryOpType::Abs => x.abs(),
        UnaryOpType::Neg => -x,
        UnaryOpType::Floor => x.floor(),
        UnaryOpType::Ceil => x.ceil(),
        UnaryOpType::Square => x * x,
        UnaryOpType::Sqrt => if x >= 0.0 { x.sqrt() } else { 0.0 },
        UnaryOpType::Rsqrt => if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 },
        UnaryOpType::Exp => x.exp(),
        UnaryOpType::Log => if x > 0.0 { x.ln() } else { f32::NEG_INFINITY },
        UnaryOpType::Sin => x.sin(),
        UnaryOpType::Cos => x.cos(),
        UnaryOpType::Tan => x.tan(),
        UnaryOpType::Reciprocal => if x != 0.0 { 1.0 / x } else { 0.0 },
        UnaryOpType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryOpType::Tanh => x.tanh(),
        UnaryOpType::HardSwish => x * ((x + 3.0) / 6.0).clamp(0.0, 1.0),
        UnaryOpType::HardSigmoid { alpha, beta } => (alpha * x + beta).clamp(0.0, 1.0),
        UnaryOpType::Gelu => {
            let x3 = x * x * x;
            let inner = 0.79788456 * (x + 0.044715 * x3);
            0.5 * x * (1.0 + inner.tanh())
        }
        UnaryOpType::GeluStandard => {
            0.5 * x * (1.0 + eval_erf(x * std::f32::consts::FRAC_1_SQRT_2))
        }
        UnaryOpType::Silu => x / (1.0 + (-x).exp()),
        UnaryOpType::Round => x.round(),
        UnaryOpType::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOpType::Erf => eval_erf(x),
        UnaryOpType::Bnll | UnaryOpType::Softplus => {
            if x > 0.0 {
                x + (1.0 + (-x).exp()).ln()
            } else {
                (1.0 + x.exp()).ln()
            }
        }
        UnaryOpType::Not => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Abramowitz and Stegun erf approximation (maximum error: 1.5e-7)
#[inline(always)]
fn eval_erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();

    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x_abs);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-x_abs * x_abs).exp();

    sign * y
}

fn process_unary_chunk(input: &[f32], output: &mut [f32], op: UnaryOpType) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                if unary_op_avx2(input, output, op) {
                    return;
                }
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            if unary_op_wasm(input, output, op) {
                return;
            }
        }
    }

    for (i, &val) in input.iter().enumerate() {
        output[i] = eval_unary(val, op);
    }
}

/// Apply unary operation in-place or from input to output slice
pub fn unary_op(input: &[f32], output: &mut [f32], op: UnaryOpType) {
    assert_eq!(input.len(), output.len(), "Unary op input and output sizes must match");

    #[cfg(feature = "parallel")]
    {
        if crate::par::current_thread_index().is_none() && input.len() >= 4096 {
            let chunk_size = 4096;
            input
                .par_chunks(chunk_size)
                .zip(output.par_chunks_mut(chunk_size))
                .for_each(|(in_sub, out_sub)| {
                    process_unary_chunk(in_sub, out_sub, op);
                });
            return;
        }
    }

    process_unary_chunk(input, output, op);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp_avx2(x: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;

    let min_x = _mm256_set1_ps(-87.33654f32);
    let max_x = _mm256_set1_ps(88.722839f32);
    let mask = _mm256_cmp_ps(x, min_x, _CMP_GE_OQ);
    let xc = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);

    let log2_e = _mm256_set1_ps(core::f32::consts::LOG2_E);
    let k = _mm256_round_ps(_mm256_mul_ps(xc, log2_e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    let ln2_hi = _mm256_set1_ps(0.693359375f32);
    let ln2_lo = _mm256_set1_ps(-2.12194440e-4f32);
    let r = _mm256_fnmadd_ps(k, ln2_lo, _mm256_fnmadd_ps(k, ln2_hi, xc));

    let c6 = _mm256_set1_ps(0.0013888889f32);
    let c5 = _mm256_set1_ps(0.008333334f32);
    let c4 = _mm256_set1_ps(0.041666668f32);
    let c3 = _mm256_set1_ps(0.16666667f32);
    let c2 = _mm256_set1_ps(0.5f32);
    let one = _mm256_set1_ps(1.0f32);

    let mut p = _mm256_fmadd_ps(c6, r, c5);
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);

    let k_i = _mm256_cvtps_epi32(k);
    let pow2_k = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_i, _mm256_set1_epi32(127)), 23));

    _mm256_and_ps(_mm256_mul_ps(p, pow2_k), mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn erf_avx2(x: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;

    let sign_mask = _mm256_set1_ps(-0.0f32);
    let sign_bit = _mm256_and_ps(x, sign_mask);
    let x_abs = _mm256_andnot_ps(sign_mask, x);

    let one = _mm256_set1_ps(1.0f32);
    let p = _mm256_set1_ps(0.3275911f32);
    let a1 = _mm256_set1_ps(0.254829592f32);
    let a2 = _mm256_set1_ps(-0.284496736f32);
    let a3 = _mm256_set1_ps(1.421413741f32);
    let a4 = _mm256_set1_ps(-1.453152027f32);
    let a5 = _mm256_set1_ps(1.061405429f32);

    let t = _mm256_div_ps(one, _mm256_fmadd_ps(p, x_abs, one));
    let mut poly = _mm256_fmadd_ps(a5, t, a4);
    poly = _mm256_fmadd_ps(poly, t, a3);
    poly = _mm256_fmadd_ps(poly, t, a2);
    poly = _mm256_fmadd_ps(poly, t, a1);
    poly = _mm256_mul_ps(poly, t);

    let neg_x2 = _mm256_xor_ps(_mm256_mul_ps(x_abs, x_abs), sign_mask);
    let exp_term = exp_avx2(neg_x2);
    let y = _mm256_fnmadd_ps(poly, exp_term, one);

    _mm256_xor_ps(y, sign_bit)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unary_op_avx2(input: &[f32], output: &mut [f32], op: UnaryOpType) -> bool {
    use core::arch::x86_64::*;

    let len = input.len();
    let chunks = len / 8;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    match op {
        UnaryOpType::Identity => {
            output[..len].copy_from_slice(&input[..len]);
            true
        }
        UnaryOpType::HardSigmoid { alpha, beta } => {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let alpha_v = _mm256_set1_ps(alpha);
            let beta_v = _mm256_set1_ps(beta);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                let linear = _mm256_fmadd_ps(v, alpha_v, beta_v);
                let res = _mm256_min_ps(one, _mm256_max_ps(zero, linear));
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = (alpha * input[i] + beta).clamp(0.0, 1.0);
            }
            true
        }
        UnaryOpType::Abs => {
            let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_and_ps(v, sign_mask));
            }
            for i in chunks * 8..len {
                output[i] = input[i].abs();
            }
            true
        }
        UnaryOpType::Neg => {
            let sign_mask = _mm256_set1_ps(-0.0f32);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_xor_ps(v, sign_mask));
            }
            for i in chunks * 8..len {
                output[i] = -input[i];
            }
            true
        }
        UnaryOpType::Floor => {
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_floor_ps(v));
            }
            for i in chunks * 8..len {
                output[i] = input[i].floor();
            }
            true
        }
        UnaryOpType::Ceil => {
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_ceil_ps(v));
            }
            for i in chunks * 8..len {
                output[i] = input[i].ceil();
            }
            true
        }
        UnaryOpType::HardSwish => {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let inv6 = _mm256_set1_ps(1.0 / 6.0);
            let three = _mm256_set1_ps(3.0);
            for i in 0..chunks {
                let x = _mm256_loadu_ps(in_ptr.add(i * 8));
                let scaled = _mm256_mul_ps(_mm256_add_ps(x, three), inv6);
                let clamped = _mm256_min_ps(one, _mm256_max_ps(zero, scaled));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_mul_ps(x, clamped));
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], UnaryOpType::HardSwish);
            }
            true
        }
        UnaryOpType::Square => {
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_mul_ps(v, v));
            }
            for i in chunks * 8..len {
                output[i] = input[i] * input[i];
            }
            true
        }
        UnaryOpType::Sqrt => {
            let zero = _mm256_setzero_ps();
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), _mm256_sqrt_ps(_mm256_max_ps(zero, v)));
            }
            for i in chunks * 8..len {
                output[i] = if input[i] >= 0.0 { input[i].sqrt() } else { 0.0 };
            }
            true
        }
        UnaryOpType::Exp => {
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), exp_avx2(v));
            }
            for i in chunks * 8..len {
                output[i] = input[i].exp();
            }
            true
        }
        UnaryOpType::Erf => {
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                _mm256_storeu_ps(out_ptr.add(i * 8), erf_avx2(v));
            }
            for i in chunks * 8..len {
                output[i] = eval_erf(input[i]);
            }
            true
        }
        UnaryOpType::GeluStandard => {
            let half = _mm256_set1_ps(0.5f32);
            let one = _mm256_set1_ps(1.0f32);
            let inv_sqrt2 = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
            for i in 0..chunks {
                let x = _mm256_loadu_ps(in_ptr.add(i * 8));
                let erf_val = erf_avx2(_mm256_mul_ps(x, inv_sqrt2));
                let res = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, erf_val));
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], UnaryOpType::GeluStandard);
            }
            true
        }
        UnaryOpType::Gelu => {
            let p0 = _mm256_set1_ps(0.044715);
            let p1 = _mm256_set1_ps(0.79788456);
            let five = _mm256_set1_ps(5.0);
            let neg_five = _mm256_set1_ps(-5.0);
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            let half = _mm256_set1_ps(0.5);

            let a0 = _mm256_set1_ps(135135.0);
            let a1 = _mm256_set1_ps(17325.0);
            let a2 = _mm256_set1_ps(378.0);

            let b0 = _mm256_set1_ps(135135.0);
            let b1 = _mm256_set1_ps(62370.0);
            let b2 = _mm256_set1_ps(3150.0);
            let b3 = _mm256_set1_ps(28.0);

            for i in 0..chunks {
                let x = _mm256_loadu_ps(in_ptr.add(i * 8));
                let x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
                let val = _mm256_mul_ps(p1, _mm256_fmadd_ps(p0, x3, x));
                let v2 = _mm256_mul_ps(val, val);

                let a_term = _mm256_fmadd_ps(v2, _mm256_fmadd_ps(v2, _mm256_add_ps(a2, v2), a1), a0);
                let num = _mm256_mul_ps(val, a_term);
                let den = _mm256_fmadd_ps(v2, _mm256_fmadd_ps(v2, _mm256_fmadd_ps(v2, b3, b2), b1), b0);
                let tanh_poly = _mm256_div_ps(num, den);

                let is_gt = _mm256_cmp_ps(val, five, _CMP_GT_OQ);
                let is_lt = _mm256_cmp_ps(val, neg_five, _CMP_LE_OQ);
                let mut tanh_val = _mm256_blendv_ps(tanh_poly, one, is_gt);
                tanh_val = _mm256_blendv_ps(tanh_val, neg_one, is_lt);

                let res = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_val)));
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], UnaryOpType::Gelu);
            }
            true
        }
        UnaryOpType::Sigmoid | UnaryOpType::Silu => {
            let one = _mm256_set1_ps(1.0f32);
            let sign_mask = _mm256_set1_ps(-0.0f32);
            let is_silu = op == UnaryOpType::Silu;

            for i in 0..chunks {
                let x = _mm256_loadu_ps(in_ptr.add(i * 8));
                let neg_x = _mm256_xor_ps(x, sign_mask);
                let denom = _mm256_add_ps(one, exp_avx2(neg_x));
                let res = if is_silu {
                    _mm256_div_ps(x, denom)
                } else {
                    _mm256_div_ps(one, denom)
                };
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], op);
            }
            true
        }
        UnaryOpType::Tanh => {
            let five = _mm256_set1_ps(5.0);
            let neg_five = _mm256_set1_ps(-5.0);
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);

            let a0 = _mm256_set1_ps(135135.0);
            let a1 = _mm256_set1_ps(17325.0);
            let a2 = _mm256_set1_ps(378.0);

            let b0 = _mm256_set1_ps(135135.0);
            let b1 = _mm256_set1_ps(62370.0);
            let b2 = _mm256_set1_ps(3150.0);
            let b3 = _mm256_set1_ps(28.0);

            for i in 0..chunks {
                let val = _mm256_loadu_ps(in_ptr.add(i * 8));
                let v2 = _mm256_mul_ps(val, val);

                let a_term = _mm256_fmadd_ps(v2, _mm256_fmadd_ps(v2, _mm256_add_ps(a2, v2), a1), a0);
                let num = _mm256_mul_ps(val, a_term);
                let den = _mm256_fmadd_ps(v2, _mm256_fmadd_ps(v2, _mm256_fmadd_ps(v2, b3, b2), b1), b0);
                let tanh_poly = _mm256_div_ps(num, den);

                let is_gt = _mm256_cmp_ps(val, five, _CMP_GT_OQ);
                let is_lt = _mm256_cmp_ps(val, neg_five, _CMP_LE_OQ);
                let mut tanh_val = _mm256_blendv_ps(tanh_poly, one, is_gt);
                tanh_val = _mm256_blendv_ps(tanh_val, neg_one, is_lt);

                _mm256_storeu_ps(out_ptr.add(i * 8), tanh_val);
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], UnaryOpType::Tanh);
            }
            true
        }
        _ => false,
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn exp_wasm(x: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    let min_x = f32x4_splat(-87.33654f32);
    let max_x = f32x4_splat(88.722839f32);
    let mask = f32x4_ge(x, min_x);
    let xc = f32x4_pmin(f32x4_pmax(x, min_x), max_x);

    let log2_e = f32x4_splat(1.4426950408889634f32);
    let k = f32x4_nearest(f32x4_mul(xc, log2_e));

    let ln2_hi = f32x4_splat(0.693359375f32);
    let ln2_lo = f32x4_splat(-2.12194440e-4f32);
    let r = f32x4_sub(f32x4_sub(xc, f32x4_mul(k, ln2_hi)), f32x4_mul(k, ln2_lo));

    let c6 = f32x4_splat(0.0013888889f32);
    let c5 = f32x4_splat(0.008333334f32);
    let c4 = f32x4_splat(0.041666668f32);
    let c3 = f32x4_splat(0.16666667f32);
    let c2 = f32x4_splat(0.5f32);
    let one = f32x4_splat(1.0f32);

    let mut p = f32x4_add(f32x4_mul(c6, r), c5);
    p = f32x4_add(f32x4_mul(p, r), c4);
    p = f32x4_add(f32x4_mul(p, r), c3);
    p = f32x4_add(f32x4_mul(p, r), c2);
    p = f32x4_add(f32x4_mul(p, r), one);
    p = f32x4_add(f32x4_mul(p, r), one);

    let k_i = i32x4_trunc_sat_f32x4(k);
    let pow2_k = i32x4_shl(i32x4_add(k_i, i32x4_splat(127)), 23);

    v128_and(f32x4_mul(p, pow2_k), mask)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn erf_wasm(x: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    let sign_mask = f32x4_splat(-0.0f32);
    let sign_bit = v128_and(x, sign_mask);
    let x_abs = f32x4_abs(x);

    let one = f32x4_splat(1.0f32);
    let p = f32x4_splat(0.3275911f32);
    let a1 = f32x4_splat(0.254829592f32);
    let a2 = f32x4_splat(-0.284496736f32);
    let a3 = f32x4_splat(1.421413741f32);
    let a4 = f32x4_splat(-1.453152027f32);
    let a5 = f32x4_splat(1.061405429f32);

    let t = f32x4_div(one, f32x4_add(f32x4_mul(p, x_abs), one));
    let mut poly = f32x4_add(f32x4_mul(a5, t), a4);
    poly = f32x4_add(f32x4_mul(poly, t), a3);
    poly = f32x4_add(f32x4_mul(poly, t), a2);
    poly = f32x4_add(f32x4_mul(poly, t), a1);
    poly = f32x4_mul(poly, t);

    let neg_x2 = f32x4_neg(f32x4_mul(x_abs, x_abs));
    let exp_term = exp_wasm(neg_x2);
    let y = f32x4_sub(one, f32x4_mul(poly, exp_term));

    v128_xor(y, sign_bit)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn tanh_pade_wasm(val: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    let five = f32x4_splat(5.0);
    let neg_five = f32x4_splat(-5.0);
    let one = f32x4_splat(1.0);
    let neg_one = f32x4_splat(-1.0);

    let a0 = f32x4_splat(135135.0);
    let a1 = f32x4_splat(17325.0);
    let a2 = f32x4_splat(378.0);

    let b0 = f32x4_splat(135135.0);
    let b1 = f32x4_splat(62370.0);
    let b2 = f32x4_splat(3150.0);
    let b3 = f32x4_splat(28.0);

    let vc = f32x4_pmin(f32x4_pmax(val, neg_five), five);
    let v2 = f32x4_mul(vc, vc);

    let a_term = f32x4_add(f32x4_mul(v2, f32x4_add(f32x4_mul(v2, f32x4_add(a2, v2)), a1)), a0);
    let num = f32x4_mul(vc, a_term);
    let den = f32x4_add(f32x4_mul(v2, f32x4_add(f32x4_mul(v2, f32x4_add(f32x4_mul(v2, b3), b2)), b1)), b0);
    let poly = f32x4_div(num, den);

    f32x4_pmin(f32x4_pmax(poly, neg_one), one)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn unary_op_wasm(input: &[f32], output: &mut [f32], op: UnaryOpType) -> bool {
    use core::arch::wasm32::*;

    let len = input.len();
    let chunks = len / 4;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    match op {
        UnaryOpType::Identity => {
            output[..len].copy_from_slice(&input[..len]);
            true
        }
        UnaryOpType::HardSigmoid { alpha, beta } => {
            let zero = f32x4_splat(0.0);
            let one = f32x4_splat(1.0);
            let alpha_v = f32x4_splat(alpha);
            let beta_v = f32x4_splat(beta);
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                let linear = f32x4_add(f32x4_mul(v, alpha_v), beta_v);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_pmin(f32x4_pmax(linear, zero), one));
            }
            for i in chunks * 4..len {
                output[i] = (alpha * input[i] + beta).clamp(0.0, 1.0);
            }
            true
        }
        UnaryOpType::Abs => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_abs(v));
            }
            for i in chunks * 4..len {
                output[i] = input[i].abs();
            }
            true
        }
        UnaryOpType::Neg => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_neg(v));
            }
            for i in chunks * 4..len {
                output[i] = -input[i];
            }
            true
        }
        UnaryOpType::Floor => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_floor(v));
            }
            for i in chunks * 4..len {
                output[i] = input[i].floor();
            }
            true
        }
        UnaryOpType::Ceil => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_ceil(v));
            }
            for i in chunks * 4..len {
                output[i] = input[i].ceil();
            }
            true
        }
        UnaryOpType::HardSwish => {
            let zero = f32x4_splat(0.0);
            let one = f32x4_splat(1.0);
            let inv6 = f32x4_splat(1.0 / 6.0);
            let three = f32x4_splat(3.0);
            for i in 0..chunks {
                let x = v128_load(in_ptr.add(i * 4) as *const v128);
                let scaled = f32x4_mul(f32x4_add(x, three), inv6);
                let clamped = f32x4_pmin(f32x4_pmax(scaled, zero), one);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_mul(x, clamped));
            }
            for i in chunks * 4..len {
                output[i] = eval_unary(input[i], UnaryOpType::HardSwish);
            }
            true
        }
        UnaryOpType::Square => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_mul(v, v));
            }
            for i in chunks * 4..len {
                output[i] = input[i] * input[i];
            }
            true
        }
        UnaryOpType::Sqrt => {
            let zero = f32x4_splat(0.0);
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, f32x4_sqrt(f32x4_pmax(v, zero)));
            }
            for i in chunks * 4..len {
                output[i] = if input[i] >= 0.0 { input[i].sqrt() } else { 0.0 };
            }
            true
        }
        UnaryOpType::Exp => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, exp_wasm(v));
            }
            for i in chunks * 4..len {
                output[i] = input[i].exp();
            }
            true
        }
        UnaryOpType::Erf => {
            for i in 0..chunks {
                let v = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, erf_wasm(v));
            }
            for i in chunks * 4..len {
                output[i] = eval_erf(input[i]);
            }
            true
        }
        UnaryOpType::GeluStandard => {
            let half = f32x4_splat(0.5f32);
            let one = f32x4_splat(1.0f32);
            let inv_sqrt2 = f32x4_splat(std::f32::consts::FRAC_1_SQRT_2);
            for i in 0..chunks {
                let x = v128_load(in_ptr.add(i * 4) as *const v128);
                let erf_val = erf_wasm(f32x4_mul(x, inv_sqrt2));
                let res = f32x4_mul(f32x4_mul(half, x), f32x4_add(one, erf_val));
                v128_store(out_ptr.add(i * 4) as *mut v128, res);
            }
            for i in chunks * 4..len {
                output[i] = eval_unary(input[i], UnaryOpType::GeluStandard);
            }
            true
        }
        UnaryOpType::Gelu => {
            let p0 = f32x4_splat(0.044715);
            let p1 = f32x4_splat(0.79788456);
            let one = f32x4_splat(1.0);
            let half = f32x4_splat(0.5);
            for i in 0..chunks {
                let x = v128_load(in_ptr.add(i * 4) as *const v128);
                let x3 = f32x4_mul(f32x4_mul(x, x), x);
                let val = f32x4_mul(p1, f32x4_add(x, f32x4_mul(p0, x3)));
                let tanh_val = tanh_pade_wasm(val);
                let res = f32x4_mul(half, f32x4_mul(x, f32x4_add(one, tanh_val)));
                v128_store(out_ptr.add(i * 4) as *mut v128, res);
            }
            for i in chunks * 4..len {
                output[i] = eval_unary(input[i], UnaryOpType::Gelu);
            }
            true
        }
        UnaryOpType::Sigmoid | UnaryOpType::Silu => {
            let one = f32x4_splat(1.0f32);
            let is_silu = op == UnaryOpType::Silu;
            for i in 0..chunks {
                let x = v128_load(in_ptr.add(i * 4) as *const v128);
                let denom = f32x4_add(one, exp_wasm(f32x4_neg(x)));
                let res = if is_silu {
                    f32x4_div(x, denom)
                } else {
                    f32x4_div(one, denom)
                };
                v128_store(out_ptr.add(i * 4) as *mut v128, res);
            }
            for i in chunks * 4..len {
                output[i] = eval_unary(input[i], op);
            }
            true
        }
        UnaryOpType::Tanh => {
            for i in 0..chunks {
                let val = v128_load(in_ptr.add(i * 4) as *const v128);
                v128_store(out_ptr.add(i * 4) as *mut v128, tanh_pade_wasm(val));
            }
            for i in chunks * 4..len {
                output[i] = eval_unary(input[i], UnaryOpType::Tanh);
            }
            true
        }
        _ => false,
    }
}
