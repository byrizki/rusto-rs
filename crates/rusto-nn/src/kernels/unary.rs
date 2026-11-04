#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Unary operations module

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
            // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let x3 = x * x * x;
            let inner = 0.79788456 * (x + 0.044715 * x3);
            0.5 * x * (1.0 + inner.tanh())
        }
        UnaryOpType::GeluStandard => {
            // Standard formulation using error function: 0.5 * x * (1 + erf(x / sqrt(2)))
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

/// Process a contiguous chunk of unary operation with AVX2 or scalar
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

    for (i, &val) in input.iter().enumerate() {
        output[i] = eval_unary(val, op);
    }
}

/// Apply unary operation in-place or from input to output slice
pub fn unary_op(input: &[f32], output: &mut [f32], op: UnaryOpType) {
    assert_eq!(input.len(), output.len(), "Unary op input and output sizes must match");

    #[cfg(feature = "rayon")]
    {
        if input.len() >= 4096 {
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
                let res = _mm256_and_ps(v, sign_mask);
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = input[i].abs();
            }
            true
        }
        UnaryOpType::Neg => {
            let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(-0x80000000i32));
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                let res = _mm256_xor_ps(v, sign_mask);
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = -input[i];
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
                let x_plus_3 = _mm256_add_ps(x, three);
                let scaled = _mm256_mul_ps(x_plus_3, inv6);
                let clamped = _mm256_min_ps(one, _mm256_max_ps(zero, scaled));
                let res = _mm256_mul_ps(x, clamped);
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], UnaryOpType::HardSwish);
            }
            true
        }
        UnaryOpType::Square => {
            for i in 0..chunks {
                let v = _mm256_loadu_ps(in_ptr.add(i * 8));
                let res = _mm256_mul_ps(v, v);
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
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
                let clamped = _mm256_max_ps(zero, v);
                let res = _mm256_sqrt_ps(clamped);
                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = if input[i] >= 0.0 { input[i].sqrt() } else { 0.0 };
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
                let x2 = _mm256_mul_ps(x, x);
                let x3 = _mm256_mul_ps(x2, x);

                let inner = _mm256_fmadd_ps(p0, x3, x);
                let val = _mm256_mul_ps(p1, inner);

                let v2 = _mm256_mul_ps(val, val);

                let a_term = _mm256_add_ps(a2, v2);
                let a_term = _mm256_fmadd_ps(v2, a_term, a1);
                let a_term = _mm256_fmadd_ps(v2, a_term, a0);
                let num = _mm256_mul_ps(val, a_term);

                let b_term = _mm256_fmadd_ps(v2, b3, b2);
                let b_term = _mm256_fmadd_ps(v2, b_term, b1);
                let den = _mm256_fmadd_ps(v2, b_term, b0);

                let tanh_poly = _mm256_div_ps(num, den);

                let is_gt = _mm256_cmp_ps(val, five, _CMP_GT_OQ);
                let is_lt = _mm256_cmp_ps(val, neg_five, _CMP_LE_OQ);
                let mut tanh_val = _mm256_blendv_ps(tanh_poly, one, is_gt);
                tanh_val = _mm256_blendv_ps(tanh_val, neg_one, is_lt);

                let one_plus_tanh = _mm256_add_ps(one, tanh_val);
                let res = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_tanh));

                _mm256_storeu_ps(out_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                output[i] = eval_unary(input[i], UnaryOpType::Gelu);
            }
            true
        }
        UnaryOpType::Sigmoid | UnaryOpType::Silu => {
            let half = _mm256_set1_ps(0.5);
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

            let is_silu = op == UnaryOpType::Silu;

            for i in 0..chunks {
                let x = _mm256_loadu_ps(in_ptr.add(i * 8));
                let val = _mm256_mul_ps(half, x);

                let v2 = _mm256_mul_ps(val, val);

                let a_term = _mm256_add_ps(a2, v2);
                let a_term = _mm256_fmadd_ps(v2, a_term, a1);
                let a_term = _mm256_fmadd_ps(v2, a_term, a0);
                let num = _mm256_mul_ps(val, a_term);

                let b_term = _mm256_fmadd_ps(v2, b3, b2);
                let b_term = _mm256_fmadd_ps(v2, b_term, b1);
                let den = _mm256_fmadd_ps(v2, b_term, b0);

                let tanh_poly = _mm256_div_ps(num, den);

                let is_gt = _mm256_cmp_ps(val, five, _CMP_GT_OQ);
                let is_lt = _mm256_cmp_ps(val, neg_five, _CMP_LE_OQ);
                let mut tanh_val = _mm256_blendv_ps(tanh_poly, one, is_gt);
                tanh_val = _mm256_blendv_ps(tanh_val, neg_one, is_lt);

                let sig = _mm256_mul_ps(half, _mm256_add_ps(one, tanh_val));
                let res = if is_silu {
                    _mm256_mul_ps(x, sig)
                } else {
                    sig
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

                let a_term = _mm256_add_ps(a2, v2);
                let a_term = _mm256_fmadd_ps(v2, a_term, a1);
                let a_term = _mm256_fmadd_ps(v2, a_term, a0);
                let num = _mm256_mul_ps(val, a_term);

                let b_term = _mm256_fmadd_ps(v2, b3, b2);
                let b_term = _mm256_fmadd_ps(v2, b_term, b1);
                let den = _mm256_fmadd_ps(v2, b_term, b0);

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
