#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpType {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    FloorDiv,
    FloorMod,
    SquaredDifference,
    MulSilu,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
}

#[inline(always)]
fn eval_binary(a: f32, b: f32, op: BinaryOpType) -> f32 {
    match op {
        BinaryOpType::Add => a + b,
        BinaryOpType::Sub => a - b,
        BinaryOpType::Mul => a * b,
        BinaryOpType::Div => {
            if b != 0.0 {
                a / b
            } else {
                0.0
            }
        }
        BinaryOpType::Pow => a.powf(b),
        BinaryOpType::Min => a.min(b),
        BinaryOpType::Max => a.max(b),
        BinaryOpType::Greater => {
            if a > b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::GreaterEqual => {
            if a >= b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::Less => {
            if a < b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::LessEqual => {
            if a <= b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::Equal => {
            if a == b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::NotEqual => {
            if a != b {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::FloorDiv => (a / b).floor(),
        BinaryOpType::FloorMod => a - (a / b).floor() * b,
        BinaryOpType::SquaredDifference => {
            let diff = a - b;
            diff * diff
        }
        BinaryOpType::MulSilu => {
            let silu_b = b / (1.0 + (-b).exp());
            a * silu_b
        }
        BinaryOpType::LogicalAnd => {
            if a != 0.0 && b != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::LogicalOr => {
            if a != 0.0 || b != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpType::LogicalXor => {
            if (a != 0.0) ^ (b != 0.0) {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_slice_avx2(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    op: BinaryOpType,
    act: FusedActivation,
) -> bool {
    use core::arch::x86_64::*;

    let len = lhs.len();
    let chunks = len / 8;
    let l_ptr = lhs.as_ptr();
    let r_ptr = rhs.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    let three = _mm256_set1_ps(3.0);
    let inv6 = _mm256_set1_ps(1.0 / 6.0);

    macro_rules! apply_avx2_op {
        ($op_expr:expr) => {
            for i in 0..chunks {
                let va = _mm256_loadu_ps(l_ptr.add(i * 8));
                let vb = _mm256_loadu_ps(r_ptr.add(i * 8));
                let mut res = $op_expr(va, vb);

                match act {
                    FusedActivation::Relu => {
                        res = _mm256_max_ps(zero, res);
                    }
                    FusedActivation::Relu6 => {
                        res = _mm256_min_ps(six, _mm256_max_ps(zero, res));
                    }
                    FusedActivation::HardSwish => {
                        let p = _mm256_add_ps(res, three);
                        let cl = _mm256_min_ps(six, _mm256_max_ps(zero, p));
                        res = _mm256_mul_ps(res, _mm256_mul_ps(cl, inv6));
                    }
                    _ => {}
                }
                _mm256_storeu_ps(o_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                let val = eval_binary(lhs[i], rhs[i], op);
                out[i] = apply_activation_scalar(val, act);
            }
        };
    }

    match op {
        BinaryOpType::Add => {
            apply_avx2_op!(|va, vb| _mm256_add_ps(va, vb));
            true
        }
        BinaryOpType::Sub => {
            apply_avx2_op!(|va, vb| _mm256_sub_ps(va, vb));
            true
        }
        BinaryOpType::Mul => {
            apply_avx2_op!(|va, vb| _mm256_mul_ps(va, vb));
            true
        }
        BinaryOpType::Div => {
            apply_avx2_op!(|va, vb| _mm256_div_ps(va, vb));
            true
        }
        BinaryOpType::Min => {
            apply_avx2_op!(|va, vb| _mm256_min_ps(va, vb));
            true
        }
        BinaryOpType::Max => {
            apply_avx2_op!(|va, vb| _mm256_max_ps(va, vb));
            true
        }
        BinaryOpType::MulSilu => {
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

            apply_avx2_op!(|va, vb| {
                let val = _mm256_mul_ps(half, vb);
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

                let sig_b = _mm256_mul_ps(half, _mm256_add_ps(one, tanh_val));
                let silu_b = _mm256_mul_ps(vb, sig_b);
                _mm256_mul_ps(va, silu_b)
            });
            true
        }
        _ => false,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_scalar_rhs_avx2(
    lhs: &[f32],
    b: f32,
    out: &mut [f32],
    op: BinaryOpType,
    act: FusedActivation,
) -> bool {
    use core::arch::x86_64::*;

    let len = lhs.len();
    let chunks = len / 8;
    let l_ptr = lhs.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let vb = _mm256_set1_ps(b);
    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    let three = _mm256_set1_ps(3.0);
    let inv6 = _mm256_set1_ps(1.0 / 6.0);

    macro_rules! apply_scalar_rhs_op {
        ($op_expr:expr) => {
            for i in 0..chunks {
                let va = _mm256_loadu_ps(l_ptr.add(i * 8));
                let mut res = $op_expr(va, vb);

                match act {
                    FusedActivation::Relu => {
                        res = _mm256_max_ps(zero, res);
                    }
                    FusedActivation::Relu6 => {
                        res = _mm256_min_ps(six, _mm256_max_ps(zero, res));
                    }
                    FusedActivation::HardSwish => {
                        let p = _mm256_add_ps(res, three);
                        let cl = _mm256_min_ps(six, _mm256_max_ps(zero, p));
                        res = _mm256_mul_ps(res, _mm256_mul_ps(cl, inv6));
                    }
                    _ => {}
                }
                _mm256_storeu_ps(o_ptr.add(i * 8), res);
            }
            for i in chunks * 8..len {
                let val = eval_binary(lhs[i], b, op);
                out[i] = apply_activation_scalar(val, act);
            }
        };
    }

    match op {
        BinaryOpType::Add => {
            apply_scalar_rhs_op!(|va, vb| _mm256_add_ps(va, vb));
            true
        }
        BinaryOpType::Sub => {
            apply_scalar_rhs_op!(|va, vb| _mm256_sub_ps(va, vb));
            true
        }
        BinaryOpType::Mul => {
            apply_scalar_rhs_op!(|va, vb| _mm256_mul_ps(va, vb));
            true
        }
        BinaryOpType::Div => {
            apply_scalar_rhs_op!(|va, vb| _mm256_div_ps(va, vb));
            true
        }
        BinaryOpType::Min => {
            apply_scalar_rhs_op!(|va, vb| _mm256_min_ps(va, vb));
            true
        }
        BinaryOpType::Max => {
            apply_scalar_rhs_op!(|va, vb| _mm256_max_ps(va, vb));
            true
        }
        BinaryOpType::MulSilu => {
            let silu_b = b / (1.0 + (-b).exp());
            let vb_silu = _mm256_set1_ps(silu_b);
            apply_scalar_rhs_op!(|va, _vb| _mm256_mul_ps(va, vb_silu));
            true
        }
        _ => false,
    }
}

pub fn binary_slice(lhs: &[f32], rhs: &[f32], out: &mut [f32], op: BinaryOpType, act: FusedActivation) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                if binary_slice_avx2(lhs, rhs, out, op, act) {
                    return;
                }
            }
        }
    }

    for i in 0..lhs.len() {
        let val = eval_binary(lhs[i], rhs[i], op);
        out[i] = apply_activation_scalar(val, act);
    }
}

pub fn binary_scalar_rhs(lhs: &[f32], b: f32, out: &mut [f32], op: BinaryOpType, act: FusedActivation) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                if binary_scalar_rhs_avx2(lhs, b, out, op, act) {
                    return;
                }
            }
        }
    }

    for i in 0..lhs.len() {
        let val = eval_binary(lhs[i], b, op);
        out[i] = apply_activation_scalar(val, act);
    }
}

/// Elementwise binary operation with standard NumPy-style broadcasting
pub fn binary_op(
    lhs_shape: &[usize],
    lhs_data: &[f32],
    rhs_shape: &[usize],
    rhs_data: &[f32],
    out_shape: &[usize],
    out_data: &mut [f32],
    op_type: BinaryOpType,
    act: FusedActivation,
) {
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0 || out_data.is_empty() || lhs_data.is_empty() || rhs_data.is_empty() {
        return;
    }
    assert!(out_data.len() >= out_len);

    // Fast path: identical shapes
    if lhs_shape == rhs_shape && lhs_shape == out_shape {
        #[cfg(feature = "rayon")]
        {
            if out_len >= 8192 {
                let chunk_size = 4096;
                out_data
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, out_sub)| {
                        let offset = chunk_idx * chunk_size;
                        let end = (offset + out_sub.len()).min(lhs_data.len());
                        binary_slice(&lhs_data[offset..end], &rhs_data[offset..end], out_sub, op_type, act);
                    });
                return;
            }
        }
        binary_slice(lhs_data, rhs_data, out_data, op_type, act);
        return;
    }

    // Fast path: RHS is scalar
    if rhs_data.len() == 1 {
        let b = rhs_data[0];
        #[cfg(feature = "rayon")]
        {
            if out_len >= 8192 {
                let chunk_size = 4096;
                out_data
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, out_sub)| {
                        let offset = chunk_idx * chunk_size;
                        let end = (offset + out_sub.len()).min(lhs_data.len());
                        binary_scalar_rhs(&lhs_data[offset..end], b, out_sub, op_type, act);
                    });
                return;
            }
        }
        binary_scalar_rhs(lhs_data, b, out_data, op_type, act);
        return;
    }

    // Fast path: LHS is scalar
    if lhs_data.len() == 1 {
        let a = lhs_data[0];
        if matches!(op_type, BinaryOpType::Add | BinaryOpType::Mul) {
            #[cfg(feature = "rayon")]
            {
                if out_len >= 8192 {
                    let chunk_size = 4096;
                    out_data
                        .par_chunks_mut(chunk_size)
                        .enumerate()
                        .for_each(|(chunk_idx, out_sub)| {
                            let offset = chunk_idx * chunk_size;
                            let end = (offset + out_sub.len()).min(rhs_data.len());
                            binary_scalar_rhs(&rhs_data[offset..end], a, out_sub, op_type, act);
                        });
                    return;
                }
            }
            binary_scalar_rhs(rhs_data, a, out_data, op_type, act);
            return;
        }

        for i in 0..out_len {
            let b = if rhs_data.len() == 1 { rhs_data[0] } else { rhs_data.get(i).copied().unwrap_or(0.0) };
            let val = eval_binary(a, b, op_type);
            out_data[i] = apply_activation_scalar(val, act);
        }
        return;
    }

    // Fast path: 4D Channel Broadcast (e.g. Squeeze-and-Excitation / BatchNorm)
    // LHS: [1, C, H, W], RHS: [1, C, 1, 1] or [C]
    if lhs_shape.len() == 4 && out_shape == lhs_shape {
        let channels = lhs_shape[1];
        let hw = lhs_shape[2] * lhs_shape[3];

        let rhs_is_channel_vec = (rhs_shape.len() == 4 && rhs_shape[0] == 1 && rhs_shape[1] == channels && rhs_shape[2] == 1 && rhs_shape[3] == 1)
            || (rhs_shape.len() == 1 && rhs_shape[0] == channels);

        if rhs_is_channel_vec {
            #[cfg(feature = "rayon")]
            {
                if channels >= 2 {
                    out_data
                        .par_chunks_exact_mut(hw)
                        .enumerate()
                        .for_each(|(i, out_sub)| {
                            let c = i % channels;
                            let b_val = rhs_data[c];
                            let l_sub = &lhs_data[i * hw..(i + 1) * hw];
                            binary_scalar_rhs(l_sub, b_val, out_sub, op_type, act);
                        });
                    return;
                }
            }

            let total_chunks = lhs_shape[0] * channels;
            for i in 0..total_chunks {
                let c = i % channels;
                let b_val = rhs_data[c];
                let l_sub = &lhs_data[i * hw..(i + 1) * hw];
                let o_sub = &mut out_data[i * hw..(i + 1) * hw];
                binary_scalar_rhs(l_sub, b_val, o_sub, op_type, act);
            }
            return;
        }
    }

    // General broadcasting fallback
    let ndim = out_shape.len();
    let mut coord = vec![0usize; ndim];

    let mut lhs_padded_shape = vec![1usize; ndim];
    let offset_l = ndim.saturating_sub(lhs_shape.len());
    lhs_padded_shape[offset_l..].copy_from_slice(lhs_shape);

    let mut rhs_padded_shape = vec![1usize; ndim];
    let offset_r = ndim.saturating_sub(rhs_shape.len());
    rhs_padded_shape[offset_r..].copy_from_slice(rhs_shape);

    let mut lhs_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        lhs_strides[i] = lhs_strides[i + 1] * lhs_padded_shape[i + 1];
    }

    let mut rhs_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        rhs_strides[i] = rhs_strides[i + 1] * rhs_padded_shape[i + 1];
    }

    for idx in 0..out_len {
        let mut lhs_idx = 0;
        let mut rhs_idx = 0;
        for d in 0..ndim {
            let l_c = if lhs_padded_shape[d] == 1 { 0 } else { coord[d] };
            let r_c = if rhs_padded_shape[d] == 1 { 0 } else { coord[d] };
            lhs_idx += l_c * lhs_strides[d];
            rhs_idx += r_c * rhs_strides[d];
        }

        let l_val = lhs_data.get(lhs_idx).copied().unwrap_or(0.0);
        let r_val = rhs_data.get(rhs_idx).copied().unwrap_or(0.0);
        let val = eval_binary(l_val, r_val, op_type);
        out_data[idx] = apply_activation_scalar(val, act);

        // Advance multi-dimensional coordinates
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < out_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
}
