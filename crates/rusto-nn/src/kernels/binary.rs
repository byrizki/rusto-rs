#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, apply_activation_slice, FusedActivation};

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

    if !matches!(
        act,
        FusedActivation::None | FusedActivation::Relu | FusedActivation::Relu6 | FusedActivation::HardSwish
    ) {
        return false;
    }

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
        BinaryOpType::SquaredDifference => {
            apply_avx2_op!(|va, vb| {
                let d = _mm256_sub_ps(va, vb);
                _mm256_mul_ps(d, d)
            });
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

    if !matches!(
        act,
        FusedActivation::None | FusedActivation::Relu | FusedActivation::Relu6 | FusedActivation::HardSwish
    ) || (op == BinaryOpType::Div && b == 0.0)
    {
        return false;
    }

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
            let inv_b = _mm256_set1_ps(1.0 / b);
            apply_scalar_rhs_op!(|va, _vb| _mm256_mul_ps(va, inv_b));
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
        BinaryOpType::SquaredDifference => {
            apply_scalar_rhs_op!(|va, vb| {
                let d = _mm256_sub_ps(va, vb);
                _mm256_mul_ps(d, d)
            });
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_scalar_lhs_avx2(
    a: f32,
    rhs: &[f32],
    out: &mut [f32],
    op: BinaryOpType,
    act: FusedActivation,
) -> bool {
    use core::arch::x86_64::*;

    if !matches!(
        act,
        FusedActivation::None | FusedActivation::Relu | FusedActivation::Relu6 | FusedActivation::HardSwish
    ) {
        return false;
    }

    let len = rhs.len();
    let chunks = len / 8;
    let r_ptr = rhs.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let va = _mm256_set1_ps(a);
    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    let three = _mm256_set1_ps(3.0);
    let inv6 = _mm256_set1_ps(1.0 / 6.0);

    macro_rules! apply_scalar_lhs_op {
        ($op_expr:expr) => {
            for i in 0..chunks {
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
                let val = eval_binary(a, rhs[i], op);
                out[i] = apply_activation_scalar(val, act);
            }
        };
    }

    match op {
        BinaryOpType::Add => {
            apply_scalar_lhs_op!(|va, vb| _mm256_add_ps(va, vb));
            true
        }
        BinaryOpType::Sub => {
            apply_scalar_lhs_op!(|va, vb| _mm256_sub_ps(va, vb));
            true
        }
        BinaryOpType::Mul => {
            apply_scalar_lhs_op!(|va, vb| _mm256_mul_ps(va, vb));
            true
        }
        BinaryOpType::Div => {
            apply_scalar_lhs_op!(|va, vb| _mm256_div_ps(va, vb));
            true
        }
        BinaryOpType::Min => {
            apply_scalar_lhs_op!(|va, vb| _mm256_min_ps(va, vb));
            true
        }
        BinaryOpType::Max => {
            apply_scalar_lhs_op!(|va, vb| _mm256_max_ps(va, vb));
            true
        }
        BinaryOpType::SquaredDifference => {
            apply_scalar_lhs_op!(|va, vb| {
                let d = _mm256_sub_ps(va, vb);
                _mm256_mul_ps(d, d)
            });
            true
        }
        _ => false,
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn binary_slice_wasm(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    op: BinaryOpType,
    act: FusedActivation,
) -> bool {
    use core::arch::wasm32::*;

    if !matches!(
        act,
        FusedActivation::None | FusedActivation::Relu | FusedActivation::Relu6 | FusedActivation::HardSwish
    ) {
        return false;
    }

    let len = lhs.len();
    let chunks = len / 4;
    let l_ptr = lhs.as_ptr();
    let r_ptr = rhs.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let zero = f32x4_splat(0.0);
    let six = f32x4_splat(6.0);
    let three = f32x4_splat(3.0);
    let inv6 = f32x4_splat(1.0 / 6.0);

    macro_rules! apply_wasm_op {
        ($op_expr:expr) => {
            for i in 0..chunks {
                let va = v128_load(l_ptr.add(i * 4) as *const v128);
                let vb = v128_load(r_ptr.add(i * 4) as *const v128);
                let mut res = $op_expr(va, vb);

                match act {
                    FusedActivation::Relu => {
                        res = f32x4_pmax(res, zero);
                    }
                    FusedActivation::Relu6 => {
                        res = f32x4_pmin(f32x4_pmax(res, zero), six);
                    }
                    FusedActivation::HardSwish => {
                        let p = f32x4_add(res, three);
                        let cl = f32x4_pmin(f32x4_pmax(p, zero), six);
                        res = f32x4_mul(res, f32x4_mul(cl, inv6));
                    }
                    _ => {}
                }
                v128_store(o_ptr.add(i * 4) as *mut v128, res);
            }
            for i in chunks * 4..len {
                let val = eval_binary(lhs[i], rhs[i], op);
                out[i] = apply_activation_scalar(val, act);
            }
        };
    }

    match op {
        BinaryOpType::Add => {
            apply_wasm_op!(|va, vb| f32x4_add(va, vb));
            true
        }
        BinaryOpType::Sub => {
            apply_wasm_op!(|va, vb| f32x4_sub(va, vb));
            true
        }
        BinaryOpType::Mul => {
            apply_wasm_op!(|va, vb| f32x4_mul(va, vb));
            true
        }
        BinaryOpType::Div => {
            apply_wasm_op!(|va, vb| f32x4_div(va, vb));
            true
        }
        BinaryOpType::Min => {
            apply_wasm_op!(|va, vb| f32x4_pmin(va, vb));
            true
        }
        BinaryOpType::Max => {
            apply_wasm_op!(|va, vb| f32x4_pmax(va, vb));
            true
        }
        BinaryOpType::SquaredDifference => {
            apply_wasm_op!(|va, vb| {
                let d = f32x4_sub(va, vb);
                f32x4_mul(d, d)
            });
            true
        }
        _ => false,
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn binary_scalar_rhs_wasm(
    lhs: &[f32],
    b: f32,
    out: &mut [f32],
    op: BinaryOpType,
    act: FusedActivation,
) -> bool {
    use core::arch::wasm32::*;

    if !matches!(
        act,
        FusedActivation::None | FusedActivation::Relu | FusedActivation::Relu6 | FusedActivation::HardSwish
    ) || (op == BinaryOpType::Div && b == 0.0)
    {
        return false;
    }

    let len = lhs.len();
    let chunks = len / 4;
    let l_ptr = lhs.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let vb = f32x4_splat(b);
    let zero = f32x4_splat(0.0);
    let six = f32x4_splat(6.0);
    let three = f32x4_splat(3.0);
    let inv6 = f32x4_splat(1.0 / 6.0);

    macro_rules! apply_wasm_rhs_op {
        ($op_expr:expr) => {
            for i in 0..chunks {
                let va = v128_load(l_ptr.add(i * 4) as *const v128);
                let mut res = $op_expr(va, vb);

                match act {
                    FusedActivation::Relu => {
                        res = f32x4_pmax(res, zero);
                    }
                    FusedActivation::Relu6 => {
                        res = f32x4_pmin(f32x4_pmax(res, zero), six);
                    }
                    FusedActivation::HardSwish => {
                        let p = f32x4_add(res, three);
                        let cl = f32x4_pmin(f32x4_pmax(p, zero), six);
                        res = f32x4_mul(res, f32x4_mul(cl, inv6));
                    }
                    _ => {}
                }
                v128_store(o_ptr.add(i * 4) as *mut v128, res);
            }
            for i in chunks * 4..len {
                let val = eval_binary(lhs[i], b, op);
                out[i] = apply_activation_scalar(val, act);
            }
        };
    }

    match op {
        BinaryOpType::Add => {
            apply_wasm_rhs_op!(|va, vb| f32x4_add(va, vb));
            true
        }
        BinaryOpType::Sub => {
            apply_wasm_rhs_op!(|va, vb| f32x4_sub(va, vb));
            true
        }
        BinaryOpType::Mul => {
            apply_wasm_rhs_op!(|va, vb| f32x4_mul(va, vb));
            true
        }
        BinaryOpType::Div => {
            let inv_b = f32x4_splat(1.0 / b);
            apply_wasm_rhs_op!(|va, _vb| f32x4_mul(va, inv_b));
            true
        }
        BinaryOpType::Min => {
            apply_wasm_rhs_op!(|va, vb| f32x4_pmin(va, vb));
            true
        }
        BinaryOpType::Max => {
            apply_wasm_rhs_op!(|va, vb| f32x4_pmax(va, vb));
            true
        }
        BinaryOpType::SquaredDifference => {
            apply_wasm_rhs_op!(|va, vb| {
                let d = f32x4_sub(va, vb);
                f32x4_mul(d, d)
            });
            true
        }
        BinaryOpType::MulSilu => {
            let silu_b = b / (1.0 + (-b).exp());
            let vb_silu = f32x4_splat(silu_b);
            apply_wasm_rhs_op!(|va, _vb| f32x4_mul(va, vb_silu));
            true
        }
        _ => false,
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn binary_scalar_lhs_wasm(
    a: f32,
    rhs: &[f32],
    out: &mut [f32],
    op: BinaryOpType,
    act: FusedActivation,
) -> bool {
    use core::arch::wasm32::*;

    if !matches!(
        act,
        FusedActivation::None | FusedActivation::Relu | FusedActivation::Relu6 | FusedActivation::HardSwish
    ) {
        return false;
    }

    let len = rhs.len();
    let chunks = len / 4;
    let r_ptr = rhs.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let va = f32x4_splat(a);
    let zero = f32x4_splat(0.0);
    let six = f32x4_splat(6.0);
    let three = f32x4_splat(3.0);
    let inv6 = f32x4_splat(1.0 / 6.0);

    macro_rules! apply_wasm_lhs_op {
        ($op_expr:expr) => {
            for i in 0..chunks {
                let vb = v128_load(r_ptr.add(i * 4) as *const v128);
                let mut res = $op_expr(va, vb);

                match act {
                    FusedActivation::Relu => {
                        res = f32x4_pmax(res, zero);
                    }
                    FusedActivation::Relu6 => {
                        res = f32x4_pmin(f32x4_pmax(res, zero), six);
                    }
                    FusedActivation::HardSwish => {
                        let p = f32x4_add(res, three);
                        let cl = f32x4_pmin(f32x4_pmax(p, zero), six);
                        res = f32x4_mul(res, f32x4_mul(cl, inv6));
                    }
                    _ => {}
                }
                v128_store(o_ptr.add(i * 4) as *mut v128, res);
            }
            for i in chunks * 4..len {
                let val = eval_binary(a, rhs[i], op);
                out[i] = apply_activation_scalar(val, act);
            }
        };
    }

    match op {
        BinaryOpType::Add => {
            apply_wasm_lhs_op!(|va, vb| f32x4_add(va, vb));
            true
        }
        BinaryOpType::Sub => {
            apply_wasm_lhs_op!(|va, vb| f32x4_sub(va, vb));
            true
        }
        BinaryOpType::Mul => {
            apply_wasm_lhs_op!(|va, vb| f32x4_mul(va, vb));
            true
        }
        BinaryOpType::Div => {
            apply_wasm_lhs_op!(|va, vb| f32x4_div(va, vb));
            true
        }
        BinaryOpType::Min => {
            apply_wasm_lhs_op!(|va, vb| f32x4_pmin(va, vb));
            true
        }
        BinaryOpType::Max => {
            apply_wasm_lhs_op!(|va, vb| f32x4_pmax(va, vb));
            true
        }
        BinaryOpType::SquaredDifference => {
            apply_wasm_lhs_op!(|va, vb| {
                let d = f32x4_sub(va, vb);
                f32x4_mul(d, d)
            });
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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            if binary_slice_wasm(lhs, rhs, out, op, act) {
                return;
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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            if binary_scalar_rhs_wasm(lhs, b, out, op, act) {
                return;
            }
        }
    }

    for i in 0..lhs.len() {
        let val = eval_binary(lhs[i], b, op);
        out[i] = apply_activation_scalar(val, act);
    }
}

pub fn binary_scalar_lhs(a: f32, rhs: &[f32], out: &mut [f32], op: BinaryOpType, act: FusedActivation) {
    if matches!(op, BinaryOpType::Add | BinaryOpType::Mul | BinaryOpType::Min | BinaryOpType::Max | BinaryOpType::SquaredDifference) {
        binary_scalar_rhs(rhs, a, out, op, act);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                if binary_scalar_lhs_avx2(a, rhs, out, op, act) {
                    return;
                }
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            if binary_scalar_lhs_wasm(a, rhs, out, op, act) {
                return;
            }
        }
    }

    for i in 0..rhs.len() {
        let val = eval_binary(a, rhs[i], op);
        out[i] = apply_activation_scalar(val, act);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BcastMode {
    Match,
    Rhs,
    Lhs,
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

    // Coalesce adjacent axes with identical broadcast behavior into contiguous SIMD segments
    let ndim = out_shape.len();
    let lhs_expected = lhs_shape.iter().product::<usize>();
    let rhs_expected = rhs_shape.iter().product::<usize>();

    if ndim <= 8
        && ndim >= lhs_shape.len()
        && ndim >= rhs_shape.len()
        && lhs_data.len() >= lhs_expected
        && rhs_data.len() >= rhs_expected
    {
        let mut lhs_pad = [1usize; 8];
        let mut rhs_pad = [1usize; 8];
        lhs_pad[ndim - lhs_shape.len()..ndim].copy_from_slice(lhs_shape);
        rhs_pad[ndim - rhs_shape.len()..ndim].copy_from_slice(rhs_shape);

        let mut seg_mode = [BcastMode::Match; 8];
        let mut seg_len = [0usize; 8];
        let mut n_segs = 0usize;
        let mut valid = true;

        for d in 0..ndim {
            let od = out_shape[d];
            let ld = lhs_pad[d];
            let rd = rhs_pad[d];
            if (ld != od && ld != 1) || (rd != od && rd != 1) {
                valid = false;
                break;
            }
            if od <= 1 {
                continue;
            }
            let mode = match (ld == od, rd == od) {
                (true, true) => BcastMode::Match,
                (true, false) => BcastMode::Rhs,
                (false, true) => BcastMode::Lhs,
                (false, false) => {
                    valid = false;
                    break;
                }
            };
            if n_segs > 0 && seg_mode[n_segs - 1] == mode {
                seg_len[n_segs - 1] *= od;
            } else {
                seg_mode[n_segs] = mode;
                seg_len[n_segs] = od;
                n_segs += 1;
            }
        }

        if valid {
            match n_segs {
                0 => {
                    let val = eval_binary(lhs_data[0], rhs_data[0], op_type);
                    out_data[0] = apply_activation_scalar(val, act);
                    return;
                }
                1 => match seg_mode[0] {
                    BcastMode::Match => {
                        #[cfg(feature = "rayon")]
                        if out_len >= 8192 {
                            let chunk_size = 4096;
                            out_data[..out_len]
                                .par_chunks_mut(chunk_size)
                                .enumerate()
                                .for_each(|(chunk_idx, out_sub)| {
                                    let offset = chunk_idx * chunk_size;
                                    let end = offset + out_sub.len();
                                    binary_slice(&lhs_data[offset..end], &rhs_data[offset..end], out_sub, op_type, act);
                                });
                            return;
                        }
                        binary_slice(&lhs_data[..out_len], &rhs_data[..out_len], &mut out_data[..out_len], op_type, act);
                        return;
                    }
                    BcastMode::Rhs => {
                        let b = rhs_data[0];
                        #[cfg(feature = "rayon")]
                        if out_len >= 8192 {
                            let chunk_size = 4096;
                            out_data[..out_len]
                                .par_chunks_mut(chunk_size)
                                .enumerate()
                                .for_each(|(chunk_idx, out_sub)| {
                                    let offset = chunk_idx * chunk_size;
                                    let end = offset + out_sub.len();
                                    binary_scalar_rhs(&lhs_data[offset..end], b, out_sub, op_type, act);
                                });
                            return;
                        }
                        binary_scalar_rhs(&lhs_data[..out_len], b, &mut out_data[..out_len], op_type, act);
                        return;
                    }
                    BcastMode::Lhs => {
                        let a = lhs_data[0];
                        #[cfg(feature = "rayon")]
                        if out_len >= 8192 {
                            let chunk_size = 4096;
                            out_data[..out_len]
                                .par_chunks_mut(chunk_size)
                                .enumerate()
                                .for_each(|(chunk_idx, out_sub)| {
                                    let offset = chunk_idx * chunk_size;
                                    let end = offset + out_sub.len();
                                    binary_scalar_lhs(a, &rhs_data[offset..end], out_sub, op_type, act);
                                });
                            return;
                        }
                        binary_scalar_lhs(a, &rhs_data[..out_len], &mut out_data[..out_len], op_type, act);
                        return;
                    }
                },
                2 => {
                    let (outer, inner) = (seg_len[0], seg_len[1]);
                    match (seg_mode[0], seg_mode[1]) {
                        (BcastMode::Rhs, BcastMode::Match) => {
                            let r_slice = &rhs_data[..inner];
                            #[cfg(feature = "rayon")]
                            if outer >= 4 && out_len >= 8192 {
                                out_data[..out_len]
                                    .par_chunks_exact_mut(inner)
                                    .enumerate()
                                    .for_each(|(o, out_sub)| {
                                        binary_slice(&lhs_data[o * inner..(o + 1) * inner], r_slice, out_sub, op_type, act);
                                    });
                                return;
                            }
                            for o in 0..outer {
                                binary_slice(
                                    &lhs_data[o * inner..(o + 1) * inner],
                                    r_slice,
                                    &mut out_data[o * inner..(o + 1) * inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Lhs, BcastMode::Match) => {
                            let l_slice = &lhs_data[..inner];
                            #[cfg(feature = "rayon")]
                            if outer >= 4 && out_len >= 8192 {
                                out_data[..out_len]
                                    .par_chunks_exact_mut(inner)
                                    .enumerate()
                                    .for_each(|(o, out_sub)| {
                                        binary_slice(l_slice, &rhs_data[o * inner..(o + 1) * inner], out_sub, op_type, act);
                                    });
                                return;
                            }
                            for o in 0..outer {
                                binary_slice(
                                    l_slice,
                                    &rhs_data[o * inner..(o + 1) * inner],
                                    &mut out_data[o * inner..(o + 1) * inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Match, BcastMode::Rhs) => {
                            #[cfg(feature = "rayon")]
                            if outer >= 2 && out_len >= 4096 {
                                out_data[..out_len]
                                    .par_chunks_exact_mut(inner)
                                    .enumerate()
                                    .for_each(|(o, out_sub)| {
                                        binary_scalar_rhs(&lhs_data[o * inner..(o + 1) * inner], rhs_data[o], out_sub, op_type, act);
                                    });
                                return;
                            }
                            for o in 0..outer {
                                binary_scalar_rhs(
                                    &lhs_data[o * inner..(o + 1) * inner],
                                    rhs_data[o],
                                    &mut out_data[o * inner..(o + 1) * inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Match, BcastMode::Lhs) => {
                            #[cfg(feature = "rayon")]
                            if outer >= 2 && out_len >= 4096 {
                                out_data[..out_len]
                                    .par_chunks_exact_mut(inner)
                                    .enumerate()
                                    .for_each(|(o, out_sub)| {
                                        binary_scalar_lhs(lhs_data[o], &rhs_data[o * inner..(o + 1) * inner], out_sub, op_type, act);
                                    });
                                return;
                            }
                            for o in 0..outer {
                                binary_scalar_lhs(
                                    lhs_data[o],
                                    &rhs_data[o * inner..(o + 1) * inner],
                                    &mut out_data[o * inner..(o + 1) * inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Rhs, BcastMode::Lhs) => {
                            let r_slice = &rhs_data[..inner];
                            for o in 0..outer {
                                binary_scalar_lhs(
                                    lhs_data[o],
                                    r_slice,
                                    &mut out_data[o * inner..(o + 1) * inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Lhs, BcastMode::Rhs) => {
                            let l_slice = &lhs_data[..inner];
                            for o in 0..outer {
                                binary_scalar_rhs(
                                    l_slice,
                                    rhs_data[o],
                                    &mut out_data[o * inner..(o + 1) * inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        _ => {}
                    }
                }
                3 => {
                    let (outer, mid, inner) = (seg_len[0], seg_len[1], seg_len[2]);
                    match (seg_mode[0], seg_mode[1], seg_mode[2]) {
                        (BcastMode::Match, BcastMode::Rhs, BcastMode::Match) => {
                            for o in 0..outer {
                                let r_slice = &rhs_data[o * inner..(o + 1) * inner];
                                for m in 0..mid {
                                    let idx = (o * mid + m) * inner;
                                    binary_slice(
                                        &lhs_data[idx..idx + inner],
                                        r_slice,
                                        &mut out_data[idx..idx + inner],
                                        op_type,
                                        act,
                                    );
                                }
                            }
                            return;
                        }
                        (BcastMode::Match, BcastMode::Lhs, BcastMode::Match) => {
                            for o in 0..outer {
                                let l_slice = &lhs_data[o * inner..(o + 1) * inner];
                                for m in 0..mid {
                                    let idx = (o * mid + m) * inner;
                                    binary_slice(
                                        l_slice,
                                        &rhs_data[idx..idx + inner],
                                        &mut out_data[idx..idx + inner],
                                        op_type,
                                        act,
                                    );
                                }
                            }
                            return;
                        }
                        (BcastMode::Rhs, BcastMode::Match, BcastMode::Rhs) => {
                            let total_chunks = outer * mid;
                            #[cfg(feature = "rayon")]
                            if total_chunks >= 2 && out_len >= 4096 {
                                out_data[..out_len]
                                    .par_chunks_exact_mut(inner)
                                    .enumerate()
                                    .for_each(|(i, out_sub)| {
                                        let m = i % mid;
                                        binary_scalar_rhs(&lhs_data[i * inner..(i + 1) * inner], rhs_data[m], out_sub, op_type, act);
                                    });
                                return;
                            }
                            for i in 0..total_chunks {
                                let m = i % mid;
                                let idx = i * inner;
                                binary_scalar_rhs(
                                    &lhs_data[idx..idx + inner],
                                    rhs_data[m],
                                    &mut out_data[idx..idx + inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Lhs, BcastMode::Match, BcastMode::Lhs) => {
                            let total_chunks = outer * mid;
                            #[cfg(feature = "rayon")]
                            if total_chunks >= 2 && out_len >= 4096 {
                                out_data[..out_len]
                                    .par_chunks_exact_mut(inner)
                                    .enumerate()
                                    .for_each(|(i, out_sub)| {
                                        let m = i % mid;
                                        binary_scalar_lhs(lhs_data[m], &rhs_data[i * inner..(i + 1) * inner], out_sub, op_type, act);
                                    });
                                return;
                            }
                            for i in 0..total_chunks {
                                let m = i % mid;
                                let idx = i * inner;
                                binary_scalar_lhs(
                                    lhs_data[m],
                                    &rhs_data[idx..idx + inner],
                                    &mut out_data[idx..idx + inner],
                                    op_type,
                                    act,
                                );
                            }
                            return;
                        }
                        (BcastMode::Match, BcastMode::Rhs, BcastMode::Lhs) => {
                            for o in 0..outer {
                                let r_slice = &rhs_data[o * inner..(o + 1) * inner];
                                for m in 0..mid {
                                    let idx = (o * mid + m) * inner;
                                    binary_scalar_lhs(
                                        lhs_data[o * mid + m],
                                        r_slice,
                                        &mut out_data[idx..idx + inner],
                                        op_type,
                                        act,
                                    );
                                }
                            }
                            return;
                        }
                        (BcastMode::Match, BcastMode::Lhs, BcastMode::Rhs) => {
                            for o in 0..outer {
                                let l_slice = &lhs_data[o * inner..(o + 1) * inner];
                                for m in 0..mid {
                                    let idx = (o * mid + m) * inner;
                                    binary_scalar_rhs(
                                        l_slice,
                                        rhs_data[o * mid + m],
                                        &mut out_data[idx..idx + inner],
                                        op_type,
                                        act,
                                    );
                                }
                            }
                            return;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // General broadcasting fallback
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

        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < out_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
}

#[inline]
fn binary_inplace_slice(lhs: &mut [f32], rhs: &[f32], op: BinaryOpType, act: FusedActivation) {
    debug_assert_eq!(lhs.len(), rhs.len());
    match op {
        BinaryOpType::Add => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a += b;
            }
        }
        BinaryOpType::Sub => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a -= b;
            }
        }
        BinaryOpType::Mul => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a *= b;
            }
        }
        BinaryOpType::Div => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = if b != 0.0 { *a / b } else { 0.0 };
            }
        }
        BinaryOpType::Min => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = a.min(b);
            }
        }
        BinaryOpType::Max => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = a.max(b);
            }
        }
        _ => {
            for (a, &b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = eval_binary(*a, b, op);
            }
        }
    }
    if act != FusedActivation::None {
        apply_activation_slice(lhs, act);
    }
}

#[inline]
fn binary_inplace_scalar_rhs(lhs: &mut [f32], b: f32, op: BinaryOpType, act: FusedActivation) {
    match op {
        BinaryOpType::Add => {
            for a in lhs.iter_mut() {
                *a += b;
            }
        }
        BinaryOpType::Sub => {
            for a in lhs.iter_mut() {
                *a -= b;
            }
        }
        BinaryOpType::Mul => {
            for a in lhs.iter_mut() {
                *a *= b;
            }
        }
        BinaryOpType::Div => {
            if b != 0.0 {
                for a in lhs.iter_mut() {
                    *a /= b;
                }
            } else {
                lhs.fill(0.0);
            }
        }
        BinaryOpType::Min => {
            for a in lhs.iter_mut() {
                *a = a.min(b);
            }
        }
        BinaryOpType::Max => {
            for a in lhs.iter_mut() {
                *a = a.max(b);
            }
        }
        _ => {
            for a in lhs.iter_mut() {
                *a = eval_binary(*a, b, op);
            }
        }
    }
    if act != FusedActivation::None {
        apply_activation_slice(lhs, act);
    }
}

/// Mutate `lhs_buf` in-place when `lhs_shape == out_shape` and broadcasting is contiguous.
pub fn try_binary_op_inplace_lhs(
    lhs_shape: &[usize],
    lhs_buf: &mut [f32],
    rhs_shape: &[usize],
    rhs_data: &[f32],
    out_shape: &[usize],
    op_type: BinaryOpType,
    act: FusedActivation,
) -> bool {
    if lhs_shape != out_shape {
        return false;
    }
    let out_len = out_shape.iter().product::<usize>();
    if out_len == 0 {
        return true;
    }
    let ndim = out_shape.len();
    let rhs_expected = rhs_shape.iter().product::<usize>();
    if ndim > 8
        || ndim < rhs_shape.len()
        || lhs_buf.len() < out_len
        || rhs_data.len() < rhs_expected
    {
        return false;
    }

    let mut rhs_pad = [1usize; 8];
    rhs_pad[ndim - rhs_shape.len()..ndim].copy_from_slice(rhs_shape);

    let mut seg_mode = [BcastMode::Match; 8];
    let mut seg_len = [0usize; 8];
    let mut n_segs = 0usize;

    for d in 0..ndim {
        let od = out_shape[d];
        let rd = rhs_pad[d];
        if rd != od && rd != 1 {
            return false;
        }
        if od <= 1 {
            continue;
        }
        let mode = if rd == od { BcastMode::Match } else { BcastMode::Rhs };
        if n_segs > 0 && seg_mode[n_segs - 1] == mode {
            seg_len[n_segs - 1] *= od;
        } else {
            seg_mode[n_segs] = mode;
            seg_len[n_segs] = od;
            n_segs += 1;
        }
    }

    let buf = &mut lhs_buf[..out_len];
    match n_segs {
        0 => {
            let val = eval_binary(buf[0], rhs_data[0], op_type);
            buf[0] = apply_activation_scalar(val, act);
            true
        }
        1 => match seg_mode[0] {
            BcastMode::Match => {
                #[cfg(feature = "rayon")]
                if out_len >= 8192 {
                    buf.par_chunks_mut(4096).enumerate().for_each(|(ci, sub)| {
                        let start = ci * 4096;
                        binary_inplace_slice(sub, &rhs_data[start..start + sub.len()], op_type, act);
                    });
                    return true;
                }
                binary_inplace_slice(buf, &rhs_data[..out_len], op_type, act);
                true
            }
            BcastMode::Rhs => {
                let b = rhs_data[0];
                #[cfg(feature = "rayon")]
                if out_len >= 8192 {
                    buf.par_chunks_mut(4096)
                        .for_each(|sub| binary_inplace_scalar_rhs(sub, b, op_type, act));
                    return true;
                }
                binary_inplace_scalar_rhs(buf, b, op_type, act);
                true
            }
            BcastMode::Lhs => false,
        },
        2 => {
            let (_outer, inner) = (seg_len[0], seg_len[1]);
            match (seg_mode[0], seg_mode[1]) {
                (BcastMode::Rhs, BcastMode::Match) => {
                    let r_slice = &rhs_data[..inner];
                    #[cfg(feature = "rayon")]
                    if _outer >= 4 && out_len >= 8192 {
                        buf.par_chunks_exact_mut(inner)
                            .for_each(|sub| binary_inplace_slice(sub, r_slice, op_type, act));
                        return true;
                    }
                    for sub in buf.chunks_exact_mut(inner) {
                        binary_inplace_slice(sub, r_slice, op_type, act);
                    }
                    true
                }
                (BcastMode::Match, BcastMode::Rhs) => {
                    #[cfg(feature = "rayon")]
                    if _outer >= 2 && out_len >= 4096 {
                        buf.par_chunks_exact_mut(inner)
                            .enumerate()
                            .for_each(|(o, sub)| {
                                binary_inplace_scalar_rhs(sub, rhs_data[o], op_type, act);
                            });
                        return true;
                    }
                    for (o, sub) in buf.chunks_exact_mut(inner).enumerate() {
                        binary_inplace_scalar_rhs(sub, rhs_data[o], op_type, act);
                    }
                    true
                }
                _ => false,
            }
        }
        3 => {
            let (outer, mid, inner) = (seg_len[0], seg_len[1], seg_len[2]);
            match (seg_mode[0], seg_mode[1], seg_mode[2]) {
                (BcastMode::Match, BcastMode::Rhs, BcastMode::Match) => {
                    for o in 0..outer {
                        let r_slice = &rhs_data[o * inner..(o + 1) * inner];
                        for m in 0..mid {
                            let idx = (o * mid + m) * inner;
                            binary_inplace_slice(&mut buf[idx..idx + inner], r_slice, op_type, act);
                        }
                    }
                    true
                }
                (BcastMode::Rhs, BcastMode::Match, BcastMode::Rhs) => {
                    #[cfg(feature = "rayon")]
                    if outer * mid >= 2 && out_len >= 4096 {
                        buf.par_chunks_exact_mut(inner)
                            .enumerate()
                            .for_each(|(i, sub)| {
                                binary_inplace_scalar_rhs(sub, rhs_data[i % mid], op_type, act);
                            });
                        return true;
                    }
                    for (i, sub) in buf.chunks_exact_mut(inner).enumerate() {
                        binary_inplace_scalar_rhs(sub, rhs_data[i % mid], op_type, act);
                    }
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_binary_op_inplace_lhs_broadcast_modes() {
        let shape = [2, 3, 17];
        let lhs_orig: Vec<f32> = (0..102).map(|i| (i as f32 - 40.0) * 0.13).collect();

        for &(rhs_shape, rhs_len) in &[
            (&[1usize][..], 1usize),
            (&[17][..], 17),
            (&[2, 3, 1][..], 6),
            (&[2, 1, 17][..], 34),
            (&[1, 3, 1][..], 3),
            (&[2, 3, 17][..], 102),
        ] {
            let rhs: Vec<f32> = (0..rhs_len).map(|i| (i as f32 + 1.5) * 0.2).collect();
            for &op in &[BinaryOpType::Add, BinaryOpType::Mul, BinaryOpType::Sub] {
                let mut expected = vec![0.0f32; 102];
                binary_op(
                    &shape,
                    &lhs_orig,
                    rhs_shape,
                    &rhs,
                    &shape,
                    &mut expected,
                    op,
                    FusedActivation::HardSwish,
                );
                let mut inplace = lhs_orig.clone();
                assert!(try_binary_op_inplace_lhs(
                    &shape,
                    &mut inplace,
                    rhs_shape,
                    &rhs,
                    &shape,
                    op,
                    FusedActivation::HardSwish,
                ));
                for (i, (&got, &exp)) in inplace.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-5,
                        "mismatch at {i} for rhs_shape={rhs_shape:?} op={op:?}: {got} vs {exp}"
                    );
                }
            }
        }
    }
}
