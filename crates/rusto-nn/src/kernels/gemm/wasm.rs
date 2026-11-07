#![cfg(target_arch = "wasm32")]

use core::arch::wasm32::*;
use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

#[target_feature(enable = "simd128")]
pub unsafe fn gemm_wasm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    alpha: f32,
    beta: f32,
    bias: Option<&[f32]>,
    act: FusedActivation,
) {
    let alpha_v = f32x4_splat(alpha);
    let beta_v = f32x4_splat(beta);

    let mut i = 0;

    // 4x8 register-tiled microkernel: loads each B vector once across 4 rows of A
    while i + 4 <= m {
        let a_ptr0 = a.as_ptr().add(i * k);
        let a_ptr1 = a.as_ptr().add((i + 1) * k);
        let a_ptr2 = a.as_ptr().add((i + 2) * k);
        let a_ptr3 = a.as_ptr().add((i + 3) * k);

        let b0_s = bias.map(|bs| bs[i]).unwrap_or(0.0);
        let b1_s = bias.map(|bs| bs[i + 1]).unwrap_or(0.0);
        let b2_s = bias.map(|bs| bs[i + 2]).unwrap_or(0.0);
        let b3_s = bias.map(|bs| bs[i + 3]).unwrap_or(0.0);

        let bias_v0 = f32x4_splat(b0_s);
        let bias_v1 = f32x4_splat(b1_s);
        let bias_v2 = f32x4_splat(b2_s);
        let bias_v3 = f32x4_splat(b3_s);

        let c_base0 = c.as_mut_ptr().add(i * n);
        let c_base1 = c.as_mut_ptr().add((i + 1) * n);
        let c_base2 = c.as_mut_ptr().add((i + 2) * n);
        let c_base3 = c.as_mut_ptr().add((i + 3) * n);

        let mut j = 0;
        while j + 8 <= n {
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            let mut b_ptr = b.as_ptr().add(j);
            for p in 0..k {
                let bv0 = v128_load(b_ptr as *const v128);
                let bv1 = v128_load(b_ptr.add(4) as *const v128);

                let av0 = f32x4_splat(*a_ptr0.add(p));
                acc00 = f32x4_add(acc00, f32x4_mul(av0, bv0));
                acc01 = f32x4_add(acc01, f32x4_mul(av0, bv1));

                let av1 = f32x4_splat(*a_ptr1.add(p));
                acc10 = f32x4_add(acc10, f32x4_mul(av1, bv0));
                acc11 = f32x4_add(acc11, f32x4_mul(av1, bv1));

                let av2 = f32x4_splat(*a_ptr2.add(p));
                acc20 = f32x4_add(acc20, f32x4_mul(av2, bv0));
                acc21 = f32x4_add(acc21, f32x4_mul(av2, bv1));

                let av3 = f32x4_splat(*a_ptr3.add(p));
                acc30 = f32x4_add(acc30, f32x4_mul(av3, bv0));
                acc31 = f32x4_add(acc31, f32x4_mul(av3, bv1));

                b_ptr = b_ptr.add(n);
            }

            store_pair_wasm(c_base0.add(j), acc00, acc01, alpha_v, beta, beta_v, bias_v0, act);
            store_pair_wasm(c_base1.add(j), acc10, acc11, alpha_v, beta, beta_v, bias_v1, act);
            store_pair_wasm(c_base2.add(j), acc20, acc21, alpha_v, beta, beta_v, bias_v2, act);
            store_pair_wasm(c_base3.add(j), acc30, acc31, alpha_v, beta, beta_v, bias_v3, act);

            j += 8;
        }

        while j + 4 <= n {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);
            let mut acc2 = f32x4_splat(0.0);
            let mut acc3 = f32x4_splat(0.0);

            let mut b_ptr = b.as_ptr().add(j);
            for p in 0..k {
                let bv = v128_load(b_ptr as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(f32x4_splat(*a_ptr0.add(p)), bv));
                acc1 = f32x4_add(acc1, f32x4_mul(f32x4_splat(*a_ptr1.add(p)), bv));
                acc2 = f32x4_add(acc2, f32x4_mul(f32x4_splat(*a_ptr2.add(p)), bv));
                acc3 = f32x4_add(acc3, f32x4_mul(f32x4_splat(*a_ptr3.add(p)), bv));
                b_ptr = b_ptr.add(n);
            }

            store_quad_wasm(c_base0.add(j), acc0, alpha_v, beta, beta_v, bias_v0, act);
            store_quad_wasm(c_base1.add(j), acc1, alpha_v, beta, beta_v, bias_v1, act);
            store_quad_wasm(c_base2.add(j), acc2, alpha_v, beta, beta_v, bias_v2, act);
            store_quad_wasm(c_base3.add(j), acc3, alpha_v, beta, beta_v, bias_v3, act);

            j += 4;
        }

        while j < n {
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;

            let mut b_ptr = b.as_ptr().add(j);
            for p in 0..k {
                let bv = *b_ptr;
                acc0 += *a_ptr0.add(p) * bv;
                acc1 += *a_ptr1.add(p) * bv;
                acc2 += *a_ptr2.add(p) * bv;
                acc3 += *a_ptr3.add(p) * bv;
                b_ptr = b_ptr.add(n);
            }

            store_scalar_wasm(c_base0.add(j), acc0, alpha, beta, b0_s, act);
            store_scalar_wasm(c_base1.add(j), acc1, alpha, beta, b1_s, act);
            store_scalar_wasm(c_base2.add(j), acc2, alpha, beta, b2_s, act);
            store_scalar_wasm(c_base3.add(j), acc3, alpha, beta, b3_s, act);

            j += 1;
        }

        i += 4;
    }

    // Remaining rows (0..3)
    while i < m {
        let a_ptr = a.as_ptr().add(i * k);
        let c_base = c.as_mut_ptr().add(i * n);
        let bias_val = bias.map(|bs| bs[i]).unwrap_or(0.0);
        let bias_v = f32x4_splat(bias_val);

        let mut j = 0;
        while j + 8 <= n {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            let mut b_ptr = b.as_ptr().add(j);
            for p in 0..k {
                let av = f32x4_splat(*a_ptr.add(p));
                let bv0 = v128_load(b_ptr as *const v128);
                let bv1 = v128_load(b_ptr.add(4) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(av, bv0));
                acc1 = f32x4_add(acc1, f32x4_mul(av, bv1));
                b_ptr = b_ptr.add(n);
            }

            store_pair_wasm(c_base.add(j), acc0, acc1, alpha_v, beta, beta_v, bias_v, act);
            j += 8;
        }

        while j + 4 <= n {
            let mut acc = f32x4_splat(0.0);
            let mut b_ptr = b.as_ptr().add(j);
            for p in 0..k {
                let av = f32x4_splat(*a_ptr.add(p));
                let bv = v128_load(b_ptr as *const v128);
                acc = f32x4_add(acc, f32x4_mul(av, bv));
                b_ptr = b_ptr.add(n);
            }

            store_quad_wasm(c_base.add(j), acc, alpha_v, beta, beta_v, bias_v, act);
            j += 4;
        }

        while j < n {
            let mut acc = 0.0f32;
            let mut b_ptr = b.as_ptr().add(j);
            for p in 0..k {
                acc += *a_ptr.add(p) * *b_ptr;
                b_ptr = b_ptr.add(n);
            }
            store_scalar_wasm(c_base.add(j), acc, alpha, beta, bias_val, act);
            j += 1;
        }

        i += 1;
    }
}

#[inline(always)]
unsafe fn store_pair_wasm(
    c_ptr: *mut f32,
    acc0: v128,
    acc1: v128,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias_v: v128,
    act: FusedActivation,
) {
    let mut out0 = f32x4_add(f32x4_mul(alpha_v, acc0), bias_v);
    let mut out1 = f32x4_add(f32x4_mul(alpha_v, acc1), bias_v);

    if beta != 0.0 {
        let old0 = v128_load(c_ptr as *const v128);
        let old1 = v128_load(c_ptr.add(4) as *const v128);
        out0 = f32x4_add(out0, f32x4_mul(beta_v, old0));
        out1 = f32x4_add(out1, f32x4_mul(beta_v, old1));
    }

    out0 = apply_activation_wasm(out0, act);
    out1 = apply_activation_wasm(out1, act);

    v128_store(c_ptr as *mut v128, out0);
    v128_store(c_ptr.add(4) as *mut v128, out1);
}

#[inline(always)]
unsafe fn store_quad_wasm(
    c_ptr: *mut f32,
    acc: v128,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias_v: v128,
    act: FusedActivation,
) {
    let mut out = f32x4_add(f32x4_mul(alpha_v, acc), bias_v);
    if beta != 0.0 {
        let old = v128_load(c_ptr as *const v128);
        out = f32x4_add(out, f32x4_mul(beta_v, old));
    }
    out = apply_activation_wasm(out, act);
    v128_store(c_ptr as *mut v128, out);
}

#[inline(always)]
unsafe fn store_scalar_wasm(
    c_ptr: *mut f32,
    acc: f32,
    alpha: f32,
    beta: f32,
    bias_val: f32,
    act: FusedActivation,
) {
    let mut out = alpha * acc + bias_val;
    if beta != 0.0 {
        out += beta * *c_ptr;
    }
    *c_ptr = apply_activation_scalar(out, act);
}

#[inline(always)]
unsafe fn apply_activation_wasm(v: v128, act: FusedActivation) -> v128 {
    match act {
        FusedActivation::None => v,
        FusedActivation::Relu => {
            let zero = f32x4_splat(0.0);
            f32x4_pmax(v, zero)
        }
        FusedActivation::Relu6 => {
            let zero = f32x4_splat(0.0);
            let six = f32x4_splat(6.0);
            f32x4_pmin(f32x4_pmax(v, zero), six)
        }
        FusedActivation::HardSwish => {
            let zero = f32x4_splat(0.0);
            let six = f32x4_splat(6.0);
            let three = f32x4_splat(3.0);
            let inv_six = f32x4_splat(1.0 / 6.0);
            let shifted = f32x4_add(v, three);
            let clamped = f32x4_pmin(f32x4_pmax(shifted, zero), six);
            f32x4_mul(v, f32x4_mul(clamped, inv_six))
        }
        _ => {
            let mut arr = [0.0f32; 4];
            v128_store(arr.as_mut_ptr() as *mut v128, v);
            for x in &mut arr {
                *x = apply_activation_scalar(*x, act);
            }
            v128_load(arr.as_ptr() as *const v128)
        }
    }
}
