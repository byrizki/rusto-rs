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

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        let bias_val = bias.map(|b| b[i]).unwrap_or(0.0);
        let bias_v = f32x4_splat(bias_val);

        let mut j = 0;
        // Process 8 floats (2x v128) per iteration
        while j + 8 <= n {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for p in 0..k {
                let a_val = f32x4_splat(*a_row.get_unchecked(p));
                let b_ptr0 = b.as_ptr().add(p * n + j);
                let b_ptr1 = b.as_ptr().add(p * n + j + 4);

                let b0 = v128_load(b_ptr0 as *const v128);
                let b1 = v128_load(b_ptr1 as *const v128);

                acc0 = f32x4_add(acc0, f32x4_mul(a_val, b0));
                acc1 = f32x4_add(acc1, f32x4_mul(a_val, b1));
            }

            let c_ptr0 = c_row.as_mut_ptr().add(j);
            let c_ptr1 = c_row.as_mut_ptr().add(j + 4);

            let mut out0 = f32x4_add(f32x4_mul(alpha_v, acc0), bias_v);
            let mut out1 = f32x4_add(f32x4_mul(alpha_v, acc1), bias_v);

            if beta != 0.0 {
                let old_c0 = v128_load(c_ptr0 as *const v128);
                let old_c1 = v128_load(c_ptr1 as *const v128);
                out0 = f32x4_add(out0, f32x4_mul(beta_v, old_c0));
                out1 = f32x4_add(out1, f32x4_mul(beta_v, old_c1));
            }

            out0 = apply_activation_wasm(out0, act);
            out1 = apply_activation_wasm(out1, act);

            v128_store(c_ptr0 as *mut v128, out0);
            v128_store(c_ptr1 as *mut v128, out1);

            j += 8;
        }

        // Process 4 floats (1x v128)
        while j + 4 <= n {
            let mut acc = f32x4_splat(0.0);
            for p in 0..k {
                let a_val = f32x4_splat(*a_row.get_unchecked(p));
                let b_ptr = b.as_ptr().add(p * n + j);
                let b_vec = v128_load(b_ptr as *const v128);
                acc = f32x4_add(acc, f32x4_mul(a_val, b_vec));
            }

            let c_ptr = c_row.as_mut_ptr().add(j);
            let mut out = f32x4_add(f32x4_mul(alpha_v, acc), bias_v);
            if beta != 0.0 {
                let old_c = v128_load(c_ptr as *const v128);
                out = f32x4_add(out, f32x4_mul(beta_v, old_c));
            }
            out = apply_activation_wasm(out, act);
            v128_store(c_ptr as *mut v128, out);

            j += 4;
        }

        // Scalar remainder
        while j < n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += *a_row.get_unchecked(p) * *b.get_unchecked(p * n + j);
            }
            let mut out = alpha * acc + bias_val;
            if beta != 0.0 {
                out += beta * *c_row.get_unchecked(j);
            }
            *c_row.get_unchecked_mut(j) = apply_activation_scalar(out, act);
            j += 1;
        }
    }
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
