#![cfg(target_arch = "wasm32")]

use core::arch::wasm32::*;
use core::cell::RefCell;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

thread_local! {
    static PACKED_B_WASM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

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
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let alpha_v = f32x4_splat(alpha);
    let beta_v = f32x4_splat(beta);

    if m == 1 {
        gemm_wasm_m1(n, k, a, b, c, alpha, alpha_v, beta, beta_v, bias, act);
        return;
    }

    let a_base = a.as_ptr();
    let b_base = b.as_ptr();
    let c_base = c.as_mut_ptr();

    PACKED_B_WASM.with(|cell| {
        let mut cache = cell.borrow_mut();
        let needed = k * 8;
        if cache.len() < needed {
            cache.resize(needed, 0.0);
        }
        let pb_ptr = cache.as_mut_ptr();

        let mut j = 0;
        while j + 8 <= n {
            let mut b_src = b_base.add(j);
            let mut b_dst = pb_ptr;
            for _ in 0..k {
                v128_store(b_dst as *mut v128, v128_load(b_src as *const v128));
                v128_store(b_dst.add(4) as *mut v128, v128_load(b_src.add(4) as *const v128));
                b_src = b_src.add(n);
                b_dst = b_dst.add(8);
            }

            let mut i = 0;
            while i + 6 <= m {
                microkernel_6x8_packed(
                    a_base.add(i * k),
                    pb_ptr,
                    c_base.add(i * n + j),
                    k,
                    n,
                    i,
                    alpha_v,
                    beta,
                    beta_v,
                    bias,
                    act,
                );
                i += 6;
            }
            if i + 4 <= m {
                microkernel_4x8_packed(
                    a_base.add(i * k),
                    pb_ptr,
                    c_base.add(i * n + j),
                    k,
                    n,
                    i,
                    alpha_v,
                    beta,
                    beta_v,
                    bias,
                    act,
                );
                i += 4;
            }
            if i + 2 <= m {
                microkernel_2x8_packed(
                    a_base.add(i * k),
                    pb_ptr,
                    c_base.add(i * n + j),
                    k,
                    n,
                    i,
                    alpha_v,
                    beta,
                    beta_v,
                    bias,
                    act,
                );
                i += 2;
            }
            if i < m {
                microkernel_1x8_packed(
                    a_base.add(i * k),
                    pb_ptr,
                    c_base.add(i * n + j),
                    k,
                    i,
                    alpha_v,
                    beta,
                    beta_v,
                    bias,
                    act,
                );
            }

            j += 8;
        }

        if j + 4 <= n {
            let mut b_src = b_base.add(j);
            let mut b_dst = pb_ptr;
            for _ in 0..k {
                v128_store(b_dst as *mut v128, v128_load(b_src as *const v128));
                b_src = b_src.add(n);
                b_dst = b_dst.add(4);
            }

            let mut i = 0;
            while i + 4 <= m {
                let a0 = a_base.add(i * k);
                let a1 = a0.add(k);
                let a2 = a1.add(k);
                let a3 = a2.add(k);

                let mut acc0 = f32x4_splat(0.0);
                let mut acc1 = f32x4_splat(0.0);
                let mut acc2 = f32x4_splat(0.0);
                let mut acc3 = f32x4_splat(0.0);

                for p in 0..k {
                    let bv = v128_load(pb_ptr.add(p * 4) as *const v128);
                    acc0 = f32x4_add(acc0, f32x4_mul(f32x4_splat(*a0.add(p)), bv));
                    acc1 = f32x4_add(acc1, f32x4_mul(f32x4_splat(*a1.add(p)), bv));
                    acc2 = f32x4_add(acc2, f32x4_mul(f32x4_splat(*a2.add(p)), bv));
                    acc3 = f32x4_add(acc3, f32x4_mul(f32x4_splat(*a3.add(p)), bv));
                }

                let b0 = f32x4_splat(bias.map(|bs| bs[i]).unwrap_or(0.0));
                let b1 = f32x4_splat(bias.map(|bs| bs[i + 1]).unwrap_or(0.0));
                let b2 = f32x4_splat(bias.map(|bs| bs[i + 2]).unwrap_or(0.0));
                let b3 = f32x4_splat(bias.map(|bs| bs[i + 3]).unwrap_or(0.0));

                store_quad_wasm(c_base.add(i * n + j), acc0, alpha_v, beta, beta_v, b0, act);
                store_quad_wasm(c_base.add((i + 1) * n + j), acc1, alpha_v, beta, beta_v, b1, act);
                store_quad_wasm(c_base.add((i + 2) * n + j), acc2, alpha_v, beta, beta_v, b2, act);
                store_quad_wasm(c_base.add((i + 3) * n + j), acc3, alpha_v, beta, beta_v, b3, act);
                i += 4;
            }

            while i < m {
                let a0 = a_base.add(i * k);
                let mut acc = f32x4_splat(0.0);
                for p in 0..k {
                    let bv = v128_load(pb_ptr.add(p * 4) as *const v128);
                    acc = f32x4_add(acc, f32x4_mul(f32x4_splat(*a0.add(p)), bv));
                }
                let bv_bias = f32x4_splat(bias.map(|bs| bs[i]).unwrap_or(0.0));
                store_quad_wasm(c_base.add(i * n + j), acc, alpha_v, beta, beta_v, bv_bias, act);
                i += 1;
            }

            j += 4;
        }

        while j < n {
            let mut b_src = b_base.add(j);
            for p in 0..k {
                *pb_ptr.add(p) = *b_src;
                b_src = b_src.add(n);
            }

            let mut i = 0;
            while i + 4 <= m {
                let a0 = a_base.add(i * k);
                let a1 = a0.add(k);
                let a2 = a1.add(k);
                let a3 = a2.add(k);
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                let mut acc3 = 0.0f32;
                for p in 0..k {
                    let bv = *pb_ptr.add(p);
                    acc0 += *a0.add(p) * bv;
                    acc1 += *a1.add(p) * bv;
                    acc2 += *a2.add(p) * bv;
                    acc3 += *a3.add(p) * bv;
                }
                store_scalar_wasm(c_base.add(i * n + j), acc0, alpha, beta, bias.map(|bs| bs[i]).unwrap_or(0.0), act);
                store_scalar_wasm(c_base.add((i + 1) * n + j), acc1, alpha, beta, bias.map(|bs| bs[i + 1]).unwrap_or(0.0), act);
                store_scalar_wasm(c_base.add((i + 2) * n + j), acc2, alpha, beta, bias.map(|bs| bs[i + 2]).unwrap_or(0.0), act);
                store_scalar_wasm(c_base.add((i + 3) * n + j), acc3, alpha, beta, bias.map(|bs| bs[i + 3]).unwrap_or(0.0), act);
                i += 4;
            }
            while i < m {
                let a0 = a_base.add(i * k);
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += *a0.add(p) * *pb_ptr.add(p);
                }
                store_scalar_wasm(c_base.add(i * n + j), acc, alpha, beta, bias.map(|bs| bs[i]).unwrap_or(0.0), act);
                i += 1;
            }

            j += 1;
        }
    });
}

#[target_feature(enable = "simd128")]
unsafe fn gemm_wasm_m1(
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    alpha: f32,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias: Option<&[f32]>,
    act: FusedActivation,
) {
    let a_ptr = a.as_ptr();
    let b_base = b.as_ptr();
    let c_ptr = c.as_mut_ptr();
    let bias_val = bias.map(|bs| bs[0]).unwrap_or(0.0);
    let bias_v = f32x4_splat(bias_val);

    let mut j = 0;
    while j + 16 <= n {
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut b_ptr = b_base.add(j);
        for p in 0..k {
            let av = f32x4_splat(*a_ptr.add(p));
            acc0 = f32x4_add(acc0, f32x4_mul(av, v128_load(b_ptr as *const v128)));
            acc1 = f32x4_add(acc1, f32x4_mul(av, v128_load(b_ptr.add(4) as *const v128)));
            acc2 = f32x4_add(acc2, f32x4_mul(av, v128_load(b_ptr.add(8) as *const v128)));
            acc3 = f32x4_add(acc3, f32x4_mul(av, v128_load(b_ptr.add(12) as *const v128)));
            b_ptr = b_ptr.add(n);
        }
        store_pair_wasm(c_ptr.add(j), acc0, acc1, alpha_v, beta, beta_v, bias_v, act);
        store_pair_wasm(c_ptr.add(j + 8), acc2, acc3, alpha_v, beta, beta_v, bias_v, act);
        j += 16;
    }

    while j + 4 <= n {
        let mut acc = f32x4_splat(0.0);
        let mut b_ptr = b_base.add(j);
        for p in 0..k {
            let av = f32x4_splat(*a_ptr.add(p));
            acc = f32x4_add(acc, f32x4_mul(av, v128_load(b_ptr as *const v128)));
            b_ptr = b_ptr.add(n);
        }
        store_quad_wasm(c_ptr.add(j), acc, alpha_v, beta, beta_v, bias_v, act);
        j += 4;
    }

    while j < n {
        let mut acc = 0.0f32;
        let mut b_ptr = b_base.add(j);
        for p in 0..k {
            acc += *a_ptr.add(p) * *b_ptr;
            b_ptr = b_ptr.add(n);
        }
        store_scalar_wasm(c_ptr.add(j), acc, alpha, beta, bias_val, act);
        j += 1;
    }
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn microkernel_6x8_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    row_idx: usize,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias: Option<&[f32]>,
    act: FusedActivation,
) {
    let a0 = a_ptr;
    let a1 = a0.add(k);
    let a2 = a1.add(k);
    let a3 = a2.add(k);
    let a4 = a3.add(k);
    let a5 = a4.add(k);

    let mut acc00 = f32x4_splat(0.0);
    let mut acc01 = f32x4_splat(0.0);
    let mut acc10 = f32x4_splat(0.0);
    let mut acc11 = f32x4_splat(0.0);
    let mut acc20 = f32x4_splat(0.0);
    let mut acc21 = f32x4_splat(0.0);
    let mut acc30 = f32x4_splat(0.0);
    let mut acc31 = f32x4_splat(0.0);
    let mut acc40 = f32x4_splat(0.0);
    let mut acc41 = f32x4_splat(0.0);
    let mut acc50 = f32x4_splat(0.0);
    let mut acc51 = f32x4_splat(0.0);

    let mut bp = b_packed;
    for p in 0..k {
        let bv0 = v128_load(bp as *const v128);
        let bv1 = v128_load(bp.add(4) as *const v128);

        let av0 = f32x4_splat(*a0.add(p));
        acc00 = f32x4_add(acc00, f32x4_mul(av0, bv0));
        acc01 = f32x4_add(acc01, f32x4_mul(av0, bv1));

        let av1 = f32x4_splat(*a1.add(p));
        acc10 = f32x4_add(acc10, f32x4_mul(av1, bv0));
        acc11 = f32x4_add(acc11, f32x4_mul(av1, bv1));

        let av2 = f32x4_splat(*a2.add(p));
        acc20 = f32x4_add(acc20, f32x4_mul(av2, bv0));
        acc21 = f32x4_add(acc21, f32x4_mul(av2, bv1));

        let av3 = f32x4_splat(*a3.add(p));
        acc30 = f32x4_add(acc30, f32x4_mul(av3, bv0));
        acc31 = f32x4_add(acc31, f32x4_mul(av3, bv1));

        let av4 = f32x4_splat(*a4.add(p));
        acc40 = f32x4_add(acc40, f32x4_mul(av4, bv0));
        acc41 = f32x4_add(acc41, f32x4_mul(av4, bv1));

        let av5 = f32x4_splat(*a5.add(p));
        acc50 = f32x4_add(acc50, f32x4_mul(av5, bv0));
        acc51 = f32x4_add(acc51, f32x4_mul(av5, bv1));

        bp = bp.add(8);
    }

    let b0 = f32x4_splat(bias.map(|bs| bs[row_idx]).unwrap_or(0.0));
    let b1 = f32x4_splat(bias.map(|bs| bs[row_idx + 1]).unwrap_or(0.0));
    let b2 = f32x4_splat(bias.map(|bs| bs[row_idx + 2]).unwrap_or(0.0));
    let b3 = f32x4_splat(bias.map(|bs| bs[row_idx + 3]).unwrap_or(0.0));
    let b4 = f32x4_splat(bias.map(|bs| bs[row_idx + 4]).unwrap_or(0.0));
    let b5 = f32x4_splat(bias.map(|bs| bs[row_idx + 5]).unwrap_or(0.0));

    store_pair_wasm(c_ptr, acc00, acc01, alpha_v, beta, beta_v, b0, act);
    store_pair_wasm(c_ptr.add(n), acc10, acc11, alpha_v, beta, beta_v, b1, act);
    store_pair_wasm(c_ptr.add(2 * n), acc20, acc21, alpha_v, beta, beta_v, b2, act);
    store_pair_wasm(c_ptr.add(3 * n), acc30, acc31, alpha_v, beta, beta_v, b3, act);
    store_pair_wasm(c_ptr.add(4 * n), acc40, acc41, alpha_v, beta, beta_v, b4, act);
    store_pair_wasm(c_ptr.add(5 * n), acc50, acc51, alpha_v, beta, beta_v, b5, act);
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn microkernel_4x8_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    row_idx: usize,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias: Option<&[f32]>,
    act: FusedActivation,
) {
    let a0 = a_ptr;
    let a1 = a0.add(k);
    let a2 = a1.add(k);
    let a3 = a2.add(k);

    let mut acc00 = f32x4_splat(0.0);
    let mut acc01 = f32x4_splat(0.0);
    let mut acc10 = f32x4_splat(0.0);
    let mut acc11 = f32x4_splat(0.0);
    let mut acc20 = f32x4_splat(0.0);
    let mut acc21 = f32x4_splat(0.0);
    let mut acc30 = f32x4_splat(0.0);
    let mut acc31 = f32x4_splat(0.0);

    let mut bp = b_packed;
    for p in 0..k {
        let bv0 = v128_load(bp as *const v128);
        let bv1 = v128_load(bp.add(4) as *const v128);

        let av0 = f32x4_splat(*a0.add(p));
        acc00 = f32x4_add(acc00, f32x4_mul(av0, bv0));
        acc01 = f32x4_add(acc01, f32x4_mul(av0, bv1));

        let av1 = f32x4_splat(*a1.add(p));
        acc10 = f32x4_add(acc10, f32x4_mul(av1, bv0));
        acc11 = f32x4_add(acc11, f32x4_mul(av1, bv1));

        let av2 = f32x4_splat(*a2.add(p));
        acc20 = f32x4_add(acc20, f32x4_mul(av2, bv0));
        acc21 = f32x4_add(acc21, f32x4_mul(av2, bv1));

        let av3 = f32x4_splat(*a3.add(p));
        acc30 = f32x4_add(acc30, f32x4_mul(av3, bv0));
        acc31 = f32x4_add(acc31, f32x4_mul(av3, bv1));

        bp = bp.add(8);
    }

    let b0 = f32x4_splat(bias.map(|bs| bs[row_idx]).unwrap_or(0.0));
    let b1 = f32x4_splat(bias.map(|bs| bs[row_idx + 1]).unwrap_or(0.0));
    let b2 = f32x4_splat(bias.map(|bs| bs[row_idx + 2]).unwrap_or(0.0));
    let b3 = f32x4_splat(bias.map(|bs| bs[row_idx + 3]).unwrap_or(0.0));

    store_pair_wasm(c_ptr, acc00, acc01, alpha_v, beta, beta_v, b0, act);
    store_pair_wasm(c_ptr.add(n), acc10, acc11, alpha_v, beta, beta_v, b1, act);
    store_pair_wasm(c_ptr.add(2 * n), acc20, acc21, alpha_v, beta, beta_v, b2, act);
    store_pair_wasm(c_ptr.add(3 * n), acc30, acc31, alpha_v, beta, beta_v, b3, act);
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn microkernel_2x8_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    row_idx: usize,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias: Option<&[f32]>,
    act: FusedActivation,
) {
    let a0 = a_ptr;
    let a1 = a0.add(k);

    let mut acc00 = f32x4_splat(0.0);
    let mut acc01 = f32x4_splat(0.0);
    let mut acc10 = f32x4_splat(0.0);
    let mut acc11 = f32x4_splat(0.0);

    let mut bp = b_packed;
    for p in 0..k {
        let bv0 = v128_load(bp as *const v128);
        let bv1 = v128_load(bp.add(4) as *const v128);

        let av0 = f32x4_splat(*a0.add(p));
        acc00 = f32x4_add(acc00, f32x4_mul(av0, bv0));
        acc01 = f32x4_add(acc01, f32x4_mul(av0, bv1));

        let av1 = f32x4_splat(*a1.add(p));
        acc10 = f32x4_add(acc10, f32x4_mul(av1, bv0));
        acc11 = f32x4_add(acc11, f32x4_mul(av1, bv1));

        bp = bp.add(8);
    }

    let b0 = f32x4_splat(bias.map(|bs| bs[row_idx]).unwrap_or(0.0));
    let b1 = f32x4_splat(bias.map(|bs| bs[row_idx + 1]).unwrap_or(0.0));
    store_pair_wasm(c_ptr, acc00, acc01, alpha_v, beta, beta_v, b0, act);
    store_pair_wasm(c_ptr.add(n), acc10, acc11, alpha_v, beta, beta_v, b1, act);
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn microkernel_1x8_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    row_idx: usize,
    alpha_v: v128,
    beta: f32,
    beta_v: v128,
    bias: Option<&[f32]>,
    act: FusedActivation,
) {
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut bp = b_packed;
    for p in 0..k {
        let av = f32x4_splat(*a_ptr.add(p));
        acc0 = f32x4_add(acc0, f32x4_mul(av, v128_load(bp as *const v128)));
        acc1 = f32x4_add(acc1, f32x4_mul(av, v128_load(bp.add(4) as *const v128)));
        bp = bp.add(8);
    }
    let b0 = f32x4_splat(bias.map(|bs| bs[row_idx]).unwrap_or(0.0));
    store_pair_wasm(c_ptr, acc0, acc1, alpha_v, beta, beta_v, b0, act);
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
        FusedActivation::Relu => f32x4_pmax(v, f32x4_splat(0.0)),
        FusedActivation::Relu6 => f32x4_pmin(f32x4_pmax(v, f32x4_splat(0.0)), f32x4_splat(6.0)),
        FusedActivation::HardSwish => {
            let zero = f32x4_splat(0.0);
            let six = f32x4_splat(6.0);
            let shifted = f32x4_add(v, f32x4_splat(3.0));
            let clamped = f32x4_pmin(f32x4_pmax(shifted, zero), six);
            f32x4_mul(v, f32x4_mul(clamped, f32x4_splat(1.0 / 6.0)))
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
