#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

#[derive(Copy, Clone)]
struct SendMutPtr(pub *mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

impl SendMutPtr {
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Copy, Clone)]
struct SendConstPtr(pub *const f32);
unsafe impl Send for SendConstPtr {}
unsafe impl Sync for SendConstPtr {}

impl SendConstPtr {
    #[inline(always)]
    fn get(self) -> *const f32 {
        self.0
    }
}

#[inline(always)]
unsafe fn store_row_2x16(
    r0: &mut __m512,
    r1: &mut __m512,
    c_row_ptr: *mut f32,
    row_idx: usize,
    alpha_vec: __m512,
    beta: f32,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    *r0 = _mm512_mul_ps(*r0, alpha_vec);
    *r1 = _mm512_mul_ps(*r1, alpha_vec);

    if beta != 0.0 {
        let prev0 = _mm512_loadu_ps(c_row_ptr);
        let prev1 = _mm512_loadu_ps(c_row_ptr.add(16));
        *r0 = _mm512_fmadd_ps(prev0, beta_vec, *r0);
        *r1 = _mm512_fmadd_ps(prev1, beta_vec, *r1);
    }

    if let Some(bias_slice) = bias {
        let bias_val = _mm512_set1_ps(bias_slice[row_idx]);
        *r0 = _mm512_add_ps(*r0, bias_val);
        *r1 = _mm512_add_ps(*r1, bias_val);
    }

    match act {
        FusedActivation::Relu => {
            *r0 = _mm512_max_ps(zero_vec, *r0);
            *r1 = _mm512_max_ps(zero_vec, *r1);
        }
        FusedActivation::Relu6 => {
            *r0 = _mm512_min_ps(six_vec, _mm512_max_ps(zero_vec, *r0));
            *r1 = _mm512_min_ps(six_vec, _mm512_max_ps(zero_vec, *r1));
        }
        FusedActivation::HardSwish => {
            let p0 = _mm512_add_ps(*r0, three_vec);
            let cl0 = _mm512_min_ps(six_vec, _mm512_max_ps(zero_vec, p0));
            *r0 = _mm512_mul_ps(*r0, _mm512_mul_ps(cl0, inv_six_vec));

            let p1 = _mm512_add_ps(*r1, three_vec);
            let cl1 = _mm512_min_ps(six_vec, _mm512_max_ps(zero_vec, p1));
            *r1 = _mm512_mul_ps(*r1, _mm512_mul_ps(cl1, inv_six_vec));
        }
        _ => {}
    }

    _mm512_storeu_ps(c_row_ptr, *r0);
    _mm512_storeu_ps(c_row_ptr.add(16), *r1);
}

#[inline(always)]
unsafe fn store_row_1x16(
    r0: &mut __m512,
    c_row_ptr: *mut f32,
    row_idx: usize,
    alpha_vec: __m512,
    beta: f32,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    *r0 = _mm512_mul_ps(*r0, alpha_vec);

    if beta != 0.0 {
        let prev0 = _mm512_loadu_ps(c_row_ptr);
        *r0 = _mm512_fmadd_ps(prev0, beta_vec, *r0);
    }

    if let Some(bias_slice) = bias {
        let bias_val = _mm512_set1_ps(bias_slice[row_idx]);
        *r0 = _mm512_add_ps(*r0, bias_val);
    }

    match act {
        FusedActivation::Relu => {
            *r0 = _mm512_max_ps(zero_vec, *r0);
        }
        FusedActivation::Relu6 => {
            *r0 = _mm512_min_ps(six_vec, _mm512_max_ps(zero_vec, *r0));
        }
        FusedActivation::HardSwish => {
            let p0 = _mm512_add_ps(*r0, three_vec);
            let cl0 = _mm512_min_ps(six_vec, _mm512_max_ps(zero_vec, p0));
            *r0 = _mm512_mul_ps(*r0, _mm512_mul_ps(cl0, inv_six_vec));
        }
        _ => {}
    }

    _mm512_storeu_ps(c_row_ptr, *r0);
}

#[inline(always)]
unsafe fn microkernel_8x32_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    alpha_vec: __m512,
    beta: f32,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    let mut c00 = _mm512_setzero_ps();
    let mut c01 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();
    let mut c20 = _mm512_setzero_ps();
    let mut c21 = _mm512_setzero_ps();
    let mut c30 = _mm512_setzero_ps();
    let mut c31 = _mm512_setzero_ps();
    let mut c40 = _mm512_setzero_ps();
    let mut c41 = _mm512_setzero_ps();
    let mut c50 = _mm512_setzero_ps();
    let mut c51 = _mm512_setzero_ps();
    let mut c60 = _mm512_setzero_ps();
    let mut c61 = _mm512_setzero_ps();
    let mut c70 = _mm512_setzero_ps();
    let mut c71 = _mm512_setzero_ps();

    for p in 0..k {
        let b0 = _mm512_loadu_ps(b_packed.add(p * 32));
        let b1 = _mm512_loadu_ps(b_packed.add(p * 32 + 16));

        let a0 = _mm512_set1_ps(*a_ptr.add(p));
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);

        let a1 = _mm512_set1_ps(*a_ptr.add(k + p));
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);

        let a2 = _mm512_set1_ps(*a_ptr.add(2 * k + p));
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);

        let a3 = _mm512_set1_ps(*a_ptr.add(3 * k + p));
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);

        let a4 = _mm512_set1_ps(*a_ptr.add(4 * k + p));
        c40 = _mm512_fmadd_ps(a4, b0, c40);
        c41 = _mm512_fmadd_ps(a4, b1, c41);

        let a5 = _mm512_set1_ps(*a_ptr.add(5 * k + p));
        c50 = _mm512_fmadd_ps(a5, b0, c50);
        c51 = _mm512_fmadd_ps(a5, b1, c51);

        let a6 = _mm512_set1_ps(*a_ptr.add(6 * k + p));
        c60 = _mm512_fmadd_ps(a6, b0, c60);
        c61 = _mm512_fmadd_ps(a6, b1, c61);

        let a7 = _mm512_set1_ps(*a_ptr.add(7 * k + p));
        c70 = _mm512_fmadd_ps(a7, b0, c70);
        c71 = _mm512_fmadd_ps(a7, b1, c71);
    }

    store_row_2x16(&mut c00, &mut c01, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c10, &mut c11, c_ptr.add(n), global_i + 1, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c20, &mut c21, c_ptr.add(2 * n), global_i + 2, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c30, &mut c31, c_ptr.add(3 * n), global_i + 3, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c40, &mut c41, c_ptr.add(4 * n), global_i + 4, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c50, &mut c51, c_ptr.add(5 * n), global_i + 5, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c60, &mut c61, c_ptr.add(6 * n), global_i + 6, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c70, &mut c71, c_ptr.add(7 * n), global_i + 7, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn microkernel_4x32_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    alpha_vec: __m512,
    beta: f32,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    let mut c00 = _mm512_setzero_ps();
    let mut c01 = _mm512_setzero_ps();
    let mut c10 = _mm512_setzero_ps();
    let mut c11 = _mm512_setzero_ps();
    let mut c20 = _mm512_setzero_ps();
    let mut c21 = _mm512_setzero_ps();
    let mut c30 = _mm512_setzero_ps();
    let mut c31 = _mm512_setzero_ps();

    for p in 0..k {
        let b0 = _mm512_loadu_ps(b_packed.add(p * 32));
        let b1 = _mm512_loadu_ps(b_packed.add(p * 32 + 16));

        let a0 = _mm512_set1_ps(*a_ptr.add(p));
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);

        let a1 = _mm512_set1_ps(*a_ptr.add(k + p));
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);

        let a2 = _mm512_set1_ps(*a_ptr.add(2 * k + p));
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);

        let a3 = _mm512_set1_ps(*a_ptr.add(3 * k + p));
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
    }

    store_row_2x16(&mut c00, &mut c01, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c10, &mut c11, c_ptr.add(n), global_i + 1, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c20, &mut c21, c_ptr.add(2 * n), global_i + 2, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x16(&mut c30, &mut c31, c_ptr.add(3 * n), global_i + 3, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn microkernel_1x32_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    global_i: usize,
    alpha_vec: __m512,
    beta: f32,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    let mut c00 = _mm512_setzero_ps();
    let mut c01 = _mm512_setzero_ps();

    for p in 0..k {
        let b0 = _mm512_loadu_ps(b_packed.add(p * 32));
        let b1 = _mm512_loadu_ps(b_packed.add(p * 32 + 16));
        let a0 = _mm512_set1_ps(*a_ptr.add(p));
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
    }

    store_row_2x16(&mut c00, &mut c01, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn microkernel_1x16_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    global_i: usize,
    alpha_vec: __m512,
    beta: f32,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    let mut c00 = _mm512_setzero_ps();

    for p in 0..k {
        let b0 = _mm512_loadu_ps(b_packed.add(p * 16));
        let a0 = _mm512_set1_ps(*a_ptr.add(p));
        c00 = _mm512_fmadd_ps(a0, b0, c00);
    }

    store_row_1x16(&mut c00, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn gemm_compute_row_block_packed(
    r_start: usize,
    r_count: usize,
    n: usize,
    k: usize,
    a_base: *const f32,
    pb_base: *const f32,
    b_raw: *const f32,
    c_base: *mut f32,
    alpha: f32,
    beta: f32,
    alpha_vec: __m512,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    let n_tiles_32 = n / 32;
    let n_packed_32 = n_tiles_32 * 32;

    let mut i = 0;
    while i + 8 <= r_count {
        let global_i = r_start + i;
        let a_ptr = a_base.add(global_i * k);
        let c_ptr = c_base.add(global_i * n);

        for jt in 0..n_tiles_32 {
            let j = jt * 32;
            let b_tile = pb_base.add(jt * k * 32);
            microkernel_8x32_packed(
                a_ptr,
                b_tile,
                c_ptr.add(j),
                k,
                n,
                global_i,
                alpha_vec,
                beta,
                beta_vec,
                bias,
                act,
                zero_vec,
                six_vec,
                three_vec,
                inv_six_vec,
            );
        }

        // Remainder columns for this 8-row block
        for col in n_packed_32..n {
            for r in 0..8 {
                let row = global_i + r;
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a_base.add(row * k + p) * *b_raw.add(p * n + col);
                }
                let mut val = alpha * sum;
                if let Some(b_slice) = bias {
                    val += b_slice[row];
                }
                if beta != 0.0 {
                    val += beta * *c_base.add(row * n + col);
                }
                *c_base.add(row * n + col) = apply_activation_scalar(val, act);
            }
        }

        i += 8;
    }

    while i + 4 <= r_count {
        let global_i = r_start + i;
        let a_ptr = a_base.add(global_i * k);
        let c_ptr = c_base.add(global_i * n);

        for jt in 0..n_tiles_32 {
            let j = jt * 32;
            let b_tile = pb_base.add(jt * k * 32);
            microkernel_4x32_packed(
                a_ptr,
                b_tile,
                c_ptr.add(j),
                k,
                n,
                global_i,
                alpha_vec,
                beta,
                beta_vec,
                bias,
                act,
                zero_vec,
                six_vec,
                three_vec,
                inv_six_vec,
            );
        }

        for col in n_packed_32..n {
            for r in 0..4 {
                let row = global_i + r;
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a_base.add(row * k + p) * *b_raw.add(p * n + col);
                }
                let mut val = alpha * sum;
                if let Some(b_slice) = bias {
                    val += b_slice[row];
                }
                if beta != 0.0 {
                    val += beta * *c_base.add(row * n + col);
                }
                *c_base.add(row * n + col) = apply_activation_scalar(val, act);
            }
        }

        i += 4;
    }

    while i < r_count {
        let global_i = r_start + i;
        let a_ptr = a_base.add(global_i * k);
        let c_ptr = c_base.add(global_i * n);

        for jt in 0..n_tiles_32 {
            let j = jt * 32;
            let b_tile = pb_base.add(jt * k * 32);
            microkernel_1x32_packed(
                a_ptr,
                b_tile,
                c_ptr.add(j),
                k,
                global_i,
                alpha_vec,
                beta,
                beta_vec,
                bias,
                act,
                zero_vec,
                six_vec,
                three_vec,
                inv_six_vec,
            );
        }

        for col in n_packed_32..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a_base.add(global_i * k + p) * *b_raw.add(p * n + col);
            }
            let mut val = alpha * sum;
            if let Some(b_slice) = bias {
                val += b_slice[global_i];
            }
            if beta != 0.0 {
                val += beta * *c_base.add(global_i * n + col);
            }
            *c_base.add(global_i * n + col) = apply_activation_scalar(val, act);
        }

        i += 1;
    }
}

#[inline(always)]
unsafe fn gemm_compute_col_panel(
    m: usize,
    n: usize,
    k: usize,
    a_base: *const f32,
    b_raw: *const f32,
    c_base: *mut f32,
    j_start: usize,
    j_end: usize,
    alpha: f32,
    beta: f32,
    alpha_vec: __m512,
    beta_vec: __m512,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m512,
    six_vec: __m512,
    three_vec: __m512,
    inv_six_vec: __m512,
) {
    let mut j = j_start;
    while j + 32 <= j_end {
        let mut i = 0;
        while i + 8 <= m {
            let a_ptr = a_base.add(i * k);
            let c_ptr = c_base.add(i * n + j);

            let mut c00 = _mm512_setzero_ps();
            let mut c01 = _mm512_setzero_ps();
            let mut c10 = _mm512_setzero_ps();
            let mut c11 = _mm512_setzero_ps();
            let mut c20 = _mm512_setzero_ps();
            let mut c21 = _mm512_setzero_ps();
            let mut c30 = _mm512_setzero_ps();
            let mut c31 = _mm512_setzero_ps();
            let mut c40 = _mm512_setzero_ps();
            let mut c41 = _mm512_setzero_ps();
            let mut c50 = _mm512_setzero_ps();
            let mut c51 = _mm512_setzero_ps();
            let mut c60 = _mm512_setzero_ps();
            let mut c61 = _mm512_setzero_ps();
            let mut c70 = _mm512_setzero_ps();
            let mut c71 = _mm512_setzero_ps();

            for p in 0..k {
                let b0 = _mm512_loadu_ps(b_raw.add(p * n + j));
                let b1 = _mm512_loadu_ps(b_raw.add(p * n + j + 16));

                let a0 = _mm512_set1_ps(*a_ptr.add(p));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);

                let a1 = _mm512_set1_ps(*a_ptr.add(k + p));
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c11 = _mm512_fmadd_ps(a1, b1, c11);

                let a2 = _mm512_set1_ps(*a_ptr.add(2 * k + p));
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c21 = _mm512_fmadd_ps(a2, b1, c21);

                let a3 = _mm512_set1_ps(*a_ptr.add(3 * k + p));
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c31 = _mm512_fmadd_ps(a3, b1, c31);

                let a4 = _mm512_set1_ps(*a_ptr.add(4 * k + p));
                c40 = _mm512_fmadd_ps(a4, b0, c40);
                c41 = _mm512_fmadd_ps(a4, b1, c41);

                let a5 = _mm512_set1_ps(*a_ptr.add(5 * k + p));
                c50 = _mm512_fmadd_ps(a5, b0, c50);
                c51 = _mm512_fmadd_ps(a5, b1, c51);

                let a6 = _mm512_set1_ps(*a_ptr.add(6 * k + p));
                c60 = _mm512_fmadd_ps(a6, b0, c60);
                c61 = _mm512_fmadd_ps(a6, b1, c61);

                let a7 = _mm512_set1_ps(*a_ptr.add(7 * k + p));
                c70 = _mm512_fmadd_ps(a7, b0, c70);
                c71 = _mm512_fmadd_ps(a7, b1, c71);
            }

            store_row_2x16(&mut c00, &mut c01, c_ptr, i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c10, &mut c11, c_ptr.add(n), i + 1, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c20, &mut c21, c_ptr.add(2 * n), i + 2, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c30, &mut c31, c_ptr.add(3 * n), i + 3, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c40, &mut c41, c_ptr.add(4 * n), i + 4, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c50, &mut c51, c_ptr.add(5 * n), i + 5, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c60, &mut c61, c_ptr.add(6 * n), i + 6, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            store_row_2x16(&mut c70, &mut c71, c_ptr.add(7 * n), i + 7, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);

            i += 8;
        }

        while i < m {
            let a_ptr = a_base.add(i * k);
            let c_ptr = c_base.add(i * n + j);

            let mut c00 = _mm512_setzero_ps();
            let mut c01 = _mm512_setzero_ps();

            for p in 0..k {
                let b0 = _mm512_loadu_ps(b_raw.add(p * n + j));
                let b1 = _mm512_loadu_ps(b_raw.add(p * n + j + 16));
                let a0 = _mm512_set1_ps(*a_ptr.add(p));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
            }

            store_row_2x16(&mut c00, &mut c01, c_ptr, i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
            i += 1;
        }

        j += 32;
    }

    while j + 16 <= j_end {
        for i in 0..m {
            let a_ptr = a_base.add(i * k);
            let c_ptr = c_base.add(i * n + j);

            let mut c00 = _mm512_setzero_ps();
            for p in 0..k {
                let b0 = _mm512_loadu_ps(b_raw.add(p * n + j));
                let a0 = _mm512_set1_ps(*a_ptr.add(p));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
            }

            store_row_1x16(&mut c00, c_ptr, i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
        }
        j += 16;
    }

    while j < j_end {
        for i in 0..m {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a_base.add(i * k + p) * *b_raw.add(p * n + j);
            }
            let mut val = alpha * sum;
            if let Some(b_slice) = bias {
                val += b_slice[i];
            }
            if beta != 0.0 {
                val += beta * *c_base.add(i * n + j);
            }
            *c_base.add(i * n + j) = apply_activation_scalar(val, act);
        }
        j += 1;
    }
}

/// Specialized AVX-512 vector-matrix multiply when M=1
#[inline(always)]
unsafe fn gemm_avx512_m1(
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
    let alpha_vec = _mm512_set1_ps(alpha);
    let beta_vec = _mm512_set1_ps(beta);
    let zero_vec = _mm512_setzero_ps();
    let six_vec = _mm512_set1_ps(6.0);
    let three_vec = _mm512_set1_ps(3.0);
    let inv_six_vec = _mm512_set1_ps(1.0 / 6.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let mut j = 0;
    while j + 32 <= n {
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for p in 0..k {
            let a_val = _mm512_set1_ps(*a_ptr.add(p));
            let b0 = _mm512_loadu_ps(b_ptr.add(p * n + j));
            let b1 = _mm512_loadu_ps(b_ptr.add(p * n + j + 16));
            acc0 = _mm512_fmadd_ps(a_val, b0, acc0);
            acc1 = _mm512_fmadd_ps(a_val, b1, acc1);
        }

        store_row_2x16(&mut acc0, &mut acc1, c_ptr.add(j), 0, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
        j += 32;
    }

    while j + 16 <= n {
        let mut acc0 = _mm512_setzero_ps();
        for p in 0..k {
            let a_val = _mm512_set1_ps(*a_ptr.add(p));
            let b0 = _mm512_loadu_ps(b_ptr.add(p * n + j));
            acc0 = _mm512_fmadd_ps(a_val, b0, acc0);
        }

        store_row_1x16(&mut acc0, c_ptr.add(j), 0, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
        j += 16;
    }

    while j < n {
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += *a_ptr.add(p) * *b_ptr.add(p * n + j);
        }
        let mut val = alpha * sum;
        if let Some(b_slice) = bias {
            val += b_slice[0];
        }
        if beta != 0.0 {
            val += beta * *c_ptr.add(j);
        }
        *c_ptr.add(j) = apply_activation_scalar(val, act);
        j += 1;
    }
}

/// AVX-512 vectorized GEMM implementation with cache paneling
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_avx512(
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

    if m == 1 {
        gemm_avx512_m1(n, k, a, b, c, alpha, beta, bias, act);
        return;
    }

    let alpha_vec = _mm512_set1_ps(alpha);
    let beta_vec = _mm512_set1_ps(beta);
    let zero_vec = _mm512_setzero_ps();
    let six_vec = _mm512_set1_ps(6.0);
    let three_vec = _mm512_set1_ps(3.0);
    let inv_six_vec = _mm512_set1_ps(1.0 / 6.0);

    let a_send = SendConstPtr(a.as_ptr());
    let b_send = SendConstPtr(b.as_ptr());
    let c_send = SendMutPtr(c.as_mut_ptr());

    #[cfg(feature = "rayon")]
    {
        if n >= 2048 || m < 16 {
            let col_chunk = 512;
            let num_col_chunks = n.div_ceil(col_chunk);

            (0..num_col_chunks).into_par_iter().for_each(move |col_block| {
                let j_start = col_block * col_chunk;
                let j_end = (j_start + col_chunk).min(n);
                gemm_compute_col_panel(
                    m, n, k, a_send.get(), b_send.get(), c_send.get(), j_start, j_end,
                    alpha, beta, alpha_vec, beta_vec, bias, act,
                    zero_vec, six_vec, three_vec, inv_six_vec,
                );
            });
            return;
        }

        let n_tiles_32 = n / 32;
        let mut packed_b = vec![0.0f32; n_tiles_32 * k * 32];
        let pb_ptr: *mut f32 = packed_b.as_mut_ptr();
        for jt in 0..n_tiles_32 {
            let j = jt * 32;
            let tile_dst = pb_ptr.add(jt * k * 32);
            for p in 0..k {
                let src = b.as_ptr().add(p * n + j);
                let dst = tile_dst.add(p * 32);
                _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
                _mm512_storeu_ps(dst.add(16), _mm512_loadu_ps(src.add(16)));
            }
        }
        let pb_send = SendConstPtr(packed_b.as_ptr());

        let num_threads = rayon::current_num_threads();
        let target_chunks = (num_threads * 4).max(1);
        let ideal_chunk = (m / target_chunks).max(8);
        let row_chunk = ideal_chunk.div_ceil(8) * 8;
        let num_row_chunks = m.div_ceil(row_chunk);

        (0..num_row_chunks).into_par_iter().for_each(move |row_block| {
            let r_start = row_block * row_chunk;
            let r_end = (r_start + row_chunk).min(m);
            let r_count = r_end - r_start;
            gemm_compute_row_block_packed(
                r_start, r_count, n, k, a_send.get(), pb_send.get(), b_send.get(), c_send.get(),
                alpha, beta, alpha_vec, beta_vec, bias, act,
                zero_vec, six_vec, three_vec, inv_six_vec,
            );
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        gemm_compute_col_panel(
            m, n, k, a_send.get(), b_send.get(), c_send.get(), 0, n,
            alpha, beta, alpha_vec, beta_vec, bias, act,
            zero_vec, six_vec, three_vec, inv_six_vec,
        );
    }
}
