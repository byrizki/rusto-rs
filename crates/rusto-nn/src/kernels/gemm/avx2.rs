#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

#[cfg(feature = "rayon")]
thread_local! {
    static PACKED_B_CACHE: core::cell::RefCell<Vec<f32>> = core::cell::RefCell::new(Vec::new());
}

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
unsafe fn store_row_2x8(
    r0: &mut __m256,
    r1: &mut __m256,
    c_row_ptr: *mut f32,
    row_idx: usize,
    alpha_vec: __m256,
    beta: f32,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    *r0 = _mm256_mul_ps(*r0, alpha_vec);
    *r1 = _mm256_mul_ps(*r1, alpha_vec);

    if beta != 0.0 {
        let prev0 = _mm256_loadu_ps(c_row_ptr);
        let prev1 = _mm256_loadu_ps(c_row_ptr.add(8));
        *r0 = _mm256_fmadd_ps(prev0, beta_vec, *r0);
        *r1 = _mm256_fmadd_ps(prev1, beta_vec, *r1);
    }

    if let Some(bias_slice) = bias {
        let bias_val = _mm256_set1_ps(bias_slice[row_idx]);
        *r0 = _mm256_add_ps(*r0, bias_val);
        *r1 = _mm256_add_ps(*r1, bias_val);
    }

    match act {
        FusedActivation::Relu => {
            *r0 = _mm256_max_ps(zero_vec, *r0);
            *r1 = _mm256_max_ps(zero_vec, *r1);
        }
        FusedActivation::Relu6 => {
            *r0 = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, *r0));
            *r1 = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, *r1));
        }
        FusedActivation::HardSwish => {
            let p0 = _mm256_add_ps(*r0, three_vec);
            let cl0 = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, p0));
            *r0 = _mm256_mul_ps(*r0, _mm256_mul_ps(cl0, inv_six_vec));

            let p1 = _mm256_add_ps(*r1, three_vec);
            let cl1 = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, p1));
            *r1 = _mm256_mul_ps(*r1, _mm256_mul_ps(cl1, inv_six_vec));
        }
        _ => {}
    }

    _mm256_storeu_ps(c_row_ptr, *r0);
    _mm256_storeu_ps(c_row_ptr.add(8), *r1);
}

#[inline(always)]
unsafe fn microkernel_6x16_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    alpha_vec: __m256,
    beta: f32,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();
    let mut c40 = _mm256_setzero_ps();
    let mut c41 = _mm256_setzero_ps();
    let mut c50 = _mm256_setzero_ps();
    let mut c51 = _mm256_setzero_ps();

    for p in 0..k {
        let b0 = _mm256_loadu_ps(b_packed.add(p * 16));
        let b1 = _mm256_loadu_ps(b_packed.add(p * 16 + 8));

        let a0 = _mm256_set1_ps(*a_ptr.add(p));
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        let a1 = _mm256_set1_ps(*a_ptr.add(k + p));
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        let a2 = _mm256_set1_ps(*a_ptr.add(2 * k + p));
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        let a3 = _mm256_set1_ps(*a_ptr.add(3 * k + p));
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        let a4 = _mm256_set1_ps(*a_ptr.add(4 * k + p));
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        let a5 = _mm256_set1_ps(*a_ptr.add(5 * k + p));
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    store_row_2x8(&mut c00, &mut c01, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c10, &mut c11, c_ptr.add(n), global_i + 1, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c20, &mut c21, c_ptr.add(2 * n), global_i + 2, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c30, &mut c31, c_ptr.add(3 * n), global_i + 3, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c40, &mut c41, c_ptr.add(4 * n), global_i + 4, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c50, &mut c51, c_ptr.add(5 * n), global_i + 5, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn microkernel_4x16_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    alpha_vec: __m256,
    beta: f32,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();

    for p in 0..k {
        let b0 = _mm256_loadu_ps(b_packed.add(p * 16));
        let b1 = _mm256_loadu_ps(b_packed.add(p * 16 + 8));

        let a0 = _mm256_set1_ps(*a_ptr.add(p));
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        let a1 = _mm256_set1_ps(*a_ptr.add(k + p));
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        let a2 = _mm256_set1_ps(*a_ptr.add(2 * k + p));
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        let a3 = _mm256_set1_ps(*a_ptr.add(3 * k + p));
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }

    store_row_2x8(&mut c00, &mut c01, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c10, &mut c11, c_ptr.add(n), global_i + 1, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c20, &mut c21, c_ptr.add(2 * n), global_i + 2, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c30, &mut c31, c_ptr.add(3 * n), global_i + 3, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn microkernel_2x16_packed(
    a_ptr: *const f32,
    b_packed: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    alpha_vec: __m256,
    beta: f32,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();

    for p in 0..k {
        let b0 = _mm256_loadu_ps(b_packed.add(p * 16));
        let b1 = _mm256_loadu_ps(b_packed.add(p * 16 + 8));

        let a0 = _mm256_set1_ps(*a_ptr.add(p));
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        let a1 = _mm256_set1_ps(*a_ptr.add(k + p));
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
    }

    store_row_2x8(&mut c00, &mut c01, c_ptr, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
    store_row_2x8(&mut c10, &mut c11, c_ptr.add(n), global_i + 1, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn microkernel_1x16_packed(
    a_row: *const f32,
    b_packed: *const f32,
    c_out: *mut f32,
    k: usize,
    global_i: usize,
    alpha_vec: __m256,
    beta: f32,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    let mut c0 = _mm256_setzero_ps();
    let mut c1 = _mm256_setzero_ps();

    for p in 0..k {
        let b0 = _mm256_loadu_ps(b_packed.add(p * 16));
        let b1 = _mm256_loadu_ps(b_packed.add(p * 16 + 8));

        let a_val = _mm256_set1_ps(*a_row.add(p));
        c0 = _mm256_fmadd_ps(a_val, b0, c0);
        c1 = _mm256_fmadd_ps(a_val, b1, c1);
    }

    store_row_2x8(&mut c0, &mut c1, c_out, global_i, alpha_vec, beta, beta_vec, bias, act, zero_vec, six_vec, three_vec, inv_six_vec);
}

#[inline(always)]
unsafe fn gemm_avx2_m1_chunk(
    j_start: usize,
    j_end: usize,
    n: usize,
    k: usize,
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    alpha: f32,
    beta: f32,
    alpha_vec: __m256,
    beta_vec: __m256,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
    bias_val: f32,
    bias_vec: __m256,
    act: FusedActivation,
) {
    let chunk_len = j_end - j_start;
    let v_chunks = chunk_len / 8;

    for vi in 0..v_chunks {
        let j = j_start + vi * 8;
        let mut acc = _mm256_setzero_ps();

        for p in 0..k {
            let a_val = _mm256_set1_ps(*a_ptr.add(p));
            let b_val = _mm256_loadu_ps(b_ptr.add(p * n + j));
            acc = _mm256_fmadd_ps(a_val, b_val, acc);
        }

        acc = _mm256_mul_ps(acc, alpha_vec);
        if beta != 0.0 {
            let prev = _mm256_loadu_ps(c_ptr.add(j));
            acc = _mm256_fmadd_ps(prev, beta_vec, acc);
        }
        acc = _mm256_add_ps(acc, bias_vec);

        match act {
            FusedActivation::Relu => {
                acc = _mm256_max_ps(zero_vec, acc);
            }
            FusedActivation::Relu6 => {
                acc = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, acc));
            }
            FusedActivation::HardSwish => {
                let p_term = _mm256_add_ps(acc, three_vec);
                let cl = _mm256_min_ps(six_vec, _mm256_max_ps(zero_vec, p_term));
                acc = _mm256_mul_ps(acc, _mm256_mul_ps(cl, inv_six_vec));
            }
            _ => {}
        }

        _mm256_storeu_ps(c_ptr.add(j), acc);
    }

    let v_done = v_chunks * 8;
    for j in (j_start + v_done)..j_end {
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += *a_ptr.add(p) * *b_ptr.add(p * n + j);
        }
        let val = if beta == 0.0 {
            alpha * sum + bias_val
        } else {
            alpha * sum + beta * *c_ptr.add(j) + bias_val
        };
        *c_ptr.add(j) = apply_activation_scalar(val, act);
    }
}

#[inline(always)]
unsafe fn gemm_avx2_m1(
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
    let alpha_vec = _mm256_set1_ps(alpha);
    let beta_vec = _mm256_set1_ps(beta);
    let zero_vec = _mm256_setzero_ps();
    let six_vec = _mm256_set1_ps(6.0);
    let three_vec = _mm256_set1_ps(3.0);
    let inv_six_vec = _mm256_set1_ps(1.0 / 6.0);
    let bias_val = bias.map(|b| b[0]).unwrap_or(0.0);
    let bias_vec = _mm256_set1_ps(bias_val);

    #[cfg(feature = "rayon")]
    {
        if n >= 1024 {
            let chunk_n = 1024;
            let num_chunks = n.div_ceil(chunk_n);
            let a_send = SendConstPtr(a.as_ptr());
            let b_send = SendConstPtr(b.as_ptr());
            let c_send = SendMutPtr(c.as_mut_ptr());

            (0..num_chunks).into_par_iter().for_each(move |chunk_idx| {
                let j_start = chunk_idx * chunk_n;
                let j_end = (j_start + chunk_n).min(n);
                gemm_avx2_m1_chunk(
                    j_start, j_end, n, k, a_send.get(), b_send.get(), c_send.get(),
                    alpha, beta, alpha_vec, beta_vec, zero_vec, six_vec, three_vec, inv_six_vec,
                    bias_val, bias_vec, act,
                );
            });
            return;
        }
    }

    gemm_avx2_m1_chunk(
        0, n, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
        alpha, beta, alpha_vec, beta_vec, zero_vec, six_vec, three_vec, inv_six_vec,
        bias_val, bias_vec, act,
    );
}

#[inline(always)]
unsafe fn gemm_compute_col_panel(
    m: usize,
    n: usize,
    k: usize,
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    j_start: usize,
    j_end: usize,
    alpha: f32,
    beta: f32,
    alpha_vec: __m256,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    let cur_n = j_end - j_start;
    let n_tiles = cur_n / 16;

    let mut stack_buf = core::mem::MaybeUninit::<[f32; 8192]>::uninit();
    let mut heap_buf;
    let packed_b_ptr = if k * 16 <= 8192 {
        stack_buf.as_mut_ptr() as *mut f32
    } else {
        heap_buf = vec![0.0f32; k * 16];
        heap_buf.as_mut_ptr()
    };

    for jt in 0..n_tiles {
        let j = j_start + jt * 16;

        for p in 0..k {
            let src = b_ptr.add(p * n + j);
            let dst = packed_b_ptr.add(p * 16);
            _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
            _mm256_storeu_ps(dst.add(8), _mm256_loadu_ps(src.add(8)));
        }

        let mut row_idx = 0;
        let m6_tiles = m / 6;
        for _ in 0..m6_tiles {
            microkernel_6x16_packed(
                a_ptr.add(row_idx * k),
                packed_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                n,
                row_idx,
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
            row_idx += 6;
        }

        let rem_m = m - row_idx;
        if rem_m >= 4 {
            microkernel_4x16_packed(
                a_ptr.add(row_idx * k),
                packed_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                n,
                row_idx,
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
            row_idx += 4;
        }

        let rem_m = m - row_idx;
        if rem_m >= 2 {
            microkernel_2x16_packed(
                a_ptr.add(row_idx * k),
                packed_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                n,
                row_idx,
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
            row_idx += 2;
        }

        let rem_m = m - row_idx;
        if rem_m == 1 {
            microkernel_1x16_packed(
                a_ptr.add(row_idx * k),
                packed_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                row_idx,
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
    }

    let n_done = n_tiles * 16;
    if n_done < cur_n {
        for local_i in 0..m {
            let a_row = std::slice::from_raw_parts(a_ptr.add(local_i * k), k);
            let c_row = std::slice::from_raw_parts_mut(c_ptr.add(local_i * n), n);
            let bias_val = bias.map(|b| b[local_i]).unwrap_or(0.0);

            for j in (j_start + n_done)..j_end {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * *b_ptr.add(p * n + j);
                }
                let val = if beta == 0.0 {
                    alpha * sum + bias_val
                } else {
                    alpha * sum + beta * c_row[j] + bias_val
                };
                c_row[j] = apply_activation_scalar(val, act);
            }
        }
    }
}

#[inline(always)]
unsafe fn gemm_compute_row_block_packed(
    r_start: usize,
    r_count: usize,
    n: usize,
    k: usize,
    a_ptr: *const f32,
    packed_b_ptr: *const f32,
    b_raw_ptr: *const f32,
    c_ptr: *mut f32,
    alpha: f32,
    beta: f32,
    alpha_vec: __m256,
    beta_vec: __m256,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: __m256,
    six_vec: __m256,
    three_vec: __m256,
    inv_six_vec: __m256,
) {
    let n_tiles = n / 16;

    for jt in 0..n_tiles {
        let j = jt * 16;
        let tile_b_ptr = packed_b_ptr.add(jt * k * 16);

        let mut local_r = 0;
        let m6_tiles = r_count / 6;
        for _ in 0..m6_tiles {
            let row_idx = r_start + local_r;
            microkernel_6x16_packed(
                a_ptr.add(row_idx * k),
                tile_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                n,
                row_idx,
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
            local_r += 6;
        }

        let rem = r_count - local_r;
        if rem >= 4 {
            let row_idx = r_start + local_r;
            microkernel_4x16_packed(
                a_ptr.add(row_idx * k),
                tile_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                n,
                row_idx,
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
            local_r += 4;
        }

        let rem = r_count - local_r;
        if rem >= 2 {
            let row_idx = r_start + local_r;
            microkernel_2x16_packed(
                a_ptr.add(row_idx * k),
                tile_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                n,
                row_idx,
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
            local_r += 2;
        }

        let rem = r_count - local_r;
        if rem == 1 {
            let row_idx = r_start + local_r;
            microkernel_1x16_packed(
                a_ptr.add(row_idx * k),
                tile_b_ptr,
                c_ptr.add(row_idx * n + j),
                k,
                row_idx,
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
    }

    let n_done = n_tiles * 16;
    if n_done < n {
        for local_i in 0..r_count {
            let row_idx = r_start + local_i;
            let a_row = std::slice::from_raw_parts(a_ptr.add(row_idx * k), k);
            let c_row = std::slice::from_raw_parts_mut(c_ptr.add(row_idx * n), n);
            let bias_val = bias.map(|b| b[row_idx]).unwrap_or(0.0);

            for j in n_done..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * *b_raw_ptr.add(p * n + j);
                }
                let val = if beta == 0.0 {
                    alpha * sum + bias_val
                } else {
                    alpha * sum + beta * c_row[j] + bias_val
                };
                c_row[j] = apply_activation_scalar(val, act);
            }
        }
    }
}

/// Dedicated AVX2 + FMA register-blocked GEMM kernel with cache blocking and Rayon multi-threading
///
/// # Safety
/// Caller must ensure that the CPU supports AVX2 and FMA instructions, and slices have sufficient capacity.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_avx2(
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
        gemm_avx2_m1(n, k, a, b, c, alpha, beta, bias, act);
        return;
    }

    let alpha_vec = _mm256_set1_ps(alpha);
    let beta_vec = _mm256_set1_ps(beta);
    let zero_vec = _mm256_setzero_ps();
    let six_vec = _mm256_set1_ps(6.0);
    let three_vec = _mm256_set1_ps(3.0);
    let inv_six_vec = _mm256_set1_ps(1.0 / 6.0);

    let a_send = SendConstPtr(a.as_ptr());
    let b_send = SendConstPtr(b.as_ptr());
    let c_send = SendMutPtr(c.as_mut_ptr());

    #[cfg(feature = "rayon")]
    {
        if n >= 2048 || m < 12 {
            let col_chunk = 256;
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

        let n_tiles = n / 16;
        let packed_b_len = n_tiles * k * 16;

        PACKED_B_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if cache.len() < packed_b_len {
                cache.resize(packed_b_len, 0.0);
            }
            let pb_ptr: *mut f32 = cache.as_mut_ptr();
            for jt in 0..n_tiles {
                let j = jt * 16;
                let tile_dst = pb_ptr.add(jt * k * 16);
                for p in 0..k {
                    let src = b.as_ptr().add(p * n + j);
                    let dst = tile_dst.add(p * 16);
                    _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
                    _mm256_storeu_ps(dst.add(8), _mm256_loadu_ps(src.add(8)));
                }
            }
            let pb_send = SendConstPtr(cache.as_ptr());

            let num_threads = rayon::current_num_threads();
            let target_chunks = (num_threads * 4).max(1);
            let ideal_chunk = (m / target_chunks).max(6);
            let row_chunk = ideal_chunk.div_ceil(6) * 6;
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
