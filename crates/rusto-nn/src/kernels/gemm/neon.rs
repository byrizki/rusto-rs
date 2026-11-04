#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

/// Execute 4x16 micro-kernel on pointers using ARM NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn microkernel_4x16(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    j: usize,
    alpha_vec: float32x4_t,
    beta: f32,
    beta_vec: float32x4_t,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: float32x4_t,
    six_vec: float32x4_t,
    three_vec: float32x4_t,
    inv_six_vec: float32x4_t,
) {
    let mut c00 = vdupq_n_f32(0.0);
    let mut c01 = vdupq_n_f32(0.0);
    let mut c02 = vdupq_n_f32(0.0);
    let mut c03 = vdupq_n_f32(0.0);

    let mut c10 = vdupq_n_f32(0.0);
    let mut c11 = vdupq_n_f32(0.0);
    let mut c12 = vdupq_n_f32(0.0);
    let mut c13 = vdupq_n_f32(0.0);

    let mut c20 = vdupq_n_f32(0.0);
    let mut c21 = vdupq_n_f32(0.0);
    let mut c22 = vdupq_n_f32(0.0);
    let mut c23 = vdupq_n_f32(0.0);

    let mut c30 = vdupq_n_f32(0.0);
    let mut c31 = vdupq_n_f32(0.0);
    let mut c32 = vdupq_n_f32(0.0);
    let mut c33 = vdupq_n_f32(0.0);

    for p in 0..k {
        let b0 = vld1q_f32(b_ptr.add(p * n + j));
        let b1 = vld1q_f32(b_ptr.add(p * n + j + 4));
        let b2 = vld1q_f32(b_ptr.add(p * n + j + 8));
        let b3 = vld1q_f32(b_ptr.add(p * n + j + 12));

        let a0 = vdupq_n_f32(*a_ptr.add(p));
        c00 = vfmaq_f32(c00, b0, a0);
        c01 = vfmaq_f32(c01, b1, a0);
        c02 = vfmaq_f32(c02, b2, a0);
        c03 = vfmaq_f32(c03, b3, a0);

        let a1 = vdupq_n_f32(*a_ptr.add(k + p));
        c10 = vfmaq_f32(c10, b0, a1);
        c11 = vfmaq_f32(c11, b1, a1);
        c12 = vfmaq_f32(c12, b2, a1);
        c13 = vfmaq_f32(c13, b3, a1);

        let a2 = vdupq_n_f32(*a_ptr.add(2 * k + p));
        c20 = vfmaq_f32(c20, b0, a2);
        c21 = vfmaq_f32(c21, b1, a2);
        c22 = vfmaq_f32(c22, b2, a2);
        c23 = vfmaq_f32(c23, b3, a2);

        let a3 = vdupq_n_f32(*a_ptr.add(3 * k + p));
        c30 = vfmaq_f32(c30, b0, a3);
        c31 = vfmaq_f32(c31, b1, a3);
        c32 = vfmaq_f32(c32, b2, a3);
        c33 = vfmaq_f32(c33, b3, a3);
    }

    let mut rows = [
        [c00, c01, c02, c03],
        [c10, c11, c12, c13],
        [c20, c21, c22, c23],
        [c30, c31, c32, c33],
    ];

    for (r, row_accs) in rows.iter_mut().enumerate() {
        let bias_v = if let Some(b_slice) = bias {
            vdupq_n_f32(b_slice[global_i + r])
        } else {
            zero_vec
        };

        for (col_idx, acc) in row_accs.iter_mut().enumerate() {
            let mut val = vmulq_f32(*acc, alpha_vec);
            if bias.is_some() {
                val = vaddq_f32(val, bias_v);
            }
            if beta != 0.0 {
                let existing = vld1q_f32(c_ptr.add(r * n + j + col_idx * 4));
                val = vfmaq_f32(val, existing, beta_vec);
            }

            match act {
                FusedActivation::Relu => {
                    val = vmaxq_f32(zero_vec, val);
                }
                FusedActivation::Relu6 => {
                    val = vminq_f32(six_vec, vmaxq_f32(zero_vec, val));
                }
                FusedActivation::HardSwish => {
                    let val_p3 = vaddq_f32(val, three_vec);
                    let clamped = vmaxq_f32(zero_vec, vminq_f32(six_vec, val_p3));
                    val = vmulq_f32(val, vmulq_f32(clamped, inv_six_vec));
                }
                _ => {}
            }

            vst1q_f32(c_ptr.add(r * n + j + col_idx * 4), val);
        }
    }
}

/// Execute 1x16 micro-kernel on pointers using ARM NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn microkernel_1x16(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
    global_i: usize,
    j: usize,
    alpha_vec: float32x4_t,
    beta: f32,
    beta_vec: float32x4_t,
    bias: Option<&[f32]>,
    act: FusedActivation,
    zero_vec: float32x4_t,
    six_vec: float32x4_t,
    three_vec: float32x4_t,
    inv_six_vec: float32x4_t,
) {
    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);

    for p in 0..k {
        let b0 = vld1q_f32(b_ptr.add(p * n + j));
        let b1 = vld1q_f32(b_ptr.add(p * n + j + 4));
        let b2 = vld1q_f32(b_ptr.add(p * n + j + 8));
        let b3 = vld1q_f32(b_ptr.add(p * n + j + 12));

        let a0 = vdupq_n_f32(*a_ptr.add(p));
        c0 = vfmaq_f32(c0, b0, a0);
        c1 = vfmaq_f32(c1, b1, a0);
        c2 = vfmaq_f32(c2, b2, a0);
        c3 = vfmaq_f32(c3, b3, a0);
    }

    let bias_v = if let Some(b_slice) = bias {
        vdupq_n_f32(b_slice[global_i])
    } else {
        zero_vec
    };

    let mut row_accs = [c0, c1, c2, c3];
    for (col_idx, acc) in row_accs.iter_mut().enumerate() {
        let mut val = vmulq_f32(*acc, alpha_vec);
        if bias.is_some() {
            val = vaddq_f32(val, bias_v);
        }
        if beta != 0.0 {
            let existing = vld1q_f32(c_ptr.add(j + col_idx * 4));
            val = vfmaq_f32(val, existing, beta_vec);
        }

        match act {
            FusedActivation::Relu => {
                val = vmaxq_f32(zero_vec, val);
            }
            FusedActivation::Relu6 => {
                val = vminq_f32(six_vec, vmaxq_f32(zero_vec, val));
            }
            FusedActivation::HardSwish => {
                let val_p3 = vaddq_f32(val, three_vec);
                let clamped = vmaxq_f32(zero_vec, vminq_f32(six_vec, val_p3));
                val = vmulq_f32(val, vmulq_f32(clamped, inv_six_vec));
            }
            _ => {}
        }

        vst1q_f32(c_ptr.add(j + col_idx * 4), val);
    }
}

/// NEON vectorized GEMM implementation with cache paneling
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_neon(
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
    let alpha_vec = vdupq_n_f32(alpha);
    let beta_vec = vdupq_n_f32(beta);
    let zero_vec = vdupq_n_f32(0.0);
    let six_vec = vdupq_n_f32(6.0);
    let three_vec = vdupq_n_f32(3.0);
    let inv_six_vec = vdupq_n_f32(1.0 / 6.0);

    let m_blocks = m / 4;
    let full_rows = m_blocks * 4;

    let nc_tiles = 16;
    let n_panels = n.div_ceil(nc_tiles * 16);

    let compute_panel = |i_block: usize, j_panel: usize, c_out: &mut [f32]| {
        let i = i_block * 4;
        let j_start = j_panel * (nc_tiles * 16);
        let j_end = (j_start + nc_tiles * 16).min(n);

        let a_ptr = a.as_ptr().add(i * k);
        let b_ptr = b.as_ptr();
        let c_ptr = c_out.as_mut_ptr();

        let mut j = j_start;
        while j + 16 <= j_end {
            microkernel_4x16(
                a_ptr,
                b_ptr,
                c_ptr,
                k,
                n,
                i,
                j,
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
            j += 16;
        }

        // Remainder columns in panel
        while j < j_end {
            for r in 0..4 {
                let global_r = i + r;
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a_ptr.add(r * k + p) * *b_ptr.add(p * n + j);
                }
                let mut val = alpha * sum;
                if let Some(b_slice) = bias {
                    val += b_slice[global_r];
                }
                if beta != 0.0 {
                    val += beta * *c_ptr.add(r * n + j);
                }
                *c_ptr.add(r * n + j) = apply_activation_scalar(val, act);
            }
            j += 1;
        }
    };

    #[cfg(feature = "rayon")]
    {
        if m * n > 4096 {
            let (c_full, _) = c.split_at_mut(full_rows * n);
            c_full
                .par_chunks_exact_mut(4 * n)
                .enumerate()
                .for_each(|(i_block, c_row)| {
                    for j_panel in 0..n_panels {
                        compute_panel(i_block, j_panel, c_row);
                    }
                });
        } else {
            for i_block in 0..m_blocks {
                let offset = i_block * 4 * n;
                for j_panel in 0..n_panels {
                    compute_panel(i_block, j_panel, &mut c[offset..offset + 4 * n]);
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for i_block in 0..m_blocks {
            let offset = i_block * 4 * n;
            for j_panel in 0..n_panels {
                compute_panel(i_block, j_panel, &mut c[offset..offset + 4 * n]);
            }
        }
    }

    // Remainder rows (M % 4 != 0)
    for i in full_rows..m {
        let a_ptr = a.as_ptr().add(i * k);
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr().add(i * n);

        let mut j = 0;
        while j + 16 <= n {
            microkernel_1x16(
                a_ptr,
                b_ptr,
                c_ptr,
                k,
                n,
                i,
                j,
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
            j += 16;
        }

        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a_ptr.add(p) * *b_ptr.add(p * n + j);
            }
            let mut val = alpha * sum;
            if let Some(b_slice) = bias {
                val += b_slice[i];
            }
            if beta != 0.0 {
                val += beta * *c_ptr.add(j);
            }
            *c_ptr.add(j) = apply_activation_scalar(val, act);
            j += 1;
        }
    }
}
