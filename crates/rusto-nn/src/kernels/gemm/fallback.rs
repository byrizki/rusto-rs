use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

/// Portable cache-blocked GEMM fallback: C = alpha * A * B + beta * C + bias
pub fn gemm_scalar(
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
    let mc = 64;
    let nc = 128;
    let kc = 128;

    for i_block in (0..m).step_by(mc) {
        let i_end = (i_block + mc).min(m);
        for j_block in (0..n).step_by(nc) {
            let j_end = (j_block + nc).min(n);

            for i in i_block..i_end {
                let a_row = &a[i * k..];
                let c_row = &mut c[i * n..];
                let bias_val = bias.map(|b| b[i]).unwrap_or(0.0);

                for j in j_block..j_end {
                    let mut sum = 0.0f32;
                    for p_block in (0..k).step_by(kc) {
                        let p_end = (p_block + kc).min(k);
                        for p in p_block..p_end {
                            sum += a_row[p] * b[p * n + j];
                        }
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
}

