pub mod fallback;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

use crate::kernels::activation::FusedActivation;

/// Generic high-performance GEMM entry point: C = alpha * A * B + beta * C + bias
pub fn gemm(
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
    assert!(a.len() >= m * k, "Matrix A buffer too small");
    assert!(b.len() >= k * n, "Matrix B buffer too small");
    assert!(c.len() >= m * n, "Matrix C buffer too small");
    if let Some(b_slice) = bias {
        assert!(b_slice.len() >= m, "Bias vector length must match rows M");
    }

    // Check pluggable hardware accelerator backend (e.g. rusto-nn-webgpu)
    if let Some(backend) = crate::backend::active_backend() {
        if backend.is_available() && m * n * k >= 65536 {
            if backend.gemm(m, n, k, a, b, c, alpha, beta, bias, act).is_ok() {
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                avx512::gemm_avx512(m, n, k, a, b, c, alpha, beta, bias, act);
                return;
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                avx2::gemm_avx2(m, n, k, a, b, c, alpha, beta, bias, act);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::gemm_neon(m, n, k, a, b, c, alpha, beta, bias, act);
            return;
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            wasm::gemm_wasm(m, n, k, a, b, c, alpha, beta, bias, act);
            return;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    fallback::gemm_scalar(m, n, k, a, b, c, alpha, beta, bias, act);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_parity_and_activations() {
        let (m, n, k) = (13, 27, 35);
        let mut a = vec![0.0f32; m * k];
        for i in 0..a.len() {
            a[i] = ((i % 17) as f32 - 8.0) * 0.1;
        }

        let mut b = vec![0.0f32; k * n];
        for i in 0..b.len() {
            b[i] = ((i % 13) as f32 - 6.0) * 0.1;
        }

        let mut bias = vec![0.0f32; m];
        for i in 0..m {
            bias[i] = (i as f32 - 5.0) * 0.2;
        }

        for act in [
            FusedActivation::None,
            FusedActivation::Relu,
            FusedActivation::Relu6,
            FusedActivation::HardSwish,
        ] {
            let mut c_scalar = vec![0.0f32; m * n];
            fallback::gemm_scalar(m, n, k, &a, &b, &mut c_scalar, 1.0, 0.0, Some(&bias), act);

            let mut c_dispatched = vec![0.0f32; m * n];
            gemm(m, n, k, &a, &b, &mut c_dispatched, 1.0, 0.0, Some(&bias), act);

            for i in 0..m * n {
                let diff = (c_scalar[i] - c_dispatched[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at idx {}: scalar={}, dispatched={}, diff={}",
                    i, c_scalar[i], c_dispatched[i], diff
                );
            }
        }
    }

    #[test]
    fn test_gemm_shapes_and_threading() {
        for (m, n, k) in [(1, 2055, 32), (4, 100, 16), (24, 512, 48)] {
            let mut a = vec![0.0f32; m * k];
            for i in 0..a.len() {
                a[i] = ((i % 19) as f32 - 9.0) * 0.05;
            }

            let mut b = vec![0.0f32; k * n];
            for i in 0..b.len() {
                b[i] = ((i % 23) as f32 - 11.0) * 0.05;
            }

            let mut bias = vec![0.0f32; m];
            for i in 0..m {
                bias[i] = (i as f32 - 3.0) * 0.1;
            }

            let mut c_scalar = vec![0.0f32; m * n];
            fallback::gemm_scalar(m, n, k, &a, &b, &mut c_scalar, 1.0, 0.0, Some(&bias), FusedActivation::Relu);

            let mut c_dispatched = vec![0.0f32; m * n];
            gemm(m, n, k, &a, &b, &mut c_dispatched, 1.0, 0.0, Some(&bias), FusedActivation::Relu);

            for i in 0..m * n {
                let diff = (c_scalar[i] - c_dispatched[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Shape ({}, {}, {}): Mismatch at idx {}: scalar={}, dispatched={}, diff={}",
                    m, n, k, i, c_scalar[i], c_dispatched[i], diff
                );
            }

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let mut c_avx2 = vec![0.0f32; m * n];
                    unsafe {
                        avx2::gemm_avx2(m, n, k, &a, &b, &mut c_avx2, 1.0, 0.0, Some(&bias), FusedActivation::Relu);
                    }
                    for i in 0..m * n {
                        let diff = (c_scalar[i] - c_avx2[i]).abs();
                        assert!(
                            diff < 1e-4,
                            "Shape ({}, {}, {}): Mismatch at idx {}: scalar={}, avx2={}, diff={}",
                            m, n, k, i, c_scalar[i], c_avx2[i], diff
                        );
                    }
                }
            }
        }
    }
}

