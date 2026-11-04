use rusto_nn::kernels::activation::FusedActivation;
use rusto_nn_cuda::CudaBackend;

#[test]
fn test_cuda_backend_initialization_or_fallback() {
    match CudaBackend::init() {
        Ok(backend) => {
            println!("CUDA GPU detected: registering backend");
            backend.register();
            assert_eq!(rusto_nn::active_backend().unwrap().name(), "CUDA");

            let (m, n, k) = (16, 16, 16);
            let a = vec![1.0f32; m * k];
            let b = vec![2.0f32; k * n];
            let mut c = vec![0.0f32; m * n];

            rusto_nn::kernels::gemm::gemm(m, n, k, &a, &b, &mut c, 1.0, 0.0, None, FusedActivation::None);
            for &val in &c {
                assert!((val - 32.0).abs() < 1e-4);
            }

            rusto_nn::clear_backend();
            assert!(rusto_nn::active_backend().is_none());
        }
        Err(err) => {
            println!("No CUDA GPU available on host (expected on non-NVIDIA machines): {}", err);
            // Verify backend remains unset or falls back gracefully
            assert!(rusto_nn::active_backend().is_none());
        }
    }
}

