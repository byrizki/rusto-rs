use rusto_nn::kernels::activation::FusedActivation;
use rusto_nn_webgpu::WebGpuBackend;

#[test]
fn test_webgpu_backend_registration() {
    if let Ok(backend) = WebGpuBackend::init() {
        backend.register();
        let active = rusto_nn::active_backend();
        assert!(active.is_some());
        assert_eq!(active.unwrap().name(), "WebGPU");

        // Test simple GEMM
        let (m, n, k) = (16, 16, 16);
        let a = vec![1.0f32; m * k];
        let b = vec![2.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        rusto_nn::kernels::gemm::gemm(m, n, k, &a, &b, &mut c, 1.0, 0.0, None, FusedActivation::None);
        for &val in &c {
            // 16 * 1.0 * 2.0 = 32.0
            assert!((val - 32.0).abs() < 1e-4);
        }

        rusto_nn::clear_backend();
        assert!(rusto_nn::active_backend().is_none());
    }
}

