use std::sync::Arc;

use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rusto_nn::kernels::activation::{apply_activation_scalar, FusedActivation};
use rusto_nn::{DeviceBackend, MnnError};

/// Pluggable NVIDIA CUDA hardware accelerator backend for rusto-nn
pub struct CudaBackend {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub cublas: Arc<CudaBlas>,
}

/// Query whether NVIDIA CUDA driver shared library is present on system
pub fn is_cuda_driver_available() -> bool {
    let candidates = [
        "libcuda.so.1",
        "libcuda.so",
        "nvcuda.dll",
        "libcuda.dylib",
    ];
    for name in &candidates {
        if unsafe { libloading::Library::new(*name) }.is_ok() {
            return true;
        }
    }
    false
}

impl CudaBackend {
    /// Initialize CUDA backend on default device ordinal (0)
    pub fn init() -> Result<Self, MnnError> {
        Self::with_device(0)
    }

    /// Initialize CUDA backend on specified device ordinal
    pub fn with_device(ordinal: usize) -> Result<Self, MnnError> {
        if !is_cuda_driver_available() {
            return Err(MnnError::BackendError(
                "NVIDIA CUDA driver shared library (libcuda.so / nvcuda.dll) not found on system".into(),
            ));
        }

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let context = CudaContext::new(ordinal)
                .map_err(|e| MnnError::BackendError(format!("Failed to initialize CUDA context on device {}: {:?}", ordinal, e)))?;
            let stream = context.default_stream();
            let cublas = CudaBlas::new(stream.clone())
                .map_err(|e| MnnError::BackendError(format!("Failed to initialize cuBLAS on device {}: {:?}", ordinal, e)))?;

            Ok(Self {
                context,
                stream,
                cublas: Arc::new(cublas),
            })
        }));

        match res {
            Ok(inner_res) => inner_res,
            Err(_) => Err(MnnError::BackendError(
                "CUDA initialization panicked (driver mismatch or device unavailable)".into(),
            )),
        }
    }

    /// Register this backend globally into rusto-nn
    pub fn register(self) {
        rusto_nn::register_backend(Arc::new(self));
    }
}

impl DeviceBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn gemm(
        &self,
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
    ) -> Result<(), MnnError> {
        let a_dev: CudaSlice<f32> = self.stream.clone_htod(a)
            .map_err(|e| MnnError::BackendError(format!("CUDA htod copy A failed: {:?}", e)))?;
        let b_dev: CudaSlice<f32> = self.stream.clone_htod(b)
            .map_err(|e| MnnError::BackendError(format!("CUDA htod copy B failed: {:?}", e)))?;
        let mut c_dev: CudaSlice<f32> = if beta != 0.0 {
            self.stream.clone_htod(c)
                .map_err(|e| MnnError::BackendError(format!("CUDA htod copy C failed: {:?}", e)))?
        } else {
            self.stream.alloc_zeros::<f32>(m * n)
                .map_err(|e| MnnError::BackendError(format!("CUDA alloc C failed: {:?}", e)))?
        };

        // cuBLAS column-major multiplication for row-major matrices C = A * B:
        // C_row = A_row * B_row <=> C_col^T = B_col^T * A_col^T
        // Dim: (N, M) = (N, K) * (K, M)
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: n as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };

        unsafe {
            self.cublas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)
                .map_err(|e| MnnError::BackendError(format!("cuBLAS gemm failed: {:?}", e)))?;
        }

        self.stream.memcpy_dtoh(&c_dev, c)
            .map_err(|e| MnnError::BackendError(format!("CUDA dtoh copy C failed: {:?}", e)))?;

        // Fused bias and activation
        if let Some(bs) = bias {
            for i in 0..m {
                let b_val = bs[i];
                let row = &mut c[i * n..(i + 1) * n];
                for val in row.iter_mut() {
                    *val = apply_activation_scalar(*val + b_val, act);
                }
            }
        } else if act != FusedActivation::None {
            for val in c.iter_mut() {
                *val = apply_activation_scalar(*val, act);
            }
        }

        Ok(())
    }
}

/// Helper to initialize CUDA and register it into rusto-nn in a single line
pub fn init_and_register() -> Result<(), MnnError> {
    let backend = CudaBackend::init()?;
    backend.register();
    Ok(())
}

/// Helper to clear the active hardware accelerator
pub fn clear() {
    rusto_nn::clear_backend();
}
