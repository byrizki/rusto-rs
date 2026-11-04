use std::sync::{Arc, RwLock};

use crate::error::MnnError;
use crate::kernels::activation::FusedActivation;

/// Hardware acceleration backend interface (WebGPU, Vulkan, Metal, NPU, etc.)
pub trait DeviceBackend: Send + Sync {
    /// Descriptive name of the backend (e.g. "WebGPU", "Metal")
    fn name(&self) -> &'static str;

    /// Whether this backend is active and ready to accept dispatches
    fn is_available(&self) -> bool {
        true
    }

    /// Dispatch general matrix multiplication: C = alpha * (A @ B) + beta * C + bias
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), MnnError>;
}

static ACTIVE_BACKEND: RwLock<Option<Arc<dyn DeviceBackend>>> = RwLock::new(None);

/// Register or replace the globally active hardware accelerator backend
pub fn register_backend(backend: Arc<dyn DeviceBackend>) {
    let mut lock = ACTIVE_BACKEND.write().unwrap();
    *lock = Some(backend);
}

/// Remove the globally active hardware accelerator backend (falls back to pure CPU)
pub fn clear_backend() {
    let mut lock = ACTIVE_BACKEND.write().unwrap();
    *lock = None;
}

/// Retrieve the currently active hardware accelerator backend, if any
pub fn active_backend() -> Option<Arc<dyn DeviceBackend>> {
    let lock = ACTIVE_BACKEND.read().unwrap();
    lock.as_ref().cloned()
}

