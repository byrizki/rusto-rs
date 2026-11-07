pub mod ffi;
pub mod session;

pub use session::{MnnCppError, MnnCppSession};

fn loader(
    path: &std::path::Path,
    cfg: &rusto_core::types::EngineConfig,
) -> Result<Box<dyn rusto_core::engine::InferenceSession>, rusto_core::error::CoreError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("mnn") {
        let session = MnnCppSession::from_file(path, cfg.intra_op_num_threads as usize)
            .map_err(|e| rusto_core::error::CoreError::Model(e.to_string()))?;
        Ok(Box::new(session))
    } else {
        Err(rusto_core::error::CoreError::Model("UNSUPPORTED_FORMAT".into()))
    }
}

/// Register the Alibaba MNN C++ backend with `rusto-core`'s engine session loader.
pub fn register() {
    rusto_core::engine::set_active_backend(rusto_core::engine::EngineBackend::MnnCpp);
    rusto_core::engine::register_session_loader(loader);
}

