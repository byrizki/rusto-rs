use thiserror::Error;

#[derive(Error, Debug)]
pub enum MnnError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid MNN model: {0}")]
    InvalidModel(String),

    #[error("Unsupported operator: {0}")]
    UnsupportedOp(String),

    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Backend error: {0}")]
    BackendError(String),
}

