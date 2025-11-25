use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum MNNError {
    #[error("Failed to create interpreter: {0}")]
    InterpreterCreation(String),
    
    #[error("Failed to create session: {0}")]
    SessionCreation(String),
    
    #[error("Failed to get tensor: {0}")]
    TensorAccess(String),
    
    #[error("Invalid input shape: {0}")]
    InvalidShape(String),
    
    #[error("Failed to copy tensor data: {0}")]
    TensorCopy(String),
    
    #[error("Failed to run session: {0}")]
    SessionRun(String),
    
    #[error("Null pointer error: {0}")]
    NullPointer(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, MNNError>;
