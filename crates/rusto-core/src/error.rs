use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Preprocess error: {0}")]
    Preprocess(String),
    #[error("Postprocess error: {0}")]
    Postprocess(String),
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    #[error("Image error: {0}")]
    Image(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

