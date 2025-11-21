use std::path::Path;

use ndarray::ArrayD;
use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider, TensorRTExecutionProvider, XNNPACKExecutionProvider}, session::{Session, builder::GraphOptimizationLevel}, value::Tensor
};

use crate::types::{DetConfig, EngineConfig, RecConfig};

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    #[cfg(feature = "use-opencv")]
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),

    #[error("Image processing error: {0}")]
    ImageError(String),

    #[error("Invalid input shape")] 
    InvalidInputShape,

    #[error("Preprocess error: {0}")]
    Preprocess(String),
}

impl From<Box<dyn std::error::Error>> for EngineError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        EngineError::ImageError(err.to_string())
    }
}

pub struct OrtSession {
    session: Session,
}

impl OrtSession {
    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    fn from_path(model_path: &Path, engine_cfg: &EngineConfig) -> Result<Self, EngineError> {        
        let mut builder = Session::builder()?
            .with_execution_providers([
                // Prefer TensorRT over CUDA.
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                // Use DirectML on Windows if NVIDIA EPs are not available
                DirectMLExecutionProvider::default().build(),
                // use ANE on Apple platforms
                CoreMLExecutionProvider::default().build(),
                // Use XNNPACK on Android
                XNNPACKExecutionProvider::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;

        if engine_cfg.intra_op_num_threads > 0 {
            builder = builder.with_intra_threads(engine_cfg.intra_op_num_threads as usize)?;
        }

        if engine_cfg.inter_op_num_threads > 0 {
            builder = builder.with_inter_threads(engine_cfg.inter_op_num_threads as usize)?;
        }

        let session = builder.commit_from_file(model_path)?;

        Ok(Self { session })
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        // Create tensor from owned array - v2 requires owned data
        let input_tensor = Tensor::from_array(input)?;

        // Run inference using ort::inputs! macro
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        // Get first output - outputs is a ValueMap, iterate to get first value
        let output = outputs
            .values()
            .next()
            .ok_or(EngineError::InvalidInputShape)?;

        // Extract array view and convert to owned ArrayD
        let array_view = output.try_extract_array::<f32>()?;
        Ok(array_view.to_owned())
    }

    pub fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        let meta = self.session.metadata().ok()?;
        let value = meta.custom(key).ok()??;
        let s = String::from_utf8_lossy(value.as_bytes());
        Some(s.lines().map(|l| l.to_string()).collect())
    }

    pub fn have_key(&self, key: &str) -> bool {
        self.session
            .metadata()
            .ok()
            .and_then(|m| m.custom(key).ok())
            .flatten()
            .is_some()
    }
}
