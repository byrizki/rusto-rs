use std::path::Path;
use std::sync::Arc;

use ndarray::{ArrayD, CowArray, IxDyn};
use ort::environment::Environment;
use ort::session::Session;
use ort::tensor::OrtOwnedTensor;
use ort::value::Value;
use ort::{GraphOptimizationLevel, OrtError, SessionBuilder};

use crate::types::{DetConfig, EngineConfig, RecConfig};

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("ORT error: {0}")]
    Ort(#[from] OrtError),

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
    #[allow(dead_code)]
    env: Arc<Environment>,
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
        let env = Environment::builder()
            .with_name("rapidocr")
            .build()? 
            .into_arc();

        let mut builder = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;

        if engine_cfg.intra_op_num_threads > 0 {
            builder = builder.with_intra_threads(engine_cfg.intra_op_num_threads as i16)?;
        }

        if engine_cfg.inter_op_num_threads > 0 {
            builder = builder.with_inter_threads(engine_cfg.inter_op_num_threads as i16)?;
        }

        let session = builder.with_model_from_file(model_path)?;

        Ok(Self { env, session })
    }

    pub fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        let allocator = self.session.allocator();

        let cow: CowArray<'_, f32, IxDyn> = CowArray::from(input);
        let input_value = Value::from_array(allocator, &cow)?;

        let outputs = self.session.run(vec![input_value])?;
        let first = outputs
            .into_iter()
            .next()
            .ok_or(EngineError::InvalidInputShape)?;

        let tensor: OrtOwnedTensor<f32, IxDyn> = first.try_extract()?;
        let view = tensor.view();
        Ok(view.to_owned())
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
