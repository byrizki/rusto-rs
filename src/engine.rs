use std::path::Path;

use ndarray::{ArrayD, Array};
use mnn::{BackendConfig, ForwardType, Interpreter, PrecisionMode, ScheduleConfig, PowerMode};

use crate::types::{DetConfig, EngineConfig, RecConfig};

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("MNN error: {0}")]
    Mnn(#[from] mnn::MNNError),

    #[cfg(feature = "use-opencv")]
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),

    #[error("Image processing error: {0}")]
    ImageError(String),

    #[error("Invalid input shape")] 
    InvalidInputShape,

    #[error("Preprocess error: {0}")]
    Preprocess(String),

    #[error("Shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),

    #[error("Output error: {0}")]
    OutputError(String),
}

impl From<Box<dyn std::error::Error>> for EngineError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        EngineError::ImageError(err.to_string())
    }
}

pub struct MnnSession {
    interpreter: Interpreter,
    session: Option<mnn::Session>,
    input_tensor_name: Option<String>,
    output_tensor_name: Option<String>,
    last_input_shape: Option<[i32; 4]>,
}

impl Drop for MnnSession {
    fn drop(&mut self) {
        // Explicitly release the session before the interpreter is dropped
        // This prevents segfault by ensuring proper cleanup order
        if let Some(session) = self.session.take() {
            // Release session before interpreter drops
            drop(session);
        }
    }
}

impl MnnSession {
    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    fn from_path(model_path: &Path, _engine_cfg: &EngineConfig) -> Result<Self, EngineError> {        
        let interpreter = Interpreter::from_file(model_path)?;
        
        Ok(Self { 
            interpreter,
            session: None,
            input_tensor_name: None,
            output_tensor_name: None,
            last_input_shape: None,
        })
    }

    fn ensure_session(&mut self) -> Result<(), EngineError> {
        if self.session.is_none() {
            let mut config = ScheduleConfig::new();
            config.set_type(ForwardType::Auto);

            let mut backend_config = BackendConfig::new();
            backend_config.set_precision_mode(PrecisionMode::High);
            backend_config.set_power_mode(PowerMode::High);

            config.set_backend_config(backend_config);

            let session = self.interpreter.create_session(config)?;
            self.session = Some(session);
        }
        Ok(())
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.ensure_session()?;
        
        // Get tensor names if not cached
        if self.input_tensor_name.is_none() || self.output_tensor_name.is_none() {
            let session = self.session.as_ref().unwrap();
            let inputs = self.interpreter.inputs(session);
            let outputs = self.interpreter.outputs(session);
            
            let input_info = inputs.iter().next().unwrap();
            let output_info = outputs.iter().next().unwrap();
            
            self.input_tensor_name = Some(input_info.name().to_string());
            self.output_tensor_name = Some(output_info.name().to_string());
        }
        
        let input_tensor_name = self.input_tensor_name.as_ref().unwrap();
        let output_tensor_name = self.output_tensor_name.as_ref().unwrap();
        
        let input_shape = input.shape();
        let new_shape: [i32; 4] = [
            input_shape[0] as i32,
            input_shape[1] as i32,
            input_shape[2] as i32,
            input_shape[3] as i32,
        ];
        
        // Resize if shape changed
        let need_resize = self.last_input_shape
            .map(|last_shape| last_shape != new_shape)
            .unwrap_or(true);
        
        if need_resize {
            let session = self.session.as_mut().unwrap();
            let mut input_tensor = unsafe {
                self.interpreter.input_unresized::<f32>(session, input_tensor_name)?
            };
            
            self.interpreter.resize_tensor(&mut input_tensor, new_shape);
            drop(input_tensor);
            self.interpreter.resize_session(session);
            
            self.last_input_shape = Some(new_shape);
        }
        
        // Run inference
        let (output_data, output_shape) = {
            let session = self.session.as_mut().unwrap();
            let mut input_tensor = self.interpreter.input::<f32>(session, input_tensor_name)?;
            
            // Copy input data
            if let Some(flat_data) = input.as_slice() {
                let mut host_tensor = input_tensor.create_host_tensor_from_device(false);
                let host_data_mut = host_tensor.host_mut();
                host_data_mut.copy_from_slice(flat_data);
                input_tensor.copy_from_host_tensor(&host_tensor)?;
            } else {
                let mut host_tensor = input_tensor.create_host_tensor_from_device(false);
                let host_data_mut = host_tensor.host_mut();
                for (i, val) in input.iter().enumerate() {
                    host_data_mut[i] = *val;
                }
                input_tensor.copy_from_host_tensor(&host_tensor)?;
            }
            
            self.interpreter.run_session(session)?;
            
            let output = self.interpreter.output::<f32>(session, output_tensor_name)?;
            output.wait(mnn::ffi::MapType::MAP_TENSOR_READ, true);
            
            let shape = output.shape();
            let output_host_tensor = output.create_host_tensor_from_device(true);
            (output_host_tensor.host().to_vec(), shape)
        };
        
        // Convert to ndarray
        let output_shape_usize: Vec<usize> = output_shape.iter().map(|&x| x as usize).collect();
        let output_array = Array::from_shape_vec(output_shape_usize, output_data)?;
        
        Ok(output_array.into_dyn())
    }

    pub fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        // MNN models don't typically embed character lists in metadata
        None
    }

    pub fn have_key(&self, _key: &str) -> bool {
        false
    }
}
