use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;

use crate::types::{DetConfig, EngineConfig, RecConfig};

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[cfg(feature = "rten")]
    #[error("RTen error: {0}")]
    Rten(#[from] rusto_rten::RtenError),

    #[error("Engine error: {0}")]
    Engine(String),

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

pub type LineNetRecOutput = (Vec<Vec<i32>>, Vec<Vec<f32>>);

/// Abstract inference session interface allowing alternative backends
/// (e.g. RTen, ONNX Runtime, Tract, Candle, or custom hardware accelerators).
pub trait InferenceSession: Send {
    /// Execute inference with a single input tensor returning primary output tensor.
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError>;

    /// Execute inference with a single input tensor returning all named output tensors.
    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError>;

    /// Execute inference with named input tensors returning all named output tensors.
    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError>;

    /// Optional metadata lookup for embedded character list.
    fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    /// Check if metadata key exists.
    fn have_key(&self, _key: &str) -> bool {
        false
    }

    /// Model input channels (1 for grayscale ink map, 3 for RGB/BGR).
    fn input_channels(&self) -> usize {
        3
    }
}

impl InferenceSession for Box<dyn InferenceSession> {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        (**self).run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        (**self).run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        (**self).run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        (**self).get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        (**self).have_key(key)
    }

    fn input_channels(&self) -> usize {
        (**self).input_channels()
    }
}

/// Generic engine session enum supporting pure-Rust Alibaba MNN (rusto-nn) by default,
/// optional pluggable RTen (rusto-rten), native Alibaba MNN C++, and custom `InferenceSession` backends.
pub enum EngineSession {
    Mnn(Box<MnnSession>),
    #[cfg(feature = "rten")]
    Rten(Box<rusto_rten::RtenSession>),
    #[cfg(feature = "mnn-cpp")]
    MnnCpp(Box<rusto_mnn::MnnCppSession>),
    Custom(Box<dyn InferenceSession>),
}

impl EngineSession {
    pub fn from_path(model_path: &Path, engine_cfg: &EngineConfig) -> Result<Self, EngineError> {
        let resolved = resolve_model_path(model_path);
        let ext = resolved.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("rten") || ext.eq_ignore_ascii_case("onnx") {
            #[cfg(feature = "rten")]
            {
                let session = rusto_rten::RtenSession::from_file(&resolved, engine_cfg.intra_op_num_threads as usize)?;
                Ok(Self::Rten(Box::new(session)))
            }
            #[cfg(not(feature = "rten"))]
            {
                Err(EngineError::Engine(format!(
                    "RTen model '{}' requested but 'rten' feature is not enabled. Enable feature 'rten' or use MNN format with default rusto-nn.",
                    resolved.display()
                )))
            }
        } else {
            let session = MnnSession::from_path(&resolved, engine_cfg)?;
            Ok(Self::Mnn(Box::new(session)))
        }
    }

    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_cls_config(cfg: &crate::types::ClsConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_orient_config(cfg: &crate::types::OrientConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_unwarp_config(cfg: &crate::types::UnwarpConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_layout_config(cfg: &crate::types::LayoutConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_mnn(session: MnnSession) -> Self {
        Self::Mnn(Box::new(session))
    }

    #[cfg(feature = "rten")]
    pub fn from_rten(session: rusto_rten::RtenSession) -> Self {
        Self::Rten(Box::new(session))
    }

    #[cfg(feature = "mnn-cpp")]
    pub fn from_mnn_cpp(session: rusto_mnn::MnnCppSession) -> Self {
        Self::MnnCpp(Box::new(session))
    }

    #[cfg(feature = "mnn-cpp")]
    pub fn from_mnn_cpp_path(model_path: &Path, num_threads: usize) -> Result<Self, EngineError> {
        let resolved = resolve_model_path(model_path);
        let session = rusto_mnn::MnnCppSession::from_file(&resolved, num_threads)
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(Self::MnnCpp(Box::new(session)))
    }

    pub fn from_custom(session: impl InferenceSession + 'static) -> Self {
        Self::Custom(Box::new(session))
    }

    pub fn from_boxed(session: Box<dyn InferenceSession>) -> Self {
        Self::Custom(session)
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        match self {
            Self::Mnn(s) => s.run(input),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.run(&input).map_err(|e| EngineError::Engine(e.to_string())),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(s) => s.run(input).map_err(|e| EngineError::Engine(e.to_string())),
            Self::Custom(s) => s.run(input),
        }
    }

    pub fn run_all(
        &mut self,
        input: ArrayD<f32>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        match self {
            Self::Mnn(s) => s.run_all(input),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.run_all(&input).map_err(|e| EngineError::Engine(e.to_string())),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(s) => s.run_all(input).map_err(|e| EngineError::Engine(e.to_string())),
            Self::Custom(s) => s.run_all(input),
        }
    }

    pub fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        match self {
            Self::Mnn(s) => s.run_with_inputs(inputs),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.run_with_inputs(&inputs).map_err(|e| EngineError::Engine(e.to_string())),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(s) => s.run_with_inputs(inputs).map_err(|e| EngineError::Engine(e.to_string())),
            Self::Custom(s) => s.run_with_inputs(inputs),
        }
    }

    pub fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        match self {
            Self::Mnn(s) => s.get_character_list(key),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.get_character_list(key),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(_) => None,
            Self::Custom(s) => s.get_character_list(key),
        }
    }

    pub fn have_key(&self, key: &str) -> bool {
        match self {
            Self::Mnn(s) => s.have_key(key),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.have_key(key),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(_) => false,
            Self::Custom(s) => s.have_key(key),
        }
    }

    pub fn input_channels(&self) -> usize {
        match self {
            Self::Mnn(s) => s.input_channels(),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.input_channels(),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(s) => s.input_channels(),
            Self::Custom(s) => s.input_channels(),
        }
    }

    pub fn is_linenet(&self) -> bool {
        self.input_channels() == 1
    }

    pub fn run_linenet_rec(
        &mut self,
        x: ndarray::Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, EngineError> {
        match self {
            Self::Mnn(s) => s.run_linenet_rec(x, widths),
            #[cfg(feature = "rten")]
            Self::Rten(s) => s.run_linenet_rec(&x, &widths).map_err(|e| EngineError::Engine(e.to_string())),
            #[cfg(feature = "mnn-cpp")]
            Self::MnnCpp(s) => s.run_linenet_rec(x, widths).map_err(|e| EngineError::Engine(e.to_string())),
            Self::Custom(_) => Err(EngineError::Engine(
                "Custom sessions do not support LineNet multi-input recognition".into(),
            )),
        }
    }
}

impl InferenceSession for EngineSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.have_key(key)
    }

    fn input_channels(&self) -> usize {
        self.input_channels()
    }
}

fn resolve_model_path(path: &Path) -> PathBuf {
    if path.exists() {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("mnn") || ext.eq_ignore_ascii_case("rten") || ext.eq_ignore_ascii_case("onnx") {
                return path.to_path_buf();
            }
        }
    }

    let mnn_cand = path.with_extension("mnn");
    if mnn_cand.exists() {
        return mnn_cand;
    }

    let rten_cand = path.with_extension("rten");
    if rten_cand.exists() {
        return rten_cand;
    }

    let onnx_cand = path.with_extension("onnx");
    if onnx_cand.exists() {
        return onnx_cand;
    }

    if path.exists() {
        return path.to_path_buf();
    }

    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
        let is_det = file_name.starts_with("det");
        let is_rec = file_name.starts_with("rec");
        let is_cls = file_name.starts_with("cls");
        let is_orient = file_name.starts_with("orient") || file_name.starts_with("rapid_orientation") || file_name.starts_with("lcnet");
        let is_unwarp = file_name.starts_with("unwarp") || file_name.starts_with("uvdoc");
        let is_layout = file_name.starts_with("layout") || file_name.starts_with("pp_doclayout") || file_name.starts_with("pp_layout");

        let candidates: &[&str] = if is_det {
            &["det"]
        } else if is_rec {
            &["rec"]
        } else if is_cls {
            &["cls", "lcnet-text", "ch_ppocr_mobile_v2.0_cls"]
        } else if is_orient {
            &["orient", "rapid_orientation", "lcnet"]
        } else if is_unwarp {
            &["unwarp", "uvdoc"]
        } else if is_layout {
            &["layout_cdla", "layout_publaynet", "layout", "layout_s", "layout_l", "pp_doclayout_plus_l", "pp_doclayout_s"]
        } else {
            &[]
        };

        if !candidates.is_empty() {
            for dir in &["models/PPOCR_v6_tiny", "models/PPOCR_v6", "models/PPOCR_v5", "models/PPOCR_v4_mobile", "models/DocOCR", "models"] {
                for candidate in candidates {
                    let p_mnn = PathBuf::from(format!("{}/{}.mnn", dir, candidate));
                    if p_mnn.exists() {
                        return p_mnn;
                    }
                    let p_rten = PathBuf::from(format!("{}/{}.rten", dir, candidate));
                    if p_rten.exists() {
                        return p_rten;
                    }
                    let p_onnx = PathBuf::from(format!("{}/{}.onnx", dir, candidate));
                    if p_onnx.exists() {
                        return p_onnx;
                    }
                }
            }
        }
    }

    path.to_path_buf()
}

#[cfg(feature = "rten")]
pub use rusto_rten::RtenSession;

#[cfg(feature = "rten")]
impl InferenceSession for rusto_rten::RtenSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.run(&input).map_err(|e| EngineError::Engine(e.to_string()))
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_all(&input).map_err(|e| EngineError::Engine(e.to_string()))
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_with_inputs(&inputs).map_err(|e| EngineError::Engine(e.to_string()))
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.have_key(key)
    }

    fn input_channels(&self) -> usize {
        self.input_channels()
    }
}

pub struct MnnSession {
    session: rusto_nn::Session<'static>,
    channels: usize,
}

impl MnnSession {
    pub fn from_path(model_path: &Path, _engine_cfg: &EngineConfig) -> Result<Self, EngineError> {
        let resolved = resolve_model_path(model_path);
        let model = rusto_nn::Model::from_file(&resolved)
            .map_err(|e| EngineError::Engine(format!("Failed to load MNN model {}: {}", resolved.display(), e)))?;
        let channels = model.input_channels();
        let session = rusto_nn::Session::from_arc(std::sync::Arc::new(model))
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(Self {
            session,
            channels,
        })
    }

    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn input_channels(&self) -> usize {
        self.channels
    }

    pub fn is_linenet(&self) -> bool {
        self.channels == 1
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let data: Vec<f32> = if let Some(slice) = input.as_slice() {
            slice.to_vec()
        } else {
            input.iter().copied().collect()
        };
        let mnn_tensor = rusto_nn::Tensor::from_floats(&shape, data, rusto_nn::TensorLayout::NCHW);
        let out_tensor = self.session
            .run_single(&mnn_tensor)
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let out_shape = out_tensor.shape().to_vec();
        let out_data = out_tensor
            .as_floats()
            .ok_or_else(|| EngineError::Engine("MNN output tensor is not float data".into()))?
            .to_vec();
        ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), out_data)
            .map_err(EngineError::ShapeError)
    }

    pub fn run_all(
        &mut self,
        input: ArrayD<f32>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let data: Vec<f32> = if let Some(slice) = input.as_slice() {
            slice.to_vec()
        } else {
            input.iter().copied().collect()
        };
        let mnn_tensor = rusto_nn::Tensor::from_floats(&shape, data, rusto_nn::TensorLayout::NCHW);
        let input_names = self.session.model().input_names();
        let in_name = input_names.first().map(|s| s.as_str()).unwrap_or("x");
        let outputs = self.session
            .run(&[(in_name, &mnn_tensor)])
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let mut results = HashMap::with_capacity(outputs.len());
        for (name, out_tensor) in outputs {
            let out_shape = out_tensor.shape().to_vec();
            if let Some(floats) = out_tensor.as_floats() {
                let array = ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), floats.to_vec())
                    .map_err(EngineError::ShapeError)?;
                results.insert(name, array);
            } else if let Some(ints) = out_tensor.as_ints() {
                let floats: Vec<f32> = ints.iter().map(|&x| x as f32).collect();
                let array = ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), floats)
                    .map_err(EngineError::ShapeError)?;
                results.insert(name, array);
            }
        }
        Ok(results)
    }

    pub fn run_with_inputs(
        &mut self,
        inputs_map: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let mut mnn_tensors = Vec::with_capacity(inputs_map.len());
        for (name, arr) in &inputs_map {
            let shape = arr.shape().to_vec();
            let data: Vec<f32> = if let Some(slice) = arr.as_slice() {
                slice.to_vec()
            } else {
                arr.iter().copied().collect()
            };
            mnn_tensors.push((name.clone(), rusto_nn::Tensor::from_floats(&shape, data, rusto_nn::TensorLayout::NCHW)));
        }

        let input_refs: Vec<(&str, &rusto_nn::Tensor)> = mnn_tensors
            .iter()
            .map(|(name, t)| (name.as_str(), t))
            .collect();

        let outputs = self.session
            .run(&input_refs)
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let mut results = HashMap::with_capacity(outputs.len());
        for (name, out_tensor) in outputs {
            let out_shape = out_tensor.shape().to_vec();
            if let Some(floats) = out_tensor.as_floats() {
                let array = ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), floats.to_vec())
                    .map_err(EngineError::ShapeError)?;
                results.insert(name, array);
            } else if let Some(ints) = out_tensor.as_ints() {
                let floats: Vec<f32> = ints.iter().map(|&x| x as f32).collect();
                let array = ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), floats)
                    .map_err(EngineError::ShapeError)?;
                results.insert(name, array);
            }
        }
        Ok(results)
    }

    pub fn run_linenet_rec(
        &mut self,
        x: ndarray::Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, EngineError> {
        let x_shape = x.shape().to_vec();
        let x_data: Vec<f32> = if let Some(slice) = x.as_slice() {
            slice.to_vec()
        } else {
            x.iter().copied().collect()
        };
        let x_tensor = rusto_nn::Tensor::from_floats(&x_shape, x_data, rusto_nn::TensorLayout::NCHW);
        let widths_tensor = rusto_nn::Tensor::from_ints(&[widths.len()], widths);

        let outputs = self.session
            .run(&[("x", &x_tensor), ("widths", &widths_tensor)])
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let ids_tensor = outputs
            .get("ids")
            .ok_or_else(|| EngineError::OutputError("Missing 'ids' in MNN LineNet rec output".into()))?;
        let conf_tensor = outputs
            .get("conf")
            .ok_or_else(|| EngineError::OutputError("Missing 'conf' in MNN LineNet rec output".into()))?;
        let lengths_tensor = outputs
            .get("lengths")
            .ok_or_else(|| EngineError::OutputError("Missing 'lengths' in MNN LineNet rec output".into()))?;

        let ids_vec = ids_tensor.to_i32_vec();
        let ids_shape = ids_tensor.shape().to_vec();
        let conf_vec = conf_tensor.to_f32_vec();
        let lengths_vec = lengths_tensor.to_i32_vec();

        let batch_size = lengths_vec.len();
        let max_len = if ids_shape.len() >= 2 {
            ids_shape[1]
        } else {
            ids_vec.len() / batch_size.max(1)
        };

        let mut batch_ids = Vec::with_capacity(batch_size);
        let mut batch_confs = Vec::with_capacity(batch_size);

        for (b, &len_val) in lengths_vec.iter().enumerate().take(batch_size) {
            let len = len_val.max(0) as usize;
            let start = b * max_len;
            let actual_len = len.min(max_len);
            let b_ids = ids_vec[start..start + actual_len].to_vec();
            let b_confs = conf_vec[start..start + actual_len].to_vec();
            batch_ids.push(b_ids);
            batch_confs.push(b_confs);
        }

        Ok((batch_ids, batch_confs))
    }

    pub fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    pub fn have_key(&self, _key: &str) -> bool {
        false
    }
}

impl InferenceSession for MnnSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.have_key(key)
    }

    fn input_channels(&self) -> usize {
        self.input_channels()
    }
}

#[cfg(feature = "mnn-cpp")]
impl InferenceSession for rusto_mnn::MnnCppSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.run(input).map_err(|e| EngineError::Engine(e.to_string()))
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_all(input).map_err(|e| EngineError::Engine(e.to_string()))
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_with_inputs(inputs).map_err(|e| EngineError::Engine(e.to_string()))
    }

    fn input_channels(&self) -> usize {
        self.input_channels()
    }
}
