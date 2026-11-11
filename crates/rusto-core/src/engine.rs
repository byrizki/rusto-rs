//! Pure Rust inference engine and high-level OCR pipeline orchestrator.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::tensor::{Array4, ArrayD, Tensor};

use crate::config::InitializeConfig;
pub use crate::doc::DocPipelineConfig;
use crate::doc::{DocBlock, DocResult, OcrRunner};
pub use crate::error::CoreError;
use crate::image::{crop_bbox, imread, Mat};
use crate::layout::{sort_layout_regions_top_down, LayoutOutput, LayoutRegion, LayoutType};
use crate::orient::{OrientOutput, Orientation};
use crate::pipeline::{OcrPipeline, PipelineOutput};
use crate::postprocess::TextDetOutput;
use crate::rec::TextRecOutput;
use crate::table::{TableDetectorConfig, TableStructure, TableStructureConfig};
use crate::types::{
    ClsConfig, DetConfig, DetectTextResult, DetectionRunOptions, EngineConfig, ImageSource,
    LayoutConfig, OcrRunOptions, OrientConfig, OutputGranularity, PostprocessRunOptions, RecConfig,
    UnwarpConfig,
};
use crate::unwarp::UnwarpOutput;
use crate::Point2f;

pub type LineNetRecOutput = (Vec<Vec<i32>>, Vec<Vec<f32>>);

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("Engine error: {0}")]
    Engine(String),

    #[error("Image processing error: {0}")]
    ImageError(String),

    #[error("Invalid input shape")]
    InvalidInputShape,

    #[error("Preprocess error: {0}")]
    Preprocess(String),

    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("Output error: {0}")]
    OutputError(String),
}


impl From<Box<dyn std::error::Error>> for EngineError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        EngineError::ImageError(err.to_string())
    }
}

impl From<CoreError> for EngineError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::Preprocess(s) => EngineError::Preprocess(s),
            CoreError::Postprocess(s) => EngineError::OutputError(s),
            CoreError::InvalidShape(_) => EngineError::InvalidInputShape,
            CoreError::Image(s) => EngineError::ImageError(s),
            CoreError::Model(s) => EngineError::Engine(s),
            CoreError::Config(s) => EngineError::Engine(s),
            CoreError::Io(e) => EngineError::Preprocess(e.to_string()),
        }
    }
}

impl From<EngineError> for CoreError {
    fn from(err: EngineError) -> Self {
        CoreError::Model(err.to_string())
    }
}

/// Abstract inference session interface allowing alternative backends
/// (e.g. RTen, ONNX Runtime, Tract, Candle, MNN, or custom hardware accelerators).
pub trait InferenceSession: Send {
    /// Execute inference with a single input tensor returning primary output tensor.
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError>;

    /// Execute inference with a single input tensor returning all named output tensors.
    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError>;

    /// Execute inference with named input tensors returning all named output tensors.
    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError>;

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

    /// Whether this session expects 1-channel grayscale LineNet inputs.
    fn is_linenet(&self) -> bool {
        self.input_channels() == 1
    }

    /// Run multi-input LineNet recognition (ink tensor + sequence widths).
    fn run_linenet_rec(
        &mut self,
        _x: Array4<f32>,
        _widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, CoreError> {
        Err(CoreError::Model(
            "LineNet multi-input recognition not supported by this session".into(),
        ))
    }

    /// Attempt to clone or spawn a session sharing underlying model weights for concurrent inference.
    fn try_clone(&self) -> Option<Box<dyn InferenceSession>> {
        None
    }
}

impl InferenceSession for Box<dyn InferenceSession> {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        (**self).run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        (**self).run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
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

    fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, CoreError> {
        (**self).run_linenet_rec(x, widths)
    }

    fn try_clone(&self) -> Option<Box<dyn InferenceSession>> {
        (**self).try_clone()
    }
}

impl<T: InferenceSession + ?Sized> InferenceSession for &mut T {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        (**self).run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        (**self).run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
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

    fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, CoreError> {
        (**self).run_linenet_rec(x, widths)
    }

    fn try_clone(&self) -> Option<Box<dyn InferenceSession>> {
        (**self).try_clone()
    }
}

pub fn resolve_model_path(path: &Path) -> PathBuf {
    if path.exists() {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("mnn")
                || ext.eq_ignore_ascii_case("rten")
                || ext.eq_ignore_ascii_case("onnx")
            {
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
        let is_orient = file_name.starts_with("orient")
            || file_name.starts_with("rapid_orientation")
            || file_name.starts_with("lcnet");
        let is_unwarp = file_name.starts_with("unwarp") || file_name.starts_with("uvdoc");
        let is_layout = file_name.starts_with("layout")
            || file_name.starts_with("pp_doclayout")
            || file_name.starts_with("pp_layout");

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
            &[
                "layout_cdla",
                "layout_publaynet",
                "layout",
                "layout_s",
                "layout_l",
                "pp_doclayout_plus_l",
                "pp_doclayout_s",
            ]
        } else {
            &[]
        };

        if !candidates.is_empty() {
            for dir in &[
                "models/PPOCR_v6_tiny",
                "models/PPOCR_v6",
                "models/PPOCR_v5",
                "models/PPOCR_v4_mobile",
                "models/DocOCR",
                "models",
            ] {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EngineBackend {
    MnnPure = 0,
    Rten = 1,
    MnnCpp = 2,
}

static CURRENT_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_active_backend(backend: EngineBackend) {
    CURRENT_BACKEND.store(backend as u8, std::sync::atomic::Ordering::Release);
}

pub fn active_backend() -> EngineBackend {
    match CURRENT_BACKEND.load(std::sync::atomic::Ordering::Acquire) {
        1 => EngineBackend::Rten,
        2 => EngineBackend::MnnCpp,
        _ => EngineBackend::MnnPure,
    }
}

pub type SessionLoader = fn(&Path, &EngineConfig) -> Result<Box<dyn InferenceSession>, CoreError>;
static SESSION_LOADER: std::sync::atomic::AtomicPtr<()> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Register a global session loader callback (e.g. for RTen or ONNX runtime).
/// If registered, `EngineSession::from_path` will first query this loader before falling back to MNN.
pub fn register_session_loader(loader: SessionLoader) {
    SESSION_LOADER.store(loader as *mut (), std::sync::atomic::Ordering::Release);
}

/// Generic engine session enum supporting pure-Rust Alibaba MNN (rusto-nn) by default,
/// pluggable RTen, and custom `InferenceSession` backends.
pub enum EngineSession {
    Mnn(Box<MnnSession>),
    Custom(Box<dyn InferenceSession>),
}

impl EngineSession {
    pub fn from_path(model_path: &Path, engine_cfg: &EngineConfig) -> Result<Self, EngineError> {
        let resolved = resolve_model_path(model_path);
        let ptr = SESSION_LOADER.load(std::sync::atomic::Ordering::Acquire);
        if !ptr.is_null() {
            let loader: SessionLoader = unsafe { std::mem::transmute(ptr) };
            match loader(&resolved, engine_cfg) {
                Ok(session) => return Ok(Self::Custom(session)),
                Err(CoreError::Model(ref msg)) if msg == "UNSUPPORTED_FORMAT" => {
                    // Fall through to MNN session
                }
                Err(err) => return Err(err.into()),
            }
        }
        let session = MnnSession::from_path(&resolved, engine_cfg)?;
        Ok(Self::Mnn(Box::new(session)))
    }

    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_cls_config(cfg: &ClsConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_orient_config(cfg: &OrientConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_unwarp_config(cfg: &UnwarpConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_layout_config(cfg: &LayoutConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_mnn(session: MnnSession) -> Self {
        Self::Mnn(Box::new(session))
    }

    pub fn from_rten(session: impl InferenceSession + 'static) -> Self {
        Self::Custom(Box::new(session))
    }

    pub fn from_mnn_cpp(session: impl InferenceSession + 'static) -> Self {
        Self::Custom(Box::new(session))
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
            Self::Custom(s) => Ok(s.run(input)?),
        }
    }

    pub fn run_all(
        &mut self,
        input: ArrayD<f32>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        match self {
            Self::Mnn(s) => s.run_all(input),
            Self::Custom(s) => Ok(s.run_all(input)?),
        }
    }

    pub fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        match self {
            Self::Mnn(s) => s.run_with_inputs(inputs),
            Self::Custom(s) => Ok(s.run_with_inputs(inputs)?),
        }
    }

    pub fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        match self {
            Self::Mnn(s) => s.get_character_list(key),
            Self::Custom(s) => s.get_character_list(key),
        }
    }

    pub fn have_key(&self, key: &str) -> bool {
        match self {
            Self::Mnn(s) => s.have_key(key),
            Self::Custom(s) => s.have_key(key),
        }
    }

    pub fn input_channels(&self) -> usize {
        match self {
            Self::Mnn(s) => s.input_channels(),
            Self::Custom(s) => s.input_channels(),
        }
    }

    pub fn is_linenet(&self) -> bool {
        self.input_channels() == 1
    }

    pub fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, EngineError> {
        match self {
            Self::Mnn(s) => s.run_linenet_rec(x, widths),
            Self::Custom(s) => s.run_linenet_rec(x, widths).map_err(Into::into),
        }
    }
}

impl InferenceSession for EngineSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        self.run(input).map_err(Into::into)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        self.run_all(input).map_err(Into::into)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        self.run_with_inputs(inputs).map_err(Into::into)
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

    fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, CoreError> {
        self.run_linenet_rec(x, widths).map_err(Into::into)
    }

    fn try_clone(&self) -> Option<Box<dyn InferenceSession>> {
        match self {
            Self::Mnn(s) => Some(Box::new(EngineSession::Mnn(Box::new((**s).clone()))) as Box<dyn InferenceSession>),
            Self::Custom(s) => s.try_clone().map(|c| Box::new(EngineSession::Custom(c)) as Box<dyn InferenceSession>),
        }
    }
}

#[derive(Clone)]
pub struct MnnSession {
    session: rusto_nn::Session<'static>,
    channels: usize,
}

impl MnnSession {
    pub fn from_path(model_path: &Path, engine_cfg: &EngineConfig) -> Result<Self, EngineError> {
        rusto_nn::init_thread_pool(engine_cfg.intra_op_num_threads.max(0) as usize);
        let resolved = resolve_model_path(model_path);
        let model = rusto_nn::Model::from_file(&resolved)
            .map_err(|e| EngineError::Engine(format!("Failed to load MNN model {}: {}", resolved.display(), e)))?;
        let channels = model.input_channels();
        let session = rusto_nn::Session::from_arc(std::sync::Arc::new(model))
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(Self { session, channels })
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
        let expected_len = shape.iter().product::<usize>();
        let (raw_vec, offset) = input.into_raw_vec_and_offset();
        let data = match offset {
            None => raw_vec,
            Some(0) if raw_vec.len() == expected_len => raw_vec,
            Some(off) if off + expected_len <= raw_vec.len() => {
                raw_vec[off..off + expected_len].to_vec()
            }
            _ => raw_vec,
        };
        let mnn_tensor = rusto_nn::Tensor::from_floats(&shape, data, rusto_nn::TensorLayout::NCHW);
        let out_tensor = self
            .session
            .run_single(&mnn_tensor)
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let out_shape = out_tensor.shape().to_vec();
        let out_data = out_tensor.into_f32_vec();
        Ok(Tensor::from_data(&out_shape, out_data))
    }

    pub fn run_all(
        &mut self,
        input: ArrayD<f32>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let expected_len = shape.iter().product::<usize>();
        let (raw_vec, offset) = input.into_raw_vec_and_offset();
        let data = match offset {
            None => raw_vec,
            Some(0) if raw_vec.len() == expected_len => raw_vec,
            Some(off) if off + expected_len <= raw_vec.len() => {
                raw_vec[off..off + expected_len].to_vec()
            }
            _ => raw_vec,
        };
        let mnn_tensor = rusto_nn::Tensor::from_floats(&shape, data, rusto_nn::TensorLayout::NCHW);
        let input_names = self.session.model().input_names();
        let in_name = input_names.first().map(|s| s.as_str()).unwrap_or("x");
        let outputs = self
            .session
            .run(&[(in_name, &mnn_tensor)])
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let mut results = HashMap::with_capacity(outputs.len());
        for (name, out_tensor) in outputs {
            let out_shape = out_tensor.shape().to_vec();
            let array = Tensor::from_data(&out_shape, out_tensor.into_f32_vec());
            results.insert(name, array);
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
            let data = arr.as_slice().to_vec();
            mnn_tensors.push((
                name.clone(),
                rusto_nn::Tensor::from_floats(&shape, data, rusto_nn::TensorLayout::NCHW),
            ));
        }

        let input_refs: Vec<(&str, &rusto_nn::Tensor)> = mnn_tensors
            .iter()
            .map(|(name, t)| (name.as_str(), t))
            .collect();

        let outputs = self
            .session
            .run(&input_refs)
            .map_err(|e| EngineError::Engine(e.to_string()))?;

        let mut results = HashMap::with_capacity(outputs.len());
        for (name, out_tensor) in outputs {
            let out_shape = out_tensor.shape().to_vec();
            if let Some(floats) = out_tensor.as_floats() {
                let array = Tensor::from_data(&out_shape, floats.to_vec());
                results.insert(name, array);
            } else if let Some(ints) = out_tensor.as_ints() {
                let floats: Vec<f32> = ints.iter().map(|&x| x as f32).collect();
                let array = Tensor::from_data(&out_shape, floats);
                results.insert(name, array);
            }
        }
        Ok(results)
    }

    pub fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, EngineError> {
        let x_shape = x.shape().to_vec();
        let x_data = x.into_raw_vec();
        let x_tensor = rusto_nn::Tensor::from_floats(&x_shape, x_data, rusto_nn::TensorLayout::NCHW);
        let widths_tensor = rusto_nn::Tensor::from_ints(&[widths.len()], widths);

        let wanted: Vec<&str> = self
            .session
            .model()
            .output_names()
            .iter()
            .map(|s| s.as_str())
            .filter(|&n| matches!(n, "ids" | "conf" | "len" | "lengths"))
            .collect();
        let outputs = if wanted.len() >= 2 {
            self.session
                .run_outputs(&[("x", &x_tensor), ("widths", &widths_tensor)], &wanted)
                .map_err(|e| EngineError::Engine(e.to_string()))?
        } else {
            self.session
                .run(&[("x", &x_tensor), ("widths", &widths_tensor)])
                .map_err(|e| EngineError::Engine(e.to_string()))?
        };

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
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, CoreError> {
        self.run(input).map_err(Into::into)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        self.run_all(input).map_err(Into::into)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, CoreError> {
        self.run_with_inputs(inputs).map_err(Into::into)
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

    fn run_linenet_rec(
        &mut self,
        x: Array4<f32>,
        widths: Vec<i32>,
    ) -> Result<LineNetRecOutput, CoreError> {
        self.run_linenet_rec(x, widths).map_err(Into::into)
    }

    fn try_clone(&self) -> Option<Box<dyn InferenceSession>> {
        Some(Box::new(self.clone()))
    }
}

pub type RustOOutput = PipelineOutput;

pub struct RustO {
    pub pipeline: OcrPipeline,
}

impl RustO {
    pub fn initialize(config: InitializeConfig) -> Result<Self, EngineError> {
        config.global.validate()?;
        let det_session = EngineSession::from_det_config(&config.det)?;
        let rec_session = EngineSession::from_rec_config(&config.rec)?;
        Self::with_engine_sessions(config, det_session, rec_session)
    }

    pub fn with_engine_sessions(
        mut config: InitializeConfig,
        det_session: EngineSession,
        rec_session: EngineSession,
    ) -> Result<Self, EngineError> {
        config.global.validate()?;
        let decoder = match &config.rec.rec_keys_path {
            Some(keys_path) => crate::ctc::CtcDecoder::from_keys_file(keys_path)
                .map_err(|e| EngineError::Engine(e.to_string()))?,
            None => {
                if let Some(char_list) = rec_session.get_character_list("character") {
                    crate::ctc::CtcDecoder::from_model_character_list(char_list)
                } else {
                    return Err(EngineError::Engine(
                        "No character dictionary found for recognition".into(),
                    ));
                }
            }
        };

        let mut pipeline = OcrPipeline::new(
            Box::new(det_session),
            Box::new(rec_session),
            decoder,
            config.global.clone(),
        )
        .with_det_config(config.det.clone());
        pipeline.rec_batch_num = config.rec.rec_batch_num.max(1) as usize;

        if let Some(orient_cfg) = config.orient.take() {
            let session = EngineSession::from_orient_config(&orient_cfg)?;
            pipeline = pipeline.with_orient(Box::new(session));
        }
        if let Some(cls_cfg) = config.cls.take() {
            let session = EngineSession::from_cls_config(&cls_cfg)?;
            pipeline = pipeline.with_cls(Box::new(session));
        }
        if let Some(unwarp_cfg) = config.unwarp.take() {
            let session = EngineSession::from_unwarp_config(&unwarp_cfg)?;
            pipeline = pipeline.with_unwarp(Box::new(session));
        }
        if let Some(layout_cfg) = config.layout.take() {
            let session = EngineSession::from_layout_config(&layout_cfg)?;
            pipeline = pipeline.with_layout(Box::new(session)).with_layout_config(layout_cfg);
        }
        if config.global.reading_order {
            if let Some(order_path) = &config.global.reading_order_model {
                let session = EngineSession::from_path(order_path, &config.det.engine_cfg)?;
                pipeline = pipeline.with_order(Box::new(session));
            }
        }

        Ok(Self { pipeline })
    }

    pub fn with_custom_engines(
        mut config: InitializeConfig,
        det_session: impl InferenceSession + 'static,
        rec_session: impl InferenceSession + 'static,
    ) -> Result<Self, EngineError> {
        config.global.validate()?;
        let decoder = match &config.rec.rec_keys_path {
            Some(keys_path) => crate::ctc::CtcDecoder::from_keys_file(keys_path)
                .map_err(|e| EngineError::Engine(e.to_string()))?,
            None => {
                if let Some(char_list) = rec_session.get_character_list("character") {
                    crate::ctc::CtcDecoder::from_model_character_list(char_list)
                } else {
                    return Err(EngineError::Engine(
                        "No character dictionary found for recognition".into(),
                    ));
                }
            }
        };

        let mut pipeline = OcrPipeline::new(
            Box::new(det_session),
            Box::new(rec_session),
            decoder,
            config.global.clone(),
        )
        .with_det_config(config.det.clone());
        pipeline.rec_batch_num = config.rec.rec_batch_num.max(1) as usize;

        if let Some(orient_cfg) = config.orient.take() {
            let session = EngineSession::from_orient_config(&orient_cfg)?;
            pipeline = pipeline.with_orient(Box::new(session));
        }
        if let Some(cls_cfg) = config.cls.take() {
            let session = EngineSession::from_cls_config(&cls_cfg)?;
            pipeline = pipeline.with_cls(Box::new(session));
        }
        if let Some(unwarp_cfg) = config.unwarp.take() {
            let session = EngineSession::from_unwarp_config(&unwarp_cfg)?;
            pipeline = pipeline.with_unwarp(Box::new(session));
        }
        if let Some(layout_cfg) = config.layout.take() {
            let session = EngineSession::from_layout_config(&layout_cfg)?;
            pipeline = pipeline.with_layout(Box::new(session)).with_layout_config(layout_cfg);
        }
        if config.global.reading_order {
            if let Some(order_path) = &config.global.reading_order_model {
                let session = EngineSession::from_path(order_path, &config.det.engine_cfg)?;
                pipeline = pipeline.with_order(Box::new(session));
            }
        }

        Ok(Self { pipeline })
    }

    pub fn with_components(
        global: crate::types::GlobalConfig,
        det: TextDetector,
        rec: TextRecognizer,
    ) -> Result<Self, EngineError> {
        let det_session = det.session;
        let rec_session = rec.session;
        let decoder = rec.decoder;
        let mut pipeline = OcrPipeline::new(
            Box::new(det_session),
            Box::new(rec_session),
            decoder,
            global,
        )
        .with_det_config(det.cfg);
        pipeline.rec_batch_num = rec.cfg.rec_batch_num.max(1) as usize;
        Ok(Self { pipeline })
    }

    pub fn with_orient(mut self, orient: OrientClassifier) -> Self {
        self.pipeline = self.pipeline.with_orient(Box::new(orient.session));
        self
    }

    pub fn with_cls(mut self, cls: OrientClassifier) -> Self {
        self.pipeline = self.pipeline.with_cls(Box::new(cls.session));
        self
    }

    pub fn with_unwarp(mut self, unwarp: DocUnwarper) -> Self {
        self.pipeline = self.pipeline.with_unwarp(Box::new(unwarp.session));
        self
    }

    pub fn with_layout(mut self, layout: LayoutDetector) -> Self {
        self.pipeline = self
            .pipeline
            .with_layout(Box::new(layout.session))
            .with_layout_config(layout.config);
        self
    }

    /// Returns whether the engine instance is initialized and ready for inference.
    pub fn is_ready(&self) -> bool {
        true
    }

    /// Returns the engine library version string.
    pub fn version(&self) -> &'static str {
        crate::version()
    }

    pub fn predict(
        &mut self,
        image_source: ImageSource,
        run_opts: Option<OcrRunOptions>,
    ) -> Result<DetectTextResult, EngineError> {
        let options = run_opts.unwrap_or_default();
        self.detect_text(&image_source, &options)
    }

    pub fn detect_text(
        &mut self,
        source: &ImageSource,
        options: &OcrRunOptions,
    ) -> Result<DetectTextResult, EngineError> {
        options.validate()?;
        let output = match source {
            ImageSource::Path(path) => self.run_with_options(path, options)?,
            ImageSource::Bytes(bytes) => {
                let image = ::image::load_from_memory(bytes)
                    .map_err(|error| EngineError::ImageError(error.to_string()))?;
                self.run_on_mat_with_options(&Mat::new(image), options)?
            }
        };
        match options.output {
            OutputGranularity::Spatial => {
                let (line_y, word_x) = options.pre_recognition.as_ref().map_or((None, None), |pr| {
                    (pr.line_y_threshold, pr.word_x_threshold)
                });
                Ok(DetectTextResult::Spatial(
                    output.to_spatial_text(line_y, word_x),
                ))
            }
            OutputGranularity::Tsv => {
                Ok(DetectTextResult::Spatial(output.to_tsv()))
            }
            OutputGranularity::Lines | OutputGranularity::Words => Ok(
                DetectTextResult::Structured(output.to_text_results_with_options(options)),
            ),
        }
    }

    pub fn detect_text_rgb(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        options: &OcrRunOptions,
    ) -> Result<DetectTextResult, EngineError> {
        options.validate()?;
        let mat = Mat::from_rgb_stride(width, height, stride, pixels)
            .map_err(|e| EngineError::ImageError(e.to_string()))?;
        let output = self.run_on_mat_with_options(&mat, options)?;
        match options.output {
            OutputGranularity::Spatial => {
                let (line_y, word_x) = options.pre_recognition.as_ref().map_or((None, None), |pr| {
                    (pr.line_y_threshold, pr.word_x_threshold)
                });
                Ok(DetectTextResult::Spatial(
                    output.to_spatial_text(line_y, word_x),
                ))
            }
            OutputGranularity::Tsv => {
                Ok(DetectTextResult::Spatial(output.to_tsv()))
            }
            OutputGranularity::Lines | OutputGranularity::Words => Ok(
                DetectTextResult::Structured(output.to_text_results_with_options(options)),
            ),
        }
    }

    pub fn run_with_options<P: AsRef<Path>>(
        &mut self,
        image_path: P,
        options: &OcrRunOptions,
    ) -> Result<PipelineOutput, EngineError> {
        let img = imread(image_path)?;
        self.run_on_mat_with_options(&img, options)
    }

    pub fn run_on_mat_with_options(
        &mut self,
        img: &Mat,
        options: &OcrRunOptions,
    ) -> Result<PipelineOutput, EngineError> {
        Ok(self.pipeline.run_with_options(img, options)?)
    }
}

impl OcrRunner for RustO {
    fn run_on_mat(&mut self, img: &Mat) -> Result<PipelineOutput, CoreError> {
        self.run_on_mat_with_options(img, &OcrRunOptions::default())
            .map_err(Into::into)
    }
}

/// Text detector engine wrapper executing DBNet or LineNet.
pub struct TextDetector {
    pub cfg: DetConfig,
    pub session: EngineSession,
}

impl TextDetector {
    pub fn new(cfg: DetConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&cfg.model_path, &cfg.engine_cfg)?;
        Ok(Self { cfg, session })
    }

    pub fn with_session(cfg: DetConfig, session: EngineSession) -> Self {
        Self { cfg, session }
    }

    pub fn with_custom_session(cfg: DetConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            cfg,
            session: EngineSession::from_custom(session),
        }
    }

    pub fn is_linenet(&self) -> bool {
        self.session.is_linenet()
    }

    pub fn run(&mut self, img: &Mat) -> Result<TextDetOutput, EngineError> {
        self.run_with_options(img, None, None)
    }

    pub fn run_with_options(
        &mut self,
        img: &Mat,
        options: Option<&DetectionRunOptions>,
        postprocess: Option<&PostprocessRunOptions>,
    ) -> Result<TextDetOutput, EngineError> {
        let mut core_det = crate::det::TextDetector::with_session(self.cfg.clone(), &mut self.session);
        core_det
            .run_with_options(img, options, postprocess)
            .map_err(Into::into)
    }

    pub fn sorted_boxes(&self, dt_boxes: &mut [[Point2f; 4]]) {
        crate::det::sort_det_boxes(dt_boxes);
    }
}

/// Text recognizer engine wrapper executing CTC recognition.
pub struct TextRecognizer {
    pub cfg: RecConfig,
    pub session: EngineSession,
    pub decoder: crate::ctc::CtcDecoder,
}

impl TextRecognizer {
    pub fn new(cfg: RecConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&cfg.model_path, &cfg.engine_cfg)?;
        let decoder = crate::rec::decoder_from_cfg(&cfg, &session)
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(Self {
            cfg,
            session,
            decoder,
        })
    }

    pub fn with_session(
        cfg: RecConfig,
        session: EngineSession,
        decoder: crate::ctc::CtcDecoder,
    ) -> Self {
        Self {
            cfg,
            session,
            decoder,
        }
    }

    pub fn with_custom_session(
        cfg: RecConfig,
        session: impl InferenceSession + 'static,
    ) -> Result<Self, EngineError> {
        let session = EngineSession::from_custom(session);
        let decoder = crate::rec::decoder_from_cfg(&cfg, &session)
            .map_err(|e| EngineError::Engine(e.to_string()))?;
        Ok(Self {
            cfg,
            session,
            decoder,
        })
    }

    pub fn is_linenet(&self) -> bool {
        self.session.is_linenet()
    }

    pub fn run(
        &mut self,
        imgs: &[Mat],
        return_word_box: bool,
    ) -> Result<TextRecOutput, EngineError> {
        let mut core_rec = crate::rec::TextRecognizer::with_session(
            self.cfg.clone(),
            &mut self.session,
            self.decoder.clone(),
        );
        core_rec.run(imgs, return_word_box).map_err(Into::into)
    }
}

/// Layout detector engine wrapper executing PicoDet or direct layout detection.
pub struct LayoutDetector {
    pub session: EngineSession,
    pub config: LayoutConfig,
}

impl LayoutDetector {
    pub fn new(config: LayoutConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    pub fn with_session(config: LayoutConfig, session: EngineSession) -> Self {
        Self { session, config }
    }

    pub fn with_custom_session(
        config: LayoutConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn detect(&mut self, image: &Mat) -> Result<LayoutOutput, EngineError> {
        let mut core_layout =
            crate::layout::LayoutDetector::with_session(self.config.clone(), &mut self.session);
        core_layout.detect(image).map_err(Into::into)
    }
}

/// Orientation classifier engine wrapper.
pub struct OrientClassifier {
    pub session: EngineSession,
    pub config: OrientConfig,
}

impl OrientClassifier {
    pub fn new(config: OrientConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    pub fn with_session(config: OrientConfig, session: EngineSession) -> Self {
        Self { session, config }
    }

    pub fn with_custom_session(
        config: OrientConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn classify(&mut self, image: &Mat) -> Result<OrientOutput, EngineError> {
        let mut core_orient =
            crate::orient::OrientClassifier::with_session(self.config.clone(), &mut self.session);
        core_orient.classify(image).map_err(Into::into)
    }
}

pub trait OrientationExt {
    fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError>;
}

impl OrientationExt for Orientation {
    fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError> {
        (*self).rotate_image(img).map_err(Into::into)
    }
}

/// Document unwarper engine wrapper.
pub struct DocUnwarper {
    pub session: EngineSession,
    pub config: UnwarpConfig,
}

impl DocUnwarper {
    pub fn new(config: UnwarpConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_unwarp_config(&config)?;
        Ok(Self { session, config })
    }

    pub fn with_session(config: UnwarpConfig, session: EngineSession) -> Self {
        Self { session, config }
    }

    pub fn with_custom_session(
        config: UnwarpConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn unwarp(&mut self, image: &Mat) -> Result<UnwarpOutput, EngineError> {
        let mut core_unwarp =
            crate::unwarp::DocUnwarper::with_session(self.config.clone(), &mut self.session);
        core_unwarp.unwarp(image).map_err(Into::into)
    }
}

/// RT-DETR table detector engine wrapper.
pub struct TableDetector {
    pub session: EngineSession,
    pub config: TableDetectorConfig,
}

impl TableDetector {
    pub fn new(config: TableDetectorConfig) -> Result<Self, EngineError> {
        let engine_config = EngineConfig::default();
        let session = EngineSession::from_path(&config.model_path, &engine_config)?;
        Ok(Self { session, config })
    }

    pub fn with_session(config: TableDetectorConfig, session: EngineSession) -> Self {
        Self { session, config }
    }

    pub fn with_custom_session(
        config: TableDetectorConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn detect(&mut self, img: &Mat) -> Result<Vec<crate::layout::BBox>, EngineError> {
        let mut core_table =
            crate::table::TableDetector::with_session(self.config.clone(), &mut self.session);
        core_table.detect(img).map_err(Into::into)
    }
}

/// SLANet table structure recognizer engine wrapper.
pub struct TableStructureRecognizer {
    pub session: EngineSession,
    pub config: TableStructureConfig,
}

impl TableStructureRecognizer {
    pub fn new(config: TableStructureConfig) -> Result<Self, EngineError> {
        let engine_config = EngineConfig::default();
        let session = EngineSession::from_path(&config.model_path, &engine_config)?;
        Ok(Self { session, config })
    }

    pub fn with_session(config: TableStructureConfig, session: EngineSession) -> Self {
        Self { session, config }
    }

    pub fn with_custom_session(
        config: TableStructureConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn recognize(&mut self, img: &Mat) -> Result<TableStructure, EngineError> {
        let mut core_table = crate::table::TableStructureRecognizer::with_session(
            self.config.clone(),
            &mut self.session,
        );
        core_table.recognize(img).map_err(Into::into)
    }
}

/// End-to-end document layout and OCR pipeline.
pub struct DocPipeline {
    pub layout: LayoutDetector,
    pub ocr: RustO,
    pub table_detector: Option<TableDetector>,
    pub table_recognizer: Option<TableStructureRecognizer>,
}

impl DocPipeline {
    pub fn new(config: DocPipelineConfig) -> Result<Self, EngineError> {
        let layout = LayoutDetector::new(config.layout)?;
        let ocr = RustO::initialize(config.ocr)?;
        let table_detector = config.table_detector.map(TableDetector::new).transpose()?;
        let table_recognizer = config
            .table_recognizer
            .map(TableStructureRecognizer::new)
            .transpose()?;

        Ok(Self {
            layout,
            ocr,
            table_detector,
            table_recognizer,
        })
    }

    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<DocResult, EngineError> {
        let img = imread(image_path)?;
        self.run_on_mat(&img)
    }

    pub fn run_on_mat(&mut self, img: &Mat) -> Result<DocResult, EngineError> {
        let layout_output = self.layout.detect(img)?;
        let mut regions = layout_output.regions;
        sort_layout_regions_top_down(&mut regions, 20);

        let mut blocks = Vec::with_capacity(regions.len());
        for region in regions {
            let block = match region.layout_type {
                LayoutType::Figure => DocBlock {
                    region,
                    text: None,
                    html: None,
                },
                LayoutType::Table
                    if self.table_detector.is_some() && self.table_recognizer.is_some() =>
                {
                    let html = self.process_table(img, &region).ok();
                    DocBlock {
                        region,
                        text: None,
                        html,
                    }
                }
                LayoutType::Table => DocBlock {
                    region,
                    text: None,
                    html: None,
                },
                _ => {
                    let crop = crop_bbox(img, &region.bbox)
                        .map_err(|e| EngineError::ImageError(e.to_string()))?;
                    let ocr_res = self
                        .ocr
                        .run_on_mat_with_options(&crop, &OcrRunOptions::default())?;
                    DocBlock {
                        region,
                        text: Some(ocr_res.txts.join(" ")),
                        html: None,
                    }
                }
            };
            blocks.push(block);
        }

        Ok(DocResult { blocks })
    }

    fn process_table(&mut self, img: &Mat, region: &LayoutRegion) -> Result<String, EngineError> {
        let table_crop = crop_bbox(img, &region.bbox)
            .map_err(|e| EngineError::ImageError(e.to_string()))?;
        let recognizer = self
            .table_recognizer
            .as_mut()
            .ok_or_else(|| EngineError::Preprocess("Table recognizer missing".to_string()))?;
        let mut structure = recognizer.recognize(&table_crop)?;
        let (img_h, img_w) = (table_crop.rows(), table_crop.cols());

        for row in &mut structure.rows {
            for cell in &mut row.cells {
                let cell_bbox = cell.scaled_bbox(img_w, img_h);
                if let Ok(cell_crop) = crop_bbox(&table_crop, &cell_bbox) {
                    let ocr_res = self
                        .ocr
                        .run_on_mat_with_options(&cell_crop, &OcrRunOptions::default());
                    cell.text = ocr_res.map(|r| r.txts.join(" ")).unwrap_or_default();
                } else {
                    cell.text.clear();
                }
            }
        }

        Ok(structure.to_html())
    }
}
