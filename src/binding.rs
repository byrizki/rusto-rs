use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum RustoBindingError {
    #[error("Initialization error: {msg}")]
    Initialization { msg: String },
    #[error("Preprocess error: {msg}")]
    Preprocess { msg: String },
    #[error("Inference error: {msg}")]
    Inference { msg: String },
    #[error("Postprocess error: {msg}")]
    Postprocess { msg: String },
    #[error("Invalid input: {msg}")]
    InvalidInput { msg: String },
}

impl From<crate::engine::EngineError> for RustoBindingError {
    fn from(err: crate::engine::EngineError) -> Self {
        match err {
            crate::engine::EngineError::Preprocess(msg) => RustoBindingError::Preprocess { msg },
            crate::engine::EngineError::OutputError(msg) => RustoBindingError::Postprocess { msg },
            crate::engine::EngineError::Load(err) => RustoBindingError::Initialization { msg: err.to_string() },
            crate::engine::EngineError::Run(err) => RustoBindingError::Inference { msg: err.to_string() },
            crate::engine::EngineError::Engine(msg) => RustoBindingError::Inference { msg },
            crate::engine::EngineError::ImageError(msg) => RustoBindingError::InvalidInput { msg },
            crate::engine::EngineError::InvalidInputShape => RustoBindingError::InvalidInput { msg: "Invalid input shape".into() },
            crate::engine::EngineError::ShapeError(err) => RustoBindingError::InvalidInput { msg: err.to_string() },
            #[cfg(feature = "use-opencv")]
            crate::engine::EngineError::OpenCvError(err) => RustoBindingError::Inference { msg: err.to_string() },
        }
    }
}

#[derive(uniffi::Record, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[derive(uniffi::Record, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub width: f32,
    pub height: f32,
    pub top: f32,
    pub left: f32,
}

#[derive(uniffi::Record, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextResult {
    pub text: String,
    pub score: f32,
    pub box_points: Vec<Point2D>,
    pub frame: Frame,
}

#[derive(uniffi::Enum, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputGranularity {
    #[default]
    Lines,
    Words,
    Spatial,
}

#[derive(uniffi::Enum, Clone, Debug)]
pub enum ImageSource {
    Path { path: String },
    Bytes { data: Vec<u8> },
}

#[derive(uniffi::Enum, Clone, Debug, PartialEq)]
pub enum DetectTextResult {
    Structured { results: Vec<TextResult> },
    Spatial { text: String },
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostprocessRunOptions {
    pub threshold: Option<f32>,
    pub box_threshold: Option<f32>,
    pub max_candidates: Option<i32>,
    pub unclip_ratio: Option<f32>,
    pub use_dilation: Option<bool>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionRunOptions {
    pub limit_side_len: Option<i32>,
    pub limit_type: Option<String>,
    pub mean: Option<Vec<f32>>,
    pub std: Option<Vec<f32>>,
    pub postprocess: Option<PostprocessRunOptions>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationOptions {
    pub descreen_strength: Option<f32>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizationOptions {
    pub crop_padding_x: Option<f32>,
    pub crop_padding_y: Option<f32>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrRunOptions {
    pub output: OutputGranularity,
    pub line_y_threshold: Option<f32>,
    pub word_x_threshold: Option<f32>,
    pub text_score: Option<f32>,
    pub classification: Option<bool>,
    pub orientation: Option<bool>,
    pub min_height: Option<f32>,
    pub max_side_len: Option<f32>,
    pub min_side_len: Option<f32>,
    pub width_height_ratio: Option<f32>,
    pub detection: Option<DetectionRunOptions>,
    pub postprocess: Option<PostprocessRunOptions>,
    pub calibration: Option<CalibrationOptions>,
    pub optimization: Option<OptimizationOptions>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionConfig {
    pub enabled: Option<bool>,
    pub model_path: Option<String>,
    pub thresh: Option<f32>,
    pub box_thresh: Option<f32>,
    pub unclip_ratio: Option<f32>,
    pub limit_side_len: Option<i32>,
    pub limit_type: Option<String>,
    pub use_dilation: Option<bool>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionConfig {
    pub enabled: Option<bool>,
    pub model_path: Option<String>,
    pub dict_path: Option<String>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationConfig {
    pub enabled: Option<bool>,
    pub model_path: Option<String>,
    pub thresh: Option<f32>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrientationConfig {
    pub enabled: Option<bool>,
    pub model_path: Option<String>,
    pub thresh: Option<f32>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnwarpConfig {
    pub enabled: Option<bool>,
    pub model_path: Option<String>,
}

#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeConfig {
    pub template: Option<String>,
    pub detection: Option<DetectionConfig>,
    pub recognition: Option<RecognitionConfig>,
    pub classification: Option<ClassificationConfig>,
    pub orientation: Option<OrientationConfig>,
    pub unwarp: Option<UnwarpConfig>,
}

#[derive(uniffi::Object)]
pub struct RustO {
    inner: Mutex<crate::rusto_ocr::RustO>,
}

#[uniffi::export]
impl RustO {
    #[uniffi::constructor]
    pub fn initialize(config: InitializeConfig) -> Result<Arc<Self>, RustoBindingError> {
        let internal_config = convert_init_config(&config)?;
        let ocr = crate::rusto_ocr::RustO::initialize(internal_config)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ocr),
        }))
    }

    #[uniffi::constructor]
    pub fn initialize_from_json(config_json: String) -> Result<Arc<Self>, RustoBindingError> {
        let internal_config = crate::config::InitializeConfig::from_json(&config_json)
            .map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })?;
        let ocr = crate::rusto_ocr::RustO::initialize(internal_config)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ocr),
        }))
    }

    pub fn detect_text(
        &self,
        source: ImageSource,
        options: OcrRunOptions,
    ) -> Result<DetectTextResult, RustoBindingError> {
        let internal_source = convert_source(&source)?;
        let internal_options = convert_options(&options)?;
        let mut guard = self.inner.lock().map_err(|_| RustoBindingError::Inference {
            msg: "Internal engine lock poisoned".into(),
        })?;
        let result = guard.detect_text(&internal_source, &internal_options)?;
        Ok(convert_result(result))
    }

    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

#[uniffi::export]
pub fn rusto_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[uniffi::export]
pub fn create_initialize_config_ppv6(
    det_model_path: String,
    rec_model_path: String,
    dict_path: String,
) -> InitializeConfig {
    InitializeConfig {
        template: Some("ppv6".to_string()),
        detection: Some(DetectionConfig {
            enabled: Some(true),
            model_path: Some(det_model_path),
            ..Default::default()
        }),
        recognition: Some(RecognitionConfig {
            enabled: Some(true),
            model_path: Some(rec_model_path),
            dict_path: Some(dict_path),
        }),
        ..Default::default()
    }
}

#[uniffi::export]
pub fn default_ocr_run_options() -> OcrRunOptions {
    OcrRunOptions {
        output: OutputGranularity::Lines,
        ..Default::default()
    }
}

fn convert_init_config(config: &InitializeConfig) -> Result<crate::config::InitializeConfig, RustoBindingError> {
    let json = serde_json::to_string(config)
        .map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })?;
    crate::config::InitializeConfig::from_json(&json)
        .map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })
}

fn convert_source(source: &ImageSource) -> Result<crate::rusto_ocr::ImageSource, RustoBindingError> {
    match source {
        ImageSource::Path { path } => {
            if path.trim().is_empty() {
                return Err(RustoBindingError::InvalidInput { msg: "Image path is empty".into() });
            }
            Ok(crate::rusto_ocr::ImageSource::Path(std::path::PathBuf::from(path)))
        }
        ImageSource::Bytes { data } => {
            if data.is_empty() {
                return Err(RustoBindingError::InvalidInput { msg: "Image data is empty".into() });
            }
            Ok(crate::rusto_ocr::ImageSource::Bytes(data.clone()))
        }
    }
}

fn convert_options(opts: &OcrRunOptions) -> Result<crate::rusto_ocr::OcrRunOptions, RustoBindingError> {
    let output = match opts.output {
        OutputGranularity::Lines => crate::rusto_ocr::OutputGranularity::Lines,
        OutputGranularity::Words => crate::rusto_ocr::OutputGranularity::Words,
        OutputGranularity::Spatial => crate::rusto_ocr::OutputGranularity::Spatial,
    };
    let detection = opts.detection.as_ref().map(|d| {
        let mean = d.mean.as_ref().and_then(|m| {
            if m.len() == 3 { Some([m[0], m[1], m[2]]) } else { None }
        });
        let std = d.std.as_ref().and_then(|s| {
            if s.len() == 3 { Some([s[0], s[1], s[2]]) } else { None }
        });
        let postprocess = d.postprocess.as_ref().map(|p| crate::rusto_ocr::PostprocessRunOptions {
            threshold: p.threshold,
            box_threshold: p.box_threshold,
            max_candidates: p.max_candidates,
            unclip_ratio: p.unclip_ratio,
            use_dilation: p.use_dilation,
        });
        crate::rusto_ocr::DetectionRunOptions {
            limit_side_len: d.limit_side_len,
            limit_type: d.limit_type.clone(),
            mean,
            std,
            postprocess,
        }
    });
    let postprocess = opts.postprocess.as_ref().map(|p| crate::rusto_ocr::PostprocessRunOptions {
        threshold: p.threshold,
        box_threshold: p.box_threshold,
        max_candidates: p.max_candidates,
        unclip_ratio: p.unclip_ratio,
        use_dilation: p.use_dilation,
    });
    let calibration = opts.calibration.as_ref().map(|c| crate::calibration::CalibrationOptions {
        descreen_strength: c.descreen_strength,
        ..Default::default()
    });
    let optimization = opts.optimization.as_ref().map(|o| crate::calibration::OptimizationOptions {
        crop_padding_x: o.crop_padding_x,
        crop_padding_y: o.crop_padding_y,
        ..Default::default()
    });

    let internal = crate::rusto_ocr::OcrRunOptions {
        output,
        line_y_threshold: opts.line_y_threshold,
        word_x_threshold: opts.word_x_threshold,
        text_score: opts.text_score,
        classification: opts.classification,
        orientation: opts.orientation,
        min_height: opts.min_height,
        max_side_len: opts.max_side_len,
        min_side_len: opts.min_side_len,
        width_height_ratio: opts.width_height_ratio,
        detection,
        postprocess,
        calibration,
        optimization,
    };
    internal.validate().map_err(RustoBindingError::from)?;
    Ok(internal)
}

fn convert_result(res: crate::rusto_ocr::DetectTextResult) -> DetectTextResult {
    match res {
        crate::rusto_ocr::DetectTextResult::Structured(items) => {
            DetectTextResult::Structured {
                results: items
                    .into_iter()
                    .map(|r| TextResult {
                        text: r.text,
                        score: r.score,
                        box_points: r
                            .box_points
                            .iter()
                            .map(|p| Point2D { x: p.0, y: p.1 })
                            .collect(),
                        frame: Frame {
                            width: r.frame.width,
                            height: r.frame.height,
                            top: r.frame.top,
                            left: r.frame.left,
                        },
                    })
                    .collect(),
            }
        }
        crate::rusto_ocr::DetectTextResult::Spatial(text) => DetectTextResult::Spatial { text },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniffi_binding_types() {
        let options = default_ocr_run_options();
        assert_eq!(options.output, OutputGranularity::Lines);
        let config = create_initialize_config_ppv6(
            "models/det.rten".to_string(),
            "models/rec.rten".to_string(),
            "models/dict.txt".to_string(),
        );
        assert_eq!(config.template.as_deref(), Some("ppv6"));
        assert_eq!(rusto_version(), env!("CARGO_PKG_VERSION"));
    }
}
