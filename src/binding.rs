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

/// 2D coordinate point representing vertex location in pixels.
#[derive(uniffi::Record, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    /// Horizontal coordinate (X).
    pub x: f32,
    /// Vertical coordinate (Y).
    pub y: f32,
}

/// Axis-aligned bounding box frame with position and size in pixels.
#[derive(uniffi::Record, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    /// Width of the bounding box.
    pub width: f32,
    /// Height of the bounding box.
    pub height: f32,
    /// Top vertical coordinate (Y).
    pub top: f32,
    /// Left horizontal coordinate (X).
    pub left: f32,
}

/// Recognized OCR text result containing text, confidence score, bounding polygon, and frame.
#[derive(uniffi::Record, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextResult {
    /// Recognized text string.
    pub text: String,
    /// Confidence score between 0.0 and 1.0.
    pub score: f32,
    /// Polygon vertex coordinates [top-left, top-right, bottom-right, bottom-left].
    pub box_points: Vec<Point2D>,
    /// Axis-aligned bounding box frame.
    pub frame: Frame,
}

/// Output format granularity for text detection.
#[derive(uniffi::Enum, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputGranularity {
    /// Group recognized tokens into lines (default).
    #[default]
    Lines,
    /// Return individual word-level tokens.
    Words,
    /// Reconstruct a 2D spatial text layout preserving visual arrangement.
    Spatial,
}

/// Universal image input source for OCR detection.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum ImageSource {
    /// Filesystem path to an image file.
    Path { path: String },
    /// Raw encoded image file bytes (e.g. JPEG, PNG).
    Bytes { data: Vec<u8> },
}

/// Detection result: structured text items or formatted 2D spatial text string.
#[derive(uniffi::Enum, Clone, Debug, PartialEq)]
pub enum DetectTextResult {
    /// Structured list of recognized text lines or words.
    Structured { results: Vec<TextResult> },
    /// Formatted 2D spatial layout text string.
    Spatial { text: String },
}

/// Post-processing options for DBNet polygon detection and binarization.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostprocessRunOptions {
    /// Binarization probability cutoff threshold. Default: 0.3.
    pub threshold: Option<f32>,
    /// Polygon contour score confidence threshold. Default: 0.6.
    pub box_threshold: Option<f32>,
    /// Maximum candidate text boxes to detect per image. Default: 1000.
    pub max_candidates: Option<i32>,
    /// Polygon unclip expansion ratio. Default: 1.5 - 2.0.
    pub unclip_ratio: Option<f32>,
    /// Enable morphological dilation to connect broken text strokes.
    pub use_dilation: Option<bool>,
    /// DB score calculation mode: "fast" (bounding box mean) or "slow" (polygon contour mean).
    pub score_mode: Option<String>,
}

/// Image detection stage options.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionRunOptions {
    /// Target side length constraint for image resizing. Default: 960.
    pub limit_side_len: Option<i32>,
    /// Side length limit behavior: "min" or "max".
    pub limit_type: Option<String>,
    /// Mean RGB normalization values. Default: [0.485, 0.456, 0.406].
    pub mean: Option<Vec<f32>>,
    /// Standard deviation RGB normalization values. Default: [0.229, 0.224, 0.225].
    pub std: Option<Vec<f32>>,
    /// Post-processing options for detection.
    pub postprocess: Option<PostprocessRunOptions>,
}

/// Pre-OCR image calibration options for descreening patterned backgrounds.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationOptions {
    /// Descreen filtering strength (0.0 - 1.0).
    pub descreen_strength: Option<f32>,
}

/// Runtime crop and resolution optimization options.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizationOptions {
    /// Horizontal padding around bounding box crops in pixels.
    pub crop_padding_x: Option<f32>,
    /// Vertical padding around bounding box crops in pixels.
    pub crop_padding_y: Option<f32>,
}

/// Per-request execution options for text detection.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrRunOptions {
    /// Output format granularity: Lines (default), Words, or Spatial.
    pub output: OutputGranularity,
    /// Vertical center tolerance multiplier for same-line grouping. Default: 0.5.
    pub line_y_threshold: Option<f32>,
    /// Horizontal gap tolerance multiplier for same-word grouping. Default: 0.4.
    pub word_x_threshold: Option<f32>,
    /// Minimum recognition confidence score threshold [0.0 - 1.0].
    pub text_score: Option<f32>,
    /// Enable text line orientation classifier stage.
    pub classification: Option<bool>,
    /// Enable document orientation angle correction stage.
    pub orientation: Option<bool>,
    /// Request-local minimum height resize override.
    pub min_height: Option<f32>,
    /// Request-local maximum side length resize override.
    pub max_side_len: Option<f32>,
    /// Request-local minimum side length resize override.
    pub min_side_len: Option<f32>,
    /// Width-to-height aspect ratio padding override.
    pub width_height_ratio: Option<f32>,
    /// Detection stage options.
    pub detection: Option<DetectionRunOptions>,
    /// Post-processing options.
    pub postprocess: Option<PostprocessRunOptions>,
    /// Pre-OCR background descreening calibration options.
    pub calibration: Option<CalibrationOptions>,
    /// Runtime crop and resolution optimization options.
    pub optimization: Option<OptimizationOptions>,
}

/// Text detection model configuration.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionConfig {
    /// Whether detection stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to detection model (.rten or .onnx).
    pub model_path: Option<String>,
    /// Segmentation binarization threshold.
    pub thresh: Option<f32>,
    /// Box candidate confidence score threshold.
    pub box_thresh: Option<f32>,
    /// Polygon unclip expansion ratio.
    pub unclip_ratio: Option<f32>,
    /// Maximum side length resize target.
    pub limit_side_len: Option<i32>,
    /// Limit type: "min" or "max".
    pub limit_type: Option<String>,
    /// Enable morphological dilation.
    pub use_dilation: Option<bool>,
    /// DB score calculation mode: "fast" or "slow".
    pub score_mode: Option<String>,
}

/// Text recognition model configuration.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecognitionConfig {
    /// Whether recognition stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to recognition model (.rten or .onnx).
    pub model_path: Option<String>,
    /// Filesystem path to dictionary vocabulary file (.txt).
    pub dict_path: Option<String>,
}

/// Line classification configuration (180° rotation detection).
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationConfig {
    /// Whether classification stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to classification model.
    pub model_path: Option<String>,
    /// Confidence threshold.
    pub thresh: Option<f32>,
}

/// Document orientation angle configuration (0°, 90°, 180°, 270°).
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrientationConfig {
    /// Whether orientation stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to orientation model.
    pub model_path: Option<String>,
    /// Confidence threshold.
    pub thresh: Option<f32>,
}

/// Document dewarping model configuration.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnwarpConfig {
    /// Whether unwarp stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to unwarp model.
    pub model_path: Option<String>,
}

/// Engine initialization configuration specifying model paths and template preset.
#[derive(uniffi::Record, Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeConfig {
    /// Template preset identifier (e.g. "ppv6", "ppv5", "ppv4", "ppv3").
    pub template: Option<String>,
    /// Detection model configuration.
    pub detection: Option<DetectionConfig>,
    /// Recognition model and dictionary configuration.
    pub recognition: Option<RecognitionConfig>,
    /// Optional classification configuration.
    pub classification: Option<ClassificationConfig>,
    /// Optional orientation configuration.
    pub orientation: Option<OrientationConfig>,
    /// Optional unwarp configuration.
    pub unwarp: Option<UnwarpConfig>,
}

/// High-level RustO OCR Engine for multi-platform bindings.
#[derive(uniffi::Object)]
pub struct RustO {
    inner: Mutex<crate::rusto_ocr::RustO>,
}

#[uniffi::export]
impl RustO {
    /// Initializes an OCR engine instance with structured InitializeConfig.
    #[uniffi::constructor]
    pub fn initialize(config: InitializeConfig) -> Result<Arc<Self>, RustoBindingError> {
        let internal_config = convert_init_config(&config)?;
        let ocr = crate::rusto_ocr::RustO::initialize(internal_config)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ocr),
        }))
    }

    /// Initializes an OCR engine instance from a JSON configuration string.
    #[uniffi::constructor]
    pub fn initialize_from_json(config_json: String) -> Result<Arc<Self>, RustoBindingError> {
        let internal_config = crate::config::InitializeConfig::from_json(&config_json)
            .map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })?;
        let ocr = crate::rusto_ocr::RustO::initialize(internal_config)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ocr),
        }))
    }

    /// Runs text detection and recognition on an image source.
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

    /// Runs text detection and recognition on raw decoded interleaved 24-bit RGB pixel bytes.
    pub fn detect_text_rgb(
        &self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        stride: u64,
        options: OcrRunOptions,
    ) -> Result<DetectTextResult, RustoBindingError> {
        let internal_options = convert_options(&options)?;
        let mut guard = self.inner.lock().map_err(|_| RustoBindingError::Inference {
            msg: "Internal engine lock poisoned".into(),
        })?;
        let stride_usize = usize::try_from(stride).map_err(|_| RustoBindingError::InvalidInput {
            msg: "stride exceeds platform address space".into(),
        })?;
        let result = guard.detect_text_rgb(&pixels, width, height, stride_usize, &internal_options)?;
        Ok(convert_result(result))
    }

    /// Returns the engine library version string.
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

/// Returns the engine library version string.
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
            score_mode: p.score_mode.clone(),
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
        score_mode: p.score_mode.clone(),
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

#[no_mangle]
pub unsafe extern "C" fn ffi_uniffi_rusto_rustbuffer_from_bytes(
    bytes: uniffi::ForeignBytes,
    out_status: &mut uniffi::RustCallStatus,
) -> uniffi::RustBuffer {
    uniffi::uniffi_rustbuffer_from_bytes(bytes, out_status)
}

#[no_mangle]
pub unsafe extern "C" fn ffi_uniffi_rusto_rustbuffer_free(
    buf: uniffi::RustBuffer,
    out_status: &mut uniffi::RustCallStatus,
) {
    uniffi::uniffi_rustbuffer_free(buf, out_status)
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
