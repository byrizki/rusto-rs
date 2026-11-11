use std::sync::{Arc, Mutex};
use nanoserde::{DeJson, SerJson};

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
            crate::engine::EngineError::Engine(msg) => RustoBindingError::Inference { msg },
            crate::engine::EngineError::ImageError(msg) => RustoBindingError::InvalidInput { msg },
            crate::engine::EngineError::InvalidInputShape => RustoBindingError::InvalidInput { msg: "Invalid input shape".into() },
            crate::engine::EngineError::ShapeError(err) => RustoBindingError::InvalidInput { msg: err.to_string() },
        }
    }
}

impl From<rusto_core::CoreError> for RustoBindingError {
    fn from(err: rusto_core::CoreError) -> Self {
        crate::engine::EngineError::from(err).into()
    }
}

/// 2D coordinate point representing vertex location in pixels.
#[derive(uniffi::Record, Clone, Copy, Debug, PartialEq, DeJson, SerJson)]
pub struct Point2D {
    /// Horizontal coordinate (X).
    pub x: f32,
    /// Vertical coordinate (Y).
    pub y: f32,
}

/// Axis-aligned bounding box frame with position and size in pixels.
#[derive(uniffi::Record, Clone, Copy, Debug, PartialEq, DeJson, SerJson)]
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
#[derive(uniffi::Record, Clone, Debug, PartialEq, DeJson, SerJson)]
pub struct TextResult {
    /// Recognized text string.
    pub text: String,
    /// Confidence score between 0.0 and 1.0.
    pub score: f32,
    /// Polygon vertex coordinates [top-left, top-right, bottom-right, bottom-left].
    pub box_points: Vec<Point2D>,
    /// Axis-aligned bounding box frame.
    pub frame: Frame,
    /// Detected document layout region index (if layout analysis enabled).
    pub region_id: Option<i32>,
    /// Detected semantic layout category (if layout analysis enabled).
    pub layout_type: Option<LayoutType>,
}

/// Output format granularity for text detection.
#[derive(uniffi::Enum, Clone, Copy, Debug, Default, PartialEq, Eq, DeJson, SerJson)]
pub enum OutputGranularity {
    /// Group recognized tokens into lines (default).
    #[default]
    Lines,
    /// Return individual word-level tokens.
    Words,
    /// Reconstruct a 2D spatial text layout preserving visual arrangement.
    Spatial,
    /// Export formatted TSV (tab-separated values).
    Tsv,
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

/// Pre-detection image conditioning and resolution optimization options.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct PreDetectionOptions {
    /// Descreen filtering strength (0.0 - 1.0).
    pub descreen_strength: Option<f32>,
    /// Optimal target maximum side dimension for downscaling (e.g. 960 or 1024).
    pub target_max_side: Option<u32>,
    /// Horizontal padding around bounding box crops in pixels.
    pub crop_padding_x: Option<f32>,
    /// Vertical padding around bounding box crops in pixels.
    pub crop_padding_y: Option<f32>,
}

/// Image detection stage options.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct DetectionRunOptions {
    /// Target side length constraint for image resizing. Default: 960.
    pub limit_side_len: Option<i32>,
    /// Side length limit behavior: "min" or "max".
    pub limit_type: Option<String>,
    /// Mean RGB normalization values. Default: [0.485, 0.456, 0.406].
    pub mean: Option<Vec<f32>>,
    /// Standard deviation RGB normalization values. Default: [0.229, 0.224, 0.225].
    pub std: Option<Vec<f32>>,
}

/// Post-detection polygon extraction, thresholding, and box filtering options.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct PostDetectionOptions {
    /// Binarization probability cutoff threshold. Default: 0.3.
    pub threshold: Option<f32>,
    /// Polygon contour score confidence threshold. Default: 0.6.
    pub box_threshold: Option<f32>,
    /// Polygon unclip expansion ratio. Default: 1.5 - 2.0.
    pub unclip_ratio: Option<f32>,
    /// Enable morphological dilation to connect broken text strokes.
    pub use_dilation: Option<bool>,
    /// DB score calculation mode: "fast" (bounding box mean) or "slow" (polygon contour mean).
    pub score_mode: Option<String>,
    /// Maximum candidate text boxes to detect per image. Default: 1000.
    pub max_candidates: Option<i32>,
    /// Minimum box pixel height filter.
    pub min_height: Option<f32>,
    /// Minimum box side length filter.
    pub min_side_len: Option<f32>,
    /// Maximum box side length filter.
    pub max_side_len: Option<f32>,
    /// Width-to-height aspect ratio padding override.
    pub width_height_ratio: Option<f32>,
}

/// Pre-recognition text line/word clustering tolerances.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct PreRecognitionOptions {
    /// Vertical center tolerance multiplier for same-line grouping. Default: 0.5.
    pub line_y_threshold: Option<f32>,
    /// Horizontal gap tolerance multiplier for same-word grouping. Default: 0.4.
    pub word_x_threshold: Option<f32>,
}

/// Text recognition inference parameters.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct RecognitionRunOptions {
    /// Recognition batch size. Default: 10.
    pub batch_size: Option<i32>,
}

/// Post-recognition confidence score filtering.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct PostRecognitionOptions {
    /// Minimum recognition confidence score threshold [0.0 - 1.0].
    pub score_threshold: Option<f32>,
}

/// Per-request execution options for text detection.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct OcrRunOptions {
    /// Output format granularity: Lines (default), Words, or Spatial.
    pub output: OutputGranularity,
    /// Enable document unwarping stage.
    pub auto_unwarp: Option<bool>,
    /// Enable document layout analysis and reading order stage.
    pub layout_reading: Option<bool>,
    /// Document layout or reading direction model identifier.
    pub layout_model: Option<String>,
    /// Enable document orientation angle correction stage.
    pub orientation_detection: Option<bool>,
    /// Enable text line orientation classifier stage.
    pub line_classification: Option<bool>,
    /// Pre-detection calibration & optimization options.
    pub pre_detection: Option<PreDetectionOptions>,
    /// Detection stage options.
    pub detection: Option<DetectionRunOptions>,
    /// Post-detection binarization, expansion, and filtering options.
    pub post_detection: Option<PostDetectionOptions>,
    /// Pre-recognition clustering options.
    pub pre_recognition: Option<PreRecognitionOptions>,
    /// Recognition stage options.
    pub recognition: Option<RecognitionRunOptions>,
    /// Post-recognition filtering options.
    pub post_recognition: Option<PostRecognitionOptions>,
}

/// Text detection model configuration.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
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
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct RecognitionConfig {
    /// Whether recognition stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to recognition model (.rten or .onnx).
    pub model_path: Option<String>,
    /// Filesystem path to dictionary vocabulary file (.txt).
    pub dict_path: Option<String>,
    /// Recognition batch size. Default: 6.
    pub rec_batch_num: Option<i32>,
}

/// Line classification configuration (180° rotation detection).
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct ClassificationConfig {
    /// Whether classification stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to classification model.
    pub model_path: Option<String>,
    /// Confidence threshold.
    pub thresh: Option<f32>,
}

/// Document orientation angle configuration (0°, 90°, 180°, 270°).
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct OrientationConfig {
    /// Whether orientation stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to orientation model.
    pub model_path: Option<String>,
    /// Confidence threshold.
    pub thresh: Option<f32>,
}

/// Document dewarping model configuration.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct UnwarpConfig {
    /// Whether unwarp stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to unwarp model.
    pub model_path: Option<String>,
}

/// Category of document layout region.
#[derive(uniffi::Enum, Clone, Copy, Debug, PartialEq, Eq, DeJson, SerJson)]
pub enum LayoutType {
    Text,
    Title,
    Figure,
    FigureCaption,
    Table,
    TableCaption,
    Header,
    Footer,
    Reference,
    Equation,
}

/// Document layout analysis model configuration.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct LayoutConfig {
    /// Whether layout stage is enabled.
    pub enabled: Option<bool>,
    /// Filesystem path to layout model (.rten or .onnx).
    pub model_path: Option<String>,
    /// Confidence threshold.
    pub conf_thresh: Option<f32>,
    /// Target input side length for resizing.
    pub target_size: Option<i32>,
    /// Target input height for resizing.
    pub target_height: Option<i32>,
    /// Target input width for resizing.
    pub target_width: Option<i32>,
}

/// Engine initialization configuration specifying model paths, preset, and optional extensions.
#[derive(uniffi::Record, Clone, Debug, Default, DeJson, SerJson)]
pub struct InitializeConfig {
    /// Model identifier or directory (e.g. "ppocrv6-tiny", "ppocrv6-tiny-fp16", "ppocrv6-small", "indraocr", "ppocrv5", "ppocrv4"). Default: "ppocrv6-tiny".
    pub model: Option<String>,
    /// Base model directory containing det, rec, and dict files.
    pub model_path: Option<String>,
    /// Legacy template preset identifier (e.g. "ppv6", "ppv5", "ppv4", "ppv3").
    pub template: Option<String>,
    /// Language code or name (e.g. "ja", "ko", "ar", "cyrillic", "devanagari", "latin").
    pub lang: Option<String>,
    /// Custom cache directory override path.
    pub cache_dir: Option<String>,
    /// Enable neural reading order recovery transformer.
    pub enable_reading_order: Option<bool>,
    /// Document layout variant: "cdla" (default 10-class) or "publaynet" (5-class).
    pub layout_variant: Option<String>,
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
    /// Optional layout configuration.
    pub layout: Option<LayoutConfig>,
}

/// High-level RustO OCR Engine for multi-platform bindings.
#[derive(uniffi::Object)]
pub struct RustO {
    inner: Mutex<crate::engine::RustO>,
}

#[uniffi::export]
impl RustO {
    /// Initializes an OCR engine instance with structured InitializeConfig.
    #[uniffi::constructor]
    pub fn initialize(config: InitializeConfig) -> Result<Arc<Self>, RustoBindingError> {
        crate::ensure_rten_registered();
        let internal_config = convert_init_config(&config)?;
        let ocr = crate::engine::RustO::initialize(internal_config)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ocr),
        }))
    }

    /// Initializes an OCR engine instance from a JSON configuration string.
    #[uniffi::constructor]
    pub fn initialize_from_json(config_json: String) -> Result<Arc<Self>, RustoBindingError> {
        crate::ensure_rten_registered();
        let internal_config = crate::download::parse_config_with_auto_download(&config_json)
            .map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })?;
        let ocr = crate::engine::RustO::initialize(internal_config)?;
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

    /// Returns whether the engine is ready for inference.
    pub fn is_ready(&self) -> bool {
        self.inner.lock().is_ok()
    }
}

/// Returns the engine library version string.
#[uniffi::export]
pub fn rusto_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Creates an `InitializeConfig` using the PP-OCRv6 preset.
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
            rec_batch_num: None,
        }),
        ..Default::default()
    }
}

/// Creates an `InitializeConfig` using the Indra OCR preset with `rec_batch_num = 1` default.
#[uniffi::export]
pub fn create_initialize_config_indra(
    det_model_path: String,
    rec_model_path: String,
    dict_path: String,
) -> InitializeConfig {
    InitializeConfig {
        template: Some("indra".to_string()),
        detection: Some(DetectionConfig {
            enabled: Some(true),
            model_path: Some(det_model_path),
            ..Default::default()
        }),
        recognition: Some(RecognitionConfig {
            enabled: Some(true),
            model_path: Some(rec_model_path),
            dict_path: Some(dict_path),
            rec_batch_num: Some(1),
        }),
        ..Default::default()
    }
}

#[uniffi::export]
pub fn create_initialize_config_for_model(
    model: Option<String>,
    lang: Option<String>,
    model_path: Option<String>,
) -> InitializeConfig {
    InitializeConfig {
        model,
        lang,
        model_path,
        ..Default::default()
    }
}

/// Downloads model assets for a specific OCR pipeline ahead of time.
#[uniffi::export]
pub fn download_model(
    model: Option<String>,
    lang: Option<String>,
    cache_dir: Option<String>,
) -> Result<Vec<String>, RustoBindingError> {
    let cache_path = rusto_core::models::get_cache_dir(cache_dir.as_deref().map(std::path::Path::new));
    let downloaded = crate::download::download_model_pipeline(
        model.as_deref(),
        lang.as_deref(),
        false,
        false,
        false,
        None,
        false,
        &cache_path,
    ).map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })?;
    Ok(downloaded.into_iter().map(|p| p.to_string_lossy().to_string()).collect())
}

#[uniffi::export]
pub fn default_ocr_run_options() -> OcrRunOptions {
    OcrRunOptions {
        output: OutputGranularity::Lines,
        ..Default::default()
    }
}

fn convert_init_config(config: &InitializeConfig) -> Result<rusto_core::config::InitializeConfig, RustoBindingError> {
    let json = nanoserde::SerJson::serialize_json(config);
    crate::download::parse_config_with_auto_download(&json)
        .map_err(|e| RustoBindingError::Initialization { msg: e.to_string() })
}

fn convert_source(source: &ImageSource) -> Result<rusto_core::types::ImageSource, RustoBindingError> {
    match source {
        ImageSource::Path { path } => {
            if path.trim().is_empty() {
                return Err(RustoBindingError::InvalidInput { msg: "Image path is empty".into() });
            }
            Ok(rusto_core::types::ImageSource::Path(std::path::PathBuf::from(path)))
        }
        ImageSource::Bytes { data } => {
            if data.is_empty() {
                return Err(RustoBindingError::InvalidInput { msg: "Image data is empty".into() });
            }
            Ok(rusto_core::types::ImageSource::Bytes(data.clone()))
        }
    }
}

fn convert_options(opts: &OcrRunOptions) -> Result<rusto_core::types::OcrRunOptions, RustoBindingError> {
    let output = match opts.output {
        OutputGranularity::Lines => rusto_core::types::OutputGranularity::Lines,
        OutputGranularity::Words => rusto_core::types::OutputGranularity::Words,
        OutputGranularity::Spatial => rusto_core::types::OutputGranularity::Spatial,
        OutputGranularity::Tsv => rusto_core::types::OutputGranularity::Tsv,
    };
    let pre_detection = opts.pre_detection.as_ref().map(|p| rusto_core::types::PreDetectionOptions {
        descreen_strength: p.descreen_strength,
        target_max_side: p.target_max_side,
        crop_padding_x: p.crop_padding_x,
        crop_padding_y: p.crop_padding_y,
    });
    let detection = opts.detection.as_ref().map(|d| {
        let mean = d.mean.as_ref().and_then(|m| {
            if m.len() == 3 { Some([m[0], m[1], m[2]]) } else { None }
        });
        let std = d.std.as_ref().and_then(|s| {
            if s.len() == 3 { Some([s[0], s[1], s[2]]) } else { None }
        });
        rusto_core::types::DetectionRunOptions {
            limit_side_len: d.limit_side_len,
            limit_type: d.limit_type.clone(),
            mean,
            std,
            postprocess: None,
        }
    });
    let post_detection = opts.post_detection.as_ref().map(|p| rusto_core::types::PostDetectionOptions {
        threshold: p.threshold,
        box_threshold: p.box_threshold,
        unclip_ratio: p.unclip_ratio,
        use_dilation: p.use_dilation,
        score_mode: p.score_mode.clone(),
        max_candidates: p.max_candidates,
        min_height: p.min_height,
        min_side_len: p.min_side_len,
        max_side_len: p.max_side_len,
        width_height_ratio: p.width_height_ratio,
    });
    let pre_recognition = opts.pre_recognition.as_ref().map(|p| rusto_core::types::PreRecognitionOptions {
        line_y_threshold: p.line_y_threshold,
        word_x_threshold: p.word_x_threshold,
    });
    let recognition = opts.recognition.as_ref().map(|r| rusto_core::types::RecognitionRunOptions {
        batch_size: r.batch_size,
    });
    let post_recognition = opts.post_recognition.as_ref().map(|p| rusto_core::types::PostRecognitionOptions {
        score_threshold: p.score_threshold,
    });

    let internal = rusto_core::types::OcrRunOptions {
        output,
        auto_unwarp: opts.auto_unwarp,
        layout_reading: opts.layout_reading,
        layout_model: opts.layout_model.clone(),
        orientation_detection: opts.orientation_detection,
        line_classification: opts.line_classification,
        pre_detection,
        detection,
        post_detection,
        pre_recognition,
        recognition,
        post_recognition,
    };
    internal.validate().map_err(RustoBindingError::from)?;
    Ok(internal)
}

fn convert_result(res: rusto_core::types::DetectTextResult) -> DetectTextResult {
    match res {
        rusto_core::types::DetectTextResult::Structured(items) => {
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
                        region_id: r.region_id,
                        layout_type: r.layout_type.map(convert_layout_type),
                    })
                    .collect(),
            }
        }
        rusto_core::types::DetectTextResult::Spatial(text) => DetectTextResult::Spatial { text },
    }
}

fn convert_layout_type(t: rusto_core::layout::LayoutType) -> LayoutType {
    match t {
        rusto_core::layout::LayoutType::Text => LayoutType::Text,
        rusto_core::layout::LayoutType::Title => LayoutType::Title,
        rusto_core::layout::LayoutType::Figure => LayoutType::Figure,
        rusto_core::layout::LayoutType::FigureCaption => LayoutType::FigureCaption,
        rusto_core::layout::LayoutType::Table => LayoutType::Table,
        rusto_core::layout::LayoutType::TableCaption => LayoutType::TableCaption,
        rusto_core::layout::LayoutType::Header => LayoutType::Header,
        rusto_core::layout::LayoutType::Footer => LayoutType::Footer,
        rusto_core::layout::LayoutType::Reference => LayoutType::Reference,
        rusto_core::layout::LayoutType::Equation => LayoutType::Equation,
    }
}

/// # Safety
/// Construct RustBuffer from foreign bytes. Caller must guarantee bytes validity.
#[no_mangle]
pub unsafe extern "C" fn ffi_uniffi_rusto_rustbuffer_from_bytes(
    bytes: uniffi::ForeignBytes,
    out_status: &mut uniffi::RustCallStatus,
) -> uniffi::RustBuffer {
    uniffi::uniffi_rustbuffer_from_bytes(bytes, out_status)
}

/// # Safety
/// Free RustBuffer previously allocated by UniFFI.
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
        let mut options = default_ocr_run_options();
        assert_eq!(options.output, OutputGranularity::Lines);
        options.recognition = Some(RecognitionRunOptions { batch_size: Some(12) });
        options.pre_detection = Some(PreDetectionOptions {
            descreen_strength: None,
            target_max_side: Some(960),
            crop_padding_x: Some(0.05),
            crop_padding_y: Some(0.08),
        });
        let converted = convert_options(&options).expect("convert_options should succeed");
        assert_eq!(converted.recognition.as_ref().unwrap().batch_size, Some(12));
        let opt = converted.pre_detection.expect("pre_detection should be present");
        assert_eq!(opt.target_max_side, Some(960));
        assert_eq!(opt.crop_padding_x, Some(0.05));
        assert_eq!(opt.crop_padding_y, Some(0.08));

        let config = create_initialize_config_ppv6(
            "models/det.rten".to_string(),
            "models/rec.rten".to_string(),
            "models/dict.txt".to_string(),
        );
        assert_eq!(config.template.as_deref(), Some("ppv6"));
        assert_eq!(rusto_version(), env!("CARGO_PKG_VERSION"));

        let model_cfg = create_initialize_config_for_model(
            None,
            None,
            Some("models/PPOCR_v6_tiny".to_string()),
        );
        let converted_init = convert_init_config(&model_cfg).expect("convert_init_config should succeed");
        let det_str = converted_init.det.model_path.to_string_lossy();
        let rec_str = converted_init.rec.model_path.to_string_lossy();
        assert!(det_str.starts_with("models/PPOCR_v6_tiny/det."));
        assert!(rec_str.starts_with("models/PPOCR_v6_tiny/rec."));
        assert_eq!(converted_init.rec.rec_keys_path, Some(std::path::PathBuf::from("models/PPOCR_v6_tiny/dict.txt")));
    }
}
