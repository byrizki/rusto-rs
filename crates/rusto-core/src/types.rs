use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::calibration::{CalibrationOptions, OptimizationOptions};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub width: f32,
    pub height: f32,
    pub top: f32,
    pub left: f32,
}

impl Frame {
    pub fn new(width: f32, height: f32, top: f32, left: f32) -> Self {
        Self {
            width,
            height,
            top,
            left,
        }
    }

    pub fn from_points(points: &[(f32, f32); 4]) -> Self {
        let min_x = points.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let max_x = points.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let min_y = points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let max_y = points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        Self {
            width: max_x - min_x,
            height: max_y - min_y,
            top: min_y,
            left: min_x,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LangRec {
    Ch,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OcrVersion {
    PpOcrV3,
    PpOcrV4,
    PpOcrV5,
    PpOcrV6,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EngineType {
    OnnxRuntime,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    Mobile,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    Det,
    Cls,
    Rec,
    Orient,
    Unwarp,
    Layout,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    pub intra_op_num_threads: i32,
    pub inter_op_num_threads: i32,
    pub enable_cpu_mem_arena: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            intra_op_num_threads: 4,
            inter_op_num_threads: 1,
            enable_cpu_mem_arena: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputGranularity {
    #[default]
    Lines,
    Words,
    Spatial,
    Tsv,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PostprocessRunOptions {
    pub threshold: Option<f32>,
    pub box_threshold: Option<f32>,
    pub max_candidates: Option<i32>,
    pub unclip_ratio: Option<f32>,
    pub use_dilation: Option<bool>,
    pub score_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DetectionRunOptions {
    pub limit_side_len: Option<i32>,
    pub limit_type: Option<String>,
    pub mean: Option<[f32; 3]>,
    pub std: Option<[f32; 3]>,
    pub postprocess: Option<PostprocessRunOptions>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetConfig {
    pub engine_type: EngineType,
    pub lang_type: LangRec,
    pub model_type: ModelType,
    pub ocr_version: OcrVersion,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub limit_side_len: i32,
    pub limit_type: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: i32,
    pub unclip_ratio: f32,
    pub use_dilation: bool,
    pub score_mode: String,
    pub engine_cfg: EngineConfig,
}

impl DetConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self::ppv6(model_path)
    }

    pub fn ppv6(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV6,
            task_type: TaskType::Det,
            model_path,
            limit_side_len: 736,
            limit_type: "min".to_string(),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            thresh: 0.3,
            box_thresh: 0.6,
            max_candidates: 1000,
            unclip_ratio: 2.0,
            use_dilation: true,
            score_mode: "slow".to_string(),
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv5(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV5,
            task_type: TaskType::Det,
            model_path,
            limit_side_len: 736,
            limit_type: "min".to_string(),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            thresh: 0.3,
            box_thresh: 0.5,
            max_candidates: 1000,
            unclip_ratio: 2.0,
            use_dilation: true,
            score_mode: "slow".to_string(),
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv4(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV4,
            task_type: TaskType::Det,
            model_path,
            limit_side_len: 736,
            limit_type: "min".to_string(),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            thresh: 0.3,
            box_thresh: 0.5,
            max_candidates: 1000,
            unclip_ratio: 1.6,
            use_dilation: true,
            score_mode: "slow".to_string(),
            engine_cfg: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecConfig {
    pub engine_type: EngineType,
    pub lang_type: LangRec,
    pub model_type: ModelType,
    pub ocr_version: OcrVersion,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub rec_keys_path: Option<PathBuf>,
    pub rec_img_shape: [i32; 3],
    pub rec_batch_num: i32,
    pub engine_cfg: EngineConfig,
}

impl RecConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self::ppv6(model_path, None)
    }

    pub fn ppv6(model_path: PathBuf, rec_keys_path: Option<PathBuf>) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV6,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv5(model_path: PathBuf, rec_keys_path: Option<PathBuf>) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV5,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv4(model_path: PathBuf, rec_keys_path: Option<PathBuf>) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV4,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
            engine_cfg: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClsConfig {
    pub engine_type: EngineType,
    pub model_type: ModelType,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub cls_image_shape: [i32; 3],
    pub cls_batch_num: i32,
    pub cls_thresh: f32,
    pub label_list: Vec<String>,
    pub engine_cfg: EngineConfig,
}

impl ClsConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            model_type: ModelType::Mobile,
            task_type: TaskType::Cls,
            model_path,
            cls_image_shape: [3, 48, 192],
            cls_batch_num: 6,
            cls_thresh: 0.9,
            label_list: vec!["0".to_string(), "180".to_string()],
            engine_cfg: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrientConfig {
    pub engine_type: EngineType,
    pub model_type: ModelType,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub orient_image_shape: [i32; 3],
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub confidence_threshold: f32,
    pub orient_batch_num: i32,
    pub orient_thresh: f32,
    pub engine_cfg: EngineConfig,
}

impl OrientConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            model_type: ModelType::Mobile,
            task_type: TaskType::Orient,
            model_path,
            orient_image_shape: [3, 224, 224],
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            confidence_threshold: 0.9,
            orient_batch_num: 1,
            orient_thresh: 0.9,
            engine_cfg: EngineConfig::default(),
        }
    }
}

impl From<ClsConfig> for OrientConfig {
    fn from(cls_cfg: ClsConfig) -> Self {
        Self {
            engine_type: cls_cfg.engine_type,
            model_type: cls_cfg.model_type,
            task_type: cls_cfg.task_type,
            model_path: cls_cfg.model_path,
            orient_image_shape: cls_cfg.cls_image_shape,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            confidence_threshold: cls_cfg.cls_thresh,
            orient_batch_num: cls_cfg.cls_batch_num,
            orient_thresh: cls_cfg.cls_thresh,
            engine_cfg: cls_cfg.engine_cfg,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnwarpConfig {
    pub engine_type: EngineType,
    pub model_type: ModelType,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub unwarp_image_shape: [i32; 3],
    pub engine_cfg: EngineConfig,
}

impl UnwarpConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            model_type: ModelType::Mobile,
            task_type: TaskType::Unwarp,
            model_path,
            unwarp_image_shape: [3, 512, 512],
            engine_cfg: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayoutConfig {
    pub engine_type: EngineType,
    pub model_type: ModelType,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub conf_thresh: f32,
    pub iou_thresh: f32,
    pub target_size: i32,
    #[serde(default)]
    pub target_height: Option<i32>,
    #[serde(default)]
    pub target_width: Option<i32>,
    pub engine_cfg: EngineConfig,
}

impl LayoutConfig {
    pub fn default(model_path: PathBuf) -> Self {
        let p_str = model_path.to_string_lossy();
        let (target_height, target_width) = if p_str.contains("doclayout_s") {
            (480, 480)
        } else if p_str.contains("doclayout_l") || p_str.contains("doclayout_plus_l") {
            (800, 800)
        } else if p_str.contains("yolo") {
            (640, 640)
        } else {
            (800, 608)
        };
        Self {
            engine_type: EngineType::OnnxRuntime,
            model_type: ModelType::Mobile,
            task_type: TaskType::Layout,
            model_path,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            conf_thresh: 0.4,
            iou_thresh: 0.5,
            target_size: target_height,
            target_height: Some(target_height),
            target_width: Some(target_width),
            engine_cfg: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub text_score: f32,
    pub use_det: bool,
    pub use_cls: bool,
    pub use_rec: bool,
    pub use_orient: bool,
    pub use_unwarp: bool,
    pub use_layout: bool,
    pub reading_order: bool,
    pub reading_order_model: Option<PathBuf>,
    pub orient_config: Option<OrientConfig>,
    pub unwarp_config: Option<UnwarpConfig>,
    pub layout_config: Option<LayoutConfig>,
    pub debug_images: bool,
    pub min_height: f32,
    pub width_height_ratio: f32,
    pub max_side_len: f32,
    pub min_side_len: f32,
    pub return_word_box: bool,
    pub return_single_char_box: bool,
    pub y_threshold_multiplier: Option<f32>,
    pub x_threshold_multiplier: Option<f32>,
    pub calibration: Option<CalibrationOptions>,
    pub optimization: Option<OptimizationOptions>,
}

impl GlobalConfig {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        for (name, value) in [
            ("minHeight", self.min_height),
            ("maxSideLen", self.max_side_len),
            ("minSideLen", self.min_side_len),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(crate::error::CoreError::Preprocess(format!("{name} must be finite and greater than zero")));
            }
        }
        if self.min_side_len > self.max_side_len {
            return Err(crate::error::CoreError::Preprocess("minSideLen must be less than or equal to maxSideLen".into()));
        }
        Ok(())
    }
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            text_score: 0.5,
            use_det: true,
            use_cls: false,
            use_rec: true,
            use_orient: false,
            use_unwarp: false,
            use_layout: false,
            reading_order: false,
            reading_order_model: None,
            orient_config: None,
            unwarp_config: None,
            layout_config: None,
            debug_images: false,
            min_height: 30.0,
            width_height_ratio: 8.0,
            max_side_len: 2000.0,
            min_side_len: 30.0,
            return_word_box: false,
            return_single_char_box: false,
            y_threshold_multiplier: None,
            x_threshold_multiplier: None,
            calibration: None,
            optimization: None,
        }
    }
}


/// OCR text result with bounding box and frame
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextResult {
    pub text: String,
    pub score: f32,
    /// Box points: [top-left, top-right, bottom-right, bottom-left]
    pub box_points: [(f32, f32); 4],
    /// Axis-aligned bounding box frame
    pub frame: Frame,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_type: Option<crate::layout::LayoutType>,
}

impl TextResult {
    pub fn from_coords(x0: f32, y0: f32, x1: f32, y1: f32, text: String, score: f32) -> Self {
        let box_w = x1 - x0;
        let box_h = y1 - y0;
        Self {
            text,
            score,
            box_points: [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            frame: Frame {
                left: (x0 * 100.0).round() / 100.0,
                top: (y0 * 100.0).round() / 100.0,
                width: (box_w * 100.0).round() / 100.0,
                height: (box_h * 100.0).round() / 100.0,
            },
            region_id: None,
            layout_type: None,
        }
    }

    pub fn from_box_points(box_points: [(f32, f32); 4], text: String, score: f32) -> Self {
        Self {
            text,
            score,
            frame: Frame::from_points(&box_points),
            box_points,
            region_id: None,
            layout_type: None,
        }
    }
}

/// Image input accepted by canonical OCR APIs
#[derive(Clone, Debug)]
pub enum ImageSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

/// Result returned by the canonical `detect_text` API
#[derive(Clone, Debug, PartialEq)]
pub enum DetectTextResult {
    Structured(Vec<TextResult>),
    Spatial(String),
}

/// Pre-detection image conditioning and resolution optimization options.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PreDetectionOptions {
    pub descreen_strength: Option<f32>,
    pub target_max_side: Option<u32>,
    pub crop_padding_x: Option<f32>,
    pub crop_padding_y: Option<f32>,
}

impl PreDetectionOptions {
    pub fn to_calibration_options(&self) -> Option<CalibrationOptions> {
        self.descreen_strength.map(|strength| CalibrationOptions {
            enabled: Some(true),
            descreen_strength: Some(strength),
            ..Default::default()
        })
    }

    pub fn to_optimization_options(&self) -> Option<OptimizationOptions> {
        if self.target_max_side.is_some()
            || self.crop_padding_x.is_some()
            || self.crop_padding_y.is_some()
        {
            Some(OptimizationOptions {
                enabled: Some(true),
                target_max_side: self.target_max_side,
                crop_padding_x: self.crop_padding_x,
                crop_padding_y: self.crop_padding_y,
            })
        } else {
            None
        }
    }
}

/// Post-detection polygon extraction, thresholding, and box filtering options.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PostDetectionOptions {
    pub threshold: Option<f32>,
    pub box_threshold: Option<f32>,
    pub unclip_ratio: Option<f32>,
    pub use_dilation: Option<bool>,
    pub score_mode: Option<String>,
    pub max_candidates: Option<i32>,
    pub min_height: Option<f32>,
    pub min_side_len: Option<f32>,
    pub max_side_len: Option<f32>,
    pub width_height_ratio: Option<f32>,
}

impl PostDetectionOptions {
    pub fn to_postprocess_run_options(&self) -> PostprocessRunOptions {
        PostprocessRunOptions {
            threshold: self.threshold,
            box_threshold: self.box_threshold,
            max_candidates: self.max_candidates,
            unclip_ratio: self.unclip_ratio,
            use_dilation: self.use_dilation,
            score_mode: self.score_mode.clone(),
        }
    }
}

/// Pre-recognition text line/word clustering tolerances.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PreRecognitionOptions {
    pub line_y_threshold: Option<f32>,
    pub word_x_threshold: Option<f32>,
}

/// Text recognition inference parameters.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecognitionRunOptions {
    pub batch_size: Option<i32>,
}

/// Post-recognition confidence score filtering.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PostRecognitionOptions {
    pub score_threshold: Option<f32>,
}

/// Unified runtime execution options for text detection.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrRunOptions {
    #[serde(default)]
    pub output: OutputGranularity,
    pub auto_unwarp: Option<bool>,
    pub layout_reading: Option<bool>,
    pub orientation_detection: Option<bool>,
    pub line_classification: Option<bool>,
    pub pre_detection: Option<PreDetectionOptions>,
    pub detection: Option<DetectionRunOptions>,
    pub post_detection: Option<PostDetectionOptions>,
    pub pre_recognition: Option<PreRecognitionOptions>,
    pub recognition: Option<RecognitionRunOptions>,
    pub post_recognition: Option<PostRecognitionOptions>,
}

impl OcrRunOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_auto_unwarp(mut self, enabled: bool) -> Self {
        self.auto_unwarp = Some(enabled);
        self
    }

    pub fn with_layout_reading(mut self, enabled: bool) -> Self {
        self.layout_reading = Some(enabled);
        self
    }

    pub fn with_orientation_detection(mut self, enabled: bool) -> Self {
        self.orientation_detection = Some(enabled);
        self
    }

    pub fn with_line_classification(mut self, enabled: bool) -> Self {
        self.line_classification = Some(enabled);
        self
    }

    pub fn with_pre_detection(mut self, opts: PreDetectionOptions) -> Self {
        self.pre_detection = Some(opts);
        self
    }

    pub fn with_detection(mut self, opts: DetectionRunOptions) -> Self {
        self.detection = Some(opts);
        self
    }

    pub fn with_post_detection(mut self, opts: PostDetectionOptions) -> Self {
        self.post_detection = Some(opts);
        self
    }

    pub fn with_pre_recognition(mut self, opts: PreRecognitionOptions) -> Self {
        self.pre_recognition = Some(opts);
        self
    }

    pub fn with_recognition(mut self, opts: RecognitionRunOptions) -> Self {
        self.recognition = Some(opts);
        self
    }

    pub fn with_post_recognition(mut self, opts: PostRecognitionOptions) -> Self {
        self.post_recognition = Some(opts);
        self
    }

    pub fn as_lines(mut self) -> Self {
        self.output = OutputGranularity::Lines;
        self
    }

    pub fn as_words(mut self) -> Self {
        self.output = OutputGranularity::Words;
        self
    }

    pub fn as_spatial(mut self) -> Self {
        self.output = OutputGranularity::Spatial;
        self
    }

    pub fn with_output(mut self, output: OutputGranularity) -> Self {
        self.output = output;
        self
    }

    /// Validate per-request options before execution.
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        let finite = |name: &str, value: Option<f32>, range: fn(f32) -> bool| {
            match value {
                Some(value) if !value.is_finite() || !range(value) => {
                    Err(crate::error::CoreError::Preprocess(format!("{name} is invalid")))
                }
                _ => Ok(()),
            }
        };

        if let Some(pre_det) = &self.pre_detection {
            finite("preDetection.descreenStrength", pre_det.descreen_strength, |v| v >= 0.0)?;
            if let Some(target) = pre_det.target_max_side {
                if target == 0 {
                    return Err(crate::error::CoreError::Preprocess("preDetection.targetMaxSide must be > 0".into()));
                }
            }
            finite("preDetection.cropPaddingX", pre_det.crop_padding_x, |v| (0.0..=1.0).contains(&v))?;
            finite("preDetection.cropPaddingY", pre_det.crop_padding_y, |v| (0.0..=1.0).contains(&v))?;
        }

        if let Some(det) = &self.detection {
            if det.limit_side_len.is_some_and(|v| v <= 0) {
                return Err(crate::error::CoreError::Preprocess("detection.limitSideLen must be > 0".into()));
            }
            if det.limit_type.as_deref().is_some_and(|v| v != "min" && v != "max") {
                return Err(crate::error::CoreError::Preprocess("detection.limitType must be min or max".into()));
            }
            for (name, values, allow_zero) in [("mean", det.mean, true), ("std", det.std, false)] {
                if let Some(values) = values {
                    if values.iter().any(|value| !value.is_finite() || (!allow_zero && *value == 0.0)) {
                        return Err(crate::error::CoreError::Preprocess(format!("detection.{name} is invalid")));
                    }
                }
            }
            if let Some(pp) = &det.postprocess {
                finite("detection.postprocess.threshold", pp.threshold, |v| (0.0..=1.0).contains(&v))?;
                finite("detection.postprocess.boxThreshold", pp.box_threshold, |v| (0.0..=1.0).contains(&v))?;
                finite("detection.postprocess.unclipRatio", pp.unclip_ratio, |v| v > 0.0)?;
                if pp.max_candidates.is_some_and(|v| v < 1) {
                    return Err(crate::error::CoreError::Preprocess("detection.postprocess.maxCandidates must be >= 1".into()));
                }
                if let Some(mode) = &pp.score_mode {
                    if mode != "fast" && mode != "slow" {
                        return Err(crate::error::CoreError::Preprocess("detection.postprocess.scoreMode must be 'fast' or 'slow'".into()));
                    }
                }
            }
        }

        if let Some(post_det) = &self.post_detection {
            finite("postDetection.threshold", post_det.threshold, |v| (0.0..=1.0).contains(&v))?;
            finite("postDetection.boxThreshold", post_det.box_threshold, |v| (0.0..=1.0).contains(&v))?;
            finite("postDetection.unclipRatio", post_det.unclip_ratio, |v| v > 0.0)?;
            if post_det.max_candidates.is_some_and(|v| v < 1) {
                return Err(crate::error::CoreError::Preprocess("postDetection.maxCandidates must be >= 1".into()));
            }
            if let Some(mode) = &post_det.score_mode {
                if mode != "fast" && mode != "slow" {
                    return Err(crate::error::CoreError::Preprocess("postDetection.scoreMode must be 'fast' or 'slow'".into()));
                }
            }
            finite("postDetection.minHeight", post_det.min_height, |v| v > 0.0)?;
            finite("postDetection.maxSideLen", post_det.max_side_len, |v| v > 0.0)?;
            finite("postDetection.minSideLen", post_det.min_side_len, |v| v > 0.0)?;
            finite("postDetection.widthHeightRatio", post_det.width_height_ratio, |v| v > 0.0 || v == -1.0)?;
            if let (Some(min), Some(max)) = (post_det.min_side_len, post_det.max_side_len) {
                if min > max {
                    return Err(crate::error::CoreError::Preprocess("postDetection.minSideLen must be <= maxSideLen".into()));
                }
            }
        }

        if let Some(pre_rec) = &self.pre_recognition {
            finite("preRecognition.lineYThreshold", pre_rec.line_y_threshold, |v| v >= 0.0)?;
            finite("preRecognition.wordXThreshold", pre_rec.word_x_threshold, |v| v >= 0.0)?;
        }

        if let Some(rec) = &self.recognition {
            if rec.batch_size.is_some_and(|v| v <= 0) {
                return Err(crate::error::CoreError::Preprocess("recognition.batchSize must be > 0".into()));
            }
        }

        if let Some(post_rec) = &self.post_recognition {
            finite("postRecognition.scoreThreshold", post_rec.score_threshold, |v| (0.0..=1.0).contains(&v))?;
        }

        Ok(())
    }
}
