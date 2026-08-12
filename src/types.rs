use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Axis-aligned bounding box frame
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

    /// Calculate axis-aligned bounding box frame from 4 corner points
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum LangRec {
    Ch,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum OcrVersion {
    PpOcrV3,
    PpOcrV4,
    PpOcrV5,
    PpOcrV6,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum EngineType {
    OnnxRuntime,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    Mobile,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    Det,
    Cls,
    Rec,
    Orient, // Document orientation classification
    Unwarp, // Text image rectification
    Layout, // Layout detection
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    pub intra_op_num_threads: i32,
    pub inter_op_num_threads: i32,
    pub enable_cpu_mem_arena: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // Auto-detect optimal thread count (use all available CPUs)
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        Self {
            intra_op_num_threads: num_threads,
            inter_op_num_threads: 1, // Keep inter-op at 1 for better cache locality
            enable_cpu_mem_arena: true, // Enable for better memory performance
        }
    }
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
            score_mode: "fast".to_string(),
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
            score_mode: "fast".to_string(),
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
            limit_side_len: 960,
            limit_type: "max".to_string(),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            thresh: 0.3,
            box_thresh: 0.6,
            max_candidates: 1000,
            unclip_ratio: 1.5,
            use_dilation: false,
            score_mode: "fast".to_string(),
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv3(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV3,
            task_type: TaskType::Det,
            model_path,
            limit_side_len: 960,
            limit_type: "max".to_string(),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            thresh: 0.3,
            box_thresh: 0.6,
            max_candidates: 1000,
            unclip_ratio: 1.5,
            use_dilation: false,
            score_mode: "fast".to_string(),
            engine_cfg: EngineConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClsConfig {
    pub engine_type: EngineType,
    pub lang_type: LangRec,
    pub model_type: ModelType,
    pub ocr_version: OcrVersion,
    pub task_type: TaskType,
    pub model_path: PathBuf,
    pub cls_image_shape: [i32; 3],
    pub cls_batch_num: i32,
    pub cls_thresh: f32,
    pub label_list: Vec<String>,
    pub engine_cfg: EngineConfig,
}

impl ClsConfig {
    pub fn ppv6(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV6,
            task_type: TaskType::Cls,
            model_path,
            cls_image_shape: [3, 48, 192],
            cls_batch_num: 1,
            cls_thresh: 0.9,
            label_list: vec!["0".to_string(), "180".to_string()],
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn default(model_path: PathBuf) -> Self {
        Self::ppv6(model_path)
    }
}

impl DetConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self::ppv6(model_path)
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
    pub fn ppv6(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV6,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path: None,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn default(model_path: PathBuf) -> Self {
        Self::ppv6(model_path)
    }


    pub fn ppv5(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV5,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path: None,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv4(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV4,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path: None,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
            engine_cfg: EngineConfig::default(),
        }
    }

    pub fn ppv3(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            lang_type: LangRec::Ch,
            model_type: ModelType::Mobile,
            ocr_version: OcrVersion::PpOcrV3,
            task_type: TaskType::Rec,
            model_path,
            rec_keys_path: None,
            rec_img_shape: [3, 48, 320],
            rec_batch_num: 6,
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
    pub confidence_threshold: f32, // Minimum confidence to apply orientation correction
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
            confidence_threshold: 0.9, // Default: 90% confidence required
            orient_batch_num: 1,
            orient_thresh: 0.9,
            engine_cfg: EngineConfig::default(),
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
pub struct GlobalConfig {
    pub text_score: f32,
    pub use_det: bool,
    pub use_cls: bool,
    pub use_rec: bool,
    pub use_orient: bool,
    pub use_unwarp: bool,
    pub orient_config: Option<OrientConfig>,
    pub unwarp_config: Option<UnwarpConfig>,
    /// Enable debug images (oriented and rectified)
    pub debug_images: bool,
    pub min_height: f32,
    pub width_height_ratio: f32,
    pub max_side_len: f32,
    pub min_side_len: f32,
    pub return_word_box: bool,
    pub return_single_char_box: bool,
    /// Configurable Y threshold multiplier for line grouping in spatial text
    pub y_threshold_multiplier: Option<f32>,
    /// Configurable X threshold multiplier for word/column gap separation in spatial text
    pub x_threshold_multiplier: Option<f32>,
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
            orient_config: None,
            unwarp_config: None,
            debug_images: false, // Disabled by default for performance
            min_height: 30.0,
            width_height_ratio: 8.0,
            max_side_len: 2000.0,
            min_side_len: 30.0,
            return_word_box: false,
            return_single_char_box: false,
            y_threshold_multiplier: None,
            x_threshold_multiplier: None,
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
    pub engine_cfg: EngineConfig,
}

impl LayoutConfig {
    pub fn default(model_path: PathBuf) -> Self {
        Self {
            engine_type: EngineType::OnnxRuntime,
            model_type: ModelType::Mobile,
            task_type: TaskType::Layout,
            model_path,
            mean: [0.485, 0.456, 0.406], // ImageNet mean
            std: [0.229, 0.224, 0.225],  // ImageNet std
            conf_thresh: 0.4,
            iou_thresh: 0.5,
            target_size: 640, // Default for PP-DocLayout_plus-L
            engine_cfg: EngineConfig::default(),
        }
    }
}
