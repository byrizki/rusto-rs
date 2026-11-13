use std::path::PathBuf;

use crate::calibration::{CalibrationOptions, OptimizationOptions};
use crate::error::CoreError;
use crate::models;
use crate::types::{ClsConfig, DetConfig, GlobalConfig, LayoutConfig, OrientConfig, RecConfig, UnwarpConfig};

/// Preset configuration for OCR model architectures
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelPreset {
    pub det_limit_side_len: i32,
    pub det_limit_type: &'static str,
    pub det_thresh: f32,
    pub det_box_thresh: f32,
    pub det_unclip_ratio: f32,
    pub det_use_dilation: bool,
    pub det_score_mode: &'static str,
    pub rec_img_shape: [i32; 3],
    pub rec_batch_num: i32,
    pub text_score: f32,
}

/// Default preset configuration for PP-OCRv6 models
pub const PPV6_MODEL_CONFIG: ModelPreset = ModelPreset {
    det_limit_side_len: 736,
    det_limit_type: "min",
    det_thresh: 0.3,
    det_box_thresh: 0.6,
    det_unclip_ratio: 2.0,
    det_use_dilation: true,
    det_score_mode: "slow",
    rec_img_shape: [3, 48, 320],
    rec_batch_num: 6,
    text_score: 0.5,
};

/// Default preset configuration for PP-OCRv5 models
pub const PPV5_MODEL_CONFIG: ModelPreset = ModelPreset {
    det_limit_side_len: 736,
    det_limit_type: "min",
    det_thresh: 0.3,
    det_box_thresh: 0.5,
    det_unclip_ratio: 2.0,
    det_use_dilation: true,
    det_score_mode: "slow",
    rec_img_shape: [3, 48, 320],
    rec_batch_num: 6,
    text_score: 0.5,
};

/// Default preset configuration for PP-OCRv4 models
pub const PPV4_MODEL_CONFIG: ModelPreset = ModelPreset {
    det_limit_side_len: 960,
    det_limit_type: "max",
    det_thresh: 0.3,
    det_box_thresh: 0.6,
    det_unclip_ratio: 1.5,
    det_use_dilation: false,
    det_score_mode: "slow",
    rec_img_shape: [3, 48, 320],
    rec_batch_num: 6,
    text_score: 0.5,
};

/// Default preset configuration for PP-OCRv3 models
pub const PPV3_MODEL_CONFIG: ModelPreset = ModelPreset {
    det_limit_side_len: 960,
    det_limit_type: "max",
    det_thresh: 0.3,
    det_box_thresh: 0.6,
    det_unclip_ratio: 1.5,
    det_use_dilation: false,
    det_score_mode: "slow",
    rec_img_shape: [3, 48, 320],
    rec_batch_num: 6,
    text_score: 0.5,
};

/// Default preset configuration for Indra OCR models (Linenet dynamic recognition with batch size 1)
pub const INDRA_MODEL_CONFIG: ModelPreset = ModelPreset {
    det_limit_side_len: 1024,
    det_limit_type: "max",
    det_thresh: 0.5,
    det_box_thresh: 0.0,
    det_unclip_ratio: 2.0,
    det_use_dilation: true,
    det_score_mode: "slow",
    rec_img_shape: [1, 96, 320],
    rec_batch_num: 1,
    text_score: 0.5,
};

/// Main configuration for OCR engine
#[derive(Clone, Debug)]
pub struct InitializeConfig {
    pub det: DetConfig,
    pub rec: RecConfig,
    pub global: GlobalConfig,
    pub orient: Option<OrientConfig>,
    pub unwarp: Option<UnwarpConfig>,
    pub cls: Option<ClsConfig>,
    pub layout: Option<LayoutConfig>,
    pub calibration: Option<CalibrationOptions>,
    pub optimization: Option<OptimizationOptions>,
}

impl Default for InitializeConfig {
    fn default() -> Self {
        Self::from_preset(PPV6_MODEL_CONFIG, "det.rten", "rec.rten", "dict.txt")
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct DetectionConfigJson {
    pub enabled: Option<bool>,
    #[nserde(rename = "useDet")]
    pub use_det_camel: Option<bool>,
    #[nserde(rename = "use_det")]
    pub use_det_snake: Option<bool>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "detModelPath")]
    pub det_model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "det_model_path")]
    pub det_model_path_snake: Option<String>,

    pub thresh: Option<f32>,
    #[nserde(rename = "detThresh")]
    pub det_thresh_camel: Option<f32>,
    #[nserde(rename = "det_thresh")]
    pub det_thresh_snake: Option<f32>,

    #[nserde(rename = "boxThresh")]
    pub box_thresh_camel: Option<f32>,
    #[nserde(rename = "detBoxThresh")]
    pub det_box_thresh_camel: Option<f32>,
    #[nserde(rename = "box_thresh")]
    pub box_thresh_snake: Option<f32>,
    #[nserde(rename = "det_box_thresh")]
    pub det_box_thresh_snake: Option<f32>,

    #[nserde(rename = "unclipRatio")]
    pub unclip_ratio_camel: Option<f32>,
    #[nserde(rename = "unclip_ratio")]
    pub unclip_ratio_snake: Option<f32>,

    #[nserde(rename = "limitSideLen")]
    pub limit_side_len_camel: Option<i32>,
    #[nserde(rename = "limit_side_len")]
    pub limit_side_len_snake: Option<i32>,

    #[nserde(rename = "limitType")]
    pub limit_type_camel: Option<String>,
    #[nserde(rename = "limit_type")]
    pub limit_type_snake: Option<String>,

    #[nserde(rename = "useDilation")]
    pub use_dilation_camel: Option<bool>,
    #[nserde(rename = "use_dilation")]
    pub use_dilation_snake: Option<bool>,

    #[nserde(rename = "scoreMode")]
    pub score_mode_camel: Option<String>,
    #[nserde(rename = "detScoreMode")]
    pub det_score_mode_camel: Option<String>,
    #[nserde(rename = "score_mode")]
    pub score_mode_snake: Option<String>,
    #[nserde(rename = "det_score_mode")]
    pub det_score_mode_snake: Option<String>,
}

impl DetectionConfigJson {
    fn enabled(&self) -> Option<bool> {
        self.enabled.or(self.use_det_camel).or(self.use_det_snake)
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.det_model_path_camel.as_deref())
            .or(self.model_path_snake.as_deref())
            .or(self.det_model_path_snake.as_deref())
    }
    fn thresh(&self) -> Option<f32> {
        self.thresh.or(self.det_thresh_camel).or(self.det_thresh_snake)
    }
    fn box_thresh(&self) -> Option<f32> {
        self.box_thresh_camel
            .or(self.det_box_thresh_camel)
            .or(self.box_thresh_snake)
            .or(self.det_box_thresh_snake)
    }
    fn unclip_ratio(&self) -> Option<f32> {
        self.unclip_ratio_camel.or(self.unclip_ratio_snake)
    }
    fn limit_side_len(&self) -> Option<i32> {
        self.limit_side_len_camel.or(self.limit_side_len_snake)
    }
    fn limit_type(&self) -> Option<&str> {
        self.limit_type_camel.as_deref().or(self.limit_type_snake.as_deref())
    }
    fn use_dilation(&self) -> Option<bool> {
        self.use_dilation_camel.or(self.use_dilation_snake)
    }
    fn score_mode(&self) -> Option<&str> {
        self.score_mode_camel.as_deref()
            .or(self.det_score_mode_camel.as_deref())
            .or(self.score_mode_snake.as_deref())
            .or(self.det_score_mode_snake.as_deref())
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct RecognitionConfigJson {
    pub enabled: Option<bool>,
    #[nserde(rename = "useRec")]
    pub use_rec_camel: Option<bool>,
    #[nserde(rename = "use_rec")]
    pub use_rec_snake: Option<bool>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "recModelPath")]
    pub rec_model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "rec_model_path")]
    pub rec_model_path_snake: Option<String>,

    #[nserde(rename = "dictPath")]
    pub dict_path_camel: Option<String>,
    #[nserde(rename = "dict_path")]
    pub dict_path_snake: Option<String>,

    #[nserde(rename = "scoreThresh")]
    pub score_thresh_camel: Option<f32>,
    #[nserde(rename = "textScore")]
    pub text_score_camel: Option<f32>,
    #[nserde(rename = "score_thresh")]
    pub score_thresh_snake: Option<f32>,
    #[nserde(rename = "text_score")]
    pub text_score_snake: Option<f32>,

    #[nserde(rename = "returnWordBox")]
    pub return_word_box_camel: Option<bool>,
    #[nserde(rename = "return_word_box")]
    pub return_word_box_snake: Option<bool>,

    #[nserde(rename = "returnSingleCharBox")]
    pub return_single_char_box_camel: Option<bool>,
    #[nserde(rename = "return_single_char_box")]
    pub return_single_char_box_snake: Option<bool>,

    #[nserde(rename = "recBatchNum")]
    pub rec_batch_num_camel: Option<i32>,
    #[nserde(rename = "batchNum")]
    pub batch_num_camel: Option<i32>,
    #[nserde(rename = "rec_batch_num")]
    pub rec_batch_num_snake: Option<i32>,
    #[nserde(rename = "batch_num")]
    pub batch_num_snake: Option<i32>,

    #[nserde(rename = "recWorkers")]
    pub rec_workers_camel: Option<usize>,
    #[nserde(rename = "rec_workers")]
    pub rec_workers_snake: Option<usize>,

    #[nserde(rename = "maxRecWorkers")]
    pub max_rec_workers_camel: Option<usize>,
    #[nserde(rename = "max_rec_workers")]
    pub max_rec_workers_snake: Option<usize>,
}

impl RecognitionConfigJson {
    fn enabled(&self) -> Option<bool> {
        self.enabled.or(self.use_rec_camel).or(self.use_rec_snake)
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.rec_model_path_camel.as_deref())
            .or(self.model_path_snake.as_deref())
            .or(self.rec_model_path_snake.as_deref())
    }
    fn dict_path(&self) -> Option<&str> {
        self.dict_path_camel.as_deref().or(self.dict_path_snake.as_deref())
    }
    fn score_thresh(&self) -> Option<f32> {
        self.score_thresh_camel
            .or(self.text_score_camel)
            .or(self.score_thresh_snake)
            .or(self.text_score_snake)
    }
    fn return_word_box(&self) -> Option<bool> {
        self.return_word_box_camel.or(self.return_word_box_snake)
    }
    fn return_single_char_box(&self) -> Option<bool> {
        self.return_single_char_box_camel.or(self.return_single_char_box_snake)
    }
    fn rec_batch_num(&self) -> Option<i32> {
        self.rec_batch_num_camel
            .or(self.batch_num_camel)
            .or(self.rec_batch_num_snake)
            .or(self.batch_num_snake)
    }
    fn rec_workers(&self) -> Option<usize> {
        self.rec_workers_camel.or(self.rec_workers_snake)
    }
    fn max_rec_workers(&self) -> Option<usize> {
        self.max_rec_workers_camel.or(self.max_rec_workers_snake)
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct ClassificationConfigJson {
    pub enabled: Option<bool>,
    #[nserde(rename = "useCls")]
    pub use_cls_camel: Option<bool>,
    #[nserde(rename = "use_cls")]
    pub use_cls_snake: Option<bool>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "clsModelPath")]
    pub cls_model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "cls_model_path")]
    pub cls_model_path_snake: Option<String>,

    pub thresh: Option<f32>,
    pub threshold: Option<f32>,
    #[nserde(rename = "clsThreshold")]
    pub cls_threshold_camel: Option<f32>,
    #[nserde(rename = "cls_thresh")]
    pub cls_thresh_snake: Option<f32>,
    #[nserde(rename = "cls_threshold")]
    pub cls_threshold_snake: Option<f32>,
}

impl ClassificationConfigJson {
    fn enabled(&self) -> Option<bool> {
        self.enabled.or(self.use_cls_camel).or(self.use_cls_snake)
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.cls_model_path_camel.as_deref())
            .or(self.model_path_snake.as_deref())
            .or(self.cls_model_path_snake.as_deref())
    }
    fn threshold(&self) -> Option<f32> {
        self.threshold
            .or(self.thresh)
            .or(self.cls_threshold_camel)
            .or(self.cls_thresh_snake)
            .or(self.cls_threshold_snake)
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct OrientationConfigJson {
    pub enabled: Option<bool>,
    #[nserde(rename = "useOrient")]
    pub use_orient_camel: Option<bool>,
    #[nserde(rename = "use_orient")]
    pub use_orient_snake: Option<bool>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "orientModelPath")]
    pub orient_model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "orient_model_path")]
    pub orient_model_path_snake: Option<String>,

    pub thresh: Option<f32>,
    pub threshold: Option<f32>,
    #[nserde(rename = "orientThreshold")]
    pub orient_threshold_camel: Option<f32>,
    #[nserde(rename = "orient_thresh")]
    pub orient_thresh_snake: Option<f32>,
    #[nserde(rename = "orient_threshold")]
    pub orient_threshold_snake: Option<f32>,
}

impl OrientationConfigJson {
    fn enabled(&self) -> Option<bool> {
        self.enabled.or(self.use_orient_camel).or(self.use_orient_snake)
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.orient_model_path_camel.as_deref())
            .or(self.model_path_snake.as_deref())
            .or(self.orient_model_path_snake.as_deref())
    }
    fn threshold(&self) -> Option<f32> {
        self.threshold
            .or(self.thresh)
            .or(self.orient_threshold_camel)
            .or(self.orient_thresh_snake)
            .or(self.orient_threshold_snake)
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct UnwarpConfigJson {
    pub enabled: Option<bool>,
    #[nserde(rename = "useUnwarp")]
    pub use_unwarp_camel: Option<bool>,
    #[nserde(rename = "use_unwarp")]
    pub use_unwarp_snake: Option<bool>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "unwarpModelPath")]
    pub unwarp_model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "unwarp_model_path")]
    pub unwarp_model_path_snake: Option<String>,
}

impl UnwarpConfigJson {
    fn enabled(&self) -> Option<bool> {
        self.enabled.or(self.use_unwarp_camel).or(self.use_unwarp_snake)
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.unwarp_model_path_camel.as_deref())
            .or(self.model_path_snake.as_deref())
            .or(self.unwarp_model_path_snake.as_deref())
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct LayoutConfigJson {
    pub enabled: Option<bool>,
    #[nserde(rename = "useLayout")]
    pub use_layout_camel: Option<bool>,
    #[nserde(rename = "use_layout")]
    pub use_layout_snake: Option<bool>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "layoutModelPath")]
    pub layout_model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "layout_model_path")]
    pub layout_model_path_snake: Option<String>,

    #[nserde(rename = "confThresh")]
    pub conf_thresh_camel: Option<f32>,
    #[nserde(rename = "conf_thresh")]
    pub conf_thresh_snake: Option<f32>,
    pub threshold: Option<f32>,

    #[nserde(rename = "targetSize")]
    pub target_size_camel: Option<i32>,
    #[nserde(rename = "target_size")]
    pub target_size_snake: Option<i32>,

    #[nserde(rename = "targetHeight")]
    pub target_height_camel: Option<i32>,
    #[nserde(rename = "target_height")]
    pub target_height_snake: Option<i32>,

    #[nserde(rename = "targetWidth")]
    pub target_width_camel: Option<i32>,
    #[nserde(rename = "target_width")]
    pub target_width_snake: Option<i32>,

    #[nserde(rename = "yThresholdMultiplier")]
    pub y_threshold_multiplier_camel: Option<f32>,
    #[nserde(rename = "y_threshold_multiplier")]
    pub y_threshold_multiplier_snake: Option<f32>,

    #[nserde(rename = "xThresholdMultiplier")]
    pub x_threshold_multiplier_camel: Option<f32>,
    #[nserde(rename = "x_threshold_multiplier")]
    pub x_threshold_multiplier_snake: Option<f32>,
}

impl LayoutConfigJson {
    fn enabled(&self) -> Option<bool> {
        self.enabled.or(self.use_layout_camel).or(self.use_layout_snake)
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.layout_model_path_camel.as_deref())
            .or(self.model_path_snake.as_deref())
            .or(self.layout_model_path_snake.as_deref())
    }
    fn conf_thresh(&self) -> Option<f32> {
        self.conf_thresh_camel.or(self.conf_thresh_snake).or(self.threshold)
    }
    fn target_size(&self) -> Option<i32> {
        self.target_size_camel.or(self.target_size_snake)
    }
    fn target_height(&self) -> Option<i32> {
        self.target_height_camel.or(self.target_height_snake)
    }
    fn target_width(&self) -> Option<i32> {
        self.target_width_camel.or(self.target_width_snake)
    }
    fn y_threshold_multiplier(&self) -> Option<f32> {
        self.y_threshold_multiplier_camel.or(self.y_threshold_multiplier_snake)
    }
    fn x_threshold_multiplier(&self) -> Option<f32> {
        self.x_threshold_multiplier_camel.or(self.x_threshold_multiplier_snake)
    }
}

#[derive(nanoserde::DeJson, nanoserde::SerJson, Default, Clone, Debug)]
struct FlatConfig {
    pub model: Option<String>,
    pub models: Option<String>,
    pub template: Option<String>,

    #[nserde(rename = "modelPath")]
    pub model_path_camel: Option<String>,
    #[nserde(rename = "model_path")]
    pub model_path_snake: Option<String>,
    #[nserde(rename = "modelDir")]
    pub model_dir_camel: Option<String>,
    #[nserde(rename = "model_dir")]
    pub model_dir_snake: Option<String>,

    pub lang: Option<String>,

    #[nserde(rename = "cacheDir")]
    pub cache_dir_camel: Option<String>,
    #[nserde(rename = "cache_dir")]
    pub cache_dir_snake: Option<String>,

    #[nserde(rename = "enableReadingOrder")]
    pub enable_reading_order_camel: Option<bool>,
    #[nserde(rename = "enable_reading_order")]
    pub enable_reading_order_snake: Option<bool>,
    #[nserde(rename = "reading_order")]
    pub reading_order: Option<bool>,

    #[nserde(rename = "layoutVariant")]
    pub layout_variant_camel: Option<String>,
    #[nserde(rename = "layout_variant")]
    pub layout_variant_snake: Option<String>,

    #[nserde(rename = "detModelPath")]
    pub det_model_path_camel: Option<String>,
    #[nserde(rename = "det_model_path")]
    pub det_model_path_snake: Option<String>,
    #[nserde(rename = "detPath")]
    pub det_path_camel: Option<String>,
    #[nserde(rename = "det_path")]
    pub det_path_snake: Option<String>,

    #[nserde(rename = "recModelPath")]
    pub rec_model_path_camel: Option<String>,
    #[nserde(rename = "rec_model_path")]
    pub rec_model_path_snake: Option<String>,
    #[nserde(rename = "recPath")]
    pub rec_path_camel: Option<String>,
    #[nserde(rename = "rec_path")]
    pub rec_path_snake: Option<String>,

    #[nserde(rename = "dictPath")]
    pub dict_path_camel: Option<String>,
    #[nserde(rename = "dict_path")]
    pub dict_path_snake: Option<String>,

    #[nserde(rename = "clsModelPath")]
    pub cls_model_path_camel: Option<String>,
    #[nserde(rename = "cls_model_path")]
    pub cls_model_path_snake: Option<String>,
    #[nserde(rename = "clsPath")]
    pub cls_path_camel: Option<String>,
    #[nserde(rename = "cls_path")]
    pub cls_path_snake: Option<String>,

    #[nserde(rename = "orientModelPath")]
    pub orient_model_path_camel: Option<String>,
    #[nserde(rename = "orient_model_path")]
    pub orient_model_path_snake: Option<String>,
    #[nserde(rename = "orientPath")]
    pub orient_path_camel: Option<String>,
    #[nserde(rename = "orient_path")]
    pub orient_path_snake: Option<String>,

    #[nserde(rename = "unwarpModelPath")]
    pub unwarp_model_path_camel: Option<String>,
    #[nserde(rename = "unwarp_model_path")]
    pub unwarp_model_path_snake: Option<String>,
    #[nserde(rename = "unwarpPath")]
    pub unwarp_path_camel: Option<String>,
    #[nserde(rename = "unwarp_path")]
    pub unwarp_path_snake: Option<String>,

    #[nserde(rename = "layoutModelPath")]
    pub layout_model_path_camel: Option<String>,
    #[nserde(rename = "layout_model_path")]
    pub layout_model_path_snake: Option<String>,
    #[nserde(rename = "layoutPath")]
    pub layout_path_camel: Option<String>,
    #[nserde(rename = "layout_path")]
    pub layout_path_snake: Option<String>,

    #[nserde(rename = "detThresh")]
    pub det_thresh_camel: Option<f32>,
    #[nserde(rename = "det_thresh")]
    pub det_thresh_snake: Option<f32>,

    #[nserde(rename = "boxThresh")]
    pub box_thresh_camel: Option<f32>,
    #[nserde(rename = "detBoxThresh")]
    pub det_box_thresh_camel: Option<f32>,
    #[nserde(rename = "box_thresh")]
    pub box_thresh_snake: Option<f32>,
    #[nserde(rename = "det_box_thresh")]
    pub det_box_thresh_snake: Option<f32>,

    #[nserde(rename = "unclipRatio")]
    pub unclip_ratio_camel: Option<f32>,
    #[nserde(rename = "unclip_ratio")]
    pub unclip_ratio_snake: Option<f32>,

    #[nserde(rename = "limitSideLen")]
    pub limit_side_len_camel: Option<i32>,
    #[nserde(rename = "limit_side_len")]
    pub limit_side_len_snake: Option<i32>,

    #[nserde(rename = "limitType")]
    pub limit_type_camel: Option<String>,
    #[nserde(rename = "limit_type")]
    pub limit_type_snake: Option<String>,

    #[nserde(rename = "useDilation")]
    pub use_dilation_camel: Option<bool>,
    #[nserde(rename = "use_dilation")]
    pub use_dilation_snake: Option<bool>,

    #[nserde(rename = "scoreMode")]
    pub score_mode_camel: Option<String>,
    #[nserde(rename = "detScoreMode")]
    pub det_score_mode_camel: Option<String>,
    #[nserde(rename = "score_mode")]
    pub score_mode_snake: Option<String>,
    #[nserde(rename = "det_score_mode")]
    pub det_score_mode_snake: Option<String>,

    #[nserde(rename = "maxCandidates")]
    pub max_candidates_camel: Option<i32>,
    #[nserde(rename = "max_candidates")]
    pub max_candidates_snake: Option<i32>,

    #[nserde(rename = "textScore")]
    pub text_score_camel: Option<f32>,
    #[nserde(rename = "scoreThresh")]
    pub score_thresh_camel: Option<f32>,
    #[nserde(rename = "score_thresh")]
    pub score_thresh_snake: Option<f32>,
    #[nserde(rename = "text_score")]
    pub text_score_snake: Option<f32>,

    #[nserde(rename = "returnWordBox")]
    pub return_word_box_camel: Option<bool>,
    #[nserde(rename = "return_word_box")]
    pub return_word_box_snake: Option<bool>,

    #[nserde(rename = "returnSingleCharBox")]
    pub return_single_char_box_camel: Option<bool>,
    #[nserde(rename = "return_single_char_box")]
    pub return_single_char_box_snake: Option<bool>,

    #[nserde(rename = "recBatchNum")]
    pub rec_batch_num_camel: Option<i32>,
    #[nserde(rename = "batchNum")]
    pub batch_num_camel: Option<i32>,
    #[nserde(rename = "rec_batch_num")]
    pub rec_batch_num_snake: Option<i32>,
    #[nserde(rename = "batch_num")]
    pub batch_num_snake: Option<i32>,

    #[nserde(rename = "recWorkers")]
    pub rec_workers_camel: Option<usize>,
    #[nserde(rename = "rec_workers")]
    pub rec_workers_snake: Option<usize>,

    #[nserde(rename = "maxRecWorkers")]
    pub max_rec_workers_camel: Option<usize>,
    #[nserde(rename = "max_rec_workers")]
    pub max_rec_workers_snake: Option<usize>,

    #[nserde(rename = "useDet")]
    pub use_det_camel: Option<bool>,
    #[nserde(rename = "use_det")]
    pub use_det_snake: Option<bool>,

    #[nserde(rename = "useRec")]
    pub use_rec_camel: Option<bool>,
    #[nserde(rename = "use_rec")]
    pub use_rec_snake: Option<bool>,

    #[nserde(rename = "useCls")]
    pub use_cls_camel: Option<bool>,
    #[nserde(rename = "use_cls")]
    pub use_cls_snake: Option<bool>,

    #[nserde(rename = "useOrient")]
    pub use_orient_camel: Option<bool>,
    #[nserde(rename = "use_orient")]
    pub use_orient_snake: Option<bool>,

    #[nserde(rename = "useUnwarp")]
    pub use_unwarp_camel: Option<bool>,
    #[nserde(rename = "use_unwarp")]
    pub use_unwarp_snake: Option<bool>,

    #[nserde(rename = "useLayout")]
    pub use_layout_camel: Option<bool>,
    #[nserde(rename = "use_layout")]
    pub use_layout_snake: Option<bool>,

    pub detection: Option<DetectionConfigJson>,
    pub recognition: Option<RecognitionConfigJson>,
    pub classification: Option<ClassificationConfigJson>,
    pub orientation: Option<OrientationConfigJson>,
    pub unwarp: Option<UnwarpConfigJson>,
    pub layout: Option<LayoutConfigJson>,
}

impl FlatConfig {
    fn model(&self) -> Option<&str> {
        self.model.as_deref().or(self.models.as_deref()).or(self.template.as_deref())
    }
    fn model_path(&self) -> Option<&str> {
        self.model_path_camel.as_deref()
            .or(self.model_path_snake.as_deref())
            .or(self.model_dir_camel.as_deref())
            .or(self.model_dir_snake.as_deref())
    }
    fn cache_dir(&self) -> Option<&str> {
        self.cache_dir_camel.as_deref().or(self.cache_dir_snake.as_deref())
    }
    fn enable_reading_order(&self) -> Option<bool> {
        self.enable_reading_order_camel.or(self.enable_reading_order_snake).or(self.reading_order)
    }
    fn layout_variant(&self) -> Option<&str> {
        self.layout_variant_camel.as_deref().or(self.layout_variant_snake.as_deref())
    }
    fn det_model_path(&self) -> Option<&str> {
        self.det_model_path_camel.as_deref()
            .or(self.det_model_path_snake.as_deref())
            .or(self.det_path_camel.as_deref())
            .or(self.det_path_snake.as_deref())
    }
    fn rec_model_path(&self) -> Option<&str> {
        self.rec_model_path_camel.as_deref()
            .or(self.rec_model_path_snake.as_deref())
            .or(self.rec_path_camel.as_deref())
            .or(self.rec_path_snake.as_deref())
    }
    fn dict_path(&self) -> Option<&str> {
        self.dict_path_camel.as_deref().or(self.dict_path_snake.as_deref())
    }
    fn cls_model_path(&self) -> Option<&str> {
        self.cls_model_path_camel.as_deref()
            .or(self.cls_model_path_snake.as_deref())
            .or(self.cls_path_camel.as_deref())
            .or(self.cls_path_snake.as_deref())
    }
    fn orient_model_path(&self) -> Option<&str> {
        self.orient_model_path_camel.as_deref()
            .or(self.orient_model_path_snake.as_deref())
            .or(self.orient_path_camel.as_deref())
            .or(self.orient_path_snake.as_deref())
    }
    fn unwarp_model_path(&self) -> Option<&str> {
        self.unwarp_model_path_camel.as_deref()
            .or(self.unwarp_model_path_snake.as_deref())
            .or(self.unwarp_path_camel.as_deref())
            .or(self.unwarp_path_snake.as_deref())
    }
    fn layout_model_path(&self) -> Option<&str> {
        self.layout_model_path_camel.as_deref()
            .or(self.layout_model_path_snake.as_deref())
            .or(self.layout_path_camel.as_deref())
            .or(self.layout_path_snake.as_deref())
    }
    fn det_thresh(&self) -> Option<f32> {
        self.det_thresh_camel.or(self.det_thresh_snake)
    }
    fn box_thresh(&self) -> Option<f32> {
        self.box_thresh_camel
            .or(self.det_box_thresh_camel)
            .or(self.box_thresh_snake)
            .or(self.det_box_thresh_snake)
    }
    fn unclip_ratio(&self) -> Option<f32> {
        self.unclip_ratio_camel.or(self.unclip_ratio_snake)
    }
    fn limit_side_len(&self) -> Option<i32> {
        self.limit_side_len_camel.or(self.limit_side_len_snake)
    }
    fn limit_type(&self) -> Option<&str> {
        self.limit_type_camel.as_deref().or(self.limit_type_snake.as_deref())
    }
    fn use_dilation(&self) -> Option<bool> {
        self.use_dilation_camel.or(self.use_dilation_snake)
    }
    fn score_mode(&self) -> Option<&str> {
        self.score_mode_camel.as_deref()
            .or(self.det_score_mode_camel.as_deref())
            .or(self.score_mode_snake.as_deref())
            .or(self.det_score_mode_snake.as_deref())
    }
    fn max_candidates(&self) -> Option<i32> {
        self.max_candidates_camel.or(self.max_candidates_snake)
    }
    fn text_score(&self) -> Option<f32> {
        self.text_score_camel
            .or(self.score_thresh_camel)
            .or(self.score_thresh_snake)
            .or(self.text_score_snake)
    }
    fn return_word_box(&self) -> Option<bool> {
        self.return_word_box_camel.or(self.return_word_box_snake)
    }
    fn return_single_char_box(&self) -> Option<bool> {
        self.return_single_char_box_camel.or(self.return_single_char_box_snake)
    }
    fn rec_batch_num(&self) -> Option<i32> {
        self.rec_batch_num_camel
            .or(self.batch_num_camel)
            .or(self.rec_batch_num_snake)
            .or(self.batch_num_snake)
    }
    fn rec_workers(&self) -> Option<usize> {
        self.rec_workers_camel.or(self.rec_workers_snake)
    }
    fn max_rec_workers(&self) -> Option<usize> {
        self.max_rec_workers_camel.or(self.max_rec_workers_snake)
    }
    fn use_det(&self) -> Option<bool> {
        self.use_det_camel.or(self.use_det_snake)
    }
    fn use_rec(&self) -> Option<bool> {
        self.use_rec_camel.or(self.use_rec_snake)
    }
    fn use_cls(&self) -> Option<bool> {
        self.use_cls_camel.or(self.use_cls_snake)
    }
    fn use_orient(&self) -> Option<bool> {
        self.use_orient_camel.or(self.use_orient_snake)
    }
    fn use_unwarp(&self) -> Option<bool> {
        self.use_unwarp_camel.or(self.use_unwarp_snake)
    }
    fn use_layout(&self) -> Option<bool> {
        self.use_layout_camel.or(self.use_layout_snake)
    }
}

impl InitializeConfig {
    /// Create a configuration with model paths using the default PPV6 template
    pub fn new<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(PPV6_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Create a configuration from a template preset
    pub fn from_preset<P: Into<PathBuf>>(
        preset: ModelPreset,
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        let det_path = det_model_path.into();
        let rec_path = rec_model_path.into();
        let dict = dict_path.into();

        let mut det = DetConfig::default(det_path);
        det.limit_side_len = preset.det_limit_side_len;
        det.limit_type = preset.det_limit_type.to_string();
        det.thresh = preset.det_thresh;
        det.box_thresh = preset.det_box_thresh;
        det.unclip_ratio = preset.det_unclip_ratio;
        det.use_dilation = preset.det_use_dilation;
        det.score_mode = preset.det_score_mode.to_string();

        let mut rec = RecConfig::default(rec_path);
        if let Ok(content) = std::fs::read_to_string(&dict) {
            #[derive(nanoserde::DeJson, Default)]
            struct MetaFields {
                det_long_side: Option<i32>,
                det_thresh: Option<f32>,
                det_box_thresh: Option<f32>,
            }
            if let Ok(meta) = <MetaFields as nanoserde::DeJson>::deserialize_json(&content) {
                if let Some(side) = meta.det_long_side {
                    det.limit_side_len = side;
                    det.limit_type = "max".to_string();
                }
                if let Some(thresh) = meta.det_thresh {
                    det.thresh = thresh;
                }
                if let Some(bthresh) = meta.det_box_thresh {
                    det.box_thresh = bthresh;
                }
            }
        }
        rec.rec_keys_path = Some(dict);
        rec.rec_img_shape = preset.rec_img_shape;
        rec.rec_batch_num = preset.rec_batch_num;

        let global = GlobalConfig {
            text_score: preset.text_score,
            ..Default::default()
        };

        Self {
            det,
            rec,
            global,
            orient: None,
            unwarp: None,
            cls: None,
            layout: None,
            calibration: None,
            optimization: None,
        }
    }

    /// Create a configuration for PP-OCRv6 models
    pub fn ppv6<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(PPV6_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Create a configuration for PP-OCRv5 models
    pub fn ppv5<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(PPV5_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Create a configuration for PP-OCRv4 models
    pub fn ppv4<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(PPV4_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Create a configuration for PP-OCRv3 models
    pub fn ppv3<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(PPV3_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Create a configuration for Indra OCR models (Linenet with rec_batch_num = 1 default)
    pub fn indra<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(INDRA_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Parse configuration from a JSON string (supports direct structured and grouped JSON formats)
    pub fn from_json(json_str: &str) -> Result<Self, CoreError> {
        let flat: FlatConfig = nanoserde::DeJson::deserialize_json(json_str)
            .map_err(|e| CoreError::Config(format!("Failed to parse config JSON: {}", e)))?;

        let cache_dir = models::get_cache_dir(flat.cache_dir().map(std::path::Path::new));

        let det_override = flat
            .det_model_path()
            .or_else(|| flat.detection.as_ref().and_then(|d| d.model_path()));
        let rec_override = flat
            .rec_model_path()
            .or_else(|| flat.recognition.as_ref().and_then(|r| r.model_path()));
        let dict_override = flat
            .dict_path()
            .or_else(|| flat.recognition.as_ref().and_then(|r| r.dict_path()));

        let effective_model = flat.model();

        let effective_dir = flat.model_path().or_else(|| {
            effective_model.and_then(|m| {
                let p = std::path::Path::new(m);
                if p.exists() {
                    Some(m)
                } else {
                    None
                }
            })
        }).map(std::path::Path::new);

        let (det_path, rec_path, dict_path) = {
            let canonical_model = models::normalize_model_name(effective_model);
            let base_assets = models::get_base_pipeline_assets(canonical_model);

            let det = if let Some(p) = det_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(
                    None,
                    effective_dir,
                    "det",
                    base_assets.det_filename,
                    &cache_dir,
                )?
            };

            let (default_rec, default_dict) = if let Some(l) = flat.lang.as_deref() {
                if let Some((rec_file, dict_file)) = models::resolve_model_language_assets(canonical_model, l) {
                    (rec_file, dict_file)
                } else {
                    (base_assets.rec_filename, base_assets.dict_filename)
                }
            } else {
                (base_assets.rec_filename, base_assets.dict_filename)
            };

            let rec = if let Some(p) = rec_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(
                    None,
                    effective_dir,
                    "rec",
                    default_rec,
                    &cache_dir,
                )?
            };

            let dict = if let Some(p) = dict_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(
                    None,
                    effective_dir,
                    "dict",
                    default_dict,
                    &cache_dir,
                )?
            };

            (det, rec, dict)
        };

        let model_str = effective_model.unwrap_or("").to_ascii_lowercase();
        let is_indra = model_str.contains("indra")
            || model_str.contains("linenet")
            || flat.template.as_deref().is_some_and(|t| t.eq_ignore_ascii_case("indra") || t.eq_ignore_ascii_case("linenet"))
            || det_path.to_string_lossy().to_ascii_lowercase().contains("indra")
            || rec_path.to_string_lossy().to_ascii_lowercase().contains("indra");

        let preset = if is_indra {
            INDRA_MODEL_CONFIG
        } else if model_str.contains("v5") || model_str.contains("ppocrv5") {
            PPV5_MODEL_CONFIG
        } else if model_str.contains("v4") || model_str.contains("ppocrv4") {
            PPV4_MODEL_CONFIG
        } else if model_str.contains("v3") || model_str.contains("ppocrv3") {
            PPV3_MODEL_CONFIG
        } else {
            PPV6_MODEL_CONFIG
        };

        let mut config = Self::from_preset(preset, det_path, rec_path, dict_path);

        let orient_override = flat
            .orient_model_path()
            .or_else(|| flat.orientation.as_ref().and_then(|o| o.model_path()));
        let orient_enabled = flat
            .use_orient()
            .or_else(|| flat.orientation.as_ref().and_then(|o| o.enabled()))
            .unwrap_or(orient_override.is_some());
        if orient_enabled {
            let orient_path = if let Some(p) = orient_override {
                PathBuf::from(p)
            } else {
                let canonical = models::normalize_model_name(effective_model);
                let orient_file = if canonical == "indraocr" {
                    "indra_orient.rten"
                } else {
                    "ppocr_orient.rten"
                };
                models::resolve_asset_local(None, effective_dir, "orient", orient_file, &cache_dir)?
            };
            if let Some(thresh) = flat.orientation.as_ref().and_then(|o| o.threshold()) {
                config = config.with_orientation_threshold(orient_path, thresh);
            } else {
                config = config.with_orientation(orient_path);
            }
        }

        let cls_override = flat
            .cls_model_path()
            .or_else(|| flat.classification.as_ref().and_then(|c| c.model_path()));
        let cls_enabled = flat
            .use_cls()
            .or_else(|| flat.classification.as_ref().and_then(|c| c.enabled()))
            .unwrap_or(cls_override.is_some());
        if cls_enabled {
            let cls_path = if let Some(p) = cls_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(None, effective_dir, "cls", "ppocr_cls.rten", &cache_dir)?
            };
            if let Some(thresh) = flat.classification.as_ref().and_then(|c| c.threshold()) {
                config = config.with_cls_threshold(cls_path, thresh);
            } else {
                config = config.with_cls(cls_path);
            }
        }

        let unwarp_override = flat
            .unwarp_model_path()
            .or_else(|| flat.unwarp.as_ref().and_then(|u| u.model_path()));
        let unwarp_enabled = flat
            .use_unwarp()
            .or_else(|| flat.unwarp.as_ref().and_then(|u| u.enabled()))
            .unwrap_or(unwarp_override.is_some());
        if unwarp_enabled {
            let unwarp_path = if let Some(p) = unwarp_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(None, effective_dir, "unwarp", "uvdoc.rten", &cache_dir)?
            };
            config = config.with_unwarp(unwarp_path);
        }

        let layout_override = flat
            .layout_model_path()
            .or_else(|| flat.layout.as_ref().and_then(|l| l.model_path()));
        let layout_enabled = flat
            .use_layout()
            .or_else(|| flat.layout.as_ref().and_then(|l| l.enabled()))
            .unwrap_or(layout_override.is_some() || flat.layout_variant().is_some());
        if layout_enabled {
            let layout_path = if let Some(p) = layout_override {
                PathBuf::from(p)
            } else {
                let layout_file = if flat.layout_variant() == Some("publaynet") { "layout_publaynet.rten" } else { "layout_cdla.rten" };
                models::resolve_asset_local(None, effective_dir, "layout", layout_file, &cache_dir)?
            };
            let mut l_cfg = LayoutConfig::default(layout_path);
            if let Some(conf) = flat.layout.as_ref().and_then(|l| l.conf_thresh()) {
                l_cfg.conf_thresh = conf;
            }
            if let Some(target_size) = flat.layout.as_ref().and_then(|l| l.target_size()) {
                l_cfg.target_size = target_size;
            }
            if let Some(target_height) = flat.layout.as_ref().and_then(|l| l.target_height()) {
                l_cfg.target_height = Some(target_height);
            }
            if let Some(target_width) = flat.layout.as_ref().and_then(|l| l.target_width()) {
                l_cfg.target_width = Some(target_width);
            }
            config = config.with_layout_config(l_cfg);
        }

        if flat.enable_reading_order().unwrap_or(false) {
            let order_path = models::resolve_asset_local(None, effective_dir, "order", "indra_order.rten", &cache_dir)?;
            config.global.reading_order = true;
            config.global.reading_order_model = Some(order_path);
        }

        if let Some(score) = flat.text_score().or_else(|| flat.recognition.as_ref().and_then(|r| r.score_thresh())) {
            config.global.text_score = score;
        }
        if let Some(thresh) = flat.det_thresh().or_else(|| flat.detection.as_ref().and_then(|d| d.thresh())) {
            config.det.thresh = thresh;
        }
        if let Some(box_thresh) = flat.box_thresh().or_else(|| flat.detection.as_ref().and_then(|d| d.box_thresh())) {
            config.det.box_thresh = box_thresh;
        }
        if let Some(side_len) = flat.limit_side_len().or_else(|| flat.detection.as_ref().and_then(|d| d.limit_side_len())) {
            config.det.limit_side_len = side_len;
        }
        if let Some(limit_type) = flat.limit_type().or_else(|| flat.detection.as_ref().and_then(|d| d.limit_type())) {
            config.det.limit_type = limit_type.to_string();
        }
        if let Some(unclip) = flat.unclip_ratio().or_else(|| flat.detection.as_ref().and_then(|d| d.unclip_ratio())) {
            config.det.unclip_ratio = unclip;
        }
        if let Some(max_cand) = flat.max_candidates() {
            config.det.max_candidates = max_cand;
        }
        if let Some(dilation) = flat.use_dilation().or_else(|| flat.detection.as_ref().and_then(|d| d.use_dilation())) {
            config.det.use_dilation = dilation;
        }
        if let Some(score_mode) = flat.score_mode().or_else(|| flat.detection.as_ref().and_then(|d| d.score_mode())) {
            config.det.score_mode = score_mode.to_string();
        }
        if let Some(use_det) = flat.use_det().or_else(|| flat.detection.as_ref().and_then(|d| d.enabled())) {
            config.global.use_det = use_det;
        }
        if let Some(use_rec) = flat.use_rec().or_else(|| flat.recognition.as_ref().and_then(|r| r.enabled())) {
            config.global.use_rec = use_rec;
        }
        if let Some(use_cls) = flat.use_cls().or_else(|| flat.classification.as_ref().and_then(|c| c.enabled())) {
            config.global.use_cls = use_cls;
        }
        if let Some(use_orient) = flat.use_orient().or_else(|| flat.orientation.as_ref().and_then(|o| o.enabled())) {
            config.global.use_orient = use_orient;
        }
        if let Some(use_unwarp) = flat.use_unwarp().or_else(|| flat.unwarp.as_ref().and_then(|u| u.enabled())) {
            config.global.use_unwarp = use_unwarp;
        }
        if let Some(use_layout) = flat.use_layout().or_else(|| flat.layout.as_ref().and_then(|l| l.enabled())) {
            config.global.use_layout = use_layout;
        }
        if let Some(word_box) = flat.return_word_box().or_else(|| flat.recognition.as_ref().and_then(|r| r.return_word_box())) {
            config.global.return_word_box = word_box;
        }
        if let Some(char_box) = flat.return_single_char_box().or_else(|| flat.recognition.as_ref().and_then(|r| r.return_single_char_box())) {
            config.global.return_single_char_box = char_box;
        }
        if let Some(batch_num) = flat.rec_batch_num().or_else(|| flat.recognition.as_ref().and_then(|r| r.rec_batch_num())) {
            config.rec.rec_batch_num = batch_num;
        }
        if let Some(workers) = flat.rec_workers().or_else(|| flat.recognition.as_ref().and_then(|r| r.rec_workers())) {
            config.rec.rec_workers = Some(workers);
        }
        if let Some(max_workers) = flat.max_rec_workers().or_else(|| flat.recognition.as_ref().and_then(|r| r.max_rec_workers())) {
            config.rec.max_rec_workers = max_workers;
        }
        if let Some(y_mult) = flat.layout.as_ref().and_then(|l| l.y_threshold_multiplier()) {
            config.global.y_threshold_multiplier = Some(y_mult);
        }
        if let Some(x_mult) = flat.layout.as_ref().and_then(|l| l.x_threshold_multiplier()) {
            config.global.x_threshold_multiplier = Some(x_mult);
        }

        Ok(config)
    }

    /// Serialize configuration to a JSON string
    pub fn to_json(&self) -> Result<String, CoreError> {
        let template = if self.det.thresh == INDRA_MODEL_CONFIG.det_thresh && self.rec.rec_batch_num == 1 {
            "indra"
        } else {
            "ppv6"
        };
        let mut flat = FlatConfig::default();
        flat.template = Some(template.to_string());
        flat.det_model_path_snake = Some(self.det.model_path.to_string_lossy().to_string());
        flat.rec_model_path_snake = Some(self.rec.model_path.to_string_lossy().to_string());
        if let Some(ref d) = self.rec.rec_keys_path {
            flat.dict_path_snake = Some(d.to_string_lossy().to_string());
        }
        flat.det_thresh_snake = Some(self.det.thresh);
        flat.box_thresh_snake = Some(self.det.box_thresh);
        flat.unclip_ratio_snake = Some(self.det.unclip_ratio);
        flat.text_score_snake = Some(self.global.text_score);
        flat.use_det_snake = Some(self.global.use_det);
        flat.use_rec_snake = Some(self.global.use_rec);
        flat.use_cls_snake = Some(self.global.use_cls);
        flat.use_orient_snake = Some(self.global.use_orient);
        flat.use_unwarp_snake = Some(self.global.use_unwarp);
        flat.use_layout_snake = Some(self.global.use_layout);
        Ok(nanoserde::SerJson::serialize_json(&flat))
    }

    /// Set detection model path
    pub fn with_det_model<P: Into<PathBuf>>(mut self, model_path: P) -> Self {
        self.det.model_path = model_path.into();
        self
    }

    /// Set recognition model path
    pub fn with_rec_model<P: Into<PathBuf>>(mut self, model_path: P) -> Self {
        self.rec.model_path = model_path.into();
        self
    }

    /// Set recognition dictionary path
    pub fn with_dict<P: Into<PathBuf>>(mut self, dict_path: P) -> Self {
        self.rec.rec_keys_path = Some(dict_path.into());
        self
    }
    
    /// Add orientation classification to the configuration
    pub fn with_orientation<P: Into<PathBuf>>(mut self, model_path: P) -> Self {
        self.orient = Some(OrientConfig::default(model_path.into()));
        self.global.use_orient = true;
        self
    }
    
    /// Add orientation classification with custom confidence threshold
    pub fn with_orientation_threshold<P: Into<PathBuf>>(mut self, model_path: P, threshold: f32) -> Self {
        let mut config = OrientConfig::default(model_path.into());
        config.confidence_threshold = threshold;
        config.orient_thresh = threshold;
        self.orient = Some(config);
        self.global.use_orient = true;
        self
    }
    
    /// Add text rectification to the configuration
    pub fn with_unwarp<P: Into<PathBuf>>(mut self, model_path: P) -> Self {
        self.unwarp = Some(UnwarpConfig::default(model_path.into()));
        self.global.use_unwarp = true;
        self
    }
    
    /// Add text classification (line orientation) to the configuration
    pub fn with_cls<P: Into<PathBuf>>(mut self, model_path: P) -> Self {
        self.cls = Some(ClsConfig::default(model_path.into()));
        self.global.use_cls = true;
        self
    }

    /// Add text classification (line orientation) with custom confidence threshold
    pub fn with_cls_threshold<P: Into<PathBuf>>(mut self, model_path: P, threshold: f32) -> Self {
        let mut config = ClsConfig::default(model_path.into());
        config.cls_thresh = threshold;
        self.cls = Some(config);
        self.global.use_cls = true;
        self
    }

    /// Add document layout detection to the configuration
    pub fn with_layout<P: Into<PathBuf>>(mut self, model_path: P) -> Self {
        self.layout = Some(LayoutConfig::default(model_path.into()));
        self.global.use_layout = true;
        self
    }

    /// Add document layout detection with custom layout configuration
    pub fn with_layout_config(mut self, layout: LayoutConfig) -> Self {
        self.layout = Some(layout);
        self.global.use_layout = true;
        self
    }
    
    /// Enable debug images (oriented and rectified images will be included in output)
    pub fn with_debug_images(mut self, enabled: bool) -> Self {
        self.global.debug_images = enabled;
        self
    }
    
    /// Set text confidence score threshold (0.0 to 1.0)
    pub fn with_text_score(mut self, score: f32) -> Self {
        self.global.text_score = score;
        self
    }
    
    /// Set minimum text box height
    pub fn with_min_height(mut self, height: f32) -> Self {
        self.global.min_height = height;
        self
    }
    
    /// Set maximum image side length for detection
    pub fn with_max_side_len(mut self, len: f32) -> Self {
        self.global.max_side_len = len;
        self
    }

    /// Set minimum image side length for detection
    pub fn with_min_side_len(mut self, len: f32) -> Self {
        self.global.min_side_len = len;
        self
    }
    
    /// Enable/disable text detection
    pub fn with_detection(mut self, enabled: bool) -> Self {
        self.global.use_det = enabled;
        self
    }
    
    /// Enable/disable text recognition
    pub fn with_recognition(mut self, enabled: bool) -> Self {
        self.global.use_rec = enabled;
        self
    }

    /// Set XY threshold multipliers for spatial text line grouping and column spacing
    pub fn with_xy_threshold(mut self, y_threshold: f32, x_threshold: f32) -> Self {
        self.global.y_threshold_multiplier = Some(y_threshold);
        self.global.x_threshold_multiplier = Some(x_threshold);
        self
    }

    /// Set Y threshold multiplier for line grouping in spatial text
    pub fn with_y_threshold(mut self, y_threshold: f32) -> Self {
        self.global.y_threshold_multiplier = Some(y_threshold);
        self
    }

    /// Set X threshold multiplier for word/column gap separation in spatial text
    pub fn with_x_threshold(mut self, x_threshold: f32) -> Self {
        self.global.x_threshold_multiplier = Some(x_threshold);
        self
    }

    /// Set detection binarization threshold (0.0 to 1.0)
    pub fn with_det_thresh(mut self, thresh: f32) -> Self {
        self.det.thresh = thresh;
        self
    }

    /// Set detection box score threshold (0.0 to 1.0)
    pub fn with_det_box_thresh(mut self, box_thresh: f32) -> Self {
        self.det.box_thresh = box_thresh;
        self
    }

    /// Set detection limit side length
    pub fn with_limit_side_len(mut self, len: i32) -> Self {
        self.det.limit_side_len = len;
        self
    }

    /// Set detection limit type ("min" or "max")
    pub fn with_limit_type(mut self, limit_type: impl Into<String>) -> Self {
        self.det.limit_type = limit_type.into();
        self
    }

    /// Set detection unclip ratio
    pub fn with_unclip_ratio(mut self, ratio: f32) -> Self {
        self.det.unclip_ratio = ratio;
        self
    }

    /// Set detection dilation flag
    pub fn with_use_dilation(mut self, use_dilation: bool) -> Self {
        self.det.use_dilation = use_dilation;
        self
    }

    /// Enable/disable return word box
    pub fn with_return_word_box(mut self, enabled: bool) -> Self {
        self.global.return_word_box = enabled;
        self
    }

    /// Enable/disable return single char box
    pub fn with_return_single_char_box(mut self, enabled: bool) -> Self {
        self.global.return_single_char_box = enabled;
        self
    }

    /// Set recognition input image shape [channels, height, width]
    pub fn with_rec_img_shape(mut self, shape: [i32; 3]) -> Self {
        self.rec.rec_img_shape = shape;
        self
    }

    /// Set recognition batch size
    pub fn with_rec_batch_num(mut self, batch_num: i32) -> Self {
        self.rec.rec_batch_num = batch_num;
        self
    }

    /// Set number of parallel workers for text recognition
    pub fn with_rec_workers(mut self, workers: usize) -> Self {
        self.rec.rec_workers = Some(workers);
        self
    }

    /// Set maximum parallel workers clamp for text recognition (default: 8)
    pub fn with_max_rec_workers(mut self, max_workers: usize) -> Self {
        self.rec.max_rec_workers = max_workers;
        self
    }

    /// Set maximum detection candidates
    pub fn with_max_candidates(mut self, max_candidates: i32) -> Self {
        self.det.max_candidates = max_candidates;
        self
    }

    /// Set detection score mode ("fast" or "slow")
    pub fn with_score_mode(mut self, score_mode: impl Into<String>) -> Self {
        self.det.score_mode = score_mode.into();
        self
    }

    /// Set detection normalization mean
    pub fn with_det_mean(mut self, mean: [f32; 3]) -> Self {
        self.det.mean = mean;
        self
    }

    /// Set detection normalization standard deviation
    pub fn with_det_std(mut self, std: [f32; 3]) -> Self {
        self.det.std = std;
        self
    }

    /// Set aspect ratio threshold for long text boxes
    pub fn with_width_height_ratio(mut self, ratio: f32) -> Self {
        self.global.width_height_ratio = ratio;
        self
    }

    /// Set pre-OCR image calibration options
    pub fn with_calibration(mut self, calibration: CalibrationOptions) -> Self {
        self.global.calibration = Some(calibration);
        self
    }

    /// Set runtime image optimization options
    pub fn with_optimization(mut self, optimization: OptimizationOptions) -> Self {
        self.global.optimization = Some(optimization);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json_with_model_path_and_override() {
        let json = r#"{
            "modelPath": "models/PPOCR_v6_tiny",
            "detection": {
                "modelPath": "Cargo.toml"
            }
        }"#;

        let config = InitializeConfig::from_json(json).unwrap();
        assert_eq!(config.det.model_path, PathBuf::from("Cargo.toml"));
        assert_eq!(config.rec.model_path, PathBuf::from("models/PPOCR_v6_tiny/rec.mnn"));
        assert_eq!(config.rec.rec_keys_path, Some(PathBuf::from("models/PPOCR_v6_tiny/dict.txt")));
    }
}
