use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::types::{ClsConfig, DetConfig, RecConfig, OrientConfig, UnwarpConfig, GlobalConfig};

/// Preset configuration for OCR model architectures
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelPreset {
    pub det_limit_side_len: i32,
    pub det_limit_type: &'static str,
    pub det_thresh: f32,
    pub det_box_thresh: f32,
    pub det_unclip_ratio: f32,
    pub det_use_dilation: bool,
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
    rec_img_shape: [3, 48, 320],
    rec_batch_num: 6,
    text_score: 0.5,
};

/// Main configuration for RustO OCR engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RustOConfig {
    /// Detection model configuration
    pub det: DetConfig,
    
    /// Recognition model configuration
    pub rec: RecConfig,
    
    /// Global OCR settings
    pub global: GlobalConfig,
    
    /// Optional: Orientation classification configuration
    pub orient: Option<OrientConfig>,
    
    /// Optional: Text rectification configuration
    pub unwarp: Option<UnwarpConfig>,
    
    /// Optional: Text classification (line orientation) configuration
    pub cls: Option<ClsConfig>,
}

impl Default for RustOConfig {
    fn default() -> Self {
        Self::from_preset(PPV6_MODEL_CONFIG, "det.mnn", "rec.mnn", "dict.txt")
    }
}

#[derive(Deserialize)]
struct FlatConfig {
    #[serde(alias = "template")]
    template: Option<String>,
    #[serde(alias = "detModelPath")]
    det_model_path: Option<PathBuf>,
    #[serde(alias = "recModelPath")]
    rec_model_path: Option<PathBuf>,
    #[serde(alias = "dictPath")]
    dict_path: Option<PathBuf>,
    #[serde(alias = "orientModelPath")]
    orient_model_path: Option<PathBuf>,
    #[serde(alias = "clsModelPath")]
    cls_model_path: Option<PathBuf>,
    #[serde(alias = "unwarpModelPath")]
    unwarp_model_path: Option<PathBuf>,
    #[serde(alias = "orientThreshold")]
    orient_threshold: Option<f32>,
    #[serde(alias = "clsThreshold")]
    cls_threshold: Option<f32>,
    #[serde(alias = "textScore")]
    text_score: Option<f32>,
    #[serde(alias = "detThresh")]
    det_thresh: Option<f32>,
    #[serde(alias = "detBoxThresh")]
    det_box_thresh: Option<f32>,
    #[serde(alias = "limitSideLen")]
    limit_side_len: Option<i32>,
    #[serde(alias = "limitType")]
    limit_type: Option<String>,
    #[serde(alias = "unclipRatio")]
    unclip_ratio: Option<f32>,
    #[serde(alias = "useDilation")]
    use_dilation: Option<bool>,
    #[serde(alias = "useDet")]
    use_det: Option<bool>,
    #[serde(alias = "useRec")]
    use_rec: Option<bool>,
    #[serde(alias = "useCls")]
    use_cls: Option<bool>,
    #[serde(alias = "useOrient")]
    use_orient: Option<bool>,
    #[serde(alias = "useUnwarp")]
    use_unwarp: Option<bool>,
    #[serde(alias = "debugImages")]
    debug_images: Option<bool>,
    #[serde(alias = "minHeight")]
    min_height: Option<f32>,
    #[serde(alias = "maxSideLen")]
    max_side_len: Option<f32>,
    #[serde(alias = "minSideLen")]
    min_side_len: Option<f32>,
    #[serde(alias = "returnWordBox")]
    return_word_box: Option<bool>,
    #[serde(alias = "returnSingleCharBox")]
    return_single_char_box: Option<bool>,
    #[serde(alias = "yThresholdMultiplier")]
    y_threshold_multiplier: Option<f32>,
    #[serde(alias = "xThresholdMultiplier")]
    x_threshold_multiplier: Option<f32>,
}

impl RustOConfig {
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

        let mut det = DetConfig::ppv6(det_path);
        det.limit_side_len = preset.det_limit_side_len;
        det.limit_type = preset.det_limit_type.to_string();
        det.thresh = preset.det_thresh;
        det.box_thresh = preset.det_box_thresh;
        det.unclip_ratio = preset.det_unclip_ratio;
        det.use_dilation = preset.det_use_dilation;

        let mut rec = RecConfig::ppv6(rec_path);
        rec.rec_keys_path = Some(dict);
        rec.rec_img_shape = preset.rec_img_shape;
        rec.rec_batch_num = preset.rec_batch_num;

        let mut global = GlobalConfig::default();
        global.text_score = preset.text_score;

        Self {
            det,
            rec,
            global,
            orient: None,
            unwarp: None,
            cls: None,
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

    /// Alias for ppv6
    pub fn new_ppv6<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::ppv6(det_model_path, rec_model_path, dict_path)
    }

    /// Create a configuration for PP-OCRv5 models
    pub fn ppv5<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::from_preset(PPV5_MODEL_CONFIG, det_model_path, rec_model_path, dict_path)
    }

    /// Alias for ppv5
    pub fn new_ppv5<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        Self::ppv5(det_model_path, rec_model_path, dict_path)
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

    /// Parse configuration from a JSON string (supports both structured and flat JSON formats)
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        // Try structured format first
        if let Ok(config) = serde_json::from_str::<RustOConfig>(json_str) {
            return Ok(config);
        }

        // Fall back to flat format
        let flat: FlatConfig = serde_json::from_str(json_str)?;
        let det_path = flat.det_model_path.unwrap_or_else(|| PathBuf::from("det.mnn"));
        let rec_path = flat.rec_model_path.unwrap_or_else(|| PathBuf::from("rec.mnn"));
        let dict_path = flat.dict_path.unwrap_or_else(|| PathBuf::from("dict.txt"));

        let preset = match flat.template.as_deref() {
            Some("ppv5") | Some("PPOCRv5") | Some("v5") | Some("pp-ocrv5") => PPV5_MODEL_CONFIG,
            Some("ppv4") | Some("PPOCRv4") | Some("v4") | Some("pp-ocrv4") => PPV4_MODEL_CONFIG,
            Some("ppv3") | Some("PPOCRv3") | Some("v3") | Some("pp-ocrv3") => PPV3_MODEL_CONFIG,
            _ => PPV6_MODEL_CONFIG,
        };

        let mut config = Self::from_preset(preset, det_path, rec_path, dict_path);

        if let Some(orient_path) = flat.orient_model_path {
            if let Some(thresh) = flat.orient_threshold {
                config = config.with_orientation_threshold(orient_path, thresh);
            } else {
                config = config.with_orientation(orient_path);
            }
        }

        if let Some(cls_path) = flat.cls_model_path {
            if let Some(thresh) = flat.cls_threshold {
                config = config.with_cls_threshold(cls_path, thresh);
            } else {
                config = config.with_cls(cls_path);
            }
        }

        if let Some(unwarp_path) = flat.unwarp_model_path {
            config = config.with_unwarp(unwarp_path);
        }

        if let Some(score) = flat.text_score {
            config.global.text_score = score;
        }
        if let Some(thresh) = flat.det_thresh {
            config.det.thresh = thresh;
        }
        if let Some(box_thresh) = flat.det_box_thresh {
            config.det.box_thresh = box_thresh;
        }
        if let Some(side_len) = flat.limit_side_len {
            config.det.limit_side_len = side_len;
        }
        if let Some(limit_type) = flat.limit_type {
            config.det.limit_type = limit_type;
        }
        if let Some(unclip) = flat.unclip_ratio {
            config.det.unclip_ratio = unclip;
        }
        if let Some(dilation) = flat.use_dilation {
            config.det.use_dilation = dilation;
        }
        if let Some(use_det) = flat.use_det {
            config.global.use_det = use_det;
        }
        if let Some(use_rec) = flat.use_rec {
            config.global.use_rec = use_rec;
        }
        if let Some(use_cls) = flat.use_cls {
            config.global.use_cls = use_cls;
        }
        if let Some(use_orient) = flat.use_orient {
            config.global.use_orient = use_orient;
        }
        if let Some(use_unwarp) = flat.use_unwarp {
            config.global.use_unwarp = use_unwarp;
        }
        if let Some(debug) = flat.debug_images {
            config.global.debug_images = debug;
        }
        if let Some(min_h) = flat.min_height {
            config.global.min_height = min_h;
        }
        if let Some(max_s) = flat.max_side_len {
            config.global.max_side_len = max_s;
        }
        if let Some(min_s) = flat.min_side_len {
            config.global.min_side_len = min_s;
        }
        if let Some(word_box) = flat.return_word_box {
            config.global.return_word_box = word_box;
        }
        if let Some(char_box) = flat.return_single_char_box {
            config.global.return_single_char_box = char_box;
        }
        if let Some(y_mult) = flat.y_threshold_multiplier {
            config.global.y_threshold_multiplier = Some(y_mult);
        }
        if let Some(x_mult) = flat.x_threshold_multiplier {
            config.global.x_threshold_multiplier = Some(x_mult);
        }

        Ok(config)
    }

    /// Serialize configuration to a JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
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
    
    /// Enable debug images (oriented and rectified images will be included in output)
    /// Note: This adds overhead and should only be used for debugging/development
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
}

