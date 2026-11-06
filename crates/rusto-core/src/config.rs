use std::path::PathBuf;
use serde::{Deserialize, Serialize};

use crate::calibration::{CalibrationOptions, OptimizationOptions};
use crate::models;
use crate::types::{ClsConfig, DetConfig, GlobalConfig, LayoutConfig, OrientConfig, RecConfig, UnwarpConfig};

/// Preset configuration for OCR model architectures
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
    det_limit_side_len: 736,
    det_limit_type: "min",
    det_thresh: 0.3,
    det_box_thresh: 0.6,
    det_unclip_ratio: 2.0,
    det_use_dilation: true,
    det_score_mode: "slow",
    rec_img_shape: [1, 48, 320],
    rec_batch_num: 1,
    text_score: 0.5,
};

/// Main configuration for OCR engine
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        Self::from_preset(PPV6_MODEL_CONFIG, "det.mnn", "rec.mnn", "dict.txt")
    }
}

#[derive(Deserialize, Default, Clone, Debug)]
struct DetectionConfigJson {
    #[serde(alias = "useDet", alias = "use_det")]
    enabled: Option<bool>,
    #[serde(alias = "modelPath", alias = "detModelPath", alias = "model_path", alias = "det_model_path")]
    model_path: Option<PathBuf>,
    #[serde(alias = "detThresh", alias = "det_thresh")]
    thresh: Option<f32>,
    #[serde(alias = "boxThresh", alias = "detBoxThresh", alias = "box_thresh", alias = "det_box_thresh")]
    box_thresh: Option<f32>,
    #[serde(alias = "unclipRatio", alias = "unclip_ratio")]
    unclip_ratio: Option<f32>,
    #[serde(alias = "limitSideLen", alias = "limit_side_len")]
    limit_side_len: Option<i32>,
    #[serde(alias = "limitType", alias = "limit_type")]
    limit_type: Option<String>,
    #[serde(alias = "useDilation", alias = "use_dilation")]
    use_dilation: Option<bool>,
    #[serde(alias = "scoreMode", alias = "score_mode", alias = "detScoreMode", alias = "det_score_mode")]
    score_mode: Option<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct RecognitionConfigJson {
    #[serde(alias = "useRec", alias = "use_rec")]
    enabled: Option<bool>,
    #[serde(alias = "modelPath", alias = "recModelPath", alias = "model_path", alias = "rec_model_path")]
    model_path: Option<PathBuf>,
    #[serde(alias = "dictPath", alias = "dict_path")]
    dict_path: Option<PathBuf>,
    #[serde(alias = "scoreThresh", alias = "textScore", alias = "score_thresh", alias = "text_score")]
    score_thresh: Option<f32>,
    #[serde(alias = "returnWordBox", alias = "return_word_box")]
    return_word_box: Option<bool>,
    #[serde(alias = "returnSingleCharBox", alias = "return_single_char_box")]
    return_single_char_box: Option<bool>,
    #[serde(alias = "recBatchNum", alias = "rec_batch_num", alias = "batchNum", alias = "batch_num")]
    rec_batch_num: Option<i32>,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct ClassificationConfigJson {
    #[serde(alias = "useCls", alias = "use_cls")]
    enabled: Option<bool>,
    #[serde(alias = "modelPath", alias = "clsModelPath", alias = "model_path", alias = "cls_model_path")]
    model_path: Option<PathBuf>,
    #[serde(alias = "thresh", alias = "clsThreshold", alias = "cls_thresh", alias = "cls_threshold")]
    threshold: Option<f32>,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct OrientationConfigJson {
    #[serde(alias = "useOrient", alias = "use_orient")]
    enabled: Option<bool>,
    #[serde(alias = "modelPath", alias = "orientModelPath", alias = "model_path", alias = "orient_model_path")]
    model_path: Option<PathBuf>,
    #[serde(alias = "thresh", alias = "orientThreshold", alias = "orient_thresh", alias = "orient_threshold")]
    threshold: Option<f32>,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct UnwarpConfigJson {
    #[serde(alias = "useUnwarp", alias = "use_unwarp")]
    enabled: Option<bool>,
    #[serde(alias = "modelPath", alias = "unwarpModelPath", alias = "model_path", alias = "unwarp_model_path")]
    model_path: Option<PathBuf>,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct LayoutConfigJson {
    #[serde(alias = "useLayout", alias = "use_layout")]
    enabled: Option<bool>,
    #[serde(alias = "modelPath", alias = "layoutModelPath", alias = "model_path", alias = "layout_model_path")]
    model_path: Option<PathBuf>,
    #[serde(alias = "confThresh", alias = "conf_thresh", alias = "threshold")]
    conf_thresh: Option<f32>,
    #[serde(alias = "targetSize", alias = "target_size")]
    target_size: Option<i32>,
    #[serde(alias = "targetHeight", alias = "target_height")]
    target_height: Option<i32>,
    #[serde(alias = "targetWidth", alias = "target_width")]
    target_width: Option<i32>,
    #[serde(alias = "yThresholdMultiplier", alias = "y_threshold_multiplier")]
    y_threshold_multiplier: Option<f32>,
    #[serde(alias = "xThresholdMultiplier", alias = "x_threshold_multiplier")]
    x_threshold_multiplier: Option<f32>,
}

#[derive(Deserialize, Default, Clone, Debug)]
struct FlatConfig {
    #[serde(alias = "models")]
    model: Option<String>,
    template: Option<String>,
    #[serde(alias = "modelPath", alias = "model_path", alias = "modelDir", alias = "model_dir")]
    model_path: Option<PathBuf>,
    lang: Option<String>,
    #[serde(alias = "cacheDir", alias = "cache_dir")]
    cache_dir: Option<PathBuf>,
    #[serde(alias = "enableReadingOrder", alias = "enable_reading_order", alias = "reading_order")]
    enable_reading_order: Option<bool>,
    #[serde(alias = "layoutVariant", alias = "layout_variant")]
    layout_variant: Option<String>,

    #[serde(alias = "detModelPath", alias = "det_model_path", alias = "detPath", alias = "det_path")]
    det_model_path: Option<PathBuf>,
    #[serde(alias = "recModelPath", alias = "rec_model_path", alias = "recPath", alias = "rec_path")]
    rec_model_path: Option<PathBuf>,
    #[serde(alias = "dictPath", alias = "dict_path")]
    dict_path: Option<PathBuf>,
    #[serde(alias = "clsModelPath", alias = "cls_model_path", alias = "clsPath", alias = "cls_path")]
    cls_model_path: Option<PathBuf>,
    #[serde(alias = "orientModelPath", alias = "orient_model_path", alias = "orientPath", alias = "orient_path")]
    orient_model_path: Option<PathBuf>,
    #[serde(alias = "unwarpModelPath", alias = "unwarp_model_path", alias = "unwarpPath", alias = "unwarp_path")]
    unwarp_model_path: Option<PathBuf>,
    #[serde(alias = "layoutModelPath", alias = "layout_model_path", alias = "layoutPath", alias = "layout_path")]
    layout_model_path: Option<PathBuf>,

    #[serde(alias = "detThresh", alias = "det_thresh")]
    det_thresh: Option<f32>,
    #[serde(alias = "detBoxThresh", alias = "box_thresh", alias = "boxThresh", alias = "det_box_thresh")]
    box_thresh: Option<f32>,
    #[serde(alias = "unclipRatio", alias = "unclip_ratio")]
    unclip_ratio: Option<f32>,
    #[serde(alias = "limitSideLen", alias = "limit_side_len")]
    limit_side_len: Option<i32>,
    #[serde(alias = "limitType", alias = "limit_type")]
    limit_type: Option<String>,
    #[serde(alias = "useDilation", alias = "use_dilation")]
    use_dilation: Option<bool>,
    #[serde(alias = "scoreMode", alias = "score_mode", alias = "detScoreMode", alias = "det_score_mode")]
    score_mode: Option<String>,
    #[serde(alias = "maxCandidates", alias = "max_candidates")]
    max_candidates: Option<i32>,

    #[serde(alias = "textScore", alias = "scoreThresh", alias = "score_thresh", alias = "text_score")]
    text_score: Option<f32>,
    #[serde(alias = "returnWordBox", alias = "return_word_box")]
    return_word_box: Option<bool>,
    #[serde(alias = "returnSingleCharBox", alias = "return_single_char_box")]
    return_single_char_box: Option<bool>,
    #[serde(alias = "recBatchNum", alias = "rec_batch_num", alias = "batchNum", alias = "batch_num")]
    rec_batch_num: Option<i32>,

    #[serde(alias = "useDet", alias = "use_det")]
    use_det: Option<bool>,
    #[serde(alias = "useRec", alias = "use_rec")]
    use_rec: Option<bool>,
    #[serde(alias = "useCls", alias = "use_cls")]
    use_cls: Option<bool>,
    #[serde(alias = "useOrient", alias = "use_orient")]
    use_orient: Option<bool>,
    #[serde(alias = "useUnwarp", alias = "use_unwarp")]
    use_unwarp: Option<bool>,
    #[serde(alias = "useLayout", alias = "use_layout")]
    use_layout: Option<bool>,

    detection: Option<DetectionConfigJson>,
    recognition: Option<RecognitionConfigJson>,
    classification: Option<ClassificationConfigJson>,
    orientation: Option<OrientationConfigJson>,
    unwarp: Option<UnwarpConfigJson>,
    layout: Option<LayoutConfigJson>,
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
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        // Try direct structured format first
        if let Ok(config) = serde_json::from_str::<InitializeConfig>(json_str) {
            return Ok(config);
        }

        // Fall back to grouped format
        let flat: FlatConfig = serde_json::from_str(json_str)?;

        let cache_dir = models::get_cache_dir(flat.cache_dir.as_deref());

        let det_override = flat
            .det_model_path
            .as_deref()
            .or_else(|| flat.detection.as_ref().and_then(|d| d.model_path.as_deref()));
        let rec_override = flat
            .rec_model_path
            .as_deref()
            .or_else(|| flat.recognition.as_ref().and_then(|r| r.model_path.as_deref()));
        let dict_override = flat
            .dict_path
            .as_deref()
            .or_else(|| flat.recognition.as_ref().and_then(|r| r.dict_path.as_deref()));

        let effective_model = flat.model.as_deref().or(flat.template.as_deref());

        let effective_dir = flat.model_path.as_deref().or_else(|| {
            effective_model.and_then(|m| {
                let p = std::path::Path::new(m);
                if p.exists() {
                    Some(p)
                } else {
                    None
                }
            })
        });

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
                ).map_err(|e| serde::de::Error::custom(e.to_string()))?
            };

            let (default_rec, default_dict) = if let Some(l) = flat.lang.as_deref() {
                if let Some((rec_file, dict_file)) = models::resolve_language_assets(l) {
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
                ).map_err(|e| serde::de::Error::custom(e.to_string()))?
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
                ).map_err(|e| serde::de::Error::custom(e.to_string()))?
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
            .orient_model_path
            .as_deref()
            .or_else(|| flat.orientation.as_ref().and_then(|o| o.model_path.as_deref()));
        let orient_enabled = flat
            .use_orient
            .or_else(|| flat.orientation.as_ref().and_then(|o| o.enabled))
            .unwrap_or(orient_override.is_some());
        if orient_enabled {
            let orient_path = if let Some(p) = orient_override {
                PathBuf::from(p)
            } else {
                let canonical = models::normalize_model_name(effective_model);
                let orient_file = if canonical == "indraocr" {
                    "indra_orient.mnn"
                } else {
                    "ppocr_orient.mnn"
                };
                models::resolve_asset_local(None, effective_dir, "orient", orient_file, &cache_dir)
                    .map_err(|e| serde::de::Error::custom(e.to_string()))?
            };
            if let Some(thresh) = flat.orientation.as_ref().and_then(|o| o.threshold) {
                config = config.with_orientation_threshold(orient_path, thresh);
            } else {
                config = config.with_orientation(orient_path);
            }
        }

        let cls_override = flat
            .cls_model_path
            .as_deref()
            .or_else(|| flat.classification.as_ref().and_then(|c| c.model_path.as_deref()));
        let cls_enabled = flat
            .use_cls
            .or_else(|| flat.classification.as_ref().and_then(|c| c.enabled))
            .unwrap_or(cls_override.is_some());
        if cls_enabled {
            let cls_path = if let Some(p) = cls_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(None, effective_dir, "cls", "ppocr_cls.mnn", &cache_dir)
                    .map_err(|e| serde::de::Error::custom(e.to_string()))?
            };
            if let Some(thresh) = flat.classification.as_ref().and_then(|c| c.threshold) {
                config = config.with_cls_threshold(cls_path, thresh);
            } else {
                config = config.with_cls(cls_path);
            }
        }

        let unwarp_override = flat
            .unwarp_model_path
            .as_deref()
            .or_else(|| flat.unwarp.as_ref().and_then(|u| u.model_path.as_deref()));
        let unwarp_enabled = flat
            .use_unwarp
            .or_else(|| flat.unwarp.as_ref().and_then(|u| u.enabled))
            .unwrap_or(unwarp_override.is_some());
        if unwarp_enabled {
            let unwarp_path = if let Some(p) = unwarp_override {
                PathBuf::from(p)
            } else {
                models::resolve_asset_local(None, effective_dir, "unwarp", "uvdoc.mnn", &cache_dir)
                    .map_err(|e| serde::de::Error::custom(e.to_string()))?
            };
            config = config.with_unwarp(unwarp_path);
        }

        let layout_override = flat
            .layout_model_path
            .as_deref()
            .or_else(|| flat.layout.as_ref().and_then(|l| l.model_path.as_deref()));
        let layout_enabled = flat
            .use_layout
            .or_else(|| flat.layout.as_ref().and_then(|l| l.enabled))
            .unwrap_or(layout_override.is_some() || flat.layout_variant.is_some());
        if layout_enabled {
            let layout_path = if let Some(p) = layout_override {
                PathBuf::from(p)
            } else {
                let layout_file = if flat.layout_variant.as_deref() == Some("publaynet") { "layout_publaynet.mnn" } else { "layout_cdla.mnn" };
                models::resolve_asset_local(None, effective_dir, "layout", layout_file, &cache_dir)
                    .map_err(|e| serde::de::Error::custom(e.to_string()))?
            };
            let mut l_cfg = LayoutConfig::default(layout_path);
            if let Some(conf) = flat.layout.as_ref().and_then(|l| l.conf_thresh) {
                l_cfg.conf_thresh = conf;
            }
            if let Some(target_size) = flat.layout.as_ref().and_then(|l| l.target_size) {
                l_cfg.target_size = target_size;
            }
            if let Some(target_height) = flat.layout.as_ref().and_then(|l| l.target_height) {
                l_cfg.target_height = Some(target_height);
            }
            if let Some(target_width) = flat.layout.as_ref().and_then(|l| l.target_width) {
                l_cfg.target_width = Some(target_width);
            }
            config = config.with_layout_config(l_cfg);
        }

        if flat.enable_reading_order.unwrap_or(false) {
            let order_path = models::resolve_asset_local(None, effective_dir, "order", "indra_order.mnn", &cache_dir)
                .map_err(|e| serde::de::Error::custom(e.to_string()))?;
            config.global.reading_order = true;
            config.global.reading_order_model = Some(order_path);
        }

        if let Some(score) = flat.text_score.or_else(|| flat.recognition.as_ref().and_then(|r| r.score_thresh)) {
            config.global.text_score = score;
        }
        if let Some(thresh) = flat.det_thresh.or_else(|| flat.detection.as_ref().and_then(|d| d.thresh)) {
            config.det.thresh = thresh;
        }
        if let Some(box_thresh) = flat.box_thresh.or_else(|| flat.detection.as_ref().and_then(|d| d.box_thresh)) {
            config.det.box_thresh = box_thresh;
        }
        if let Some(side_len) = flat.limit_side_len.or_else(|| flat.detection.as_ref().and_then(|d| d.limit_side_len)) {
            config.det.limit_side_len = side_len;
        }
        if let Some(limit_type) = flat.limit_type.or_else(|| flat.detection.as_ref().and_then(|d| d.limit_type.clone())) {
            config.det.limit_type = limit_type;
        }
        if let Some(unclip) = flat.unclip_ratio.or_else(|| flat.detection.as_ref().and_then(|d| d.unclip_ratio)) {
            config.det.unclip_ratio = unclip;
        }
        if let Some(max_cand) = flat.max_candidates {
            config.det.max_candidates = max_cand;
        }
        if let Some(dilation) = flat.use_dilation.or_else(|| flat.detection.as_ref().and_then(|d| d.use_dilation)) {
            config.det.use_dilation = dilation;
        }
        if let Some(score_mode) = flat.score_mode.or_else(|| flat.detection.as_ref().and_then(|d| d.score_mode.clone())) {
            config.det.score_mode = score_mode;
        }
        if let Some(use_det) = flat.use_det.or_else(|| flat.detection.as_ref().and_then(|d| d.enabled)) {
            config.global.use_det = use_det;
        }
        if let Some(use_rec) = flat.use_rec.or_else(|| flat.recognition.as_ref().and_then(|r| r.enabled)) {
            config.global.use_rec = use_rec;
        }
        if let Some(use_cls) = flat.use_cls.or_else(|| flat.classification.as_ref().and_then(|c| c.enabled)) {
            config.global.use_cls = use_cls;
        }
        if let Some(use_orient) = flat.use_orient.or_else(|| flat.orientation.as_ref().and_then(|o| o.enabled)) {
            config.global.use_orient = use_orient;
        }
        if let Some(use_unwarp) = flat.use_unwarp.or_else(|| flat.unwarp.as_ref().and_then(|u| u.enabled)) {
            config.global.use_unwarp = use_unwarp;
        }
        if let Some(use_layout) = flat.use_layout.or_else(|| flat.layout.as_ref().and_then(|l| l.enabled)) {
            config.global.use_layout = use_layout;
        }
        if let Some(word_box) = flat.return_word_box.or_else(|| flat.recognition.as_ref().and_then(|r| r.return_word_box)) {
            config.global.return_word_box = word_box;
        }
        if let Some(char_box) = flat.return_single_char_box.or_else(|| flat.recognition.as_ref().and_then(|r| r.return_single_char_box)) {
            config.global.return_single_char_box = char_box;
        }
        if let Some(batch_num) = flat.rec_batch_num.or_else(|| flat.recognition.as_ref().and_then(|r| r.rec_batch_num)) {
            config.rec.rec_batch_num = batch_num;
        }
        if let Some(y_mult) = flat.layout.as_ref().and_then(|l| l.y_threshold_multiplier) {
            config.global.y_threshold_multiplier = Some(y_mult);
        }
        if let Some(x_mult) = flat.layout.as_ref().and_then(|l| l.x_threshold_multiplier) {
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
