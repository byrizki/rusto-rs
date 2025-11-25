use std::path::PathBuf;
use crate::types::{ClsConfig, DetConfig, RecConfig, OrientConfig, UnwarpConfig, GlobalConfig};

/// Main configuration for RustO OCR engine
#[derive(Clone, Debug)]
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

impl RustOConfig {
    /// Create a basic configuration for PPOCRv5 models
    /// This is the simplest way to get started with reasonable defaults
    pub fn new_ppv5<P: Into<PathBuf>>(
        det_model_path: P,
        rec_model_path: P,
        dict_path: P,
    ) -> Self {
        let det_path = det_model_path.into();
        let rec_path = rec_model_path.into();
        let dict = dict_path.into();
        
        let det_config = DetConfig::ppv5(det_path);
        let mut rec_config = RecConfig::ppv5(rec_path);
        rec_config.rec_keys_path = Some(dict);
        
        Self {
            det: det_config,
            rec: rec_config,
            global: GlobalConfig::default(),
            orient: None,
            unwarp: None,
            cls: None,
        }
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
}
