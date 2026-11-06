//! # RustO! - Pure Rust OCR Library
//!
//! RustO! is a high-performance OCR library written in pure Rust,
//! powered by PaddleOCR models with RTen inference engine.
//!
//! ## Features
//!
//! - **Pure Rust**: Zero OpenCV or C++ dependencies (optional OpenCV backend available)
//! - **High Accuracy**: 99.3% parity with OpenCV-based implementations
//! - **Fast Performance**: Optimized with LTO and aggressive compilation settings
//! - **Cross-Platform**: Linux, macOS, Windows, Android, iOS, WASM support
//! - **Memory Safe**: Leverages Rust's safety guarantees
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rusto::{DetectTextResult, ImageSource, OcrRunOptions, RustO, InitializeConfig};
//!
//! let mut ocr = RustO::initialize(InitializeConfig::ppv5(
//!     "models/det.rten", "models/rec.rten", "models/dict.txt",
//! ))?;
//! let result = ocr.detect_text(
//!     &ImageSource::Path("image.jpg".into()),
//!     &OcrRunOptions::default(),
//! )?;
//!
//! if let DetectTextResult::Structured(results) = result {
//!     for result in results {
//!         println!("{}: {:.3}", result.text, result.score);
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Canonical modules
pub mod download;
pub mod engine {
    pub use rusto_core::engine::*;
}
pub use rusto_core::models;

// Canonical model downloading APIs
pub use download::{
    download_asset, download_model_pipeline, download_target, stream_url_to_file,
};

/// Returns the engine library version string.
pub fn version() -> &'static str {
    rusto_core::version()
}


// Canonical public types and structs
pub use rusto_core::calibration::{CalibrationOptions, OptimizationOptions};
pub use rusto_core::config::{
    InitializeConfig, ModelPreset, INDRA_MODEL_CONFIG, PPV3_MODEL_CONFIG, PPV4_MODEL_CONFIG,
    PPV5_MODEL_CONFIG, PPV6_MODEL_CONFIG,
};
pub use rusto_core::doc::{DocBlock, DocResult};
pub use rusto_core::engine::{
    CoreError, DocPipeline, DocPipelineConfig, DocUnwarper, EngineError, EngineSession,
    InferenceSession, LayoutDetector, MnnSession, OrientClassifier, OrientationExt,
    register_session_loader, RustO, RustOOutput, SessionLoader, TableDetector,
    TableStructureRecognizer, TextDetector, TextRecognizer,
};
pub use rusto_core::layout::{BBox, LayoutOutput, LayoutRegion, LayoutType};
pub use rusto_core::orient::{OrientOutput, Orientation};
pub use rusto_core::rec::{TextRecOutput, WordInfo, WordType};
pub use rusto_core::table::{
    TableCell, TableDetectorConfig, TableModelType, TableRow, TableStructure,
    TableStructureConfig,
};
pub use rusto_core::types::{
    ClsConfig, DetConfig, DetectTextResult, DetectionRunOptions, Frame, GlobalConfig,
    ImageSource, LayoutConfig, OcrRunOptions, OrientConfig, OutputGranularity,
    PostDetectionOptions, PostRecognitionOptions, PostprocessRunOptions, PreDetectionOptions,
    PreRecognitionOptions, RecConfig, RecognitionRunOptions, TextResult, UnwarpConfig,
};

enum BuilderInput<'a> {
    Source(ImageSource),
    Rgb {
        pixels: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
    },
}

/// Fluent builder for constructing and executing OCR requests on a `RustO` engine instance.
pub struct DetectTextBuilder<'a> {
    engine: &'a mut rusto_core::engine::RustO,
    input: BuilderInput<'a>,
    options: OcrRunOptions,
}

impl<'a> DetectTextBuilder<'a> {
    pub fn new(engine: &'a mut rusto_core::engine::RustO, source: ImageSource) -> Self {
        Self {
            engine,
            input: BuilderInput::Source(source),
            options: OcrRunOptions::default(),
        }
    }

    pub fn new_rgb(
        engine: &'a mut rusto_core::engine::RustO,
        pixels: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Self {
        Self {
            engine,
            input: BuilderInput::Rgb {
                pixels,
                width,
                height,
                stride,
            },
            options: OcrRunOptions::default(),
        }
    }

    pub fn with_auto_unwarp(mut self, enabled: bool) -> Self {
        self.options.auto_unwarp = Some(enabled);
        self
    }

    pub fn with_layout_reading(mut self, enabled: bool) -> Self {
        self.options.layout_reading = Some(enabled);
        self
    }

    pub fn with_orientation_detection(mut self, enabled: bool) -> Self {
        self.options.orientation_detection = Some(enabled);
        self
    }

    pub fn with_line_classification(mut self, enabled: bool) -> Self {
        self.options.line_classification = Some(enabled);
        self
    }

    pub fn with_pre_detection(mut self, opts: PreDetectionOptions) -> Self {
        self.options.pre_detection = Some(opts);
        self
    }

    pub fn with_detection(mut self, opts: DetectionRunOptions) -> Self {
        self.options.detection = Some(opts);
        self
    }

    pub fn with_post_detection(mut self, opts: PostDetectionOptions) -> Self {
        self.options.post_detection = Some(opts);
        self
    }

    pub fn with_pre_recognition(mut self, opts: PreRecognitionOptions) -> Self {
        self.options.pre_recognition = Some(opts);
        self
    }

    pub fn with_recognition(mut self, opts: RecognitionRunOptions) -> Self {
        self.options.recognition = Some(opts);
        self
    }

    pub fn with_post_recognition(mut self, opts: PostRecognitionOptions) -> Self {
        self.options.post_recognition = Some(opts);
        self
    }

    pub fn as_lines(mut self) -> Self {
        self.options.output = OutputGranularity::Lines;
        self
    }

    pub fn as_words(mut self) -> Self {
        self.options.output = OutputGranularity::Words;
        self
    }

    pub fn as_spatial(mut self) -> Self {
        self.options.output = OutputGranularity::Spatial;
        self
    }

    pub fn as_tsv(mut self) -> Self {
        self.options.output = OutputGranularity::Tsv;
        self
    }

    pub fn with_output(mut self, output: OutputGranularity) -> Self {
        self.options.output = output;
        self
    }

    pub fn with_options(mut self, options: OcrRunOptions) -> Self {
        self.options = options;
        self
    }

    pub fn run(self) -> Result<DetectTextResult, rusto_core::engine::EngineError> {
        match self.input {
            BuilderInput::Source(source) => self.engine.detect_text(&source, &self.options),
            BuilderInput::Rgb {
                pixels,
                width,
                height,
                stride,
            } => self
                .engine
                .detect_text_rgb(pixels, width, height, stride, &self.options),
        }
    }
}

pub trait DetectTextBuilderExt {
    fn detect_text_builder(&mut self, source: ImageSource) -> DetectTextBuilder<'_>;
    fn detect_text_rgb_builder<'a>(
        &'a mut self,
        pixels: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> DetectTextBuilder<'a>;
}

impl DetectTextBuilderExt for rusto_core::engine::RustO {
    fn detect_text_builder(&mut self, source: ImageSource) -> DetectTextBuilder<'_> {
        DetectTextBuilder::new(self, source)
    }

    fn detect_text_rgb_builder<'a>(
        &'a mut self,
        pixels: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> DetectTextBuilder<'a> {
        DetectTextBuilder::new_rgb(self, pixels, width, height, stride)
    }
}

pub use rusto_core::unwarp::UnwarpOutput;

#[cfg(feature = "rten")]
pub use rusto_rten::RtenSession;

// Runtime initialization
#[cfg(feature = "rten")]
pub fn ensure_rten_registered() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        rusto_rten::register();
    });
}

#[cfg(not(feature = "rten"))]
pub fn ensure_rten_registered() {}

/// Initialize RustO runtime backends (registers RTen backend if enabled).
pub fn init() {
    ensure_rten_registered();
}

// C FFI module
#[cfg(feature = "ffi")]
pub mod ffi;

// UniFFI bindings
#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

#[cfg(feature = "uniffi")]
pub mod ffi_bindgen;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_run_options_builder_pattern() {
        let opts = OcrRunOptions::new()
            .with_auto_unwarp(true)
            .with_layout_reading(true)
            .with_orientation_detection(true)
            .with_line_classification(true)
            .with_pre_detection(PreDetectionOptions {
                descreen_strength: Some(0.5),
                target_max_side: Some(1024),
                crop_padding_x: Some(0.05),
                crop_padding_y: Some(0.05),
            })
            .with_detection(DetectionRunOptions {
                limit_side_len: Some(960),
                limit_type: Some("max".into()),
                ..Default::default()
            })
            .with_post_detection(PostDetectionOptions {
                threshold: Some(0.3),
                box_threshold: Some(0.6),
                unclip_ratio: Some(1.6),
                score_mode: Some("fast".into()),
                ..Default::default()
            })
            .with_pre_recognition(PreRecognitionOptions {
                line_y_threshold: Some(0.5),
                word_x_threshold: Some(0.4),
            })
            .with_recognition(RecognitionRunOptions {
                batch_size: Some(8),
            })
            .with_post_recognition(PostRecognitionOptions {
                score_threshold: Some(0.7),
            })
            .as_spatial();

        assert_eq!(opts.auto_unwarp, Some(true));
        assert_eq!(opts.layout_reading, Some(true));
        assert_eq!(opts.orientation_detection, Some(true));
        assert_eq!(opts.line_classification, Some(true));
        assert_eq!(opts.output, OutputGranularity::Spatial);
        assert_eq!(opts.pre_detection.as_ref().unwrap().target_max_side, Some(1024));
        assert_eq!(opts.post_detection.as_ref().unwrap().box_threshold, Some(0.6));
        assert_eq!(opts.pre_recognition.as_ref().unwrap().line_y_threshold, Some(0.5));
        assert_eq!(opts.recognition.as_ref().unwrap().batch_size, Some(8));
        assert_eq!(opts.post_recognition.as_ref().unwrap().score_threshold, Some(0.7));
        assert!(opts.validate().is_ok());
    }
}
