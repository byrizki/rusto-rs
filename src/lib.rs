//! # RustO! - Pure Rust OCR Library
//!
//! RustO! is a high-performance OCR library written in pure Rust,
//! powered by PaddleOCR models with MNN inference engine.
//!
//! ## Features
//!
//! - **Pure Rust**: Zero OpenCV dependency (optional OpenCV backend available)
//! - **High Accuracy**: 99.3% parity with OpenCV-based implementations
//! - **Fast Performance**: Optimized with LTO and aggressive compilation settings
//! - **Cross-Platform**: Linux, macOS, Windows, Android, iOS support
//! - **Memory Safe**: Leverages Rust's safety guarantees
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rusto::{DetectTextResult, ImageSource, OcrRunOptions, RustO, InitializeConfig};
//!
//! let mut ocr = RustO::initialize(InitializeConfig::ppv5(
//!     "models/det.mnn", "models/rec.mnn", "models/dict.txt",
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

// Core modules
mod cal_rec_boxes;
pub mod config;
mod det;
pub mod doc_pipeline;
mod engine;
mod geometry;
pub mod image_impl;
pub mod layout;
mod orient;
mod postprocess;
mod preprocess;
mod rec;
pub mod rusto_ocr;
pub mod table;
mod types;

#[cfg(not(feature = "use-opencv"))]
mod contours;

// FFI module for C bindings
#[cfg(feature = "ffi")]
pub mod ffi;

// Public API exports
pub use config::{ModelPreset, InitializeConfig, PPV3_MODEL_CONFIG, PPV4_MODEL_CONFIG, PPV5_MODEL_CONFIG, PPV6_MODEL_CONFIG};
pub use rusto_ocr::{DetectionRunOptions, PostprocessRunOptions, PreprocessingRunOptions};
pub use det::TextDetector;
pub use doc_pipeline::{DocBlock, DocPipeline, DocPipelineConfig, DocResult};
pub use layout::{LayoutDetector, LayoutOutput, LayoutRegion, LayoutType};
pub use orient::{OrientClassifier, OrientOutput, Orientation};
pub use rec::{TextRecOutput, TextRecognizer, WordInfo, WordType};
pub use rusto_ocr::{DetectTextResult, ImageSource, OcrRunOptions, OutputGranularity, RustO};
pub use table::{TableDetector, TableDetectorConfig, TableModelType, TableStructure, TableStructureConfig, TableStructureRecognizer};
pub use types::{ClsConfig, DetConfig, Frame, GlobalConfig, LayoutConfig, OrientConfig, RecConfig};

// Re-export for easier access
pub use crate::engine::EngineError;

/// OCR text result with bounding box and frame
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TextResult {
    pub text: String,
    pub score: f32,
    /// Box points: [top-left, top-right, bottom-right, bottom-left]
    pub box_points: [(f32, f32); 4],
    /// Axis-aligned bounding box frame
    pub frame: Frame,
}
