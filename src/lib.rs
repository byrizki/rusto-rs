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
//! use rusto::RustO;
//!
//! let mut ocr = RustO::new_ppv5("models/det.mnn", "models/rec.mnn", "models/dict.txt")?;
//! let result = ocr.run("image.jpg")?;
//!
//! for (text, score) in result.txts.iter().zip(result.scores.iter()) {
//!     println!("{}: {:.3}", text, score);
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
pub use config::RustOConfig;
pub use det::TextDetector;
pub use doc_pipeline::{DocBlock, DocPipeline, DocPipelineConfig, DocResult};
pub use layout::{LayoutDetector, LayoutOutput, LayoutRegion, LayoutType};
pub use orient::{OrientClassifier, OrientOutput, Orientation};
pub use rec::{TextRecOutput, TextRecognizer, WordInfo, WordType};
pub use rusto_ocr::{RustO, RustOOutput};
pub use table::{TableDetector, TableDetectorConfig, TableModelType, TableStructure, TableStructureConfig, TableStructureRecognizer};
pub use types::{ClsConfig, DetConfig, GlobalConfig, LayoutConfig, OrientConfig, RecConfig};

// Alias for compatibility
pub use RustO as RapidOcr;

// Re-export for easier access
pub use crate::engine::EngineError;

/// OCR text result with bounding box
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextResult {
    pub text: String,
    pub score: f32,
    /// Box points: [top-left, top-right, bottom-right, bottom-left]
    pub box_points: [(f32, f32); 4],
}
