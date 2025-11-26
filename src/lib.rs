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
//! use rusto::{RustO, RustOConfig};
//!
//! let config = RustOConfig {
//!     det_model_path: "models/det.mnn".to_string(),
//!     rec_model_path: "models/rec.mnn".to_string(),
//!     dict_path: "models/dict.txt".to_string(),
//! };
//!
//! let ocr = RustO::new(config)?;
//! let results = ocr.ocr("image.jpg")?;
//!
//! for result in results {
//!     println!("{}: {:.3}", result.text, result.score);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Core modules
mod cal_rec_boxes;
pub mod config;
mod det;
mod engine;
mod geometry;
pub mod image_impl;
mod orient;
mod postprocess;
mod preprocess;
mod rec;
pub mod rusto_ocr;
mod types;

#[cfg(not(feature = "use-opencv"))]
mod contours;

// FFI module for C bindings
#[cfg(feature = "ffi")]
pub mod ffi;

// Public API exports
pub use config::RustOConfig;
pub use det::TextDetector;
pub use orient::{OrientClassifier, OrientOutput, Orientation};
pub use rec::{TextRecOutput, TextRecognizer, WordInfo, WordType};
pub use rusto_ocr::{RustO, RustOOutput};
pub use types::{ClsConfig, DetConfig, GlobalConfig, OrientConfig, RecConfig};

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
