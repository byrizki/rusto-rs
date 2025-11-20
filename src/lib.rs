//! # RustO! - Pure Rust OCR Library
//!
//! RustO! is a high-performance OCR library written in pure Rust,
//! powered by PaddleOCR models with ONNX Runtime inference.
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
//!     det_model_path: "models/det.onnx".to_string(),
//!     rec_model_path: "models/rec.onnx".to_string(),
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
mod engine;
mod geometry;
mod image_impl;
mod postprocess;
mod preprocess;
mod det;
mod rec;
mod rusto_ocr;
mod cal_rec_boxes;
mod types;

#[cfg(not(feature = "use-opencv"))]
mod contours;

// FFI module for C bindings
#[cfg(feature = "ffi")]
pub mod ffi;

// Public API exports
pub use crate::rusto_ocr::RustO as RapidOcr;
pub use crate::types::{DetConfig, GlobalConfig, RecConfig};

// Re-export for easier access
use crate::engine::EngineError;
use std::path::Path;

/// Configuration for RustO
#[derive(Debug, Clone)]
pub struct RustOConfig {
    pub det_model_path: String,
    pub rec_model_path: String,
    pub dict_path: String,
}

impl Default for RustOConfig {
    fn default() -> Self {
        Self {
            det_model_path: String::new(),
            rec_model_path: String::new(),
            dict_path: String::new(),
        }
    }
}

/// OCR text result with bounding box
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextResult {
    pub text: String,
    pub score: f32,
    /// Box points: [top-left, top-right, bottom-right, bottom-left]
    pub box_points: [(f32, f32); 4],
}

/// Main RustO interface
pub struct RustO {
    inner: RapidOcr,
}

impl RustO {
    /// Create a new RustO instance
    pub fn new(config: RustOConfig) -> Result<Self, EngineError> {
        let inner = crate::rusto_ocr::RustO::new_ppv5(
            &config.det_model_path,
            &config.rec_model_path,
            &config.dict_path,
        )?;

        Ok(Self { inner })
    }

    /// Run OCR on an image file
    pub fn ocr<P: AsRef<Path>>(&mut self, image_path: P) -> Result<Vec<TextResult>, EngineError> {
        let results = self.inner.run(image_path)?;
        
        // Convert RapidOcrOutput to Vec<TextResult>
        Ok(results.boxes.into_iter()
            .zip(results.txts.into_iter().zip(results.scores.into_iter()))
            .map(|(boxes, (text, score))| TextResult {
                text,
                score,
                box_points: [
                    (boxes[0].x, boxes[0].y),
                    (boxes[1].x, boxes[1].y),
                    (boxes[2].x, boxes[2].y),
                    (boxes[3].x, boxes[3].y),
                ],
            }).collect())
    }

    /// Run OCR on image data in memory
    pub fn ocr_from_bytes(&mut self, image_data: &[u8]) -> Result<Vec<TextResult>, EngineError> {
        // Load image from bytes using image crate
        use image::ImageReader;
        use std::io::Cursor;
        
        let img = ImageReader::new(Cursor::new(image_data))
            .with_guessed_format()
            .map_err(|e| EngineError::ImageError(e.to_string()))?
            .decode()
            .map_err(|e| EngineError::ImageError(e.to_string()))?;
        
        // Save to temp file and process
        let temp_path = std::env::temp_dir().join(format!("rusto_{}.jpg", std::process::id()));
        img.save(&temp_path)
            .map_err(|e| EngineError::ImageError(e.to_string()))?;
        
        let result = self.ocr(&temp_path);
        let _ = std::fs::remove_file(&temp_path);
        result
    }
}
