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

// Core modules
pub mod engine {
    pub use rusto_core::engine::*;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::models instead")]
pub mod model_manager {
    pub use rusto_core::models::*;
    pub use rusto_core::models::resolve_asset_local as resolve_asset;

    pub fn resolve_asset_with_dir(
        explicit_path: Option<&std::path::Path>,
        model_dir: Option<&std::path::Path>,
        asset_kind: &str,
        asset_filename: &str,
        cache_dir: &std::path::Path,
    ) -> Result<std::path::PathBuf, rusto_core::error::CoreError> {
        rusto_core::models::resolve_asset_local(
            explicit_path,
            model_dir,
            asset_kind,
            asset_filename,
            cache_dir,
        )
    }

    pub fn download_model_pipeline(
        model_name: Option<&str>,
        lang: Option<&str>,
        include_cls: bool,
        include_orient: bool,
        include_unwarp: bool,
        layout_variant: Option<&str>,
        include_reading_order: bool,
        cache_dir: &std::path::Path,
    ) -> Result<Vec<std::path::PathBuf>, rusto_core::error::CoreError> {
        let targets = rusto_core::models::plan_pipeline_download(
            model_name,
            lang,
            include_cls,
            include_orient,
            include_unwarp,
            layout_variant,
            include_reading_order,
            cache_dir,
        );
        let mut paths = Vec::with_capacity(targets.len());
        for target in &targets {
            paths.push(download_target(target)?);
        }
        Ok(paths)
    }

    pub fn download_all_models(cache_dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, rusto_core::error::CoreError> {
        let targets = rusto_core::models::plan_all_models_download(cache_dir);
        let mut paths = Vec::with_capacity(targets.len());
        for target in &targets {
            paths.push(download_target(target)?);
        }
        Ok(paths)
    }

    pub fn download_asset(asset_filename: &str, target_path: &std::path::Path) -> Result<std::path::PathBuf, rusto_core::error::CoreError> {
        let parent = target_path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let mut targets = rusto_core::models::plan_assets_download(&[asset_filename], parent);
        let mut target = targets.remove(0);
        target.target_path = target_path.to_path_buf();
        target.already_exists = target_path.exists() && std::fs::metadata(target_path).map(|m| m.len() > 0).unwrap_or(false);
        download_target(&target)
    }

    pub fn download_target(target: &rusto_core::models::ModelDownloadTarget) -> Result<std::path::PathBuf, rusto_core::error::CoreError> {
        if target.already_exists && target.target_path.exists() {
            return Ok(target.target_path.clone());
        }

        if let Some(parent) = target.target_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        for url in &target.urls {
            if stream_url_to_file(url, &target.target_path, &target.filename).is_ok() {
                return Ok(target.target_path.clone());
            }
        }

        Err(rusto_core::error::CoreError::Model(format!(
            "Failed to download model '{}'",
            target.filename
        )))
    }

    fn stream_url_to_file(download_url: &str, target_path: &std::path::Path, asset_filename: &str) -> Result<(), rusto_core::error::CoreError> {
        let response = ureq::get(download_url).call().map_err(|e| {
            rusto_core::error::CoreError::Model(format!(
                "Failed to download model '{}' from {}: {}",
                asset_filename, download_url, e
            ))
        })?;

        let part_path = target_path.with_extension("part");
        let mut file = std::fs::File::create(&part_path).map_err(|e| {
            rusto_core::error::CoreError::Model(format!(
                "Failed to create temporary file '{}': {}",
                part_path.display(),
                e
            ))
        })?;

        if let Err(e) = std::io::copy(&mut response.into_reader(), &mut file) {
            let _ = std::fs::remove_file(&part_path);
            return Err(rusto_core::error::CoreError::Model(format!(
                "Network/disk write error while downloading {}: {}",
                asset_filename, e
            )));
        }
        drop(file);

        std::fs::rename(&part_path, target_path).map_err(|e| {
            let _ = std::fs::remove_file(&part_path);
            rusto_core::error::CoreError::Model(format!(
                "Failed to finalize model file '{}': {}",
                target_path.display(),
                e
            ))
        })?;

        Ok(())
    }
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::pipeline instead")]
pub use rusto_core::pipeline as pipelines;
#[deprecated(since = "0.3.3", note = "Directly use rusto_core::pipeline instead")]
pub use rusto_core::pipeline as rusto_ocr;

// Deprecated component shims: directly use rusto-core or crate::engine instead
#[deprecated(since = "0.3.3", note = "Directly use rusto_core::det or crate::engine::TextDetector instead")]
pub mod det {
    pub use rusto_core::det::*;
    pub use crate::engine::TextDetector;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::rec or crate::engine::TextRecognizer instead")]
pub mod rec {
    pub use rusto_core::rec::*;
    pub use crate::engine::TextRecognizer;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::layout or crate::engine::LayoutDetector instead")]
pub mod layout {
    pub use rusto_core::layout::*;
    pub use crate::engine::LayoutDetector;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::orient or crate::engine::OrientClassifier instead")]
pub mod orient {
    pub use rusto_core::orient::*;
    pub use crate::engine::{OrientClassifier, OrientationExt};
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::unwarp or crate::engine::DocUnwarper instead")]
pub mod unwarp {
    pub use rusto_core::unwarp::*;
    pub use crate::engine::DocUnwarper;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::table or crate::engine::TableDetector instead")]
pub mod table {
    pub use rusto_core::table::*;
    pub use crate::engine::{TableDetector, TableStructureRecognizer};
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::doc or crate::engine::DocPipeline instead")]
pub mod doc_pipeline {
    pub use rusto_core::doc::*;
    pub use crate::engine::{DocPipeline, DocPipelineConfig};
}

// Deprecated shims: directly use rusto-core / rusto-cpp instead
#[deprecated(since = "0.3.3", note = "Directly use rusto_core::config instead")]
pub use rusto_core::config;

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::types instead")]
pub use rusto_core::types;

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::contours instead")]
pub use rusto_core::contours;

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::calibration instead")]
pub use rusto_core::calibration;

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::geometry or rusto_cpp::geometry instead")]
pub mod geometry {
    #[cfg(feature = "use-opencv")]
    pub use rusto_cpp::geometry::*;
    #[cfg(not(feature = "use-opencv"))]
    pub use rusto_core::geometry::*;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::postprocess or rusto_cpp::postprocess instead")]
pub mod postprocess {
    #[cfg(feature = "use-opencv")]
    pub use rusto_cpp::postprocess::*;
    #[cfg(not(feature = "use-opencv"))]
    pub use rusto_core::postprocess::*;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::preprocess or rusto_cpp::preprocess instead")]
pub mod preprocess {
    #[cfg(feature = "use-opencv")]
    pub use rusto_cpp::preprocess::*;
    #[cfg(not(feature = "use-opencv"))]
    pub use rusto_core::preprocess::*;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::image or rusto_cpp::image instead")]
pub mod image_impl {
    #[cfg(feature = "use-opencv")]
    pub use rusto_cpp::image::*;
    #[cfg(not(feature = "use-opencv"))]
    pub use rusto_core::image::*;
}

#[deprecated(since = "0.3.3", note = "Directly use rusto_core::cal_rec_boxes::CalRecBoxes or rusto_cpp::cal_rec_boxes::CalRecBoxes instead")]
pub mod cal_rec_boxes {
    #[cfg(feature = "use-opencv")]
    pub use rusto_cpp::cal_rec_boxes::*;
    #[cfg(not(feature = "use-opencv"))]
    pub use rusto_core::cal_rec_boxes::*;
}

// FFI module for C bindings
#[cfg(feature = "ffi")]
pub mod ffi;

// UniFFI bindings
#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

#[cfg(feature = "uniffi")]
pub mod ffi_bindgen;
#[cfg(feature = "uniffi")]
#[deprecated(since = "0.3.3", note = "Renamed to ffi_bindgen")]
pub use ffi_bindgen as binding;

// Public API exports
pub use calibration::{CalibrationOptions, OptimizationOptions};
pub use config::{ModelPreset, InitializeConfig, PPV3_MODEL_CONFIG, PPV4_MODEL_CONFIG, PPV5_MODEL_CONFIG, PPV6_MODEL_CONFIG, INDRA_MODEL_CONFIG};
pub use rusto_core::types::{DetectionRunOptions, PostprocessRunOptions};
pub use rusto_core::engine::{
    DocPipeline, DocPipelineConfig, DocUnwarper, LayoutDetector, OrientClassifier,
    OrientationExt, RustO, RustOOutput, TableDetector, TableStructureRecognizer, TextDetector,
    TextRecognizer,
};
pub use rusto_core::doc::{DocBlock, DocResult};
pub use rusto_core::layout::{BBox, LayoutOutput, LayoutRegion, LayoutType};
pub use rusto_core::orient::{OrientOutput, Orientation};
pub use rusto_core::rec::{TextRecOutput, WordInfo, WordType};
pub use rusto_core::types::{DetectTextResult, ImageSource, OcrRunOptions, OutputGranularity};
pub use rusto_core::table::{TableCell, TableDetectorConfig, TableModelType, TableRow, TableStructure, TableStructureConfig};
pub use types::{ClsConfig, DetConfig, Frame, GlobalConfig, LayoutConfig, OrientConfig, RecConfig, TextResult, UnwarpConfig};
pub use rusto_core::unwarp::UnwarpOutput;

// Re-export for easier access
pub use rusto_core::engine::{
    CoreError, EngineError, EngineSession, InferenceSession, MnnSession,
    register_session_loader, SessionLoader,
};
#[cfg(feature = "rten")]
pub use rusto_rten::RtenSession;

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
