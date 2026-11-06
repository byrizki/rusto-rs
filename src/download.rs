//! Centralized model downloading and caching utilities for RustO.

use std::fs;
use std::path::{Path, PathBuf};
use rusto_core::error::CoreError;
use rusto_core::models::{
    plan_assets_download, plan_pipeline_download, ModelDownloadTarget,
};

/// Downloads a model asset from URL directly into target file path using a temporary file.
pub fn stream_url_to_file(
    download_url: &str,
    target_path: &Path,
    asset_filename: &str,
) -> Result<(), CoreError> {
    let response = ureq::get(download_url).call().map_err(|e| {
        CoreError::Model(format!(
            "Failed to download model '{}' from {}: {}",
            asset_filename, download_url, e
        ))
    })?;

    let part_path = target_path.with_extension("part");
    let mut file = fs::File::create(&part_path).map_err(|e| {
        CoreError::Model(format!(
            "Failed to create temporary file '{}': {}",
            part_path.display(),
            e
        ))
    })?;

    if let Err(e) = std::io::copy(&mut response.into_reader(), &mut file) {
        let _ = fs::remove_file(&part_path);
        return Err(CoreError::Model(format!(
            "Network/disk write error while downloading {}: {}",
            asset_filename, e
        )));
    }
    drop(file);

    fs::rename(&part_path, target_path).map_err(|e| {
        let _ = fs::remove_file(&part_path);
        CoreError::Model(format!(
            "Failed to finalize model file '{}': {}",
            target_path.display(),
            e
        ))
    })?;

    Ok(())
}

/// Downloads a single planned target with fallback URLs and idempotency check.
pub fn download_target(target: &ModelDownloadTarget) -> Result<PathBuf, CoreError> {
    if target.already_exists && target.target_path.exists() {
        return Ok(target.target_path.clone());
    }

    if let Some(parent) = target.target_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    for url in &target.urls {
        if stream_url_to_file(url, &target.target_path, &target.filename).is_ok() {
            return Ok(target.target_path.clone());
        }
    }

    Err(CoreError::Model(format!(
        "Failed to download model '{}'",
        target.filename
    )))
}

/// Downloads all model assets required for a specific OCR pipeline configuration.
pub fn download_model_pipeline(
    model_name: Option<&str>,
    lang: Option<&str>,
    include_cls: bool,
    include_orient: bool,
    include_unwarp: bool,
    layout_variant: Option<&str>,
    include_reading_order: bool,
    cache_dir: &Path,
) -> Result<Vec<PathBuf>, CoreError> {
    let targets = plan_pipeline_download(
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

/// Downloads a specific single asset file into target directory.
pub fn download_asset(asset_filename: &str, target_path: &Path) -> Result<PathBuf, CoreError> {
    let parent = target_path.parent().unwrap_or_else(|| Path::new("."));
    let mut targets = plan_assets_download(&[asset_filename], parent);
    let mut target = targets.remove(0);
    target.target_path = target_path.to_path_buf();
    target.already_exists = target_path.exists() && fs::metadata(target_path).map(|m| m.len() > 0).unwrap_or(false);
    download_target(&target)
}

/// Parses an `InitializeConfig` from JSON, automatically downloading missing canonical model
/// assets into the local cache directory when no explicit missing local path/directory is set.
pub fn parse_config_with_auto_download(
    json_str: &str,
) -> Result<rusto_core::config::InitializeConfig, CoreError> {
    let first_err = match rusto_core::config::InitializeConfig::from_json(json_str) {
        Ok(cfg) => return Ok(cfg),
        Err(e) => e,
    };

    let err_msg = first_err.to_string();
    if !err_msg.contains("not found in cache or local directories") {
        return Err(CoreError::Model(format!("Invalid config JSON: {first_err}")));
    }

    let val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| CoreError::Model(format!("Invalid config JSON: {e}")))?;

    let str_field = |keys: &[&str]| -> Option<&str> {
        keys.iter()
            .find_map(|k| val.get(*k).and_then(|v| v.as_str()))
            .filter(|s| !s.trim().is_empty())
    };
    let bool_field = |keys: &[&str]| -> Option<bool> {
        keys.iter().find_map(|k| val.get(*k).and_then(|v| v.as_bool()))
    };
    let nested_enabled = |section: &str, top_keys: &[&str]| -> bool {
        bool_field(top_keys)
            .or_else(|| {
                val.get(section)
                    .and_then(|s| s.get("enabled").or_else(|| s.get("use")))
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false)
    };

    let model = str_field(&["model", "models", "template"]);
    let lang = str_field(&["lang"]);
    let cache_dir = rusto_core::models::get_cache_dir(
        str_field(&["cacheDir", "cache_dir"]).map(Path::new),
    );
    let include_cls = nested_enabled("classification", &["useCls", "use_cls"]);
    let include_orient = nested_enabled("orientation", &["useOrient", "use_orient"]);
    let include_unwarp = nested_enabled("unwarp", &["useUnwarp", "use_unwarp"]);
    let layout_variant_str = str_field(&["layoutVariant", "layout_variant"]);
    let include_layout = nested_enabled("layout", &["useLayout", "use_layout"]) || layout_variant_str.is_some();
    let layout_variant = if include_layout {
        Some(layout_variant_str.unwrap_or("cdla"))
    } else {
        None
    };
    let include_reading_order = bool_field(&[
        "enableReadingOrder",
        "enable_reading_order",
        "reading_order",
    ])
    .unwrap_or(false);

    download_model_pipeline(
        model,
        lang,
        include_cls,
        include_orient,
        include_unwarp,
        layout_variant,
        include_reading_order,
        &cache_dir,
    )?;

    rusto_core::config::InitializeConfig::from_json(json_str)
        .map_err(|e| CoreError::Model(format!("Invalid config JSON: {e}")))
}

