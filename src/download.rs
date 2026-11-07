//! Centralized model downloading and caching utilities for RustO.

use std::fs;
use std::path::{Path, PathBuf};
use rusto_core::error::CoreError;
use rusto_core::models::{
    plan_assets_download, plan_pipeline_download, ModelDownloadTarget,
};

fn run_curl(url: &str, part_path: &Path) -> Result<(), String> {
    let mut cmd = std::process::Command::new("curl");
    cmd.arg("-fsSL")
        .arg("-o")
        .arg(part_path)
        .arg(url);

    match cmd.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("curl exited with code {}", status.code().unwrap_or(-1))),
        Err(e) => Err(format!("failed to spawn curl: {e}")),
    }
}

#[cfg(unix)]
fn run_wget(url: &str, part_path: &Path) -> Result<(), String> {
    let mut cmd = std::process::Command::new("wget");
    cmd.arg("-q")
        .arg("-O")
        .arg(part_path)
        .arg(url);

    match cmd.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("wget exited with code {}", status.code().unwrap_or(-1))),
        Err(e) => Err(format!("failed to spawn wget: {e}")),
    }
}

#[cfg(windows)]
fn run_powershell(url: &str, part_path: &Path) -> Result<(), String> {
    let script = format!(
        "$ProgressPreference = 'SilentlyContinue'; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('{}', '{}')",
        url.replace('\'', "''"),
        part_path.display().to_string().replace('\'', "''")
    );
    let mut cmd = std::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-Command", &script]);

    match cmd.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("powershell exited with code {}", status.code().unwrap_or(-1))),
        Err(e) => Err(format!("failed to spawn powershell: {e}")),
    }
}

fn download_native(url: &str, part_path: &Path) -> Result<(), String> {
    match run_curl(url, part_path) {
        Ok(()) => return Ok(()),
        Err(curl_err) => {
            let _ = fs::remove_file(part_path);

            #[cfg(unix)]
            {
                if let Ok(()) = run_wget(url, part_path) {
                    return Ok(());
                }
                let _ = fs::remove_file(part_path);
            }

            #[cfg(windows)]
            {
                if let Ok(()) = run_powershell(url, part_path) {
                    return Ok(());
                }
                let _ = fs::remove_file(part_path);
            }

            Err(format!(
                "OS native download failed ({curl_err}). Ensure 'curl' or 'wget' is installed and network connectivity is active."
            ))
        }
    }
}

/// Downloads a model asset from URL directly into target file path using a temporary file.
pub fn stream_url_to_file(
    download_url: &str,
    target_path: &Path,
    asset_filename: &str,
) -> Result<(), CoreError> {
    if let Some(parent) = target_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let part_path = target_path.with_extension("part");

    download_native(download_url, &part_path).map_err(|e| {
        let _ = fs::remove_file(&part_path);
        CoreError::Model(format!(
            "Failed to download model '{}' from {}: {}",
            asset_filename, download_url, e
        ))
    })?;

    let meta = fs::metadata(&part_path).map_err(|e| {
        let _ = fs::remove_file(&part_path);
        CoreError::Model(format!(
            "Downloaded model file '{}' is inaccessible: {}",
            asset_filename, e
        ))
    })?;
    if meta.len() == 0 {
        let _ = fs::remove_file(&part_path);
        return Err(CoreError::Model(format!(
            "Downloaded model file '{}' is empty (0 bytes)",
            asset_filename
        )));
    }

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

#[derive(Default, nanoserde::DeJson)]
struct NestedSubSection {
    enabled: Option<bool>,
    #[nserde(rename = "use")]
    use_flag: Option<bool>,
}

#[derive(Default, nanoserde::DeJson)]
struct AutoDownloadMeta {
    model: Option<String>,
    models: Option<String>,
    template: Option<String>,
    lang: Option<String>,
    #[nserde(rename = "cacheDir")]
    cache_dir_camel: Option<String>,
    #[nserde(rename = "cache_dir")]
    cache_dir_snake: Option<String>,
    #[nserde(rename = "useCls")]
    use_cls_camel: Option<bool>,
    #[nserde(rename = "use_cls")]
    use_cls_snake: Option<bool>,
    classification: Option<NestedSubSection>,
    #[nserde(rename = "useOrient")]
    use_orient_camel: Option<bool>,
    #[nserde(rename = "use_orient")]
    use_orient_snake: Option<bool>,
    orientation: Option<NestedSubSection>,
    #[nserde(rename = "useUnwarp")]
    use_unwarp_camel: Option<bool>,
    #[nserde(rename = "use_unwarp")]
    use_unwarp_snake: Option<bool>,
    unwarp: Option<NestedSubSection>,
    #[nserde(rename = "useLayout")]
    use_layout_camel: Option<bool>,
    #[nserde(rename = "use_layout")]
    use_layout_snake: Option<bool>,
    layout: Option<NestedSubSection>,
    #[nserde(rename = "layoutVariant")]
    layout_variant_camel: Option<String>,
    #[nserde(rename = "layout_variant")]
    layout_variant_snake: Option<String>,
    #[nserde(rename = "enableReadingOrder")]
    enable_reading_order_camel: Option<bool>,
    #[nserde(rename = "enable_reading_order")]
    enable_reading_order_snake: Option<bool>,
    reading_order: Option<bool>,
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

    let meta: AutoDownloadMeta = nanoserde::DeJson::deserialize_json(json_str)
        .map_err(|e| CoreError::Model(format!("Invalid config JSON: {e}")))?;

    let model = meta
        .model
        .as_deref()
        .or(meta.models.as_deref())
        .or(meta.template.as_deref())
        .filter(|s| !s.trim().is_empty());
    let lang = meta.lang.as_deref().filter(|s| !s.trim().is_empty());
    let cache_dir_str = meta
        .cache_dir_camel
        .as_deref()
        .or(meta.cache_dir_snake.as_deref())
        .filter(|s| !s.trim().is_empty());
    let cache_dir = rusto_core::models::get_cache_dir(cache_dir_str.map(Path::new));

    let include_cls = meta
        .use_cls_camel
        .or(meta.use_cls_snake)
        .or_else(|| meta.classification.as_ref().and_then(|c| c.enabled.or(c.use_flag)))
        .unwrap_or(false);
    let include_orient = meta
        .use_orient_camel
        .or(meta.use_orient_snake)
        .or_else(|| meta.orientation.as_ref().and_then(|c| c.enabled.or(c.use_flag)))
        .unwrap_or(false);
    let include_unwarp = meta
        .use_unwarp_camel
        .or(meta.use_unwarp_snake)
        .or_else(|| meta.unwarp.as_ref().and_then(|c| c.enabled.or(c.use_flag)))
        .unwrap_or(false);
    let layout_variant_str = meta
        .layout_variant_camel
        .as_deref()
        .or(meta.layout_variant_snake.as_deref());
    let include_layout = meta
        .use_layout_camel
        .or(meta.use_layout_snake)
        .or_else(|| meta.layout.as_ref().and_then(|c| c.enabled.or(c.use_flag)))
        .unwrap_or(false)
        || layout_variant_str.is_some();
    let layout_variant = if include_layout {
        Some(layout_variant_str.unwrap_or("cdla"))
    } else {
        None
    };
    let include_reading_order = meta
        .enable_reading_order_camel
        .or(meta.enable_reading_order_snake)
        .or(meta.reading_order)
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

