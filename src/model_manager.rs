use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::engine::EngineError;

pub const DEFAULT_MODELS_TAG: &str = "models-rten-v0.4.0";
pub const DEFAULT_BASE_URL: &str = "https://github.com/byrizki/rusto-rs/releases/download";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasePipelineAssets {
    pub det_filename: &'static str,
    pub rec_filename: &'static str,
    pub dict_filename: &'static str,
}

/// Normalizes model identifier to canonical model family/tier.
pub fn normalize_model_name(name: Option<&str>) -> &'static str {
    let clean = name.map(|s| {
        s.trim()
            .to_ascii_lowercase()
            .replace('_', "-")
            .replace("pp-ocr", "ppocr")
    });
    match clean.as_deref() {
        Some("indraocr") | Some("indra") | Some("indra-ocr") | Some("indra-base") => "indraocr",
        Some("ppocrv6-small") | Some("v6-small") | Some("small") => "ppocrv6-small",
        Some("ppocrv6-small-int8") | Some("small-int8") => "ppocrv6-small-int8",
        Some("ppocrv6-medium") | Some("v6-medium") | Some("medium") => "ppocrv6-medium",
        Some("ppocrv6-medium-int8") | Some("medium-int8") => "ppocrv6-medium-int8",
        Some("ppocrv6") | Some("v6") | Some("ppv6") | Some("ppocrv6-tiny") | Some("v6-tiny") | Some("tiny") => {
            "ppocrv6-tiny"
        }
        Some("ppocrv6-tiny-int8") | Some("tiny-int8") => "ppocrv6-tiny-int8",
        Some("ppocrv5") | Some("ppv5") | Some("ppocrv5-mobile") | Some("v5") => "ppocrv5-mobile",
        Some("ppocrv5-server") | Some("v5-server") => "ppocrv5-server",
        Some("ppocrv4") | Some("ppv4") | Some("ppocrv4-mobile") | Some("v4") => "ppocrv4-mobile",
        Some("ppocrv4-server") | Some("v4-server") => "ppocrv4-server",
        Some("ppocrv3") | Some("ppv3") | Some("ppocrv3-mobile") | Some("v3") => "ppocrv3-mobile",
        // Default flagship
        _ => "indraocr",
    }
}

/// Normalizes language identifier to canonical asset names.
pub fn resolve_language_assets(lang: &str) -> Option<(&'static str, &'static str)> {
    match lang.trim().to_ascii_lowercase().as_str() {
        "ja" | "jp" | "jpn" | "japan" | "japanese" => {
            Some(("ppocrv4_rec_japan.rten", "ppocrv4_dict_japan.txt"))
        }
        "ko" | "kor" | "korean" => {
            Some(("ppocrv5_rec_korean.rten", "ppocrv5_dict_korean.txt"))
        }
        "ar" | "ara" | "arabic" => {
            Some(("ppocrv5_rec_arabic.rten", "ppocrv5_dict_arabic.txt"))
        }
        "ru" | "rus" | "cyrillic" => {
            Some(("ppocrv5_rec_cyrillic.rten", "ppocrv5_dict_cyrillic.txt"))
        }
        "hi" | "hin" | "devanagari" => {
            Some(("ppocrv5_rec_devanagari.rten", "ppocrv5_dict_devanagari.txt"))
        }
        "cht" | "zh-hant" | "chinese-cht" | "traditional-chinese" => {
            Some(("ppocrv4_rec_chinese_cht.rten", "ppocrv4_dict_chinese_cht.txt"))
        }
        "el" | "ell" | "greek" => {
            Some(("ppocrv5_rec_el.rten", "ppocrv5_dict_el.txt"))
        }
        "eslav" | "east-slavic" => {
            Some(("ppocrv5_rec_eslav.rten", "ppocrv5_dict_eslav.txt"))
        }
        "ta" | "tam" | "tamil" => {
            Some(("ppocrv5_rec_ta.rten", "ppocrv5_dict_ta.txt"))
        }
        "te" | "tel" | "telugu" => {
            Some(("ppocrv5_rec_te.rten", "ppocrv5_dict_te.txt"))
        }
        "th" | "tha" | "thai" => {
            Some(("ppocrv5_rec_th.rten", "ppocrv5_dict_th.txt"))
        }
        "ka" | "kan" | "kannada" => {
            Some(("ppocrv4_rec_kannada.rten", "ppocrv4_dict_kannada.txt"))
        }
        "cjk" => {
            Some(("indra_rec_cjk.rten", "indra_meta.json"))
        }
        _ => None,
    }
}

/// Returns default asset names for a base model.
pub fn get_base_pipeline_assets(model_name: &str) -> BasePipelineAssets {
    match model_name {
        "indraocr" | "indra" => BasePipelineAssets {
            det_filename: "indra_det.rten",
            rec_filename: "indra_rec.rten",
            dict_filename: "indra_meta.json",
        },
        "ppocrv6-small" => BasePipelineAssets {
            det_filename: "ppocrv6_small_det.rten",
            rec_filename: "ppocrv6_small_rec.rten",
            dict_filename: "ppocrv6_dict.txt",
        },
        "ppocrv6-small-int8" => BasePipelineAssets {
            det_filename: "ppocrv6_small_int8_det.rten",
            rec_filename: "ppocrv6_small_int8_rec.rten",
            dict_filename: "ppocrv6_dict.txt",
        },
        "ppocrv6-medium" => BasePipelineAssets {
            det_filename: "ppocrv6_medium_det.rten",
            rec_filename: "ppocrv6_medium_rec.rten",
            dict_filename: "ppocrv6_dict.txt",
        },
        "ppocrv6-medium-int8" => BasePipelineAssets {
            det_filename: "ppocrv6_medium_int8_det.rten",
            rec_filename: "ppocrv6_medium_int8_rec.rten",
            dict_filename: "ppocrv6_dict.txt",
        },
        "ppocrv6-tiny-int8" => BasePipelineAssets {
            det_filename: "ppocrv6_tiny_int8_det.rten",
            rec_filename: "ppocrv6_tiny_int8_rec.rten",
            dict_filename: "ppocrv6_dict.txt",
        },
        "ppocrv5-mobile" => BasePipelineAssets {
            det_filename: "ppocrv5_mobile_det.rten",
            rec_filename: "ppocrv5_mobile_rec.rten",
            dict_filename: "ppocrv5_dict.txt",
        },
        "ppocrv5-server" => BasePipelineAssets {
            det_filename: "ppocrv5_server_det.rten",
            rec_filename: "ppocrv5_server_rec.rten",
            dict_filename: "ppocrv5_dict.txt",
        },
        "ppocrv4-mobile" => BasePipelineAssets {
            det_filename: "ppocrv4_mobile_det.rten",
            rec_filename: "ppocrv4_mobile_rec.rten",
            dict_filename: "ppocrv4_dict.txt",
        },
        "ppocrv4-server" => BasePipelineAssets {
            det_filename: "ppocrv4_server_det.rten",
            rec_filename: "ppocrv4_server_rec.rten",
            dict_filename: "ppocrv4_dict.txt",
        },
        "ppocrv6-tiny" | "ppocrv6" => BasePipelineAssets {
            det_filename: "ppocrv6_tiny_det.rten",
            rec_filename: "ppocrv6_tiny_rec.rten",
            dict_filename: "ppocrv6_dict.txt",
        },
        _ => BasePipelineAssets {
            det_filename: "indra_det.rten",
            rec_filename: "indra_rec.rten",
            dict_filename: "indra_meta.json",
        },
    }
}

/// Resolves the default cache directory for RustO models.
pub fn get_cache_dir(custom: Option<&Path>) -> PathBuf {
    if let Some(custom_path) = custom {
        return custom_path.to_path_buf();
    }

    if let Ok(env_path) = std::env::var("RUSTO_CACHE_DIR") {
        if !env_path.trim().is_empty() {
            return PathBuf::from(env_path);
        }
    }

    if let Some(cache) = dirs::cache_dir() {
        return cache.join("rusto").join("models");
    }

    PathBuf::from(".cache").join("rusto").join("models")
}

/// Finds local checkout repository fallback candidates for an asset filename.
fn find_local_repo_candidates(asset_filename: &str) -> Vec<PathBuf> {
    let mut candidates = vec![
        PathBuf::from("models").join(asset_filename),
    ];

    if asset_filename == "ppocrv6_dict.txt" {
        candidates.push(PathBuf::from("models/PPOCR_v6_tiny/dict.txt"));
        candidates.push(PathBuf::from("models/PPOCR_v6/ppocrv6_tiny_dict.txt"));
        candidates.push(PathBuf::from("models/PPOCR_v6/dict.txt"));
        candidates.push(PathBuf::from("models/PPOCR_v6_small/dict.txt"));
    } else if asset_filename == "ppocrv5_dict.txt" {
        candidates.push(PathBuf::from("models/PPOCR_v5_mobile/dict.txt"));
        candidates.push(PathBuf::from("models/PPOCR_v5/dict.txt"));
    } else if asset_filename == "ppocrv4_dict.txt" {
        candidates.push(PathBuf::from("models/PPOCR_v4_mobile/dict.txt"));
        candidates.push(PathBuf::from("models/PPOCR_v4/dict.txt"));
    } else if let Some(sub) = asset_filename.strip_prefix("indra_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/indra/{}", asset_filename)));
        candidates.push(PathBuf::from(format!("models/indra/{}", sub)));
        candidates.push(PathBuf::from(format!("models/indra/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/indra/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv6_tiny_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}.onnx", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv6_tiny_int8_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny_int8/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny_int8/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny_int8/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv6_small_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_small/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_small/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_small/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv6_small_int8_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_small_int8/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_small_int8/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_small_int8/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv6_medium_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_medium/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_medium/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_medium/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv5_mobile_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}.onnx", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5/{}.rten", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv4_mobile_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}.onnx", base)));
    } else if asset_filename == "ppocr_cls.rten" {
        candidates.push(PathBuf::from("models/cls.rten"));
        candidates.push(PathBuf::from("models/cls.onnx"));
        candidates.push(PathBuf::from("models/PPOCR_v6_tiny/cls.onnx"));
    } else if asset_filename == "ppocr_orient.rten" {
        candidates.push(PathBuf::from("models/orient.rten"));
        candidates.push(PathBuf::from("models/orient.onnx"));
        candidates.push(PathBuf::from("models/PPOCR_v6_tiny/orient.onnx"));
    } else if asset_filename == "uvdoc.rten" {
        candidates.push(PathBuf::from("models/uvdoc.rten"));
        candidates.push(PathBuf::from("models/uvdoc.onnx"));
        candidates.push(PathBuf::from("models/PPOCR_v6_tiny/uvdoc.onnx"));
    } else if asset_filename == "layout_cdla.rten" {
        candidates.push(PathBuf::from("models/layout_cdla.rten"));
        candidates.push(PathBuf::from("models/layout_cdla.onnx"));
        candidates.push(PathBuf::from("models/PPOCR_v6_tiny/layout_cdla.onnx"));
    } else if asset_filename == "layout_publaynet.rten" {
        candidates.push(PathBuf::from("models/layout_publaynet.rten"));
        candidates.push(PathBuf::from("models/layout_publaynet.onnx"));
        candidates.push(PathBuf::from("models/PPOCR_v6_tiny/layout_publaynet.onnx"));
    }

    candidates
}

/// Attempts to find a matching model file in a custom model directory.
pub fn find_in_model_dir(dir: &Path, kind: &str) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }

    // Direct filename checks first
    let direct_checks: &[&str] = match kind {
        "det" => &["det.rten", "det.onnx", "detection.rten", "detection.onnx"],
        "rec" => &["rec.rten", "rec.onnx", "recognition.rten", "recognition.onnx"],
        "dict" => &["dict.txt", "meta.json", "ppocr_keys_v1.txt", "keys.txt", "vocab.txt"],
        "cls" => &["cls.rten", "cls.onnx", "classifier.rten", "classifier.onnx"],
        "orient" => &["orient.rten", "orient.onnx", "orientation.rten", "orientation.onnx"],
        "unwarp" => &["uvdoc.rten", "uvdoc.onnx", "unwarp.rten", "unwarp.onnx"],
        "layout" => &["layout_cdla.rten", "layout_publaynet.rten", "layout.rten", "layout.onnx"],
        "order" => &["indra_order.rten", "indra_order.onnx", "order.rten", "order.onnx", "reading_order.rten", "reading_order.onnx"],
        _ => &[],
    };

    for &name in direct_checks {
        let p = dir.join(name);
        if p.exists() && fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false) {
            return Some(p);
        }
    }

    // Pattern search in directory if direct names were not found
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let matches = match kind {
                    "det" => (name.contains("det") || name.contains("detection")) && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    "rec" => (name.contains("rec") || name.contains("recognition")) && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    "dict" => (name.contains("dict") || name.contains("keys") || name.contains("meta") || name.contains("vocab")) && (name.ends_with(".txt") || name.ends_with(".json")),
                    "cls" => name.contains("cls") && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    "orient" => name.contains("orient") && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    "unwarp" => (name.contains("unwarp") || name.contains("uvdoc")) && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    "layout" => name.contains("layout") && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    "order" => name.contains("order") && (name.ends_with(".rten") || name.ends_with(".onnx")),
                    _ => false,
                };
                if matches && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false) {
                    return Some(path);
                }
            }
        }
    }

    None
}

fn resolve_default_model_dir_asset(dir: &Path, kind: &str, filename: &str) -> PathBuf {
    let default_name = match kind {
        "det" => "det.rten",
        "rec" => "rec.rten",
        "dict" => {
            if filename.ends_with(".json") {
                "meta.json"
            } else {
                "dict.txt"
            }
        }
        "cls" => "cls.rten",
        "orient" => "orient.rten",
        "unwarp" => "uvdoc.rten",
        "layout" => "layout_cdla.rten",
        "order" => "indra_order.rten",
        _ => filename,
    };
    dir.join(default_name)
}

/// Resolves a single model asset with optional custom model directory support.
///
/// Flow:
/// 1. User specified explicit path: If provided and exists, return it.
/// 2. Custom model directory: If model_dir is provided and contains matching asset, return it.
/// 3. Cache Path: If present in cache directory, return it.
/// 4. Local checkout candidates: If present in local repository checkout, return it.
/// 5. Model download: Download from GitHub Release directly into cache directory.
pub fn resolve_asset_with_dir(
    explicit_path: Option<&Path>,
    model_dir: Option<&Path>,
    asset_kind: &str,
    asset_filename: &str,
    cache_dir: &Path,
) -> Result<PathBuf, EngineError> {
    // 1. User specified explicit path
    if let Some(path) = explicit_path {
        if !path.as_os_str().is_empty() {
            if path.exists() {
                return Ok(path.to_path_buf());
            }
            return Err(EngineError::Engine(format!(
                "Specified model path does not exist: {}",
                path.display()
            )));
        }
    }

    // 2. Custom model directory resolution
    if let Some(dir) = model_dir {
        let dir_path = if dir.is_file() {
            dir.parent().unwrap_or(dir)
        } else {
            dir
        };
        if let Some(found) = find_in_model_dir(dir_path, asset_kind) {
            return Ok(found);
        }
        let direct = dir_path.join(asset_filename);
        if direct.exists() && fs::metadata(&direct).map(|m| m.len() > 0).unwrap_or(false) {
            return Ok(direct);
        }
        return Ok(resolve_default_model_dir_asset(dir_path, asset_kind, asset_filename));
    }

    // 3. Cache Path
    let cached_file = cache_dir.join(asset_filename);
    if cached_file.exists() && fs::metadata(&cached_file).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(cached_file);
    }

    // 4. Local checkout repository candidate fallback
    for cand in find_local_repo_candidates(asset_filename) {
        if cand.exists() && fs::metadata(&cand).map(|m| m.len() > 0).unwrap_or(false) {
            return Ok(cand);
        }
    }

    // 5. Model download
    download_asset(asset_filename, &cached_file)
}

/// Resolves a single model asset using the default resolution flow.
pub fn resolve_asset(
    explicit_path: Option<&Path>,
    asset_filename: &str,
    cache_dir: &Path,
) -> Result<PathBuf, EngineError> {
    resolve_asset_with_dir(explicit_path, None, "", asset_filename, cache_dir)
}

/// Downloads a single asset from the GitHub Release tag directly to target path.
pub fn download_asset(asset_filename: &str, target_path: &Path) -> Result<PathBuf, EngineError> {
    if target_path.exists() && fs::metadata(target_path).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(target_path.to_path_buf());
    }

    // If candidate exists in local checkout, copy it into target cache to avoid unnecessary network transfer
    for cand in find_local_repo_candidates(asset_filename) {
        if cand != target_path && cand.exists() && fs::metadata(&cand).map(|m| m.len() > 0).unwrap_or(false) {
            if let Some(parent) = target_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if fs::copy(&cand, target_path).is_ok() {
                eprintln!("✓ Populated {} from local {}", target_path.display(), cand.display());
                return Ok(target_path.to_path_buf());
            }
        }
    }

    let tag = std::env::var("RUSTO_MODELS_TAG").unwrap_or_else(|_| DEFAULT_MODELS_TAG.to_string());
    let base_url = std::env::var("RUSTO_MODELS_URL").unwrap_or_else(|_| {
        format!("{}/{}/{}", DEFAULT_BASE_URL, tag, asset_filename)
    });

    let download_url = if base_url.ends_with('/') {
        format!("{}{}", base_url, asset_filename)
    } else if base_url.contains(asset_filename) {
        base_url
    } else {
        format!("{}/{}", base_url, asset_filename)
    };

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            EngineError::Engine(format!(
                "Failed to create cache directory '{}': {}",
                parent.display(),
                e
            ))
        })?;
    }

    eprintln!("Downloading {} from {}...", asset_filename, download_url);

    let response = ureq::get(&download_url)
        .call()
        .map_err(|e| {
            EngineError::Engine(format!(
                "Failed to download model '{}' from {} (release tag may be unreleased or remote asset missing): {}",
                asset_filename, download_url, e
            ))
        })?;

    let part_path = target_path.with_extension("part");
    let mut file = File::create(&part_path).map_err(|e| {
        EngineError::Engine(format!(
            "Failed to create temporary download file '{}': {}",
            part_path.display(),
            e
        ))
    })?;

    let mut reader = response.into_reader();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = reader.read(&mut buffer).map_err(|e| {
            let _ = fs::remove_file(&part_path);
            EngineError::Engine(format!("Network read error while downloading {}: {}", asset_filename, e))
        })?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read]).map_err(|e| {
            let _ = fs::remove_file(&part_path);
            EngineError::Engine(format!("Disk write error while saving {}: {}", asset_filename, e))
        })?;
    }

    file.flush().map_err(|e| {
        let _ = fs::remove_file(&part_path);
        EngineError::Engine(format!("Failed to flush {}: {}", asset_filename, e))
    })?;
    drop(file);

    fs::rename(&part_path, target_path).map_err(|e| {
        let _ = fs::remove_file(&part_path);
        EngineError::Engine(format!(
            "Failed to finalize model file '{}': {}",
            target_path.display(),
            e
        ))
    })?;

    eprintln!("✓ Saved {} to {}", asset_filename, target_path.display());
    Ok(target_path.to_path_buf())
}

/// Resolves complete base OCR pipeline (detection, recognition, dictionary) paths.
pub fn resolve_base_pipeline(
    model_name: Option<&str>,
    model_dir: Option<&Path>,
    lang: Option<&str>,
    det_override: Option<&Path>,
    rec_override: Option<&Path>,
    dict_override: Option<&Path>,
    cache_dir: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf), EngineError> {
    // If model_name itself is an existing directory or file, treat it as model_dir if model_dir is None
    let effective_dir = model_dir.or_else(|| {
        model_name.and_then(|m| {
            let p = Path::new(m);
            if p.exists() {
                Some(p)
            } else {
                None
            }
        })
    });

    let canonical_model = normalize_model_name(model_name);
    let base_assets = get_base_pipeline_assets(canonical_model);

    // Resolve detection model
    let det_path = resolve_asset_with_dir(det_override, effective_dir, "det", base_assets.det_filename, cache_dir)?;

    // Resolve recognition & dictionary model (checking language add-on if specified)
    let (rec_filename, dict_filename) = if let Some(l) = lang {
        if let Some((lang_rec, lang_dict)) = resolve_language_assets(l) {
            (lang_rec, lang_dict)
        } else {
            (base_assets.rec_filename, base_assets.dict_filename)
        }
    } else {
        (base_assets.rec_filename, base_assets.dict_filename)
    };

    let rec_path = resolve_asset_with_dir(rec_override, effective_dir, "rec", rec_filename, cache_dir)?;
    let dict_path = resolve_asset_with_dir(dict_override, effective_dir, "dict", dict_filename, cache_dir)?;

    Ok((det_path, rec_path, dict_path))
}

/// All canonical model assets published on GitHub Releases.
pub const ALL_MODEL_ASSETS: &[&str] = &[
    "ppocrv6_tiny_det.rten", "ppocrv6_tiny_rec.rten", "ppocrv6_dict.txt",
    "ppocrv6_tiny_int8_det.rten", "ppocrv6_tiny_int8_rec.rten",
    "ppocrv6_small_det.rten", "ppocrv6_small_rec.rten",
    "ppocrv6_small_int8_det.rten", "ppocrv6_small_int8_rec.rten",
    "ppocrv6_medium_det.rten", "ppocrv6_medium_rec.rten",
    "ppocrv6_medium_int8_det.rten", "ppocrv6_medium_int8_rec.rten",
    "ppocrv5_mobile_det.rten", "ppocrv5_mobile_rec.rten", "ppocrv5_dict.txt",
    "ppocrv5_server_det.rten", "ppocrv5_server_rec.rten",
    "ppocrv4_mobile_det.rten", "ppocrv4_mobile_rec.rten", "ppocrv4_dict.txt",
    "ppocrv4_server_det.rten", "ppocrv4_server_rec.rten",
    "indra_det.rten", "indra_rec.rten", "indra_meta.json",
    "indra_order.rten", "indra_orient.rten", "indra_rec_cjk.rten",
    "indra_rec_rtl.rten", "indra_rec_world.rten", "indra_script.rten",
    "indra_mathcls.rten", "indra_math_enc.rten", "indra_math_dec_init.rten",
    "indra_math_dec_step.rten", "indra_math_meta.json",
    "ppocr_cls.rten", "ppocr_orient.rten", "uvdoc.rten",
    "layout_cdla.rten", "layout_publaynet.rten",
    "ppocrv5_rec_arabic.rten", "ppocrv5_dict_arabic.txt",
    "ppocrv5_rec_cyrillic.rten", "ppocrv5_dict_cyrillic.txt",
    "ppocrv5_rec_devanagari.rten", "ppocrv5_dict_devanagari.txt",
    "ppocrv5_rec_el.rten", "ppocrv5_dict_el.txt",
    "ppocrv5_rec_eslav.rten", "ppocrv5_dict_eslav.txt",
    "ppocrv5_rec_korean.rten", "ppocrv5_dict_korean.txt",
    "ppocrv5_rec_latin.rten", "ppocrv5_dict_latin.txt",
    "ppocrv5_rec_ta.rten", "ppocrv5_dict_ta.txt",
    "ppocrv5_rec_te.rten", "ppocrv5_dict_te.txt",
    "ppocrv5_rec_th.rten", "ppocrv5_dict_th.txt",
    "ppocrv4_rec_japan.rten", "ppocrv4_dict_japan.txt",
    "ppocrv4_rec_chinese_cht.rten", "ppocrv4_dict_chinese_cht.txt",
    "ppocrv4_rec_kannada.rten", "ppocrv4_dict_kannada.txt",
];

/// Downloads all assets for a given model pipeline and optional extras.
#[allow(clippy::too_many_arguments)]
pub fn download_model_pipeline(
    model_name: Option<&str>,
    lang: Option<&str>,
    include_cls: bool,
    include_orient: bool,
    include_unwarp: bool,
    layout_variant: Option<&str>,
    include_reading_order: bool,
    cache_dir: &Path,
) -> Result<Vec<PathBuf>, EngineError> {
    let canonical_model = normalize_model_name(model_name);
    let base_assets = get_base_pipeline_assets(canonical_model);
    let mut downloaded = Vec::new();

    let det_file = cache_dir.join(base_assets.det_filename);
    downloaded.push(download_asset(base_assets.det_filename, &det_file)?);

    let (rec_filename, dict_filename) = if let Some(l) = lang {
        if let Some((lang_rec, lang_dict)) = resolve_language_assets(l) {
            (lang_rec, lang_dict)
        } else {
            (base_assets.rec_filename, base_assets.dict_filename)
        }
    } else {
        (base_assets.rec_filename, base_assets.dict_filename)
    };

    let rec_file = cache_dir.join(rec_filename);
    downloaded.push(download_asset(rec_filename, &rec_file)?);

    let dict_file = cache_dir.join(dict_filename);
    downloaded.push(download_asset(dict_filename, &dict_file)?);

    if include_cls {
        let cls_file = cache_dir.join("ppocr_cls.rten");
        downloaded.push(download_asset("ppocr_cls.rten", &cls_file)?);
    }

    if include_orient {
        let orient_file = if canonical_model == "indraocr" {
            "indra_orient.rten"
        } else {
            "ppocr_orient.rten"
        };
        let target = cache_dir.join(orient_file);
        downloaded.push(download_asset(orient_file, &target)?);
    }

    if include_unwarp {
        let target = cache_dir.join("uvdoc.rten");
        downloaded.push(download_asset("uvdoc.rten", &target)?);
    }

    if let Some(variant) = layout_variant {
        let layout_file = if variant == "publaynet" {
            "layout_publaynet.rten"
        } else {
            "layout_cdla.rten"
        };
        let target = cache_dir.join(layout_file);
        downloaded.push(download_asset(layout_file, &target)?);
    }

    if include_reading_order || canonical_model == "indraocr" {
        let target = cache_dir.join("indra_order.rten");
        downloaded.push(download_asset("indra_order.rten", &target)?);
    }

    Ok(downloaded)
}

/// Downloads all available canonical models to the cache directory.
pub fn download_all_models(cache_dir: &Path) -> Result<Vec<PathBuf>, EngineError> {
    let mut downloaded = Vec::new();
    for asset in ALL_MODEL_ASSETS {
        let target = cache_dir.join(asset);
        downloaded.push(download_asset(asset, &target)?);
    }
    Ok(downloaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_name() {
        assert_eq!(normalize_model_name(None), "indraocr");
        assert_eq!(normalize_model_name(Some("")), "indraocr");
        assert_eq!(normalize_model_name(Some("ppocrv6")), "ppocrv6-tiny");
        assert_eq!(normalize_model_name(Some("ppv6")), "ppocrv6-tiny");
        assert_eq!(normalize_model_name(Some("PP-OCRv6-Tiny")), "ppocrv6-tiny");
        assert_eq!(normalize_model_name(Some("ppocrv6_small")), "ppocrv6-small");
        assert_eq!(normalize_model_name(Some("ppocrv5")), "ppocrv5-mobile");
        assert_eq!(normalize_model_name(Some("ppv5")), "ppocrv5-mobile");
        assert_eq!(normalize_model_name(Some("ppv4")), "ppocrv4-mobile");
        assert_eq!(normalize_model_name(Some("ppv3")), "ppocrv3-mobile");
        assert_eq!(normalize_model_name(Some("ppocrv4_server")), "ppocrv4-server");
        assert_eq!(normalize_model_name(Some("indraocr")), "indraocr");
        assert_eq!(normalize_model_name(Some("indra")), "indraocr");
        assert_eq!(normalize_model_name(Some("indra-ocr")), "indraocr");
    }

    #[test]
    fn test_resolve_language_assets() {
        let (rec_ja, dict_ja) = resolve_language_assets("ja").unwrap();
        assert_eq!(rec_ja, "ppocrv4_rec_japan.rten");
        assert_eq!(dict_ja, "ppocrv4_dict_japan.txt");

        let (rec_ko, dict_ko) = resolve_language_assets("korean").unwrap();
        assert_eq!(rec_ko, "ppocrv5_rec_korean.rten");
        assert_eq!(dict_ko, "ppocrv5_dict_korean.txt");

        let (rec_ar, dict_ar) = resolve_language_assets("ar").unwrap();
        assert_eq!(rec_ar, "ppocrv5_rec_arabic.rten");
        assert_eq!(dict_ar, "ppocrv5_dict_arabic.txt");

        assert!(resolve_language_assets("unknown-lang").is_none());
    }

    #[test]
    fn test_resolve_asset_explicit_override() {
        let dummy_path = PathBuf::from("Cargo.toml");
        let cache_dir = PathBuf::from("/tmp/rusto_cache_test");
        let result = resolve_asset(Some(&dummy_path), "nonexistent.rten", &cache_dir);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dummy_path);

        let missing = PathBuf::from("definitely_missing_file_12345.rten");
        let err = resolve_asset(Some(&missing), "test.rten", &cache_dir);
        assert!(err.is_err());
    }

    #[test]
    fn test_resolve_asset_flow_uses_cache() {
        let cache_dir = PathBuf::from("models/PPOCR_v6_tiny");
        let result = resolve_asset(None, "dict.txt", &cache_dir);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_base_pipeline_from_model_dir() {
        let model_dir = PathBuf::from("models/PPOCR_v6_tiny");
        let cache_dir = PathBuf::from("/tmp/rusto_empty_cache_dir");
        let (det, rec, dict) = resolve_base_pipeline(
            None,
            Some(&model_dir),
            None,
            None,
            None,
            None,
            &cache_dir,
        ).unwrap();

        assert_eq!(det, model_dir.join("det.rten"));
        assert_eq!(rec, model_dir.join("rec.rten"));
        assert_eq!(dict, model_dir.join("dict.txt"));
    }

    #[test]
    fn test_resolve_base_pipeline_with_det_override() {
        let model_dir = PathBuf::from("models/PPOCR_v6_tiny");
        let custom_det = PathBuf::from("Cargo.toml");
        let cache_dir = PathBuf::from("/tmp/rusto_empty_cache_dir");
        let (det, rec, dict) = resolve_base_pipeline(
            None,
            Some(&model_dir),
            None,
            Some(&custom_det),
            None,
            None,
            &cache_dir,
        ).unwrap();

        assert_eq!(det, custom_det);
        assert_eq!(rec, model_dir.join("rec.rten"));
        assert_eq!(dict, model_dir.join("dict.txt"));
    }
}
