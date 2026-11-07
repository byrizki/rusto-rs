use std::fs;
use std::path::{Path, PathBuf};

use crate::error::CoreError;

pub const DEFAULT_HF_BASE_URL: &str = "https://huggingface.co/byrizki/rusto-models/resolve/main";
pub const DEFAULT_BASE_URL: &str = DEFAULT_HF_BASE_URL;

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
        _ => "indraocr",
    }
}

/// Normalizes language identifier to canonical asset names.
pub fn resolve_language_assets(lang: &str) -> Option<(&'static str, &'static str)> {
    match lang.trim().to_ascii_lowercase().as_str() {
        "latin" | "en" | "eng" | "english" => {
            Some(("ppocrv5_rec_latin.rten", "ppocrv5_dict_latin.txt"))
        }
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
        "rtl" => {
            Some(("indra_rec_rtl.rten", "indra_meta.json"))
        }
        "world" => {
            Some(("indra_rec_world.rten", "indra_meta.json"))
        }
        _ => None,
    }
}

/// Resolves language-specific recognition assets taking the canonical model family into account.
pub fn resolve_model_language_assets(
    canonical_model: &str,
    lang: &str,
) -> Option<(&'static str, &'static str)> {
    let normalized = lang.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "latin" | "en" | "eng" | "english") {
        if canonical_model == "indraocr" || canonical_model.starts_with("ppocrv6") {
            return None;
        }
    }
    resolve_language_assets(&normalized)
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

/// Resolves the default cache directory for models.
pub fn get_cache_dir(custom: Option<&Path>) -> PathBuf {
    if let Some(custom_path) = custom {
        return custom_path.to_path_buf();
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(env_path) = std::env::var("RUSTO_CACHE_DIR") {
            if !env_path.trim().is_empty() {
                return PathBuf::from(env_path);
            }
        }

        if let Some(cache) = dirs::cache_dir() {
            return cache.join("rusto").join("models");
        }
    }

    PathBuf::from(".cache").join("rusto").join("models")
}

/// Finds local checkout repository fallback candidates for an asset filename.
pub fn find_local_repo_candidates(asset_filename: &str) -> Vec<PathBuf> {
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
        let base = sub.strip_suffix(".mnn").or_else(|| sub.strip_suffix(".rten")).unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}.mnn", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_tiny/{}.onnx", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6/{}.mnn", base)));
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
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv6_medium_int8_") {
        let base = sub.strip_suffix(".rten").unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_medium_int8/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_medium_int8/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v6_medium_int8/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv5_mobile_") {
        let base = sub.strip_suffix(".mnn").or_else(|| sub.strip_suffix(".rten")).unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}.mnn", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_mobile/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv5_server_") {
        let base = sub.strip_suffix(".mnn").or_else(|| sub.strip_suffix(".rten")).unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_server/{}.mnn", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_server/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_server/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v5_server/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv4_mobile_") {
        let base = sub.strip_suffix(".mnn").or_else(|| sub.strip_suffix(".rten")).unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}.mnn", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_mobile/{}.onnx", base)));
    } else if let Some(sub) = asset_filename.strip_prefix("ppocrv4_server_") {
        let base = sub.strip_suffix(".mnn").or_else(|| sub.strip_suffix(".rten")).unwrap_or(sub);
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_server/{}.mnn", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_server/{}", sub)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_server/{}.rten", base)));
        candidates.push(PathBuf::from(format!("models/PPOCR_v4_server/{}.onnx", base)));
    } else if asset_filename == "uvdoc.rten" || asset_filename == "uvdoc.mnn" {
        candidates.push(PathBuf::from("models/uvdoc/uvdoc.mnn"));
        candidates.push(PathBuf::from("models/uvdoc/uvdoc.rten"));
        candidates.push(PathBuf::from("models/uvdoc.mnn"));
        candidates.push(PathBuf::from("models/uvdoc.rten"));
    } else if asset_filename == "ppocr_orient.rten" || asset_filename == "ppocr_orient.mnn" || asset_filename == "orient.mnn" {
        candidates.push(PathBuf::from("models/orient/orient.mnn"));
        candidates.push(PathBuf::from("models/orient/orient.rten"));
        candidates.push(PathBuf::from("models/ppocr_orient.mnn"));
        candidates.push(PathBuf::from("models/ppocr_orient.rten"));
        candidates.push(PathBuf::from("models/orient.mnn"));
        candidates.push(PathBuf::from("models/orient.rten"));
    } else if asset_filename == "ppocr_cls.rten" || asset_filename == "ppocr_cls.mnn" || asset_filename == "cls.mnn" {
        candidates.push(PathBuf::from("models/cls/cls.mnn"));
        candidates.push(PathBuf::from("models/cls/cls.rten"));
        candidates.push(PathBuf::from("models/ppocr_cls.mnn"));
        candidates.push(PathBuf::from("models/ppocr_cls.rten"));
        candidates.push(PathBuf::from("models/cls.mnn"));
        candidates.push(PathBuf::from("models/cls.rten"));
    } else if asset_filename.contains("layout") {
        candidates.push(PathBuf::from(format!("models/layout/{}", asset_filename)));
        candidates.push(PathBuf::from(format!("models/{}", asset_filename)));
    }

    candidates
}

/// Searches a directory for model assets matching the specified kind.
pub fn find_in_model_dir(dir: &Path, kind: &str) -> Option<PathBuf> {
    let mut check_dirs = vec![dir.to_path_buf()];
    if !dir.is_absolute() {
        check_dirs.push(PathBuf::from("..").join(dir));
        check_dirs.push(PathBuf::from("../..").join(dir));
    }

    let exact_names: &[&str] = match kind {
        "det" => &["det.mnn", "det.rten", "det.onnx", "model.mnn", "model.rten"],
        "rec" => &["rec.mnn", "rec.rten", "rec.onnx"],
        "dict" => &["dict.txt", "ppocr_keys_v1.txt", "keys.txt", "vocab.txt", "meta.json", "indra_meta.json"],
        "cls" => &["cls.mnn", "cls.rten", "cls.onnx", "ppocr_cls.mnn", "ppocr_cls.rten"],
        "orient" => &["orient.mnn", "orient.rten", "orient.onnx", "ppocr_orient.mnn", "ppocr_orient.rten"],
        "unwarp" => &["uvdoc.mnn", "uvdoc.rten", "unwarp.mnn", "unwarp.rten"],
        "layout" => &["layout_cdla.mnn", "layout_cdla.rten", "layout_publaynet.mnn", "layout_publaynet.rten", "layout.mnn", "layout.rten"],
        "order" => &["indra_order.mnn", "indra_order.rten", "order.mnn", "order.rten"],
        _ => &[],
    };

    for check_dir in &check_dirs {
        for &name in exact_names {
            let p = check_dir.join(name);
            if p.exists() && fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false) {
                return Some(dir.join(name));
            }
        }
    }

    for check_dir in &check_dirs {
        if let Ok(entries) = fs::read_dir(check_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    let matches = match kind {
                        "det" => (name.contains("det") || name.contains("detection")) && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        "rec" => (name.contains("rec") || name.contains("recognition")) && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        "dict" => (name.contains("dict") || name.contains("keys") || name.contains("meta") || name.contains("vocab")) && (name.ends_with(".txt") || name.ends_with(".json")),
                        "cls" => name.contains("cls") && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        "orient" => name.contains("orient") && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        "unwarp" => (name.contains("unwarp") || name.contains("uvdoc")) && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        "layout" => name.contains("layout") && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        "order" => name.contains("order") && (name.ends_with(".mnn") || name.ends_with(".rten") || name.ends_with(".onnx")),
                        _ => false,
                    };
                    if matches && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false) {
                        if let Some(file_name) = path.file_name() {
                            return Some(dir.join(file_name));
                        }
                    }
                }
            }
        }
    }

    None
}


pub fn resolve_default_model_dir_asset(dir: &Path, kind: &str, filename: &str) -> PathBuf {
    let default_name = match kind {
        "det" => "det.mnn",
        "rec" => "rec.mnn",
        "dict" => {
            if filename.ends_with(".json") {
                "meta.json"
            } else {
                "dict.txt"
            }
        }
        "cls" => "cls.mnn",
        "orient" => "orient.mnn",
        "unwarp" => "uvdoc.mnn",
        "layout" => "layout_cdla.mnn",
        "order" => "indra_order.mnn",
        _ => filename,
    };
    dir.join(default_name)
}

/// Resolves a single model asset locally without network access.
pub fn resolve_asset_local(
    explicit_path: Option<&Path>,
    model_dir: Option<&Path>,
    asset_kind: &str,
    asset_filename: &str,
    cache_dir: &Path,
) -> Result<PathBuf, CoreError> {
    // 1. Explicit path
    if let Some(path) = explicit_path {
        if !path.as_os_str().is_empty() {
            if path.exists() {
                return Ok(path.to_path_buf());
            }
            return Err(CoreError::Model(format!(
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

    // 3. Cache Path (including .rten / .mnn stem fallback)
    let cached_file = cache_dir.join(asset_filename);
    if cached_file.exists() && fs::metadata(&cached_file).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(cached_file);
    }
    if let Some(stem) = asset_filename
        .strip_suffix(".mnn")
        .or_else(|| asset_filename.strip_suffix(".rten"))
    {
        for ext in ["rten", "mnn"] {
            let alt = cache_dir.join(format!("{stem}.{ext}"));
            if alt.exists() && fs::metadata(&alt).map(|m| m.len() > 0).unwrap_or(false) {
                return Ok(alt);
            }
        }
    }

    // 4. Local checkout repository candidate fallback
    for cand in find_local_repo_candidates(asset_filename) {
        if cand.exists() && fs::metadata(&cand).map(|m| m.len() > 0).unwrap_or(false) {
            return Ok(cand);
        }
    }

    Err(CoreError::Model(format!(
        "Model asset '{}' not found in cache or local directories",
        asset_filename
    )))
}

/// Resolves complete base OCR pipeline paths locally.
pub fn resolve_base_pipeline(
    model_name: Option<&str>,
    model_dir: Option<&Path>,
    lang: Option<&str>,
    det_override: Option<&Path>,
    rec_override: Option<&Path>,
    dict_override: Option<&Path>,
    cache_dir: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf), CoreError> {
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

    let det_path = resolve_asset_local(det_override, effective_dir, "det", base_assets.det_filename, cache_dir)?;

    let (rec_filename, dict_filename) = if let Some(l) = lang {
        if let Some((lang_rec, lang_dict)) = resolve_model_language_assets(canonical_model, l) {
            (lang_rec, lang_dict)
        } else {
            (base_assets.rec_filename, base_assets.dict_filename)
        }
    } else {
        (base_assets.rec_filename, base_assets.dict_filename)
    };

    let rec_path = resolve_asset_local(rec_override, effective_dir, "rec", rec_filename, cache_dir)?;
    let dict_path = resolve_asset_local(dict_override, effective_dir, "dict", dict_filename, cache_dir)?;

    Ok((det_path, rec_path, dict_path))
}

/// Maps a model asset filename to its organized path in the Hugging Face repository.
pub fn resolve_hf_asset_path(asset_filename: &str) -> &str {
    match asset_filename {
        "indra_det.rten" => "indraocr/det.rten",
        "indra_rec.rten" => "indraocr/rec.rten",
        "indra_meta.json" => "indraocr/meta.json",
        "indra_order.rten" => "indraocr/order.rten",
        "indra_orient.rten" => "indraocr/orient.rten",
        "indra_rec_cjk.rten" => "indraocr/rec_cjk.rten",
        "indra_rec_rtl.rten" => "indraocr/rec_rtl.rten",
        "indra_rec_world.rten" => "indraocr/rec_world.rten",
        "indra_script.rten" => "indraocr/script.rten",
        "indra_mathcls.rten" => "indraocr/mathcls.rten",
        "indra_math_enc.rten" => "indraocr/math_enc.rten",
        "indra_math_dec_init.rten" => "indraocr/math_dec_init.rten",
        "indra_math_dec_step.rten" => "indraocr/math_dec_step.rten",
        "indra_math_meta.json" => "indraocr/math_meta.json",

        "ppocrv6_tiny_det.rten" => "ppocrv6-tiny/det.rten",
        "ppocrv6_tiny_rec.rten" => "ppocrv6-tiny/rec.rten",
        "ppocrv6_tiny_int8_det.rten" => "ppocrv6-tiny-int8/det.rten",
        "ppocrv6_tiny_int8_rec.rten" => "ppocrv6-tiny-int8/rec.rten",
        "ppocrv6_dict.txt" => "ppocrv6-tiny/dict.txt",
        "ppocrv6_small_det.rten" => "ppocrv6-small/det.rten",
        "ppocrv6_small_rec.rten" => "ppocrv6-small/rec.rten",
        "ppocrv6_small_int8_det.rten" => "ppocrv6-small-int8/det.rten",
        "ppocrv6_small_int8_rec.rten" => "ppocrv6-small-int8/rec.rten",
        "ppocrv6_medium_det.rten" => "ppocrv6-medium/det.rten",
        "ppocrv6_medium_rec.rten" => "ppocrv6-medium/rec.rten",
        "ppocrv6_medium_int8_det.rten" => "ppocrv6-medium-int8/det.rten",
        "ppocrv6_medium_int8_rec.rten" => "ppocrv6-medium-int8/rec.rten",

        "ppocrv5_mobile_det.rten" => "ppocrv5-mobile/det.rten",
        "ppocrv5_mobile_rec.rten" => "ppocrv5-mobile/rec.rten",
        "ppocrv5_server_det.rten" => "ppocrv5-server/det.rten",
        "ppocrv5_server_rec.rten" => "ppocrv5-server/rec.rten",
        "ppocrv5_dict.txt" => "ppocrv5-mobile/dict.txt",

        "ppocrv4_mobile_det.rten" => "ppocrv4-mobile/det.rten",
        "ppocrv4_mobile_rec.rten" => "ppocrv4-mobile/rec.rten",
        "ppocrv4_server_det.rten" => "ppocrv4-server/det.rten",
        "ppocrv4_server_rec.rten" => "ppocrv4-server/rec.rten",
        "ppocrv4_dict.txt" => "ppocrv4-mobile/dict.txt",

        "ppocrv5_rec_arabic.rten" => "multilingual/ppocrv5_rec_arabic.rten",
        "ppocrv5_dict_arabic.txt" => "multilingual/ppocrv5_dict_arabic.txt",
        "ppocrv5_rec_cyrillic.rten" => "multilingual/ppocrv5_rec_cyrillic.rten",
        "ppocrv5_dict_cyrillic.txt" => "multilingual/ppocrv5_dict_cyrillic.txt",
        "ppocrv5_rec_devanagari.rten" => "multilingual/ppocrv5_rec_devanagari.rten",
        "ppocrv5_dict_devanagari.txt" => "multilingual/ppocrv5_dict_devanagari.txt",
        "ppocrv5_rec_el.rten" => "multilingual/ppocrv5_rec_el.rten",
        "ppocrv5_dict_el.txt" => "multilingual/ppocrv5_dict_el.txt",
        "ppocrv5_rec_eslav.rten" => "multilingual/ppocrv5_rec_eslav.rten",
        "ppocrv5_dict_eslav.txt" => "multilingual/ppocrv5_dict_eslav.txt",
        "ppocrv5_rec_korean.rten" => "multilingual/ppocrv5_rec_korean.rten",
        "ppocrv5_dict_korean.txt" => "multilingual/ppocrv5_dict_korean.txt",
        "ppocrv5_rec_latin.rten" => "multilingual/ppocrv5_rec_latin.rten",
        "ppocrv5_dict_latin.txt" => "multilingual/ppocrv5_dict_latin.txt",
        "ppocrv5_rec_ta.rten" => "multilingual/ppocrv5_rec_ta.rten",
        "ppocrv5_dict_ta.txt" => "multilingual/ppocrv5_dict_ta.txt",
        "ppocrv5_rec_te.rten" => "multilingual/ppocrv5_rec_te.rten",
        "ppocrv5_dict_te.txt" => "multilingual/ppocrv5_dict_te.txt",
        "ppocrv5_rec_th.rten" => "multilingual/ppocrv5_rec_th.rten",
        "ppocrv5_dict_th.txt" => "multilingual/ppocrv5_dict_th.txt",
        "ppocrv4_rec_japan.rten" => "multilingual/ppocrv4_rec_japan.rten",
        "ppocrv4_dict_japan.txt" => "multilingual/ppocrv4_dict_japan.txt",
        "ppocrv4_rec_chinese_cht.rten" => "multilingual/ppocrv4_rec_chinese_cht.rten",
        "ppocrv4_dict_chinese_cht.txt" => "multilingual/ppocrv4_dict_chinese_cht.txt",
        "ppocrv4_rec_kannada.rten" => "multilingual/ppocrv4_rec_kannada.rten",
        "ppocrv4_dict_kannada.txt" => "multilingual/ppocrv4_dict_kannada.txt",

        "ppocr_cls.rten" | "cls.rten" => "modules/cls.rten",
        "ppocr_orient.rten" | "orient.rten" => "modules/orient.rten",
        "uvdoc.rten" => "modules/uvdoc.rten",
        "layout_cdla.rten" => "modules/layout_cdla.rten",
        "layout_publaynet.rten" => "modules/layout_publaynet.rten",

        _ => asset_filename,
    }
}

pub fn get_pipeline_assets(
    model_name: Option<&str>,
    lang: Option<&str>,
    include_cls: bool,
    include_orient: bool,
    include_unwarp: bool,
    layout_variant: Option<&str>,
    include_reading_order: bool,
) -> Vec<&'static str> {
    let canonical_model = normalize_model_name(model_name);
    let base_assets = get_base_pipeline_assets(canonical_model);
    let mut assets = Vec::with_capacity(8);

    assets.push(base_assets.det_filename);

    let (rec_filename, dict_filename) = if let Some(l) = lang {
        if let Some((lang_rec, lang_dict)) = resolve_model_language_assets(canonical_model, l) {
            (lang_rec, lang_dict)
        } else {
            (base_assets.rec_filename, base_assets.dict_filename)
        }
    } else {
        (base_assets.rec_filename, base_assets.dict_filename)
    };
    assets.push(rec_filename);
    assets.push(dict_filename);

    if include_cls {
        assets.push("ppocr_cls.rten");
    }
    if include_orient {
        if canonical_model == "indraocr" {
            assets.push("indra_orient.rten");
        } else {
            assets.push("ppocr_orient.rten");
        }
    }
    if include_unwarp {
        assets.push("uvdoc.rten");
    }
    if let Some(variant) = layout_variant {
        if variant == "publaynet" {
            assets.push("layout_publaynet.rten");
        } else {
            assets.push("layout_cdla.rten");
        }
    }
    if include_reading_order {
        assets.push("indra_order.rten");
    }

    assets
}

pub fn get_asset_download_urls(asset_filename: &str) -> Vec<String> {
    if let Ok(custom) = std::env::var("RUSTO_MODELS_URL") {
        if custom.ends_with('/') {
            vec![format!("{}{}", custom, asset_filename)]
        } else if custom.contains(asset_filename) {
            vec![custom]
        } else {
            vec![format!("{}/{}", custom, asset_filename)]
        }
    } else {
        let hf_path = resolve_hf_asset_path(asset_filename);
        vec![format!("{}/{}", DEFAULT_HF_BASE_URL, hf_path)]
    }
}

pub fn populate_from_local_repo(asset_filename: &str, target_path: &Path) -> bool {
    for cand in find_local_repo_candidates(asset_filename) {
        if cand != target_path && cand.exists() && fs::metadata(&cand).map(|m| m.len() > 0).unwrap_or(false) {
            if let Some(parent) = target_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if fs::copy(&cand, target_path).is_ok() {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadTarget {
    pub filename: String,
    pub target_path: PathBuf,
    pub urls: Vec<String>,
    pub already_exists: bool,
}

pub fn plan_pipeline_download(
    model_name: Option<&str>,
    lang: Option<&str>,
    include_cls: bool,
    include_orient: bool,
    include_unwarp: bool,
    layout_variant: Option<&str>,
    include_reading_order: bool,
    cache_dir: &Path,
) -> Vec<ModelDownloadTarget> {
    let assets = get_pipeline_assets(
        model_name,
        lang,
        include_cls,
        include_orient,
        include_unwarp,
        layout_variant,
        include_reading_order,
    );
    plan_assets_download(&assets, cache_dir)
}

pub fn plan_all_models_download(cache_dir: &Path) -> Vec<ModelDownloadTarget> {
    plan_assets_download(ALL_MODEL_ASSETS, cache_dir)
}

pub fn plan_assets_download<S: AsRef<str>>(assets: &[S], cache_dir: &Path) -> Vec<ModelDownloadTarget> {
    assets
        .iter()
        .map(|asset| {
            let name = asset.as_ref();
            let target = cache_dir.join(name);
            let already_exists = if target.exists() && fs::metadata(&target).map(|m| m.len() > 0).unwrap_or(false) {
                true
            } else {
                populate_from_local_repo(name, &target)
            };
            let urls = get_asset_download_urls(name);
            ModelDownloadTarget {
                filename: name.to_string(),
                target_path: target,
                urls,
                already_exists,
            }
        })
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_pipeline_assets() {
        let assets = get_pipeline_assets(
            Some("indraocr"),
            Some("cjk"),
            true,
            true,
            true,
            Some("publaynet"),
            true,
        );
        assert_eq!(assets[0], "indra_det.rten");
        assert_eq!(assets[1], "indra_rec_cjk.rten");
        assert_eq!(assets[2], "indra_meta.json");
        assert!(assets.contains(&"ppocr_cls.rten"));
        assert!(assets.contains(&"indra_orient.rten"));
        assert!(assets.contains(&"uvdoc.rten"));
        assert!(assets.contains(&"layout_publaynet.rten"));
        assert!(assets.contains(&"indra_order.rten"));
    }

    #[test]
    fn test_plan_pipeline_download() {
        let temp_dir = std::env::temp_dir().join("rusto_test_models_plan");
        let plan = plan_pipeline_download(
            Some("ppocrv6-tiny"),
            None,
            false,
            false,
            false,
            None,
            false,
            &temp_dir,
        );
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].filename, "ppocrv6_tiny_det.rten");
        assert_eq!(plan[1].filename, "ppocrv6_tiny_rec.rten");
        assert_eq!(plan[2].filename, "ppocrv6_dict.txt");
        assert!(!plan[0].urls.is_empty());
    }
}
