use crate::calibration::{CalibrationOptions, OptimizationOptions};
use crate::image::Mat;
use crate::types::DetConfig;

/// Profile of optical, dimensional, and contrast characteristics extracted from an input image.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageProfile {
    pub width: usize,
    pub height: usize,
    pub aspect_ratio: f32,
    pub mean_luma: f32,
    pub std_luma: f32,
    pub illumination_variance: f32,
    pub bg_min: f32,
    pub bg_max: f32,
    pub bg_var: f32,
    pub bg_chroma_ratio: f32,
    pub gradient_energy: f32,
}

/// Profiles an image in ~1ms to detect lighting gradients, halftone noise, relief/carving, blur, and scale.
pub fn profile_image(img: &Mat) -> ImageProfile {
    let (w, h) = (img.cols().max(0) as usize, img.rows().max(0) as usize);
    if w == 0 || h == 0 {
        return ImageProfile {
            width: 0,
            height: 0,
            aspect_ratio: 1.0,
            mean_luma: 128.0,
            std_luma: 0.0,
            illumination_variance: 0.0,
            bg_min: 255.0,
            bg_max: 255.0,
            bg_var: 0.0,
            bg_chroma_ratio: 0.0,
            gradient_energy: 0.0,
        };
    }

    let aspect_ratio = w as f32 / h as f32;

    let owned_rgb;
    let (raw, stride_w) = if let Some(rgb) = img.as_dynamic().as_rgb8() {
        (rgb.as_raw().as_slice(), rgb.width() as usize)
    } else {
        owned_rgb = img.to_rgb8();
        (owned_rgb.as_raw().as_slice(), owned_rgb.width() as usize)
    };

    // Subsample step to keep profiling < 1ms even on 4K images
    let step_x = (w / 128).max(1);
    let step_y = (h / 128).max(1);

    let mut sum_luma = 0.0f64;
    let mut sum_luma_sq = 0.0f64;
    let mut sampled_count = 0u32;
    let mut bg_chroma_count = 0u32;
    let mut light_pixel_count = 0u32;

    // 4x4 grid block statistics for illumination gradient detection
    let mut grid_sums = [[0.0f64; 4]; 4];
    let mut grid_counts = [0u32; 16];
    let mut grid_hist = [[0u16; 16]; 16];

    for y in (0..h).step_by(step_y) {
        let gy = (y * 4 / h).min(3);
        let row_off = y * stride_w * 3;
        for x in (0..w).step_by(step_x) {
            let gx = (x * 4 / w).min(3);
            let idx = row_off + x * 3;
            if idx + 2 >= raw.len() {
                continue;
            }
            let r = raw[idx] as f32;
            let g = raw[idx + 1] as f32;
            let b = raw[idx + 2] as f32;

            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            sum_luma += luma as f64;
            sum_luma_sq += (luma as f64) * (luma as f64);
            sampled_count += 1;

            grid_sums[gy][gx] += luma as f64;
            let cell_idx = gy * 4 + gx;
            grid_counts[cell_idx] += 1;
            let bin = ((luma / 16.0) as usize).min(15);
            grid_hist[cell_idx][bin] = grid_hist[cell_idx][bin].saturating_add(1);

            if luma > 70.0 {
                light_pixel_count += 1;
                let max_c = r.max(g.max(b));
                let min_c = r.min(g.min(b));
                if (max_c - min_c) >= 5.0 {
                    bg_chroma_count += 1;
                }
            }
        }
    }

    let mean_luma = if sampled_count > 0 {
        (sum_luma / sampled_count as f64) as f32
    } else {
        128.0
    };

    let variance = if sampled_count > 1 {
        let var = (sum_luma_sq / sampled_count as f64) - (mean_luma as f64 * mean_luma as f64);
        var.max(0.0) as f32
    } else {
        0.0
    };
    let std_luma = variance.sqrt();

    // Illumination variance across 4x4 grid
    let mut min_block_mean = 255.0f32;
    let mut max_block_mean = 0.0f32;
    let mut cell_bgs = [128.0f32; 16];
    for gy in 0..4 {
        for gx in 0..4 {
            let cell_idx = gy * 4 + gx;
            let count = grid_counts[cell_idx];
            if count > 0 {
                let bm = (grid_sums[gy][gx] / count as f64) as f32;
                min_block_mean = min_block_mean.min(bm);
                max_block_mean = max_block_mean.max(bm);

                // 85th percentile background estimate of this cell
                let target = (count * 85) / 100;
                let mut acc = 0;
                for b in 0..16 {
                    acc += grid_hist[cell_idx][b] as u32;
                    if acc >= target {
                        cell_bgs[cell_idx] = (b * 16 + 8) as f32;
                        break;
                    }
                }
            }
        }
    }
    let illumination_variance = (max_block_mean - min_block_mean).max(0.0);

    let mut bg_min = 255.0f32;
    let mut bg_max = 0.0f32;
    for &bg in &cell_bgs {
        bg_min = bg_min.min(bg);
        bg_max = bg_max.max(bg);
    }
    let bg_var = (bg_max - bg_min).max(0.0);

    let bg_chroma_ratio = if light_pixel_count > 0 {
        bg_chroma_count as f32 / light_pixel_count as f32
    } else {
        0.0
    };

    // Fast discrete gradient variance (horizontal differences)
    let mut grad_sum = 0.0f64;
    let mut grad_count = 0u32;
    let g_step_x = (w / 64).max(1);
    let g_step_y = (h / 64).max(1);
    for y in (0..h).step_by(g_step_y) {
        let row_off = y * stride_w * 3;
        for x in (0..w.saturating_sub(g_step_x)).step_by(g_step_x) {
            let idx1 = row_off + x * 3;
            let idx2 = row_off + (x + g_step_x) * 3;
            if idx2 + 2 < raw.len() {
                let l1 = 0.299 * raw[idx1] as f32 + 0.587 * raw[idx1 + 1] as f32 + 0.114 * raw[idx1 + 2] as f32;
                let l2 = 0.299 * raw[idx2] as f32 + 0.587 * raw[idx2 + 1] as f32 + 0.114 * raw[idx2 + 2] as f32;
                grad_sum += (l2 - l1).abs() as f64;
                grad_count += 1;
            }
        }
    }
    let gradient_energy = if grad_count > 0 {
        (grad_sum / grad_count as f64) as f32
    } else {
        0.0
    };

    ImageProfile {
        width: w,
        height: h,
        aspect_ratio,
        mean_luma,
        std_luma,
        illumination_variance,
        bg_min,
        bg_max,
        bg_var,
        bg_chroma_ratio,
        gradient_energy,
    }
}

/// High-level document archetype for classification and targeted calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentArchetype {
    /// Pristine digital documents: vector PDF exports, math exams, academic papers with crisp dark text on uniform white/light background.
    /// Strict "Do No Harm": Zero calibration is applied.
    CleanDigital,
    /// ID cards, driver licenses, banknotes, or certificates with security background patterns or guilloche watermarks.
    PatternedID,
    /// Physical low-contrast relief: engraved stone, gravestones, metal plaques.
    CarvedRelief,
    /// Mobile or camera captures with uneven illumination gradients and casting shadows.
    CameraCapture,
    /// Degraded printed pages: historical newspapers, microfilms, aged books with faint ink or eroded strokes.
    DegradedPrint,
    /// General documents not matching specific specialized archetypes.
    General,
}

impl DocumentArchetype {
    /// Classifies an image profile into an actionable archetype.
    pub fn classify(profile: &ImageProfile) -> Self {
        // 1. Physical Carved Relief: Very high average brightness but almost zero variance (no dark ink)
        if profile.mean_luma > 185.0 && profile.std_luma < 25.0 {
            return DocumentArchetype::CarvedRelief;
        }

        // 2. Patterned ID: Saturated colored security patterns across the entire card face
        // Standard ID card dimensions (w, h <= 2400), card aspect ratio (~1.1 to 1.9 or 0.55 to 0.90),
        // no large white margins (bg_max <= 230.0), high chroma ratio.
        let is_card_aspect = (profile.aspect_ratio >= 1.1 && profile.aspect_ratio <= 1.9)
            || (profile.aspect_ratio >= 0.55 && profile.aspect_ratio <= 0.90);
        let is_card_size = profile.width <= 2400 && profile.height <= 2400;
        if profile.bg_chroma_ratio > 0.25 && is_card_aspect && is_card_size && profile.bg_max <= 230.0 {
            return DocumentArchetype::PatternedID;
        }

        // 3. Pristine Clean Digital: High white background paper across all grid cells, flat illumination.
        // If the background paper is bright white (bg_max >= 230.0) and uniform across cells (bg_var <= 35.0),
        // this is a digital document or clean scan.
        if profile.bg_max >= 230.0 && profile.bg_var <= 35.0 && profile.std_luma >= 15.0 {
            return DocumentArchetype::CleanDigital;
        }

        // 4. Mobile Camera / Severe Uneven Illumination: Significant lighting gradient on paper
        if profile.bg_var > 50.0 && profile.bg_min < 185.0 {
            return DocumentArchetype::CameraCapture;
        }

        DocumentArchetype::General
    }
}

/// Automatically infers optimal detection, optimization, and calibration settings from the image profile,
/// while respecting any explicit options provided by the caller.
pub fn resolve_auto_params(
    profile: &ImageProfile,
    base_det: &DetConfig,
    base_opt: Option<&OptimizationOptions>,
    base_calib: Option<&CalibrationOptions>,
) -> (DetConfig, OptimizationOptions, CalibrationOptions) {
    let mut det = base_det.clone();
    let mut opt = base_opt.cloned().unwrap_or_default();
    let mut calib = base_calib.cloned().unwrap_or_default();

    let archetype = DocumentArchetype::classify(profile);

    match archetype {
        DocumentArchetype::CleanDigital => {
            // "Do No Harm": Pristine digital documents require NO calibration.
            // Any calibration (sharpening halos, morphological stroke dilation) degrades accuracy.
            if calib.enabled.is_none() {
                calib.enabled = Some(false);
            }
        }
        DocumentArchetype::PatternedID => {
            if calib.descreen_background.is_none() {
                calib.descreen_background = Some(true);
                if calib.descreen_strength.is_none() {
                    calib.descreen_strength = Some((profile.bg_chroma_ratio * 1.5).clamp(0.8, 1.25));
                }
                calib.enabled = Some(true);
            }
            if opt.crop_padding_x.is_none() {
                opt.crop_padding_x = Some(0.03);
                opt.enabled = Some(true);
            }
            if opt.crop_padding_y.is_none() {
                opt.crop_padding_y = Some(0.03);
                opt.enabled = Some(true);
            }
            if calib.flatten_illumination.is_none() && profile.bg_var > 45.0 {
                calib.flatten_illumination = Some(true);
                calib.enabled = Some(true);
            }
        }
        DocumentArchetype::CarvedRelief => {
            if calib.relief_enhancement.is_none() {
                calib.relief_enhancement = Some(true);
                if calib.relief_strength.is_none() {
                    calib.relief_strength = Some((30.0 / profile.std_luma.max(5.0)).clamp(0.8, 2.5));
                }
                calib.enabled = Some(true);
            }
            if calib.flatten_illumination.is_none() && profile.bg_var > 45.0 {
                calib.flatten_illumination = Some(true);
                calib.enabled = Some(true);
            }
        }
        DocumentArchetype::CameraCapture => {
            if calib.flatten_illumination.is_none() {
                calib.flatten_illumination = Some(true);
                calib.enabled = Some(true);
            }
            if calib.auto_levels.is_none() && profile.std_luma < 35.0 {
                calib.auto_levels = Some(true);
                calib.enabled = Some(true);
            }
        }
        DocumentArchetype::DegradedPrint => {
            if calib.flatten_illumination.is_none() && profile.bg_var > 45.0 {
                calib.flatten_illumination = Some(true);
                calib.enabled = Some(true);
            }
        }
        DocumentArchetype::General => {
            if calib.flatten_illumination.is_none() && profile.bg_var > 50.0 && profile.bg_min < 185.0 {
                calib.flatten_illumination = Some(true);
                calib.enabled = Some(true);
            }
            if calib.auto_levels.is_none() && profile.std_luma < 30.0 && profile.bg_max < 210.0 {
                calib.auto_levels = Some(true);
                calib.enabled = Some(true);
            }
            // Only sharpen if image is blurry AND not on a flat light background
            if calib.sharpen.is_none()
                && profile.gradient_energy < 15.0
                && profile.width >= 400
                && profile.mean_luma < 230.0
            {
                calib.sharpen = Some(true);
                calib.enabled = Some(true);
            }
        }
    }

    // Dynamic scale-aware crop padding:
    // Standard document pages must NEVER have padding added (padding introduces bleed from adjacent lines/columns).
    // Only small isolated image chips or ID cards with faint text near borders benefit from padding.
    let total_pixels = profile.width * profile.height;
    let is_small_isolated = profile.width.min(profile.height) < 350 || total_pixels < 250_000;
    if opt.crop_padding_x.is_none() {
        if is_small_isolated {
            opt.crop_padding_x = Some(0.06);
            opt.enabled = Some(true);
        }
    }
    if opt.crop_padding_y.is_none() {
        if is_small_isolated {
            opt.crop_padding_y = Some(0.05);
            opt.enabled = Some(true);
        }
    }

    // Geometry & Detection limits:
    // PP-OCR DBNet is trained at 960. Do NOT blindly upscale limit_side_len based on image dimensions,
    // as pipeline adaptive_resolution already scales limit_side_len if detected text line height is tiny (< 9px).
    let max_dim = profile.width.max(profile.height);
    let min_dim = profile.width.min(profile.height);

    if archetype == DocumentArchetype::PatternedID && max_dim <= 2400 {
        det.limit_type = "max".to_string();
        det.limit_side_len = 960;
        det.use_dilation = false;
        if (det.box_thresh - 0.6).abs() < 1e-4 {
            det.box_thresh = 0.40;
        }
        if (det.thresh - 0.3).abs() < 1e-4 {
            det.thresh = 0.20;
        }
        if (det.unclip_ratio - 1.5).abs() < 1e-4 || (det.unclip_ratio - 2.0).abs() < 1e-4 {
            det.unclip_ratio = 1.7;
        }
    } else if min_dim < 600 {
        if det.limit_type == "max" {
            det.limit_type = "min".to_string();
        }
    }

    if profile.aspect_ratio > 2.5 || profile.aspect_ratio < 0.4 {
        det.limit_type = "max".to_string();
    }

    (det, opt, calib)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_image_detects_illumination_gradient() {
        // Left half dark, right half bright
        let mut img = image::RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let luma = if x < 50 { 50 } else { 200 };
                img.put_pixel(x, y, image::Rgb([luma, luma, luma]));
            }
        }
        let mat = Mat::new(image::DynamicImage::ImageRgb8(img));
        let profile = profile_image(&mat);
        assert!(profile.illumination_variance > 45.0, "Illumination variance was {}", profile.illumination_variance);
    }

    #[test]
    fn test_profile_image_detects_carved_relief() {
        // High luma (~230), very low variance (~5)
        let mut img = image::RgbImage::from_pixel(100, 100, image::Rgb([230, 230, 230]));
        for y in 40..60 {
            for x in 40..60 {
                img.put_pixel(x, y, image::Rgb([220, 220, 220]));
            }
        }
        let mat = Mat::new(image::DynamicImage::ImageRgb8(img));
        let profile = profile_image(&mat);
        assert!(profile.mean_luma > 185.0);
        assert!(profile.std_luma < 25.0);

        let det = DetConfig::default(std::path::PathBuf::new());
        let (_, _, calib) = resolve_auto_params(&profile, &det, None, None);
        assert_eq!(calib.relief_enhancement, Some(true));
        assert!(calib.relief_strength.unwrap() >= 0.8);
    }

    #[test]
    fn test_resolve_auto_params_preserves_explicit_user_options() {
        let profile = ImageProfile {
            width: 3000,
            height: 2000,
            aspect_ratio: 1.5,
            mean_luma: 230.0,
            std_luma: 10.0,
            illumination_variance: 50.0,
            bg_min: 160.0,
            bg_max: 240.0,
            bg_var: 80.0,
            bg_chroma_ratio: 0.20,
            gradient_energy: 5.0,
        };
        let det = DetConfig::default(std::path::PathBuf::new());
        let explicit_calib = CalibrationOptions {
            relief_enhancement: Some(false), // User explicitly disabled relief
            relief_strength: Some(1.75),
            ..Default::default()
        };
        let (_, _, calib) = resolve_auto_params(&profile, &det, None, Some(&explicit_calib));
        assert_eq!(calib.relief_enhancement, Some(false), "Explicit user setting was overridden!");
        assert_eq!(calib.relief_strength, Some(1.75), "Explicit user setting was overridden!");
        assert_eq!(calib.flatten_illumination, Some(true), "Unset field was not auto-configured!");
    }

    #[test]
    fn test_clean_digital_archetype_strictly_disables_calibration() {
        let profile = ImageProfile {
            width: 1700,
            height: 2200,
            aspect_ratio: 0.77,
            mean_luma: 248.5,
            std_luma: 35.0,
            illumination_variance: 25.0,
            bg_min: 255.0,
            bg_max: 255.0,
            bg_var: 0.0,
            bg_chroma_ratio: 0.0,
            gradient_energy: 5.5,
        };
        assert_eq!(DocumentArchetype::classify(&profile), DocumentArchetype::CleanDigital);

        let det = DetConfig::default(std::path::PathBuf::new());
        let (_, _, calib) = resolve_auto_params(&profile, &det, None, None);
        assert_eq!(calib.enabled, Some(false));
        assert_eq!(calib.stroke_repair, None);
        assert_eq!(calib.sharpen, None);
    }

    #[test]
    fn test_patterned_id_archetype_triggers_descreen() {
        let profile = ImageProfile {
            width: 1024,
            height: 635,
            aspect_ratio: 1.61,
            mean_luma: 145.0,
            std_luma: 37.0,
            illumination_variance: 30.0,
            bg_min: 168.0,
            bg_max: 184.0,
            bg_var: 16.0,
            bg_chroma_ratio: 0.95,
            gradient_energy: 22.0,
        };
        assert_eq!(DocumentArchetype::classify(&profile), DocumentArchetype::PatternedID);

        let det = DetConfig::default(std::path::PathBuf::new());
        let (_, _, calib) = resolve_auto_params(&profile, &det, None, None);
        assert_eq!(calib.enabled, Some(true));
        assert_eq!(calib.descreen_background, Some(true));
    }
}

