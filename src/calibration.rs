use serde::{Deserialize, Serialize};

#[cfg(feature = "use-opencv")]
use opencv::core::{Mat, Point2f};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Point2f};

use crate::engine::EngineError;

/// Configuration for pre-OCR image calibration.
///
/// Designed to improve detection and character recognition on challenging documents
/// (e.g. ID cards, banknotes, forms with patterned/colored security guilloche lines).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CalibrationOptions {
    /// Enable or disable pre-OCR image calibration.
    pub enabled: Option<bool>,

    /// Attenuate colored security patterns (e.g. cyan/blue guilloche lines).
    pub descreen_background: Option<bool>,

    /// Strength multiplier for background descreening (default: 1.15).
    pub descreen_strength: Option<f32>,

    /// Flatten uneven illumination, shadows, and flash glare.
    pub flatten_illumination: Option<bool>,

    /// Dynamic range normalization and contrast stretching (auto-levels).
    pub auto_levels: Option<bool>,

    /// Thresholded edge sharpening targeting character strokes while suppressing noise.
    pub sharpen: Option<bool>,

    /// Spectral red-channel contrast conversion (maximizes contrast on cyan/blue documents).
    pub red_channel_contrast: Option<bool>,
}

impl CalibrationOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preconfigured calibration profile for ID cards (e.g. Indonesian KTP).
    pub fn for_id_card() -> Self {
        Self {
            enabled: Some(true),
            descreen_background: Some(true),
            descreen_strength: Some(1.15),
            flatten_illumination: Some(false),
            auto_levels: Some(false),
            sharpen: Some(false),
            red_channel_contrast: Some(false),
        }
    }

    /// Preconfigured composite calibration pipeline (Descreening + Flattening + Sharpening).
    pub fn composite() -> Self {
        Self {
            enabled: Some(true),
            descreen_background: Some(true),
            descreen_strength: Some(1.15),
            flatten_illumination: Some(true),
            auto_levels: Some(false),
            sharpen: Some(true),
            red_channel_contrast: Some(false),
        }
    }

    pub fn with_descreen(mut self, enabled: bool) -> Self {
        self.descreen_background = Some(enabled);
        self
    }

    pub fn with_descreen_strength(mut self, strength: f32) -> Self {
        self.descreen_strength = Some(strength);
        self
    }

    pub fn with_flatten_illumination(mut self, enabled: bool) -> Self {
        self.flatten_illumination = Some(enabled);
        self
    }

    pub fn with_auto_levels(mut self, enabled: bool) -> Self {
        self.auto_levels = Some(enabled);
        self
    }

    pub fn with_sharpen(mut self, enabled: bool) -> Self {
        self.sharpen = Some(enabled);
        self
    }

    pub fn with_red_channel(mut self, enabled: bool) -> Self {
        self.red_channel_contrast = Some(enabled);
        self
    }
}

/// Configuration for runtime image optimization.
///
/// Provides resolution scaling and recognition crop padding.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OptimizationOptions {
    /// Enable or disable image optimization.
    pub enabled: Option<bool>,

    /// Optimal target maximum side dimension for downscaling (e.g. 960 or 1024).
    /// Scaling down high-resolution inputs to model receptive fields improves speed and accuracy.
    pub target_max_side: Option<u32>,

    /// Horizontal crop padding expansion ratio applied to bounding boxes (e.g. 0.06 = 6%).
    /// Prevents outer characters from being truncated at detection box boundaries.
    pub crop_padding_x: Option<f32>,

    /// Vertical crop padding expansion ratio applied to bounding boxes (e.g. 0.08 = 8%).
    pub crop_padding_y: Option<f32>,
}

impl OptimizationOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preconfigured optimization settings for ID card and mobile document OCR.
    pub fn for_id_card() -> Self {
        Self {
            enabled: Some(true),
            target_max_side: Some(800),
            crop_padding_x: Some(0.0),
            crop_padding_y: Some(0.02),
        }
    }

    pub fn with_target_max_side(mut self, max_side: u32) -> Self {
        self.target_max_side = Some(max_side);
        self
    }

    pub fn with_crop_padding(mut self, pad_x: f32, pad_y: f32) -> Self {
        self.crop_padding_x = Some(pad_x);
        self.crop_padding_y = Some(pad_y);
        self
    }
}

/// Expands a 4-point bounding box polygon outward along its principal axes.
pub fn expand_box_padding(
    b: &[Point2f; 4],
    pad_x: f32,
    pad_y: f32,
    img_w: f32,
    img_h: f32,
) -> [Point2f; 4] {
    let mut out = *b;

    // Width vector b[0] -> b[1]
    let vx = b[1].x - b[0].x;
    let vy = b[1].y - b[0].y;
    let w = vx.hypot(vy);

    // Height vector b[0] -> b[3]
    let hx = b[3].x - b[0].x;
    let hy = b[3].y - b[0].y;
    let h = hx.hypot(hy);

    if w < 1e-4 || h < 1e-4 {
        return out;
    }

    // Normalized directional unit vectors
    let u_wx = vx / w;
    let u_wy = vy / w;
    let u_hx = hx / h;
    let u_hy = hy / h;

    let dx = w * pad_x;
    let dy = h * pad_y;

    let off_x0 = -dx * u_wx - dy * u_hx;
    let off_y0 = -dx * u_wy - dy * u_hy;

    let off_x1 = dx * u_wx - dy * u_hx;
    let off_y1 = dx * u_wy - dy * u_hy;

    let off_x2 = dx * u_wx + dy * u_hx;
    let off_y2 = dx * u_wy + dy * u_hy;

    let off_x3 = -dx * u_wx + dy * u_hx;
    let off_y3 = -dx * u_wy + dy * u_hy;

    out[0].x = (b[0].x + off_x0).clamp(0.0, img_w);
    out[0].y = (b[0].y + off_y0).clamp(0.0, img_h);

    out[1].x = (b[1].x + off_x1).clamp(0.0, img_w);
    out[1].y = (b[1].y + off_y1).clamp(0.0, img_h);

    out[2].x = (b[2].x + off_x2).clamp(0.0, img_w);
    out[2].y = (b[2].y + off_y2).clamp(0.0, img_h);

    out[3].x = (b[3].x + off_x3).clamp(0.0, img_w);
    out[3].y = (b[3].y + off_y3).clamp(0.0, img_h);

    out
}

#[cfg(not(feature = "use-opencv"))]
pub fn apply_calibration(mat: &Mat, options: &CalibrationOptions) -> Result<Mat, EngineError> {
    if !options.enabled.unwrap_or(true) {
        return Ok(mat.clone());
    }

    let mut img = mat.to_rgb8();

    if options.descreen_background.unwrap_or(false) {
        let strength = options.descreen_strength.unwrap_or(1.15);
        descreen_background(&mut img, strength);
    }

    if options.flatten_illumination.unwrap_or(false) {
        img = flatten_illumination(&img);
    }

    if options.auto_levels.unwrap_or(false) {
        img = auto_levels(&img, 1.0);
    }

    if options.sharpen.unwrap_or(false) {
        img = thresholded_sharpen(&img, 1.0, 6, 0.75);
    }

    if options.red_channel_contrast.unwrap_or(false) {
        img = red_channel_contrast(&img);
    }

    Ok(Mat::new(image::DynamicImage::ImageRgb8(img)))
}

#[cfg(feature = "use-opencv")]
pub fn apply_calibration(mat: &Mat, _options: &CalibrationOptions) -> Result<Mat, EngineError> {
    // Pass-through when OpenCV backend is configured
    Ok(mat.clone())
}

#[cfg(not(feature = "use-opencv"))]
fn descreen_background(img: &mut image::RgbImage, strength: f32) {
    for p in img.pixels_mut() {
        let r = p[0] as f32;
        let g = p[1] as f32;
        let b = p[2] as f32;

        let bg_bias = ((b + g) * 0.5 - r).max(0.0);
        let lift = (bg_bias * strength).clamp(0.0, 160.0);

        let nr = (r + lift * 1.25).clamp(0.0, 255.0) as u8;
        let ng = (g + lift * 0.85).clamp(0.0, 255.0) as u8;
        let nb = (b + lift * 0.50).clamp(0.0, 255.0) as u8;

        p[0] = nr;
        p[1] = ng;
        p[2] = nb;
    }
}

#[cfg(not(feature = "use-opencv"))]
fn flatten_illumination(img: &image::RgbImage) -> image::RgbImage {
    let small_w = (img.width() / 8).max(1);
    let small_h = (img.height() / 8).max(1);
    let small = image::imageops::resize(img, small_w, small_h, image::imageops::FilterType::Nearest);
    let small_blurred = image::imageops::blur(&small, 3.0);
    let bg_estimate = image::imageops::resize(&small_blurred, img.width(), img.height(), image::imageops::FilterType::Triangle);

    let mut out = image::RgbImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let bp = bg_estimate.get_pixel(x, y);
        let mut new_p = [0u8; 3];
        for c in 0..3 {
            let bg = bp[c].max(1) as f32;
            let fg = p[c] as f32;
            let normalized = (fg / bg) * 230.0;
            new_p[c] = normalized.clamp(0.0, 255.0) as u8;
        }
        out.put_pixel(x, y, image::Rgb(new_p));
    }
    out
}

#[cfg(not(feature = "use-opencv"))]
fn auto_levels(img: &image::RgbImage, clip_percent: f32) -> image::RgbImage {
    let mut hist = [0u32; 256];
    let total_pixels = img.width() * img.height();

    for p in img.pixels() {
        let luma = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as usize;
        hist[luma] += 1;
    }

    let clip_count = (total_pixels as f32 * clip_percent / 100.0) as u32;
    let mut accum = 0u32;
    let mut min_val = 0u8;
    for (i, count) in hist.iter().enumerate() {
        accum += count;
        if accum >= clip_count {
            min_val = i as u8;
            break;
        }
    }

    accum = 0;
    let mut max_val = 255u8;
    for (i, count) in hist.iter().enumerate().rev() {
        accum += count;
        if accum >= clip_count {
            max_val = i as u8;
            break;
        }
    }

    if max_val <= min_val {
        return img.clone();
    }

    let range = (max_val - min_val) as f32;
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = (((i.clamp(min_val as usize, max_val as usize) as f32 - min_val as f32) / range) * 255.0).clamp(0.0, 255.0) as u8;
    }

    let mut out = image::RgbImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        out.put_pixel(x, y, image::Rgb([lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]]));
    }
    out
}

#[cfg(not(feature = "use-opencv"))]
fn thresholded_sharpen(img: &image::RgbImage, sigma: f32, threshold: i32, amount: f32) -> image::RgbImage {
    let blurred = image::imageops::blur(img, sigma);
    let mut out = image::RgbImage::new(img.width(), img.height());

    for (x, y, p) in img.enumerate_pixels() {
        let bp = blurred.get_pixel(x, y);
        let mut new_p = [0u8; 3];
        for c in 0..3 {
            let diff = p[c] as i32 - bp[c] as i32;
            if diff.abs() >= threshold {
                let sharp = p[c] as f32 + diff as f32 * amount;
                new_p[c] = sharp.clamp(0.0, 255.0) as u8;
            } else {
                new_p[c] = p[c];
            }
        }
        out.put_pixel(x, y, image::Rgb(new_p));
    }
    out
}

#[cfg(not(feature = "use-opencv"))]
fn red_channel_contrast(img: &image::RgbImage) -> image::RgbImage {
    let mut out = image::RgbImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let v = (p[0] as f32 * 0.85 + p[1] as f32 * 0.15).clamp(0.0, 255.0) as u8;
        out.put_pixel(x, y, image::Rgb([v, v, v]));
    }
    out
}
