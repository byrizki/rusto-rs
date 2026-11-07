use nanoserde::{DeJson, SerJson};
use crate::error::CoreError;
use crate::image::{Mat, Point2f};

#[derive(Clone, Debug, Default, DeJson, SerJson, PartialEq)]
pub struct CalibrationOptions {
    pub enabled: Option<bool>,
    #[nserde(rename = "descreenBackground")]
    pub descreen_background: Option<bool>,
    #[nserde(rename = "descreenStrength")]
    pub descreen_strength: Option<f32>,
    #[nserde(rename = "flattenIllumination")]
    pub flatten_illumination: Option<bool>,
    #[nserde(rename = "autoLevels")]
    pub auto_levels: Option<bool>,
    pub sharpen: Option<bool>,
    #[nserde(rename = "redChannelContrast")]
    pub red_channel_contrast: Option<bool>,
}

impl CalibrationOptions {
    pub fn new() -> Self {
        Self::default()
    }

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

#[derive(Clone, Debug, Default, DeJson, SerJson, PartialEq)]
pub struct OptimizationOptions {
    pub enabled: Option<bool>,
    #[nserde(rename = "targetMaxSide")]
    pub target_max_side: Option<u32>,
    #[nserde(rename = "cropPaddingX")]
    pub crop_padding_x: Option<f32>,
    #[nserde(rename = "cropPaddingY")]
    pub crop_padding_y: Option<f32>,
}

impl OptimizationOptions {
    pub fn new() -> Self {
        Self::default()
    }

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

pub fn apply_calibration(mat: &Mat, options: &CalibrationOptions) -> Result<Mat, CoreError> {
    if !options.enabled.unwrap_or(false) {
        return Ok(mat.clone());
    }

    let mut current = mat.to_rgb8();

    if options.descreen_background.unwrap_or(false) {
        let strength = options.descreen_strength.unwrap_or(1.15);
        descreen_background(&mut current, strength);
    }

    if options.flatten_illumination.unwrap_or(false) {
        current = flatten_illumination(&current);
    }

    if options.auto_levels.unwrap_or(false) {
        current = auto_levels(&current, 0.5);
    }

    if options.sharpen.unwrap_or(false) {
        current = thresholded_sharpen(&current, 1.0, 4, 1.2);
    }

    if options.red_channel_contrast.unwrap_or(false) {
        current = red_channel_contrast(&current);
    }

    Ok(Mat::new(image::DynamicImage::ImageRgb8(current)))
}

pub fn descreen_background(img: &mut image::RgbImage, strength: f32) {
    if strength <= 0.0 {
        return;
    }

    for p in img.pixels_mut() {
        let r = p[0] as f32;
        let g = p[1] as f32;
        let b = p[2] as f32;

        let lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if lum < 70.0 {
            continue;
        }

        let max_c = r.max(g.max(b));
        let min_c = r.min(g.min(b));
        let chroma = max_c - min_c;
        if chroma < 4.0 {
            continue;
        }

        let weight = ((lum - 70.0) / 50.0).clamp(0.0, 1.0);
        let factor = (strength * weight).clamp(0.0, 2.0);

        let r_lift = (max_c - r) * factor;
        let g_lift = (max_c - g) * factor;
        let b_lift = (max_c - b) * factor;

        let white_lift = (255.0 - max_c) * (chroma / 255.0) * (factor * 0.25);

        p[0] = (r + r_lift + white_lift).clamp(0.0, 255.0) as u8;
        p[1] = (g + g_lift + white_lift).clamp(0.0, 255.0) as u8;
        p[2] = (b + b_lift + white_lift).clamp(0.0, 255.0) as u8;
    }
}

pub fn descreen_background_rgba(rgba: &mut [u8], strength: f32) {
    if strength <= 0.0 {
        return;
    }

    for chunk in rgba.chunks_exact_mut(4) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;

        let lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if lum < 70.0 {
            continue;
        }

        let max_c = r.max(g.max(b));
        let min_c = r.min(g.min(b));
        let chroma = max_c - min_c;
        if chroma < 4.0 {
            continue;
        }

        let weight = ((lum - 70.0) / 50.0).clamp(0.0, 1.0);
        let factor = (strength * weight).clamp(0.0, 2.0);

        let r_lift = (max_c - r) * factor;
        let g_lift = (max_c - g) * factor;
        let b_lift = (max_c - b) * factor;

        let white_lift = (255.0 - max_c) * (chroma / 255.0) * (factor * 0.25);

        chunk[0] = (r + r_lift + white_lift).clamp(0.0, 255.0) as u8;
        chunk[1] = (g + g_lift + white_lift).clamp(0.0, 255.0) as u8;
        chunk[2] = (b + b_lift + white_lift).clamp(0.0, 255.0) as u8;
    }
}

pub fn flatten_illumination(img: &image::RgbImage) -> image::RgbImage {
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

pub fn auto_levels(img: &image::RgbImage, clip_percent: f32) -> image::RgbImage {
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

pub fn thresholded_sharpen(img: &image::RgbImage, sigma: f32, threshold: i32, amount: f32) -> image::RgbImage {
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

pub fn red_channel_contrast(img: &image::RgbImage) -> image::RgbImage {
    let mut out = image::RgbImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let v = (p[0] as f32 * 0.85 + p[1] as f32 * 0.15).clamp(0.0, 255.0) as u8;
        out.put_pixel(x, y, image::Rgb([v, v, v]));
    }
    out
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


