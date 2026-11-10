use crate::tensor::{Array2, Array3, Array4};
use crate::error::CoreError;
use crate::image::{self, Mat, Size, INTER_LINEAR};

#[inline]
fn build_bilinear_table(
    out_len: usize,
    start: f32,
    step: f32,
    src_len: usize,
    elem_stride: usize,
) -> Vec<(usize, usize, f32, f32)> {
    let max_coord = (src_len.saturating_sub(1)) as f32;
    let max_idx = src_len.saturating_sub(1);
    let mut tab = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let s = (start + (i as f32 + 0.5) * step - 0.5).clamp(0.0, max_coord);
        let i0 = s.floor() as usize;
        let i1 = (i0 + 1).min(max_idx);
        let f1 = s - i0 as f32;
        tab.push((i0 * elem_stride, i1 * elem_stride, 1.0 - f1, f1));
    }
    tab
}

pub struct DetPreProcess {
    pub limit_side_len: i32,
    pub limit_type: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl DetPreProcess {
    pub fn new(limit_side_len: i32, limit_type: String, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            limit_side_len,
            limit_type,
            mean,
            std,
        }
    }

    pub fn run(&self, img: &Mat) -> Result<Array4<f32>, CoreError> {
        let resized = self.resize(img)?;
        self.normalize_and_permute(&resized)
    }

    pub fn resize(&self, img: &Mat) -> Result<Mat, CoreError> {
        let h = img.rows();
        let w = img.cols();

        let ratio = if self.limit_type == "max" {
            let max_side = h.max(w) as f32;
            if max_side > self.limit_side_len as f32 {
                self.limit_side_len as f32 / max_side
            } else {
                1.0
            }
        } else {
            let min_side = h.min(w) as f32;
            if min_side < self.limit_side_len as f32 {
                self.limit_side_len as f32 / min_side
            } else {
                1.0
            }
        };

        let mut resize_h = (h as f32 * ratio) as i32;
        let mut resize_w = (w as f32 * ratio) as i32;

        resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
        resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

        if resize_h <= 0 || resize_w <= 0 {
            return Err(CoreError::Preprocess("resize_h or resize_w <= 0".to_string()));
        }

        let mut dst = Mat::default();
        image::resize(
            img,
            &mut dst,
            Size::new(resize_w, resize_h),
            INTER_LINEAR,
        )?;

        Ok(dst)
    }

    pub fn normalize_and_permute(&self, img: &Mat) -> Result<Array4<f32>, CoreError> {
        let size = img.size()?;
        let h = size.height as usize;
        let w = size.width as usize;

        let mut out = Array4::<f32>::zeros((1, 3, h, w));
        let scale_b = 1.0 / (255.0 * self.std[0]);
        let bias_b = -self.mean[0] / self.std[0];
        let scale_g = 1.0 / (255.0 * self.std[1]);
        let bias_g = -self.mean[1] / self.std[1];
        let scale_r = 1.0 / (255.0 * self.std[2]);
        let bias_r = -self.mean[2] / self.std[2];

        if let Some(rgb) = img.as_dynamic().as_rgb8() {
            let raw = rgb.as_raw();
            let plane_size = h * w;
            let out_slice = out.as_mut_slice();
            let (b_plane, rest) = out_slice.split_at_mut(plane_size);
            let (g_plane, r_plane) = rest.split_at_mut(plane_size);
            for i in 0..plane_size {
                let px = i * 3;
                r_plane[i] = raw[px] as f32 * scale_r + bias_r;
                g_plane[i] = raw[px + 1] as f32 * scale_g + bias_g;
                b_plane[i] = raw[px + 2] as f32 * scale_b + bias_b;
            }
            return Ok(out);
        }

        for y in 0..h {
            for x in 0..w {
                let pix = img.get_pixel(x as u32, y as u32);
                out[[0, 0, y, x]] = pix[2] as f32 * scale_b + bias_b;
                out[[0, 1, y, x]] = pix[1] as f32 * scale_g + bias_g;
                out[[0, 2, y, x]] = pix[0] as f32 * scale_r + bias_r;
            }
        }

        Ok(out)
    }

    pub fn preprocess_rgba_raw(
        rgba: &[u8],
        orig_w: usize,
        orig_h: usize,
        limit_side_len: usize,
        limit_type: &str,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> (Vec<f32>, usize, usize, f32, f32) {
        let ratio = if limit_type == "max" {
            (limit_side_len as f32 / (orig_w.max(orig_h) as f32)).min(1.0)
        } else {
            let min_side = orig_w.min(orig_h) as f32;
            if min_side < limit_side_len as f32 {
                limit_side_len as f32 / min_side
            } else {
                1.0
            }
        };

        let target_w = ((orig_w as f32 * ratio / 32.0).round() as usize * 32).max(32);
        let target_h = ((orig_h as f32 * ratio / 32.0).round() as usize * 32).max(32);
        let scale_w = target_w as f32 / orig_w as f32;
        let scale_h = target_h as f32 / orig_h as f32;

        let plane_size = target_w * target_h;
        let mut det_input = vec![0.0f32; plane_size * 3];
        if orig_w == 0 || orig_h == 0 || rgba.len() < orig_w * orig_h * 4 {
            return (det_input, target_w, target_h, scale_w, scale_h);
        }

        let x_tab = build_bilinear_table(target_w, 0.0, 1.0 / scale_w, orig_w, 4);
        let y_tab = build_bilinear_table(target_h, 0.0, 1.0 / scale_h, orig_h, orig_w * 4);

        let scale_b = 1.0 / (255.0 * std[0]);
        let bias_b = -mean[0] / std[0];
        let scale_g = 1.0 / (255.0 * std[1]);
        let bias_g = -mean[1] / std[1];
        let scale_r = 1.0 / (255.0 * std[2]);
        let bias_r = -mean[2] / std[2];

        let row_bytes = orig_w * 4;
        let (b_plane, rest) = det_input.split_at_mut(plane_size);
        let (g_plane, r_plane) = rest.split_at_mut(plane_size);

        for (ty, &(ry0, ry1, wy0, wy1)) in y_tab.iter().enumerate() {
            let row0 = &rgba[ry0..ry0 + row_bytes];
            let row1 = &rgba[ry1..ry1 + row_bytes];
            let dst_off = ty * target_w;
            let b_row = &mut b_plane[dst_off..dst_off + target_w];
            let g_row = &mut g_plane[dst_off..dst_off + target_w];
            let r_row = &mut r_plane[dst_off..dst_off + target_w];

            for (tx, &(ox0, ox1, wx0, wx1)) in x_tab.iter().enumerate() {
                let r = (row0[ox0] as f32 * wx0 + row0[ox1] as f32 * wx1) * wy0
                    + (row1[ox0] as f32 * wx0 + row1[ox1] as f32 * wx1) * wy1;
                let g = (row0[ox0 + 1] as f32 * wx0 + row0[ox1 + 1] as f32 * wx1) * wy0
                    + (row1[ox0 + 1] as f32 * wx0 + row1[ox1 + 1] as f32 * wx1) * wy1;
                let b = (row0[ox0 + 2] as f32 * wx0 + row0[ox1 + 2] as f32 * wx1) * wy0
                    + (row1[ox0 + 2] as f32 * wx0 + row1[ox1 + 2] as f32 * wx1) * wy1;

                b_row[tx] = b * scale_b + bias_b;
                g_row[tx] = g * scale_g + bias_g;
                r_row[tx] = r * scale_r + bias_r;
            }
        }

        (det_input, target_w, target_h, scale_w, scale_h)
    }
}

pub fn preprocess_rgba_raw(
    rgba: &[u8],
    orig_w: usize,
    orig_h: usize,
    limit_side_len: usize,
    limit_type: &str,
    mean: [f32; 3],
    std: [f32; 3],
) -> (Vec<f32>, usize, usize, f32, f32) {
    DetPreProcess::preprocess_rgba_raw(rgba, orig_w, orig_h, limit_side_len, limit_type, mean, std)
}

pub fn calculate_linenet_dims(
    ori_w: i32,
    ori_h: i32,
    limit_side_len: i32,
    limit_type: &str,
) -> (usize, usize, f32, f32) {
    let scale = if limit_type == "max" {
        (limit_side_len as f32 / (ori_h.max(ori_w) as f32)).min(1.0)
    } else {
        1.0
    };

    let nh = ((ori_h as f32 * scale).round() as usize).max(1);
    let nw = ((ori_w as f32 * scale).round() as usize).max(1);

    let ph = ((nh + 31) / 32) * 32;
    let pw = ((nw + 31) / 32) * 32;

    (pw, ph, scale, scale)
}

pub fn prepare_linenet_det_input(
    img: &Mat,
    target_w: usize,
    target_h: usize,
) -> Result<Array4<f32>, CoreError> {
    let orig_w_u = img.cols().max(0) as usize;
    let orig_h_u = img.rows().max(0) as usize;
    let mut input = Array4::<f32>::zeros((1, 1, target_h, target_w));
    if orig_w_u == 0 || orig_h_u == 0 || target_w == 0 || target_h == 0 {
        return Ok(input);
    }

    let orig_w = orig_w_u as f32;
    let orig_h = orig_h_u as f32;
    let scale = (target_w as f32 / orig_w).min(target_h as f32 / orig_h).min(1.0);
    let nw = ((orig_w * scale).round() as usize).clamp(1, target_w);
    let nh = ((orig_h * scale).round() as usize).clamp(1, target_h);

    if let Some(rgb) = img.as_dynamic().as_rgb8() {
        let raw = rgb.as_raw();
        let row_bytes = orig_w_u * 3;
        let x_tab = build_bilinear_table(nw, 0.0, 1.0 / scale, orig_w_u, 3);
        let y_tab = build_bilinear_table(nh, 0.0, 1.0 / scale, orig_h_u, row_bytes);
        let out_slice = input.as_mut_slice();

        const WR: f32 = 0.299 / 255.0;
        const WG: f32 = 0.587 / 255.0;
        const WB: f32 = 0.114 / 255.0;

        for (ty, &(ry0, ry1, wy0, wy1)) in y_tab.iter().enumerate() {
            let row0 = &raw[ry0..ry0 + row_bytes];
            let row1 = &raw[ry1..ry1 + row_bytes];
            let dst_row = &mut out_slice[ty * target_w..(ty * target_w + nw)];
            for (tx, &(ox0, ox1, wx0, wx1)) in x_tab.iter().enumerate() {
                let l00 = row0[ox0] as f32 * WR + row0[ox0 + 1] as f32 * WG + row0[ox0 + 2] as f32 * WB;
                let l10 = row0[ox1] as f32 * WR + row0[ox1 + 1] as f32 * WG + row0[ox1 + 2] as f32 * WB;
                let l01 = row1[ox0] as f32 * WR + row1[ox0 + 1] as f32 * WG + row1[ox0 + 2] as f32 * WB;
                let l11 = row1[ox1] as f32 * WR + row1[ox1 + 1] as f32 * WG + row1[ox1 + 2] as f32 * WB;
                let gray_norm = (l00 * wx0 + l10 * wx1) * wy0 + (l01 * wx0 + l11 * wx1) * wy1;
                dst_row[tx] = (1.0 - gray_norm).clamp(0.0, 1.0);
            }
        }
        return Ok(input);
    }

    for ty in 0..nh {
        let sy = ((ty as f32 + 0.5) / scale - 0.5).clamp(0.0, orig_h - 1.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(img.rows() as u32 - 1);
        let fy = sy - y0 as f32;

        for tx in 0..nw {
            let sx = ((tx as f32 + 0.5) / scale - 0.5).clamp(0.0, orig_w - 1.0);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(img.cols() as u32 - 1);
            let fx = sx - x0 as f32;

            let p00 = img.get_pixel(x0, y0);
            let p10 = img.get_pixel(x1, y0);
            let p01 = img.get_pixel(x0, y1);
            let p11 = img.get_pixel(x1, y1);

            let r = (p00[0] as f32 * (1.0 - fx) + p10[0] as f32 * fx) * (1.0 - fy)
                + (p01[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
            let g = (p00[1] as f32 * (1.0 - fx) + p10[1] as f32 * fx) * (1.0 - fy)
                + (p01[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
            let b = (p00[2] as f32 * (1.0 - fx) + p10[2] as f32 * fx) * (1.0 - fy)
                + (p01[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
            input[[0, 0, ty, tx]] = ink;
        }
    }

    Ok(input)
}


/// Computes target dimensions (rec_w, rec_h) for LineNet recognition crop without pixel allocation.
pub fn compute_linenet_rec_crop_dims(
    orig_w: usize,
    orig_h: usize,
    box_coords: [f32; 4],
) -> Option<(usize, usize)> {
    let [x0, y0, x1, y1] = box_coords;
    let box_h = y1 - y0;

    let exp_x0 = (x0 - 0.5 * box_h).max(0.0);
    let exp_x1 = (x1 + 0.5 * box_h).min(orig_w as f32);
    let exp_y0 = (y0 - 1.6 * box_h).max(0.0);
    let exp_y1 = (y1 + 1.6 * box_h).min(orig_h as f32);
    let crop_w = exp_x1 - exp_x0;
    let crop_h = exp_y1 - exp_y0;
    if crop_w < 4.0 || crop_h < 4.0 || orig_w == 0 || orig_h == 0 {
        return None;
    }

    let rec_h = 96usize;
    let target_w = (96.0 * (crop_w / crop_h)).round() as usize;
    let rec_w = ((target_w + 7) / 8 * 8).clamp(8, 4096);
    Some((rec_w, rec_h))
}

/// Computes target dimensions (rec_w, rec_h) for PP-OCR recognition crop without pixel allocation.
pub fn compute_ppocr_rec_crop_dims(
    orig_w: usize,
    orig_h: usize,
    box_coords: [f32; 4],
) -> Option<(usize, usize)> {
    let [x0, y0, x1, y1] = box_coords;
    let box_w = x1 - x0;
    let box_h = y1 - y0;

    let pad_x_ratio = 0.035f32;
    let pad_y_ratio = 0.045f32;

    let exp_x0 = (x0 - box_w * pad_x_ratio).max(0.0);
    let exp_x1 = (x1 + box_w * pad_x_ratio).min(orig_w as f32);
    let exp_y0 = (y0 - box_h * pad_y_ratio).max(0.0);
    let exp_y1 = (y1 + box_h * pad_y_ratio).min(orig_h as f32);

    let crop_w = exp_x1 - exp_x0;
    let crop_h = exp_y1 - exp_y0;
    if crop_w < 4.0 || crop_h < 4.0 || orig_w == 0 || orig_h == 0 {
        return None;
    }

    let rec_h = 48usize;
    let target_w = (48.0 * (crop_w / crop_h)).ceil() as usize;
    let rec_w = ((target_w as f32 / 32.0).ceil() as usize * 32).clamp(64, 960);
    Some((rec_w, rec_h))
}

/// Crop and prepare LineNet recognition crop from raw RGBA buffer
pub fn prepare_linenet_rec_crop_rgba(
    rgba: &[u8],
    orig_w: usize,
    orig_h: usize,
    box_coords: [f32; 4],
) -> Option<(Vec<f32>, usize, usize)> {
    let (rec_w, rec_h) = compute_linenet_rec_crop_dims(orig_w, orig_h, box_coords)?;
    if rgba.len() < orig_w * orig_h * 4 {
        return None;
    }

    let [x0, y0, x1, y1] = box_coords;
    let box_h = y1 - y0;
    let exp_x0 = (x0 - 0.5 * box_h).max(0.0);
    let exp_x1 = (x1 + 0.5 * box_h).min(orig_w as f32);
    let exp_y0 = (y0 - 1.6 * box_h).max(0.0);
    let exp_y1 = (y1 + 1.6 * box_h).min(orig_h as f32);
    let crop_w = exp_x1 - exp_x0;
    let crop_h = exp_y1 - exp_y0;

    let x_tab = build_bilinear_table(rec_w, exp_x0, crop_w / rec_w as f32, orig_w, 4);
    let y_tab = build_bilinear_table(rec_h, exp_y0, crop_h / rec_h as f32, orig_h, orig_w * 4);

    const WR: f32 = 0.299 / 255.0;
    const WG: f32 = 0.587 / 255.0;
    const WB: f32 = 0.114 / 255.0;

    let row_bytes = orig_w * 4;
    let mut rec_input = vec![0.0f32; rec_h * rec_w];
    for (ry, &(ry0, ry1, wy0, wy1)) in y_tab.iter().enumerate() {
        let row0 = &rgba[ry0..ry0 + row_bytes];
        let row1 = &rgba[ry1..ry1 + row_bytes];
        let dst_row = &mut rec_input[ry * rec_w..(ry + 1) * rec_w];

        for (rx, &(ox0, ox1, wx0, wx1)) in x_tab.iter().enumerate() {
            let l00 = row0[ox0] as f32 * WR + row0[ox0 + 1] as f32 * WG + row0[ox0 + 2] as f32 * WB;
            let l10 = row0[ox1] as f32 * WR + row0[ox1 + 1] as f32 * WG + row0[ox1 + 2] as f32 * WB;
            let l01 = row1[ox0] as f32 * WR + row1[ox0 + 1] as f32 * WG + row1[ox0 + 2] as f32 * WB;
            let l11 = row1[ox1] as f32 * WR + row1[ox1 + 1] as f32 * WG + row1[ox1 + 2] as f32 * WB;
            let gray_norm = (l00 * wx0 + l10 * wx1) * wy0 + (l01 * wx0 + l11 * wx1) * wy1;
            dst_row[rx] = (1.0 - gray_norm).clamp(0.0, 1.0);
        }
    }

    Some((rec_input, rec_w, rec_h))
}

/// Crop and prepare PP-OCR SVTR recognition crop from raw RGBA buffer
pub fn prepare_ppocr_rec_crop_rgba(
    rgba: &[u8],
    orig_w: usize,
    orig_h: usize,
    box_coords: [f32; 4],
) -> Option<(Vec<f32>, usize, usize)> {
    let (rec_w, rec_h) = compute_ppocr_rec_crop_dims(orig_w, orig_h, box_coords)?;
    if rgba.len() < orig_w * orig_h * 4 {
        return None;
    }

    let [x0, y0, x1, y1] = box_coords;
    let box_w = x1 - x0;
    let box_h = y1 - y0;

    let pad_x_ratio = 0.035f32;
    let pad_y_ratio = 0.045f32;

    let exp_x0 = (x0 - box_w * pad_x_ratio).max(0.0);
    let exp_x1 = (x1 + box_w * pad_x_ratio).min(orig_w as f32);
    let exp_y0 = (y0 - box_h * pad_y_ratio).max(0.0);
    let exp_y1 = (y1 + box_h * pad_y_ratio).min(orig_h as f32);

    let crop_w = exp_x1 - exp_x0;
    let crop_h = exp_y1 - exp_y0;

    let x_tab = build_bilinear_table(rec_w, exp_x0, crop_w / rec_w as f32, orig_w, 4);
    let y_tab = build_bilinear_table(rec_h, exp_y0, crop_h / rec_h as f32, orig_h, orig_w * 4);

    const SCALE: f32 = 2.0 / 255.0;
    let plane_size = rec_h * rec_w;
    let row_bytes = orig_w * 4;
    let mut rec_input = vec![0.0f32; plane_size * 3];
    let (b_plane, rest) = rec_input.split_at_mut(plane_size);
    let (g_plane, r_plane) = rest.split_at_mut(plane_size);

    for (ry, &(ry0, ry1, wy0, wy1)) in y_tab.iter().enumerate() {
        let row0 = &rgba[ry0..ry0 + row_bytes];
        let row1 = &rgba[ry1..ry1 + row_bytes];
        let dst_off = ry * rec_w;
        let b_row = &mut b_plane[dst_off..dst_off + rec_w];
        let g_row = &mut g_plane[dst_off..dst_off + rec_w];
        let r_row = &mut r_plane[dst_off..dst_off + rec_w];

        for (rx, &(ox0, ox1, wx0, wx1)) in x_tab.iter().enumerate() {
            let r = (row0[ox0] as f32 * wx0 + row0[ox1] as f32 * wx1) * wy0
                + (row1[ox0] as f32 * wx0 + row1[ox1] as f32 * wx1) * wy1;
            let g = (row0[ox0 + 1] as f32 * wx0 + row0[ox1 + 1] as f32 * wx1) * wy0
                + (row1[ox0 + 1] as f32 * wx0 + row1[ox1 + 1] as f32 * wx1) * wy1;
            let b = (row0[ox0 + 2] as f32 * wx0 + row0[ox1 + 2] as f32 * wx1) * wy0
                + (row1[ox0 + 2] as f32 * wx0 + row1[ox1 + 2] as f32 * wx1) * wy1;

            b_row[rx] = b * SCALE - 1.0;
            g_row[rx] = g * SCALE - 1.0;
            r_row[rx] = r * SCALE - 1.0;
        }
    }

    Some((rec_input, rec_w, rec_h))
}

pub fn prepare_linenet_rec_crop(
    img: &Mat,
    target_h: usize,
    target_w: usize,
) -> Result<Array2<f32>, CoreError> {
    let mut out = Array2::<f32>::zeros((target_h, target_w));
    let orig_w_u = img.cols().max(0) as usize;
    let orig_h_u = img.rows().max(0) as usize;
    if orig_w_u == 0 || orig_h_u == 0 || target_w == 0 || target_h == 0 {
        return Ok(out);
    }
    let orig_w = orig_w_u as f32;
    let orig_h = orig_h_u as f32;
    let scale_w = target_w as f32 / orig_w;
    let scale_h = target_h as f32 / orig_h;

    if orig_h_u > target_h {
        let rgb = img.to_rgb8();
        let ink: Vec<f32> = rgb
            .as_raw()
            .chunks_exact(3)
            .map(|p| 1.0 - (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0)
            .collect();
        let resized = crate::resample::area_resize_gray(&ink, orig_w_u, orig_h_u, target_w, target_h);
        for (dst, v) in out.as_mut_slice().iter_mut().zip(resized) {
            *dst = v.clamp(0.0, 1.0);
        }
        return Ok(out);
    }

    if let Some(rgb) = img.as_dynamic().as_rgb8() {
        let raw = rgb.as_raw();
        let row_bytes = orig_w_u * 3;
        let x_tab = build_bilinear_table(target_w, 0.0, 1.0 / scale_w, orig_w_u, 3);
        let y_tab = build_bilinear_table(target_h, 0.0, 1.0 / scale_h, orig_h_u, row_bytes);
        let out_slice = out.as_mut_slice();

        const WR: f32 = 0.299 / 255.0;
        const WG: f32 = 0.587 / 255.0;
        const WB: f32 = 0.114 / 255.0;

        for (ty, &(ry0, ry1, wy0, wy1)) in y_tab.iter().enumerate() {
            let row0 = &raw[ry0..ry0 + row_bytes];
            let row1 = &raw[ry1..ry1 + row_bytes];
            let dst_row = &mut out_slice[ty * target_w..(ty + 1) * target_w];
            for (tx, &(ox0, ox1, wx0, wx1)) in x_tab.iter().enumerate() {
                let l00 = row0[ox0] as f32 * WR + row0[ox0 + 1] as f32 * WG + row0[ox0 + 2] as f32 * WB;
                let l10 = row0[ox1] as f32 * WR + row0[ox1 + 1] as f32 * WG + row0[ox1 + 2] as f32 * WB;
                let l01 = row1[ox0] as f32 * WR + row1[ox0 + 1] as f32 * WG + row1[ox0 + 2] as f32 * WB;
                let l11 = row1[ox1] as f32 * WR + row1[ox1 + 1] as f32 * WG + row1[ox1 + 2] as f32 * WB;
                let gray_norm = (l00 * wx0 + l10 * wx1) * wy0 + (l01 * wx0 + l11 * wx1) * wy1;
                dst_row[tx] = (1.0 - gray_norm).clamp(0.0, 1.0);
            }
        }
        return Ok(out);
    }

    for ty in 0..target_h {
        let sy = ((ty as f32 + 0.5) / scale_h - 0.5).clamp(0.0, orig_h - 1.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(img.rows() as u32 - 1);
        let fy = sy - y0 as f32;

        for tx in 0..target_w {
            let sx = ((tx as f32 + 0.5) / scale_w - 0.5).clamp(0.0, orig_w - 1.0);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(img.cols() as u32 - 1);
            let fx = sx - x0 as f32;

            let p00 = img.get_pixel(x0, y0);
            let p10 = img.get_pixel(x1, y0);
            let p01 = img.get_pixel(x0, y1);
            let p11 = img.get_pixel(x1, y1);

            let r = (p00[0] as f32 * (1.0 - fx) + p10[0] as f32 * fx) * (1.0 - fy)
                + (p01[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
            let g = (p00[1] as f32 * (1.0 - fx) + p10[1] as f32 * fx) * (1.0 - fy)
                + (p01[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
            let b = (p00[2] as f32 * (1.0 - fx) + p10[2] as f32 * fx) * (1.0 - fy)
                + (p01[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
            out[[ty, tx]] = ink;
        }
    }

    Ok(out)
}

pub fn build_linenet_rec_batch(
    img_list: &[Mat],
    chunk: &[usize],
    target_widths: &[usize],
    rec_h: usize,
) -> Result<(Array4<f32>, Vec<i32>), CoreError> {
    let n = chunk.len();
    let max_w = chunk
        .iter()
        .map(|&idx| target_widths[idx])
        .max()
        .unwrap_or(rec_h)
        .div_ceil(8)
        * 8;
    let mut batch_tensor = Array4::<f32>::zeros((n, 1, rec_h, max_w));
    let mut widths = Vec::with_capacity(n);
    let batch_slice = batch_tensor.as_mut_slice();
    let item_len = rec_h * max_w;

    for (local_i, &idx) in chunk.iter().enumerate() {
        let w = target_widths[idx];
        widths.push(w as i32);
        let ink_crop = prepare_linenet_rec_crop(&img_list[idx], rec_h, w)?;
        let item_dest = &mut batch_slice[local_i * item_len..(local_i + 1) * item_len];
        let crop_slice = ink_crop.as_slice();
        for y in 0..rec_h {
            let src_row = &crop_slice[y * w..(y + 1) * w];
            let dst_row = &mut item_dest[y * max_w..y * max_w + w];
            dst_row.copy_from_slice(src_row);
        }
    }

    Ok((batch_tensor, widths))
}

/// Preprocesses raw RGBA bytes into a 1-channel inverted grayscale tensor input for LineNet detection.
pub fn preprocess_linenet_det_rgba(
    rgba: &[u8],
    orig_w: usize,
    orig_h: usize,
    limit: i32,
    limit_type: &str,
) -> (Vec<f32>, usize, usize, f32, f32) {
    let (target_w, target_h, scale_w, scale_h) =
        calculate_linenet_dims(orig_w as i32, orig_h as i32, limit, limit_type);

    let mut det_input = vec![0.0f32; target_w * target_h];
    if orig_w == 0 || orig_h == 0 || rgba.len() < orig_w * orig_h * 4 {
        return (det_input, target_w, target_h, scale_w, scale_h);
    }

    let scale = scale_w.min(scale_h);
    let nw = ((orig_w as f32 * scale).round() as usize).clamp(1, target_w);
    let nh = ((orig_h as f32 * scale).round() as usize).clamp(1, target_h);

    let row_bytes = orig_w * 4;
    let x_tab = build_bilinear_table(nw, 0.0, 1.0 / scale, orig_w, 4);
    let y_tab = build_bilinear_table(nh, 0.0, 1.0 / scale, orig_h, row_bytes);

    const WR: f32 = 0.299 / 255.0;
    const WG: f32 = 0.587 / 255.0;
    const WB: f32 = 0.114 / 255.0;

    for (ty, &(ry0, ry1, wy0, wy1)) in y_tab.iter().enumerate() {
        let row0 = &rgba[ry0..ry0 + row_bytes];
        let row1 = &rgba[ry1..ry1 + row_bytes];
        let dst_row = &mut det_input[ty * target_w..(ty * target_w + nw)];

        for (tx, &(ox0, ox1, wx0, wx1)) in x_tab.iter().enumerate() {
            let l00 = row0[ox0] as f32 * WR + row0[ox0 + 1] as f32 * WG + row0[ox0 + 2] as f32 * WB;
            let l10 = row0[ox1] as f32 * WR + row0[ox1 + 1] as f32 * WG + row0[ox1 + 2] as f32 * WB;
            let l01 = row1[ox0] as f32 * WR + row1[ox0 + 1] as f32 * WG + row1[ox0 + 2] as f32 * WB;
            let l11 = row1[ox1] as f32 * WR + row1[ox1 + 1] as f32 * WG + row1[ox1 + 2] as f32 * WB;
            let gray_norm = (l00 * wx0 + l10 * wx1) * wy0 + (l01 * wx0 + l11 * wx1) * wy1;
            dst_row[tx] = (1.0 - gray_norm).clamp(0.0, 1.0);
        }
    }

    (det_input, target_w, target_h, scale_w, scale_h)
}

static NORM_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        *item = (i as f32 / 255.0 - 0.5) / 0.5;
    }
    lut
});

/// Resizes and normalizes an image crop for PP-OCR recognition models directly into a target slice.
pub fn resize_norm_rec_img_to_slice(
    img: &Mat,
    img_c: usize,
    img_h: usize,
    target_w: usize,
    dest: &mut [f32],
) -> Result<(), CoreError> {
    let h = img.rows();
    let w = img.cols();
    if h <= 0 || w <= 0 {
        return Err(CoreError::Preprocess("invalid image size".to_string()));
    }

    let img_width = target_w as i32;
    let ratio = w as f32 / h as f32;
    let resized_w = if ((img_h as f32) * ratio).ceil() as i32 > img_width {
        img_width
    } else {
        ((img_h as f32) * ratio).ceil() as i32
    };

    let mut resized = Mat::default();
    image::resize(
        img,
        &mut resized,
        Size::new(resized_w, img_h as i32),
        INTER_LINEAR,
    )?;

    let size = resized.size()?;
    let h2 = size.height as usize;
    let w2 = size.width as usize;

    let plane_size = img_h * target_w;
    let valid_w = w2.min(target_w);
    let valid_h = h2.min(img_h);

    if img_c == 3 && resized.as_dynamic().as_rgb8().is_some() {
        let rgb = resized.as_dynamic().as_rgb8().unwrap();
        let raw: &[u8] = rgb.as_raw();
        let stride = w2 * 3;
        let (b_plane, rest) = dest.split_at_mut(plane_size);
        let (g_plane, r_plane) = rest.split_at_mut(plane_size);

        for y in 0..valid_h {
            let row = &raw[y * stride..y * stride + valid_w * 3];
            let b_row = &mut b_plane[y * target_w..y * target_w + valid_w];
            let g_row = &mut g_plane[y * target_w..y * target_w + valid_w];
            let r_row = &mut r_plane[y * target_w..y * target_w + valid_w];

            for x in 0..valid_w {
                b_row[x] = NORM_LUT[row[x * 3 + 2] as usize];
                g_row[x] = NORM_LUT[row[x * 3 + 1] as usize];
                r_row[x] = NORM_LUT[row[x * 3] as usize];
            }
        }
    } else {
        for y in 0..valid_h {
            for x in 0..valid_w {
                let pix = resized.get_pixel(x as u32, y as u32);
                dest[y * target_w + x] = NORM_LUT[pix[2] as usize];
                dest[plane_size + y * target_w + x] = NORM_LUT[pix[1] as usize];
                dest[2 * plane_size + y * target_w + x] = NORM_LUT[pix[0] as usize];
            }
        }
    }

    Ok(())
}

/// Resizes and normalizes an image crop for PP-OCR recognition models.
pub fn resize_norm_rec_img(
    img: &Mat,
    img_c: usize,
    img_h: usize,
    max_wh_ratio: f32,
) -> Result<Array3<f32>, CoreError> {
    let img_width = (img_h as f32 * max_wh_ratio).round() as usize;
    let mut out = Array3::<f32>::zeros((img_c, img_h, img_width));
    resize_norm_rec_img_to_slice(img, img_c, img_h, img_width, out.as_mut_slice())?;
    Ok(out)
}

/// Builds a stacked PP-OCR recognition batch tensor directly without intermediate allocations.
pub fn build_rec_batch(
    crop_imgs: &[Mat],
    indices: &[usize],
    img_c: usize,
    img_h: usize,
    max_wh_ratio: f32,
) -> Result<Array4<f32>, CoreError> {
    let n = indices.len();
    let batch_img_width = (img_h as f32 * max_wh_ratio).round() as usize;
    let item_len = img_c * img_h * batch_img_width;
    let mut batch = Array4::<f32>::zeros((n, img_c, img_h, batch_img_width));
    let batch_slice = batch.as_mut_slice();

    for (i, &idx) in indices.iter().enumerate() {
        let dest = &mut batch_slice[i * item_len..(i + 1) * item_len];
        resize_norm_rec_img_to_slice(&crop_imgs[idx], img_c, img_h, batch_img_width, dest)?;
    }

    Ok(batch)
}

#[derive(Debug, Clone)]
pub struct ResolvedDetParams {
    pub limit_side_len: i32,
    pub limit_type: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: i32,
    pub unclip_ratio: f32,
    pub use_dilation: bool,
    pub score_mode: String,
}

pub fn resolve_det_params(
    cfg: &crate::types::DetConfig,
    options: Option<&crate::types::DetectionRunOptions>,
    postprocess: Option<&crate::types::PostprocessRunOptions>,
    ori_w: i32,
    ori_h: i32,
) -> ResolvedDetParams {
    let options = options.cloned().unwrap_or_default();
    let limit_type = options.limit_type.unwrap_or_else(|| cfg.limit_type.clone());
    let configured_limit_side_len = options.limit_side_len.unwrap_or(cfg.limit_side_len);
    let mean = options.mean.unwrap_or(cfg.mean);
    let std = options.std.unwrap_or(cfg.std);

    let pp_thresh = postprocess
        .and_then(|p| p.threshold)
        .or_else(|| options.postprocess.as_ref().and_then(|p| p.threshold))
        .unwrap_or(cfg.thresh);
    let pp_box_thresh = postprocess
        .and_then(|p| p.box_threshold)
        .or_else(|| options.postprocess.as_ref().and_then(|p| p.box_threshold))
        .unwrap_or(cfg.box_thresh);
    let pp_max_candidates = postprocess
        .and_then(|p| p.max_candidates)
        .or_else(|| options.postprocess.as_ref().and_then(|p| p.max_candidates))
        .unwrap_or(cfg.max_candidates);
    let pp_unclip_ratio = postprocess
        .and_then(|p| p.unclip_ratio)
        .or_else(|| options.postprocess.as_ref().and_then(|p| p.unclip_ratio))
        .unwrap_or(cfg.unclip_ratio);
    let pp_use_dilation = postprocess
        .and_then(|p| p.use_dilation)
        .or_else(|| options.postprocess.as_ref().and_then(|p| p.use_dilation))
        .unwrap_or(cfg.use_dilation);
    let pp_score_mode = postprocess
        .and_then(|p| p.score_mode.clone())
        .or_else(|| options.postprocess.as_ref().and_then(|p| p.score_mode.clone()))
        .unwrap_or_else(|| cfg.score_mode.clone());

    let max_wh = ori_h.max(ori_w);
    let limit_side_len = if let Some(explicit_limit) = options.limit_side_len {
        explicit_limit
    } else if limit_type == "min" {
        configured_limit_side_len
    } else if cfg.limit_side_len > 0 {
        cfg.limit_side_len
    } else if max_wh < 960 {
        960
    } else if max_wh < 1500 {
        1500
    } else {
        2000
    };

    ResolvedDetParams {
        limit_side_len,
        limit_type,
        mean,
        std,
        thresh: pp_thresh,
        box_thresh: pp_box_thresh,
        max_candidates: pp_max_candidates,
        unclip_ratio: pp_unclip_ratio,
        use_dilation: pp_use_dilation,
        score_mode: pp_score_mode,
    }
}

pub fn prepare_det_input(
    img: &Mat,
    is_linenet: bool,
    params: &ResolvedDetParams,
    ori_w: i32,
    ori_h: i32,
) -> Result<(Array4<f32>, f32, f32), CoreError> {
    if is_linenet {
        let (target_w, target_h, scale_w, scale_h) =
            calculate_linenet_dims(ori_w, ori_h, params.limit_side_len, &params.limit_type);
        let input = prepare_linenet_det_input(img, target_w, target_h)?;
        Ok((input, scale_w, scale_h))
    } else {
        let pre = DetPreProcess::new(
            params.limit_side_len,
            params.limit_type.clone(),
            params.mean,
            params.std,
        );
        let input = pre.run(img)?;
        Ok((input, 1.0, 1.0))
    }
}

pub fn calculate_linenet_rec_target_widths(sizes: &[(f32, f32)], rec_h: usize) -> Vec<usize> {
    let rec_h_f32 = rec_h as f32;
    sizes
        .iter()
        .map(|&(w, h)| {
            let target_w = if h > 0.0 {
                (rec_h_f32 * (w / h)).round() as usize
            } else {
                rec_h
            };
            target_w.clamp(8, 4096)
        })
        .collect()
}

pub fn stack_rec_batch(
    norm_batch: &[Array3<f32>],
    img_c: usize,
    img_h: usize,
    batch_img_width: usize,
) -> Array4<f32> {
    let n = norm_batch.len();
    let item_len = img_c * img_h * batch_img_width;
    let mut batch = Array4::<f32>::zeros((n, img_c, img_h, batch_img_width));
    let batch_slice = batch.as_mut_slice();
    for (i, arr) in norm_batch.iter().enumerate() {
        let src = arr.as_slice();
        let copy_len = src.len().min(item_len);
        batch_slice[i * item_len..i * item_len + copy_len].copy_from_slice(&src[..copy_len]);
    }
    batch
}

#[derive(Debug, Clone)]
pub struct RecBatchPlan {
    pub sorted_indices: Vec<usize>,
    pub batches: Vec<RecBatchItem>,
}

#[derive(Debug, Clone)]
pub struct RecBatchItem {
    pub indices: Vec<usize>,
    pub max_wh_ratio: f32,
    pub wh_ratio_list: Vec<f32>,
}

pub fn plan_rec_batches(sizes: &[(f32, f32)], batch_num: usize, default_wh_ratio: f32) -> RecBatchPlan {
    let mut width_list = Vec::with_capacity(sizes.len());
    for &(w, h) in sizes {
        width_list.push(if h > 0.0 { w / h } else { 1.0 });
    }

    let mut indices: Vec<usize> = (0..sizes.len()).collect();
    indices.sort_by(|&a, &b| width_list[a].partial_cmp(&width_list[b]).unwrap_or(std::cmp::Ordering::Equal));

    let img_num = sizes.len();
    let batch_num = batch_num.max(1);
    let mut batches = Vec::new();

    let mut beg = 0usize;
    while beg < img_num {
        let end = (beg + batch_num).min(img_num);
        let mut max_wh_ratio = default_wh_ratio;
        let mut wh_ratio_list = Vec::with_capacity(end - beg);
        for &idx in &indices[beg..end] {
            let wh_ratio = width_list[idx];
            if wh_ratio > max_wh_ratio {
                max_wh_ratio = wh_ratio;
            }
            wh_ratio_list.push(wh_ratio);
        }
        batches.push(RecBatchItem {
            indices: indices[beg..end].to_vec(),
            max_wh_ratio,
            wh_ratio_list,
        });
        beg = end;
    }

    RecBatchPlan {
        sorted_indices: indices,
        batches,
    }
}

/// Group LineNet recognition crop indices into batches of identical target width.
///
/// Right-padding shorter crops would alter the patch-embedding border of the padded
/// samples, so only exact-width crops share a batch and results match single-crop runs.
pub fn plan_linenet_rec_batches(target_widths: &[usize], batch_num: usize) -> Vec<Vec<usize>> {
    let n = target_widths.len();
    if n == 0 {
        return Vec::new();
    }
    let batch_cap = batch_num.max(1);

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&idx| (target_widths[idx], idx));

    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current_w = 0usize;
    for idx in sorted {
        let w = target_widths[idx];
        match batches.last_mut() {
            Some(last) if w.div_ceil(8) == current_w && last.len() < batch_cap => last.push(idx),
            _ => {
                current_w = w.div_ceil(8);
                batches.push(vec![idx]);
            }
        }
    }
    batches
}

/// Estimates text line skew angle in degrees using ink projection profile variance.
pub fn estimate_skew(img: &Mat) -> f32 {
    let (h_orig, w_orig) = (img.rows(), img.cols());
    if h_orig < 32 || w_orig < 32 {
        return 0.0;
    }

    let max_dim = h_orig.max(w_orig) as f32;
    let scale = (384.0 / max_dim).min(1.0);
    let target_w = ((w_orig as f32 * scale).round() as i32).max(32);
    let target_h = ((h_orig as f32 * scale).round() as i32).max(32);

    let mut thumb = Mat::default();
    if image::resize(img, &mut thumb, Size::new(target_w, target_h), image::INTER_LINEAR).is_err() {
        return 0.0;
    }

    let tw = target_w as usize;
    let th = target_h as usize;

    let mut ink = vec![0.0f32; tw * th];
    if let Some(rgb) = thumb.as_dynamic().as_rgb8() {
        let raw = rgb.as_raw();
        for i in 0..tw * th {
            let px = i * 3;
            let gray = 0.299 * raw[px] as f32 + 0.587 * raw[px + 1] as f32 + 0.114 * raw[px + 2] as f32;
            ink[i] = (255.0 - gray).max(0.0);
        }
    } else {
        for y in 0..th {
            let y_off = y * tw;
            for x in 0..tw {
                let px = thumb.get_pixel(x as u32, y as u32);
                let gray = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
                ink[y_off + x] = (255.0 - gray).max(0.0);
            }
        }
    }

    let cx = tw as f32 / 2.0;
    let cy = th as f32 / 2.0;

    let calc_variance = |angle_deg: f32| -> f32 {
        let rad = angle_deg.to_radians();
        let sin_a = rad.sin();
        let cos_a = rad.cos();

        let mut bins = vec![0.0f32; th];
        for y in 0..th {
            let dy = y as f32 - cy;
            let y_off = y * tw;
            for x in 0..tw {
                let dx = x as f32 - cx;
                let proj_y = (dx * sin_a + dy * cos_a + cy).round() as isize;
                if proj_y >= 0 && (proj_y as usize) < th {
                    bins[proj_y as usize] += ink[y_off + x];
                }
            }
        }

        let mean = bins.iter().sum::<f32>() / th as f32;
        bins.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / th as f32
    };

    let mut best_angle = 0.0f32;
    let mut best_var = f32::NEG_INFINITY;

    let mut angle = -10.0f32;
    while angle <= 10.0 {
        let var = calc_variance(angle);
        if var > best_var {
            best_var = var;
            best_angle = angle;
        }
        angle += 0.5;
    }

    let fine_start = (best_angle - 0.6).max(-10.0);
    let fine_end = (best_angle + 0.6).min(10.0);
    let mut fine_angle = fine_start;
    while fine_angle <= fine_end + 1e-4 {
        let var = calc_variance(fine_angle);
        if var > best_var {
            best_var = var;
            best_angle = fine_angle;
        }
        fine_angle += 0.1;
    }

    best_angle
}

/// Deskews image by rotating around center by specified angle in degrees.
pub fn rotate_skew(img: &Mat, angle_deg: f32) -> Result<Mat, CoreError> {
    if angle_deg.abs() < 1e-4 {
        return Ok(img.clone());
    }
    let (h, w) = (img.rows(), img.cols());
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);

    let rad = angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let dst_corners = [
        crate::image::Point2f::new(0.0, 0.0),
        crate::image::Point2f::new(w as f32, 0.0),
        crate::image::Point2f::new(w as f32, h as f32),
        crate::image::Point2f::new(0.0, h as f32),
    ];

    let mut src_corners = [crate::image::Point2f::default(); 4];
    for (i, p) in dst_corners.iter().enumerate() {
        let dx = p.x - cx;
        let dy = p.y - cy;
        src_corners[i] = crate::image::Point2f::new(
            cx + dx * cos_a + dy * sin_a,
            cy - dx * sin_a + dy * cos_a,
        );
    }

    let m = image::get_perspective_transform(&src_corners, &dst_corners)?;
    let mut dst = Mat::default();
    image::warp_perspective(
        img,
        &mut dst,
        &m,
        Size::new(w, h),
        image::INTER_LINEAR,
        image::BORDER_REPLICATE,
    )?;
    Ok(dst)
}

/// Rotates 4 quad points back around image center by skew angle in degrees.
pub fn rotate_box_points_skew_back(pts: &mut [crate::image::Point2f; 4], cx: f32, cy: f32, angle_deg: f32) {
    let rad = -angle_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    for p in pts.iter_mut() {
        let dx = p.x - cx;
        let dy = p.y - cy;
        p.x = cx + dx * cos_a - dy * sin_a;
        p.y = cy + dx * sin_a + dy * cos_a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_rec_batches() {
        let sizes = vec![(100.0, 50.0), (300.0, 50.0), (50.0, 50.0)];
        let plan = plan_rec_batches(&sizes, 2, 2.0);
        assert_eq!(plan.sorted_indices, vec![2, 0, 1]);
        assert_eq!(plan.batches.len(), 2);
        assert_eq!(plan.batches[0].indices, vec![2, 0]);
        assert_eq!(plan.batches[1].indices, vec![1]);
        assert_eq!(plan.batches[1].max_wh_ratio, 6.0);
    }

    #[test]
    fn test_plan_linenet_rec_batches_groups_exact_widths() {
        let widths = vec![640, 80, 80, 88, 640, 80, 1280];
        let batches = plan_linenet_rec_batches(&widths, 2);
        assert_eq!(
            batches,
            vec![vec![1, 2], vec![5], vec![3], vec![0, 4], vec![6]]
        );
        assert_eq!(plan_linenet_rec_batches(&widths, 1).len(), widths.len());
    }

    #[test]
    fn test_preprocess_rgba_raw_matches_sample_bilinear() {
        let orig_w = 37usize;
        let orig_h = 23usize;
        let rgba: Vec<u8> = (0..orig_w * orig_h * 4)
            .map(|i| ((i * 73 + 19) % 256) as u8)
            .collect();
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let (got, tw, th, sw, sh) =
            DetPreProcess::preprocess_rgba_raw(&rgba, orig_w, orig_h, 64, "max", mean, std);
        for ty in 0..th {
            let sy = ((ty as f32 + 0.5) / sh - 0.5).clamp(0.0, (orig_h - 1) as f32);
            for tx in 0..tw {
                let sx = ((tx as f32 + 0.5) / sw - 0.5).clamp(0.0, (orig_w - 1) as f32);
                let (r, g, b) = image::sample_bilinear(&rgba, orig_w, orig_h, sx, sy);
                let idx = ty * tw + tx;
                assert!((got[idx] - (b / 255.0 - mean[0]) / std[0]).abs() < 1e-5);
                assert!((got[th * tw + idx] - (g / 255.0 - mean[1]) / std[1]).abs() < 1e-5);
                assert!((got[2 * th * tw + idx] - (r / 255.0 - mean[2]) / std[2]).abs() < 1e-5);
            }
        }
    }
}

