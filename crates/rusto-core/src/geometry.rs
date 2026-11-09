use std::collections::BTreeMap;
use crate::error::CoreError;
use crate::image::{self, Mat, Point2f, Size, INTER_LINEAR, ROTATE_90_CLOCKWISE};

pub type OpRecord = BTreeMap<String, BTreeMap<String, f32>>;

pub fn map_boxes_to_original(
    dt_boxes: &mut [[Point2f; 4]],
    op_record: &OpRecord,
    ori_h: i32,
    ori_w: i32,
) {
    for (op, v) in op_record.iter().rev() {
        if op.contains("padding") {
            let top = *v.get("top").unwrap_or(&0.0);
            let left = *v.get("left").unwrap_or(&0.0);
            for box_pts in dt_boxes.iter_mut() {
                for p in box_pts.iter_mut() {
                    p.x -= left;
                    p.y -= top;
                }
            }
        } else if op.contains("preprocess") {
            let ratio_h = *v.get("ratio_h").unwrap_or(&1.0);
            let ratio_w = *v.get("ratio_w").unwrap_or(&1.0);
            for box_pts in dt_boxes.iter_mut() {
                for p in box_pts.iter_mut() {
                    p.x *= ratio_w;
                    p.y *= ratio_h;
                }
            }
        }
    }

    for box_pts in dt_boxes.iter_mut() {
        for p in box_pts.iter_mut() {
            if p.x < 0.0 {
                p.x = 0.0;
            }
            if p.y < 0.0 {
                p.y = 0.0;
            }
            if p.x > ori_w as f32 {
                p.x = ori_w as f32;
            }
            if p.y > ori_h as f32 {
                p.y = ori_h as f32;
            }
        }
    }
}

pub fn apply_vertical_padding(
    img: &Mat,
    mut op_record: OpRecord,
    width_height_ratio: f32,
    min_height: f32,
) -> Result<(Mat, OpRecord), CoreError> {
    let h = img.rows();
    let w = img.cols();

    let use_limit_ratio = if (width_height_ratio - (-1.0)).abs() < f32::EPSILON {
        false
    } else {
        (w as f32) / (h as f32) > width_height_ratio
    };

    if (h as f32) <= min_height || use_limit_ratio {
        let padding_h = get_padding_h(h, w, width_height_ratio, min_height);
        let padded = add_round_letterbox(img, (padding_h, padding_h, 0, 0))?;
        let mut m = BTreeMap::new();
        m.insert("top".to_string(), padding_h as f32);
        m.insert("left".to_string(), 0.0);
        op_record.insert("padding_1".to_string(), m);
        Ok((padded, op_record))
    } else {
        let mut m = BTreeMap::new();
        m.insert("top".to_string(), 0.0);
        m.insert("left".to_string(), 0.0);
        op_record.insert("padding_1".to_string(), m);
        Ok((img.clone(), op_record))
    }
}

fn get_padding_h(h: i32, w: i32, width_height_ratio: f32, min_height: f32) -> i32 {
    let new_h = ((w as f32 / width_height_ratio) as i32).max(min_height as i32) * 2;
    (new_h - h).abs() / 2
}

pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> Result<Mat, CoreError> {
    let w1 = (points[0].x - points[1].x).hypot(points[0].y - points[1].y);
    let w2 = (points[2].x - points[3].x).hypot(points[2].y - points[3].y);
    let img_crop_width = w1.max(w2) as i32;

    let h1 = (points[0].x - points[3].x).hypot(points[0].y - points[3].y);
    let h2 = (points[1].x - points[2].x).hypot(points[1].y - points[2].y);
    let img_crop_height = h1.max(h2) as i32;

    let pts_src = [
        points[0],
        points[1],
        points[2],
        points[3],
    ];

    let pts_dst = [
        Point2f::new(0.0, 0.0),
        Point2f::new(img_crop_width as f32, 0.0),
        Point2f::new(img_crop_width as f32, img_crop_height as f32),
        Point2f::new(0.0, img_crop_height as f32),
    ];

    let m = image::get_perspective_transform(&pts_src, &pts_dst)?;
    let mut dst = Mat::default();
    image::warp_perspective(
        img,
        &mut dst,
        &m,
        Size::new(img_crop_width, img_crop_height),
        2,
        image::BORDER_REPLICATE,
    )?;

    let size = dst.size()?;
    let dst_h = size.height as f32;
    let dst_w = size.width as f32;
    if dst_h / dst_w >= 1.5 {
        let mut rotated = Mat::default();
        image::rotate(&dst, &mut rotated, ROTATE_90_CLOCKWISE)?;
        Ok(rotated)
    } else {
        Ok(dst)
    }
}

pub fn resize_image_within_bounds(
    img: &Mat,
    min_side_len: f32,
    max_side_len: f32,
) -> Result<(Mat, f32, f32), CoreError> {
    let size = img.size()?;
    let mut h = size.height;
    let mut w = size.width;

    let mut ratio_h = 1.0f32;
    let mut ratio_w = 1.0f32;

    let max_value = h.max(w) as f32;
    let mut img_out = img.clone();
    if max_value > max_side_len {
        let (resized, rh, rw) = reduce_max_side(&img_out, max_side_len)?;
        img_out = resized;
        ratio_h = rh;
        ratio_w = rw;
    }

    let size2 = img_out.size()?;
    h = size2.height;
    w = size2.width;
    let min_value = h.min(w) as f32;
    if min_value < min_side_len {
        let (resized, rh, rw) = increase_min_side(&img_out, min_side_len)?;
        img_out = resized;
        ratio_h = rh;
        ratio_w = rw;
    }

    Ok((img_out, ratio_h, ratio_w))
}

pub fn reduce_max_side(img: &Mat, max_side_len: f32) -> Result<(Mat, f32, f32), CoreError> {
    let size = img.size()?;
    let h = size.height as f32;
    let w = size.width as f32;

    let mut ratio = 1.0f32;
    if h.max(w) > max_side_len {
        ratio = if h > w {
            max_side_len / h
        } else {
            max_side_len / w
        };
    }

    let mut resize_h = (h * ratio) as i32;
    let mut resize_w = (w * ratio) as i32;

    resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
    resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

    if resize_w <= 0 || resize_h <= 0 {
        return Err(CoreError::Preprocess(
            "resize_w or resize_h is less than or equal to 0".to_string(),
        ));
    }

    let mut dst = Mat::default();
    image::resize(img, &mut dst, Size::new(resize_w, resize_h), INTER_LINEAR)?;

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

pub fn increase_min_side(img: &Mat, min_side_len: f32) -> Result<(Mat, f32, f32), CoreError> {
    let size = img.size()?;
    let h = size.height as f32;
    let w = size.width as f32;

    let mut ratio = 1.0f32;
    if h.min(w) < min_side_len {
        ratio = if h < w {
            min_side_len / h
        } else {
            min_side_len / w
        };
    }

    let mut resize_h = (h * ratio) as i32;
    let mut resize_w = (w * ratio) as i32;

    resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
    resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

    if resize_w <= 0 || resize_h <= 0 {
        return Err(CoreError::Preprocess(
            "resize_w or resize_h is less than or equal to 0".to_string(),
        ));
    }

    let mut dst = Mat::default();
    image::resize(img, &mut dst, Size::new(resize_w, resize_h), INTER_LINEAR)?;

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

pub fn add_round_letterbox(img: &Mat, padding: (i32, i32, i32, i32)) -> Result<Mat, CoreError> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let new_width = width + padding.2 as u32 + padding.3 as u32;
    let new_height = height + padding.0 as u32 + padding.1 as u32;

    let mut new_img = ::image::ImageBuffer::from_pixel(new_width, new_height, ::image::Rgb([0, 0, 0]));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            new_img.put_pixel(x + padding.3 as u32, y + padding.0 as u32, *pixel);
        }
    }

    Ok(Mat::new(::image::DynamicImage::ImageRgb8(new_img)))
}

pub fn iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0];
    let y1 = box1[1];
    let x2 = box1[2];
    let y2 = box1[3];

    let x1_p = box2[0];
    let y1_p = box2[1];
    let x2_p = box2[2];
    let y2_p = box2[3];

    let x1_i = x1.max(x1_p);
    let y1_i = y1.max(y1_p);
    let x2_i = x2.min(x2_p);
    let y2_i = y2.min(y2_p);

    let inter_area = (x2_i - x1_i + 1.0).max(0.0) * (y2_i - y1_i + 1.0).max(0.0);

    let box1_area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
    let box2_area = (x2_p - x1_p + 1.0) * (y2_p - y1_p + 1.0);

    inter_area / (box1_area + box2_area - inter_area)
}

pub fn nms(boxes: &[[f32; 6]], iou_same: f32, iou_diff: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b][1].partial_cmp(&boxes[a][1]).unwrap());

    let mut selected_indices = Vec::new();

    while let Some(&current) = indices.first() {
        selected_indices.push(current);
        let current_box = &boxes[current];
        let current_class = current_box[0];
        let current_coords = [
            current_box[2],
            current_box[3],
            current_box[4],
            current_box[5],
        ];

        indices.remove(0);

        indices.retain(|&i| {
            let box_ = &boxes[i];
            let box_class = box_[0];
            let box_coords = [box_[2], box_[3], box_[4], box_[5]];
            let iou_value = iou(&current_coords, &box_coords);
            let threshold = if current_class == box_class {
                iou_same
            } else {
                iou_diff
            };

            iou_value < threshold
        });
    }

    selected_indices
}

pub fn calculate_crop_box(
    b: &[Point2f; 4],
    is_linenet: bool,
    pad_x: f32,
    pad_y: f32,
    pad_x_factor: f32,
    pad_y_factor: f32,
    working_w: f32,
    working_h: f32,
) -> [Point2f; 4] {
    if is_linenet {
        let vx = b[1].x - b[0].x;
        let vy = b[1].y - b[0].y;
        let w = vx.hypot(vy);

        let hx = b[3].x - b[0].x;
        let hy = b[3].y - b[0].y;
        let h = hx.hypot(hy);

        if w < 1e-4 || h < 1e-4 {
            *b
        } else {
            let u_wx = vx / w;
            let u_wy = vy / w;
            let u_hx = hx / h;
            let u_hy = hy / h;

            let dx = h * pad_x_factor;
            let dy = h * pad_y_factor;

            let off_x0 = -dx * u_wx - dy * u_hx;
            let off_y0 = -dx * u_wy - dy * u_hy;
            let off_x1 = dx * u_wx - dy * u_hx;
            let off_y1 = dx * u_wy - dy * u_hy;
            let off_x2 = dx * u_wx + dy * u_hx;
            let off_y2 = dx * u_wy + dy * u_hy;
            let off_x3 = -dx * u_wx + dy * u_hx;
            let off_y3 = -dx * u_wy + dy * u_hy;

            [
                Point2f::new((b[0].x + off_x0).clamp(0.0, working_w), (b[0].y + off_y0).clamp(0.0, working_h)),
                Point2f::new((b[1].x + off_x1).clamp(0.0, working_w), (b[1].y + off_y1).clamp(0.0, working_h)),
                Point2f::new((b[2].x + off_x2).clamp(0.0, working_w), (b[2].y + off_y2).clamp(0.0, working_h)),
                Point2f::new((b[3].x + off_x3).clamp(0.0, working_w), (b[3].y + off_y3).clamp(0.0, working_h)),
            ]
        }
    } else if pad_x > 0.0 || pad_y > 0.0 {
        crate::calibration::expand_box_padding(
            b,
            pad_x,
            pad_y,
            working_w,
            working_h,
        )
    } else {
        *b
    }
}

fn hist_percentile(hist: &[u32; 256], count: u32, q: f32) -> f32 {
    let value_at = |rank: u32| -> f32 {
        let mut acc = 0u32;
        for (bin, &c) in hist.iter().enumerate() {
            acc += c;
            if acc > rank {
                return bin as f32;
            }
        }
        255.0
    };
    let x = q * count.saturating_sub(1) as f32;
    let i = x.floor();
    let g = x - i;
    let lo = value_at(i as u32);
    let hi = value_at((i as u32 + 1).min(count.saturating_sub(1)));
    let d = hi - lo;
    if g >= 0.5 {
        hi - d * (1.0 - g)
    } else {
        lo + d * g
    }
}

/// Returns `(ix0, cw)`: the left edge and width in page pixels of the padded line crop.
pub fn linenet_crop_x_span(b: &[Point2f; 4], pad_x_factor: f32) -> (f32, f32) {
    let x0 = b[0].x.min(b[3].x);
    let y0 = b[0].y.min(b[1].y);
    let x1 = b[1].x.max(b[2].x);
    let y1 = b[2].y.max(b[3].y);
    let h = (y1 - y0).max(1.0);
    let ix0 = (x0 - pad_x_factor * h).floor();
    let ix1 = (x1 + pad_x_factor * h).ceil();
    (ix0, (ix1 - ix0).max(1.0))
}

pub fn crop_linenet_line_image(
    img: &Mat,
    b: &[Point2f; 4],
    pad_x_factor: f32,
    pad_y_factor: f32,
) -> Result<Mat, CoreError> {
    let w_img = img.cols();
    let h_img = img.rows();
    let x0 = b[0].x.min(b[3].x);
    let y0 = b[0].y.min(b[1].y);
    let x1 = b[1].x.max(b[2].x);
    let y1 = b[2].y.max(b[3].y);
    let h = (y1 - y0).max(1.0);

    let owned_rgb;
    let (raw_rgb, stride_w) = if let Some(rgb_ref) = img.as_dynamic().as_rgb8() {
        (rgb_ref.as_raw().as_slice(), rgb_ref.width() as usize)
    } else {
        owned_rgb = img.to_rgb8();
        (owned_rgb.as_raw().as_slice(), owned_rgb.width() as usize)
    };

    let ux0 = (x0 as i32).clamp(0, w_img);
    let uy0 = (y0 as i32).clamp(0, h_img);
    let ux1 = (x1 as i32 + 1).clamp(0, w_img);
    let uy1 = (y1 as i32 + 1).clamp(0, h_img);

    let mut dark = false;
    if ux1 > ux0 && uy1 > uy0 {
        let count = ((ux1 - ux0) * (uy1 - uy0)) as u32;
        let mut hist = [0u32; 256];
        for sy in uy0..uy1 {
            let row_off = sy as usize * stride_w * 3;
            for sx in ux0..ux1 {
                let idx = row_off + sx as usize * 3;
                let gray = (299 * raw_rgb[idx] as u32
                    + 587 * raw_rgb[idx + 1] as u32
                    + 114 * raw_rgb[idx + 2] as u32
                    + 500)
                    / 1000;
                hist[gray as usize] += 1;
            }
        }
        let lo = hist_percentile(&hist, count, 0.02);
        let med = hist_percentile(&hist, count, 0.50);
        let hi = hist_percentile(&hist, count, 0.98);
        dark = (hi - med) > 2.0 * (med - lo) + 8.0;
    }

    let ix0 = (x0 - pad_x_factor * h).floor() as i32;
    let iy0 = (y0 - pad_y_factor * h).floor() as i32;
    let ix1 = (x1 + pad_x_factor * h).ceil() as i32;
    let iy1 = (y1 + pad_y_factor * h).ceil() as i32;

    let cw = (ix1 - ix0).max(1) as usize;
    let ch = (iy1 - iy0).max(1) as usize;

    let sx0 = ix0.max(0);
    let sy0 = iy0.max(0);
    let sx1 = ix1.min(w_img);
    let sy1 = iy1.min(h_img);

    let mut out = vec![255u8; cw * ch * 3];
    if sx1 > sx0 && sy1 > sy0 {
        for sy in sy0..sy1 {
            let src_row = sy as usize * stride_w * 3;
            let dst_row = (sy - iy0) as usize * cw * 3;
            for sx in sx0..sx1 {
                let s_idx = src_row + sx as usize * 3;
                let d_idx = dst_row + (sx - ix0) as usize * 3;
                if dark {
                    out[d_idx] = 255 - raw_rgb[s_idx];
                    out[d_idx + 1] = 255 - raw_rgb[s_idx + 1];
                    out[d_idx + 2] = 255 - raw_rgb[s_idx + 2];
                } else {
                    out[d_idx] = raw_rgb[s_idx];
                    out[d_idx + 1] = raw_rgb[s_idx + 1];
                    out[d_idx + 2] = raw_rgb[s_idx + 2];
                }
            }
        }
    }

    Mat::from_rgb8(cw as u32, ch as u32, out)
}

#[cfg(test)]
mod linenet_crop_tests {
    use super::*;

    fn flat_page(w: u32, h: u32, value: u8) -> Mat {
        Mat::from_rgb8(w, h, vec![value; (w * h * 3) as usize]).unwrap()
    }

    fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [Point2f; 4] {
        [
            Point2f::new(x0, y0),
            Point2f::new(x1, y0),
            Point2f::new(x1, y1),
            Point2f::new(x0, y1),
        ]
    }

    #[test]
    fn test_crop_pads_outside_page_with_white() {
        let page = flat_page(100, 100, 0);
        let crop = crop_linenet_line_image(&page, &quad(10.0, 2.0, 60.0, 12.0), 0.3, 1.6).unwrap();
        let (ix0, cw) = linenet_crop_x_span(&quad(10.0, 2.0, 60.0, 12.0), 0.3);
        assert_eq!(ix0, 7.0);
        assert_eq!(crop.cols(), cw as i32);
        assert_eq!(crop.rows(), (12.0f32 + 16.0).ceil() as i32 - (2.0f32 - 16.0).floor() as i32);
        assert_eq!(crop.get_pixel(0, 0), [255, 255, 255]);
    }

    #[test]
    fn test_crop_inverts_light_text_on_dark_ground() {
        let mut data = vec![20u8; 200 * 60 * 3];
        for y in 25..35usize {
            for x in 50..150usize {
                if (x / 3) % 2 == 0 {
                    let i = (y * 200 + x) * 3;
                    data[i..i + 3].fill(240);
                }
            }
        }
        let page = Mat::from_rgb8(200, 60, data).unwrap();
        let crop = crop_linenet_line_image(&page, &quad(50.0, 25.0, 150.0, 35.0), 0.3, 1.6).unwrap();
        let ground = crop.get_pixel(crop.cols() as u32 / 2, 1);
        assert_eq!(ground, [235, 235, 235]);
    }

    #[test]
    fn test_crop_keeps_dark_text_on_light_ground() {
        let page = flat_page(100, 100, 250);
        let crop = crop_linenet_line_image(&page, &quad(10.0, 40.0, 60.0, 50.0), 0.3, 1.6).unwrap();
        assert_eq!(crop.get_pixel(5, 20), [250, 250, 250]);
    }
}
