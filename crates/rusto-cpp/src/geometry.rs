//! Geometry operations accelerated with OpenCV

#[cfg(feature = "opencv")]
use std::collections::BTreeMap;
#[cfg(feature = "opencv")]
use opencv::{core, imgproc, prelude::*};
#[cfg(feature = "opencv")]
use opencv::core::{Mat, Point2f};
#[cfg(feature = "opencv")]
use rusto_core::CoreError;

pub use rusto_core::geometry::{iou, map_boxes_to_original, nms, OpRecord};

#[cfg(feature = "opencv")]
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

#[cfg(feature = "opencv")]
fn get_padding_h(h: i32, w: i32, width_height_ratio: f32, min_height: f32) -> i32 {
    let new_h = ((w as f32 / width_height_ratio) as i32).max(min_height as i32) * 2;
    (new_h - h).abs() / 2
}

#[cfg(feature = "opencv")]
pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> Result<Mat, CoreError> {
    let w1 = (points[0].x - points[1].x).hypot(points[0].y - points[1].y);
    let w2 = (points[2].x - points[3].x).hypot(points[2].y - points[3].y);
    let img_crop_width = w1.max(w2) as i32;

    let h1 = (points[0].x - points[3].x).hypot(points[0].y - points[3].y);
    let h2 = (points[1].x - points[2].x).hypot(points[1].y - points[2].y);
    let img_crop_height = h1.max(h2) as i32;

    let pts_src = core::Mat::from_slice_2d(&[
        [points[0].x, points[0].y],
        [points[1].x, points[1].y],
        [points[2].x, points[2].y],
        [points[3].x, points[3].y],
    ]).map_err(|e| CoreError::Image(e.to_string()))?;

    let pts_dst = core::Mat::from_slice_2d(&[
        [0.0f32, 0.0f32],
        [img_crop_width as f32, 0.0f32],
        [img_crop_width as f32, img_crop_height as f32],
        [0.0f32, img_crop_height as f32],
    ]).map_err(|e| CoreError::Image(e.to_string()))?;

    let m = imgproc::get_perspective_transform(&pts_src, &pts_dst, 0)
        .map_err(|e| CoreError::Image(e.to_string()))?;
    let mut dst = Mat::default();
    imgproc::warp_perspective(
        img,
        &mut dst,
        &m,
        core::Size::new(img_crop_width, img_crop_height),
        imgproc::INTER_CUBIC,
        core::BORDER_REPLICATE,
        core::Scalar::all(0.0),
    ).map_err(|e| CoreError::Image(e.to_string()))?;

    let size = dst.size().map_err(|e| CoreError::Image(e.to_string()))?;
    let dst_h = size.height as f32;
    let dst_w = size.width as f32;
    if dst_h / dst_w >= 1.5 {
        let mut rotated = Mat::default();
        core::rotate(&dst, &mut rotated, core::ROTATE_90_CLOCKWISE)
            .map_err(|e| CoreError::Image(e.to_string()))?;
        Ok(rotated)
    } else {
        Ok(dst)
    }
}

#[cfg(feature = "opencv")]
pub fn resize_image_within_bounds(
    img: &Mat,
    min_side_len: f32,
    max_side_len: f32,
) -> Result<(Mat, f32, f32), CoreError> {
    let size = img.size().map_err(|e| CoreError::Image(e.to_string()))?;
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

    let size2 = img_out.size().map_err(|e| CoreError::Image(e.to_string()))?;
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

#[cfg(feature = "opencv")]
pub fn reduce_max_side(img: &Mat, max_side_len: f32) -> Result<(Mat, f32, f32), CoreError> {
    let size = img.size().map_err(|e| CoreError::Image(e.to_string()))?;
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
    imgproc::resize(
        img,
        &mut dst,
        core::Size::new(resize_w, resize_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    ).map_err(|e| CoreError::Image(e.to_string()))?;

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

#[cfg(feature = "opencv")]
pub fn increase_min_side(img: &Mat, min_side_len: f32) -> Result<(Mat, f32, f32), CoreError> {
    let size = img.size().map_err(|e| CoreError::Image(e.to_string()))?;
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
    imgproc::resize(
        img,
        &mut dst,
        core::Size::new(resize_w, resize_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    ).map_err(|e| CoreError::Image(e.to_string()))?;

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

#[cfg(feature = "opencv")]
pub fn add_round_letterbox(img: &Mat, padding: (i32, i32, i32, i32)) -> Result<Mat, CoreError> {
    let mut dst = Mat::default();
    core::copy_make_border(
        img,
        &mut dst,
        padding.0,
        padding.1,
        padding.2,
        padding.3,
        core::BORDER_CONSTANT,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    ).map_err(|e| CoreError::Image(e.to_string()))?;
    Ok(dst)
}

#[cfg(feature = "opencv")]
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
    let core_b = [
        rusto_core::Point2f::new(b[0].x, b[0].y),
        rusto_core::Point2f::new(b[1].x, b[1].y),
        rusto_core::Point2f::new(b[2].x, b[2].y),
        rusto_core::Point2f::new(b[3].x, b[3].y),
    ];
    let res = rusto_core::geometry::calculate_crop_box(
        &core_b,
        is_linenet,
        pad_x,
        pad_y,
        pad_x_factor,
        pad_y_factor,
        working_w,
        working_h,
    );
    [
        Point2f::new(res[0].x, res[0].y),
        Point2f::new(res[1].x, res[1].y),
        Point2f::new(res[2].x, res[2].y),
        Point2f::new(res[3].x, res[3].y),
    ]
}
