use std::collections::BTreeMap;

use crate::engine::EngineError;

#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(feature = "use-opencv")]
use opencv::core::{Mat, Point2f};

#[cfg(feature = "use-opencv")]
type ImgResult<T> = opencv::Result<T>;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{self, Mat, Point2f, Result as ImgResult, Size, INTER_LINEAR, ROTATE_90_CLOCKWISE};

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
            if p.x < 0.0 { p.x = 0.0; }
            if p.y < 0.0 { p.y = 0.0; }
            if p.x > ori_w as f32 { p.x = ori_w as f32; }
            if p.y > ori_h as f32 { p.y = ori_h as f32; }
        }
    }
}

pub fn apply_vertical_padding(
    img: &Mat,
    mut op_record: OpRecord,
    width_height_ratio: f32,
    min_height: f32,
) -> Result<(Mat, OpRecord), EngineError> {
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
    // Match Python: max(int(w / width_height_ratio), min_height) * 2
    let new_h = ((w as f32 / width_height_ratio) as i32).max(min_height as i32) * 2;
    ((new_h - h).abs() / 2) as i32
}

#[cfg(feature = "use-opencv")]
pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> ImgResult<Mat> {
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
    ])?;

    let pts_dst = core::Mat::from_slice_2d(&[
        [0.0f32, 0.0f32],
        [img_crop_width as f32, 0.0f32],
        [img_crop_width as f32, img_crop_height as f32],
        [0.0f32, img_crop_height as f32],
    ])?;

    let m = imgproc::get_perspective_transform(&pts_src, &pts_dst, 0)?;
    let mut dst = Mat::default();
    imgproc::warp_perspective(
        img,
        &mut dst,
        &m,
        core::Size::new(img_crop_width, img_crop_height),
        imgproc::INTER_CUBIC,
        core::BORDER_REPLICATE,
        core::Scalar::all(0.0),
    )?;

    let size = dst.size()?;
    let dst_h = size.height as f32;
    let dst_w = size.width as f32;
    if dst_h / dst_w >= 1.5 {
        let mut rotated = Mat::default();
        core::rotate(&dst, &mut rotated, core::ROTATE_90_CLOCKWISE)?;
        Ok(rotated)
    } else {
        Ok(dst)
    }
}

#[cfg(not(feature = "use-opencv"))]
pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> ImgResult<Mat> {
    let w1 = (points[0].x - points[1].x).hypot(points[0].y - points[1].y);
    let w2 = (points[2].x - points[3].x).hypot(points[2].y - points[3].y);
    let img_crop_width = w1.max(w2) as i32;

    let h1 = (points[0].x - points[3].x).hypot(points[0].y - points[3].y);
    let h2 = (points[1].x - points[2].x).hypot(points[1].y - points[2].y);
    let img_crop_height = h1.max(h2) as i32;

    let pts_src = [
        [points[0].x, points[0].y],
        [points[1].x, points[1].y],
        [points[2].x, points[2].y],
        [points[3].x, points[3].y],
    ];

    let pts_dst = [
        [0.0f32, 0.0f32],
        [img_crop_width as f32, 0.0f32],
        [img_crop_width as f32, img_crop_height as f32],
        [0.0f32, img_crop_height as f32],
    ];

    let m = image_impl::get_perspective_transform(&pts_src, &pts_dst)?;
    let mut dst = Mat::default();
    image_impl::warp_perspective(
        img,
        &mut dst,
        &m,
        Size::new(img_crop_width, img_crop_height),
        2, // INTER_CUBIC
        image_impl::BORDER_REPLICATE,
    )?;

    let size = dst.size()?;
    let dst_h = size.height as f32;
    let dst_w = size.width as f32;
    if dst_h / dst_w >= 1.5 {
        let mut rotated = Mat::default();
        image_impl::rotate(&dst, &mut rotated, ROTATE_90_CLOCKWISE)?;
        Ok(rotated)
    } else {
        Ok(dst)
    }
}

pub fn resize_image_within_bounds(
    img: &Mat,
    min_side_len: f32,
    max_side_len: f32,
) -> Result<(Mat, f32, f32), EngineError> {
    let size = img.size()?;
    let mut h = size.height as i32;
    let mut w = size.width as i32;

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
    h = size2.height as i32;
    w = size2.width as i32;
    let min_value = h.min(w) as f32;
    if min_value < min_side_len {
        let (resized, rh, rw) = increase_min_side(&img_out, min_side_len)?;
        img_out = resized;
        ratio_h = rh;
        ratio_w = rw;
    }

    Ok((img_out, ratio_h, ratio_w))
}

pub fn reduce_max_side(img: &Mat, max_side_len: f32) -> Result<(Mat, f32, f32), EngineError> {
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

    // Match Python: int(h * ratio) truncates, not rounds
    let mut resize_h = (h * ratio) as i32;
    let mut resize_w = (w * ratio) as i32;

    resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
    resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

    if resize_w <= 0 || resize_h <= 0 {
        return Err(EngineError::Preprocess(
            "resize_w or resize_h is less than or equal to 0".to_string(),
        ));
    }

    #[cfg(feature = "use-opencv")]
    let dst = {
        let mut d = Mat::default();
        imgproc::resize(
            img,
            &mut d,
            core::Size::new(resize_w, resize_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        d
    };
    
    #[cfg(not(feature = "use-opencv"))]
    let dst = {
        let mut d = Mat::default();
        image_impl::resize(
            img,
            &mut d,
            Size::new(resize_w, resize_h),
            INTER_LINEAR,
        )?;
        d
    };

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

pub fn increase_min_side(img: &Mat, min_side_len: f32) -> Result<(Mat, f32, f32), EngineError> {
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

    // Match Python: int(h * ratio) truncates
    let mut resize_h = (h * ratio) as i32;
    let mut resize_w = (w * ratio) as i32;

    resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as i32;
    resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as i32;

    if resize_w <= 0 || resize_h <= 0 {
        return Err(EngineError::Preprocess(
            "resize_w or resize_h is less than or equal to 0".to_string(),
        ));
    }

    #[cfg(feature = "use-opencv")]
    let dst = {
        let mut d = Mat::default();
        imgproc::resize(
            img,
            &mut d,
            core::Size::new(resize_w, resize_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        d
    };
    
    #[cfg(not(feature = "use-opencv"))]
    let dst = {
        let mut d = Mat::default();
        image_impl::resize(
            img,
            &mut d,
            Size::new(resize_w, resize_h),
            INTER_LINEAR,
        )?;
        d
    };

    let ratio_h = h / resize_h as f32;
    let ratio_w = w / resize_w as f32;
    Ok((dst, ratio_h, ratio_w))
}

#[cfg(feature = "use-opencv")]
pub fn add_round_letterbox(
    img: &Mat,
    padding: (i32, i32, i32, i32),
) -> Result<Mat, EngineError> {
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
    )?;
    Ok(dst)
}

#[cfg(not(feature = "use-opencv"))]
pub fn add_round_letterbox(
    img: &Mat,
    padding: (i32, i32, i32, i32),
) -> Result<Mat, EngineError> {
    use image::{RgbImage, Rgb};
    
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    
    let new_width = width + padding.2 as u32 + padding.3 as u32;
    let new_height = height + padding.0 as u32 + padding.1 as u32;
    
    let mut new_img = RgbImage::from_pixel(new_width, new_height, Rgb([0, 0, 0]));
    
    // Copy original image to center
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            new_img.put_pixel(x + padding.3 as u32, y + padding.0 as u32, *pixel);
        }
    }
    
    Ok(Mat::new(image::DynamicImage::ImageRgb8(new_img)))
}

