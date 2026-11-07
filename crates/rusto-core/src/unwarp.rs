use std::time::Instant;
use crate::tensor::{Array3, Array4, ArrayD, Axis};
use ::image::{DynamicImage, ImageBuffer, Rgb};
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::{self, Mat, Size, INTER_LINEAR};
use crate::types::UnwarpConfig;

pub struct UnwarpOutput {
    pub unwarped_image: Mat,
    pub elapse: f64,
}

/// Preprocess image for document unwarping model by resizing and normalizing to [0, 1].
pub fn preprocess_unwarp(
    image: &Mat,
    target_h: usize,
    target_w: usize,
) -> Result<Array4<f32>, CoreError> {
    let mut resized = Mat::default();
    image::resize(
        image,
        &mut resized,
        Size::new(target_w as i32, target_h as i32),
        INTER_LINEAR,
    )?;

    let hw = target_h * target_w;
    let mut data = vec![0.0f32; 3 * hw];

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = y * target_w + x;
            data[idx] = pixel[0] as f32 / 255.0;
            data[hw + idx] = pixel[1] as f32 / 255.0;
            data[2 * hw + idx] = pixel[2] as f32 / 255.0;
        }
    }

    let arr = Array3::from_shape_vec((3, target_h, target_w), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;

    Ok(arr.insert_axis(Axis(0)))
}

/// Decode direct RGB unwarped image from model output tensor.
pub fn decode_direct_unwarp(
    output: &ArrayD<f32>,
    orig_w: i32,
    orig_h: i32,
) -> Result<Mat, CoreError> {
    let shape = output.shape();
    let (out_h, out_w) = if shape.len() == 4 {
        (shape[2] as i32, shape[3] as i32)
    } else {
        (shape[1] as i32, shape[2] as i32)
    };

    let mut unwarped_img = ImageBuffer::new(out_w as u32, out_h as u32);

    for y in 0..out_h as usize {
        for x in 0..out_w as usize {
            let r = if shape.len() == 4 {
                (*output.get([0, 0, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8
            } else {
                (*output.get([0, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8
            };
            let g = if shape.len() == 4 {
                (*output.get([0, 1, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8
            } else {
                (*output.get([1, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8
            };
            let b = if shape.len() == 4 {
                (*output.get([0, 2, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8
            } else {
                (*output.get([2, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8
            };

            unwarped_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    let dyn_img = DynamicImage::ImageRgb8(unwarped_img);
    let final_img = if out_h != orig_h || out_w != orig_w {
        dyn_img.resize_exact(
            orig_w as u32,
            orig_h as u32,
            ::image::imageops::FilterType::Triangle,
        )
    } else {
        dyn_img
    };

    Ok(Mat::new(final_img))
}

/// Remap source image using a predicted displacement/coordinate grid (e.g. UVDoc).
pub fn grid_sample_remap(
    src_image: &Mat,
    grid: &ArrayD<f32>,
    gh: usize,
    gw: usize,
    channels_first: bool,
    orig_w: i32,
    orig_h: i32,
) -> Result<Mat, CoreError> {
    let mut out_img = ImageBuffer::new(orig_w as u32, orig_h as u32);
    let scale_x = (gw - 1) as f32 / (orig_w - 1).max(1) as f32;
    let scale_y = (gh - 1) as f32 / (orig_h - 1).max(1) as f32;

    for y in 0..orig_h as usize {
        let gy = (y as f32 * scale_y).clamp(0.0, (gh - 1) as f32);
        let gy0 = gy.floor() as usize;
        let gy1 = (gy0 + 1).min(gh - 1);
        let wy = gy - gy0 as f32;

        for x in 0..orig_w as usize {
            let gx = (x as f32 * scale_x).clamp(0.0, (gw - 1) as f32);
            let gx0 = gx.floor() as usize;
            let gx1 = (gx0 + 1).min(gw - 1);
            let wx = gx - gx0 as f32;

            let sample_coord = |c: usize, r: usize| -> (f32, f32) {
                if channels_first {
                    let sx = if grid.ndim() == 4 {
                        *grid.get([0, 0, r, c]).unwrap_or(&0.0)
                    } else {
                        *grid.get([0, r, c]).unwrap_or(&0.0)
                    };
                    let sy = if grid.ndim() == 4 {
                        *grid.get([0, 1, r, c]).unwrap_or(&0.0)
                    } else {
                        *grid.get([1, r, c]).unwrap_or(&0.0)
                    };
                    (sx, sy)
                } else {
                    let sx = *grid.get([0, r, c, 0]).unwrap_or(&0.0);
                    let sy = *grid.get([0, r, c, 1]).unwrap_or(&0.0);
                    (sx, sy)
                }
            };

            let (c00_x, c00_y) = sample_coord(gx0, gy0);
            let (c10_x, c10_y) = sample_coord(gx1, gy0);
            let (c01_x, c01_y) = sample_coord(gx0, gy1);
            let (c11_x, c11_y) = sample_coord(gx1, gy1);

            let norm_x = (1.0 - wx) * (1.0 - wy) * c00_x
                + wx * (1.0 - wy) * c10_x
                + (1.0 - wx) * wy * c01_x
                + wx * wy * c11_x;

            let norm_y = (1.0 - wx) * (1.0 - wy) * c00_y
                + wx * (1.0 - wy) * c10_y
                + (1.0 - wx) * wy * c01_y
                + wx * wy * c11_y;

            let src_px = if norm_x.abs() <= 1.05 && norm_x >= -1.05 {
                if norm_x < 0.0 {
                    ((norm_x + 1.0) * 0.5 * (orig_w - 1) as f32).clamp(0.0, (orig_w - 1) as f32)
                } else {
                    (norm_x * (orig_w - 1) as f32).clamp(0.0, (orig_w - 1) as f32)
                }
            } else {
                norm_x.clamp(0.0, (orig_w - 1) as f32)
            };

            let src_py = if norm_y.abs() <= 1.05 && norm_y >= -1.05 {
                if norm_y < 0.0 {
                    ((norm_y + 1.0) * 0.5 * (orig_h - 1) as f32).clamp(0.0, (orig_h - 1) as f32)
                } else {
                    (norm_y * (orig_h - 1) as f32).clamp(0.0, (orig_h - 1) as f32)
                }
            } else {
                norm_y.clamp(0.0, (orig_h - 1) as f32)
            };

            let sx0 = src_px.floor() as u32;
            let sx1 = (sx0 + 1).min(orig_w as u32 - 1);
            let sy0 = src_py.floor() as u32;
            let sy1 = (sy0 + 1).min(orig_h as u32 - 1);

            let fx = src_px - sx0 as f32;
            let fy = src_py - sy0 as f32;

            let p00 = src_image.get_pixel(sx0, sy0);
            let p10 = src_image.get_pixel(sx1, sy0);
            let p01 = src_image.get_pixel(sx0, sy1);
            let p11 = src_image.get_pixel(sx1, sy1);

            let mut out_rgb = [0u8; 3];
            for c in 0..3 {
                let val = (1.0 - fx) * (1.0 - fy) * p00[c] as f32
                    + fx * (1.0 - fy) * p10[c] as f32
                    + (1.0 - fx) * fy * p01[c] as f32
                    + fx * fy * p11[c] as f32;
                out_rgb[c] = val.round().clamp(0.0, 255.0) as u8;
            }

            out_img.put_pixel(x as u32, y as u32, Rgb(out_rgb));
        }
    }

    Ok(Mat::new(DynamicImage::ImageRgb8(out_img)))
}

/// Unified postprocess dispatcher for unwarping models (handles both direct RGB and displacement grid).
pub fn postprocess_unwarp(
    src_image: &Mat,
    output: &ArrayD<f32>,
    orig_w: i32,
    orig_h: i32,
) -> Result<Mat, CoreError> {
    let shape = output.shape();

    let is_direct_image = (shape.len() == 4 && shape[0] == 1 && shape[1] == 3)
        || (shape.len() == 3 && shape[0] == 3);

    if is_direct_image {
        return decode_direct_unwarp(output, orig_w, orig_h);
    }

    let (grid_c, grid_h, grid_w, channels_first) = if shape.len() == 4 && shape[1] == 2 {
        (shape[1], shape[2], shape[3], true)
    } else if shape.len() == 3 && shape[0] == 2 {
        (shape[0], shape[1], shape[2], true)
    } else if shape.len() == 4 && shape[3] == 2 {
        (shape[3], shape[1], shape[2], false)
    } else {
        return Err(CoreError::Postprocess(format!(
            "Unexpected unwarp model output shape: {:?}",
            shape
        )));
    };

    if grid_c != 2 {
        return Err(CoreError::Postprocess(format!(
            "Expected 2 coordinate channels, got shape: {:?}",
            shape
        )));
    }

    grid_sample_remap(src_image, output, grid_h, grid_w, channels_first, orig_w, orig_h)
}

/// Document unwarping pipeline component.
pub struct DocUnwarper<S = Box<dyn InferenceSession>> {
    pub session: S,
    pub config: UnwarpConfig,
}

impl<S: InferenceSession> DocUnwarper<S> {
    pub fn with_session(config: UnwarpConfig, session: S) -> Self {
        Self { session, config }
    }

    pub fn unwarp(&mut self, image: &Mat) -> Result<UnwarpOutput, CoreError> {
        let start = Instant::now();
        let (orig_h, orig_w) = (image.rows(), image.cols());
        let [_, h, w] = self.config.unwarp_image_shape;

        let input_tensor = preprocess_unwarp(image, h as usize, w as usize)?;
        let output = self.session.run(input_tensor.into_dyn())?;
        let unwarped_image = postprocess_unwarp(image, &output, orig_w, orig_h)?;

        Ok(UnwarpOutput {
            unwarped_image,
            elapse: start.elapsed().as_secs_f64(),
        })
    }
}

