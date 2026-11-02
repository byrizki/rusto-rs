use std::time::Instant;
use ndarray::{Array3, Array4, ArrayD};

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{EngineError, EngineSession, InferenceSession};
use crate::types::UnwarpConfig;

pub struct UnwarpOutput {
    pub unwarped_image: Mat,
    pub elapse: f64,
}

pub struct DocUnwarper {
    session: EngineSession,
    pub config: UnwarpConfig,
}

impl DocUnwarper {
    pub fn new(config: UnwarpConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_unwarp_config(&config)?;
        Ok(Self { session, config })
    }

    /// Construct doc unwarper with a custom inference session
    pub fn with_custom_session(config: UnwarpConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
    }

    pub fn unwarp(&mut self, image: &Mat) -> Result<UnwarpOutput, EngineError> {
        let start = Instant::now();

        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());

        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);

        // Preprocess image
        let input_tensor = self.preprocess(image)?;

        // Run inference
        let output = self.session.run(input_tensor.into_dyn())?;

        // Postprocess - handles direct RGB or displacement grid
        let unwarped_image = self.postprocess(image, output, orig_w, orig_h)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(UnwarpOutput {
            unwarped_image,
            elapse,
        })
    }

    fn preprocess(&self, image: &Mat) -> Result<Array4<f32>, EngineError> {
        let [_, h, w] = self.config.unwarp_image_shape;

        #[cfg(feature = "use-opencv")]
        {
            use opencv::{core, imgproc};

            let mut resized = Mat::default();
            imgproc::resize(
                image,
                &mut resized,
                core::Size::new(w, h),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            let rows = resized.rows() as usize;
            let cols = resized.cols() as usize;
            let hw = rows * cols;

            let mut data = vec![0.0f32; 3 * hw];

            for i in 0..rows {
                for j in 0..cols {
                    let pix = resized.at_2d::<core::Vec3b>(i as i32, j as i32)?;
                    let idx = i * cols + j;
                    data[idx] = pix[0] as f32 / 255.0;
                    data[hw + idx] = pix[1] as f32 / 255.0;
                    data[2 * hw + idx] = pix[2] as f32 / 255.0;
                }
            }

            let arr = Array3::from_shape_vec((3, h as usize, w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok(arr.insert_axis(ndarray::Axis(0)))
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            use crate::image_impl::{resize, Size, INTER_LINEAR};

            let mut resized = Mat::default();
            resize(image, &mut resized, Size::new(w, h), INTER_LINEAR)?;

            let hw = h as usize * w as usize;
            let mut data = vec![0.0f32; 3 * hw];

            for y in 0..h as usize {
                for x in 0..w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let idx = y * w as usize + x;
                    data[idx] = pixel[0] as f32 / 255.0;
                    data[hw + idx] = pixel[1] as f32 / 255.0;
                    data[2 * hw + idx] = pixel[2] as f32 / 255.0;
                }
            }

            let arr = Array3::from_shape_vec((3, h as usize, w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok(arr.insert_axis(ndarray::Axis(0)))
        }
    }

    fn postprocess(
        &self,
        src_image: &Mat,
        output: ArrayD<f32>,
        orig_w: i32,
        orig_h: i32,
    ) -> Result<Mat, EngineError> {
        let shape = output.shape();

        // Check if output is a direct RGB image: [1, 3, H, W] or [3, H, W]
        let is_direct_image = (shape.len() == 4 && shape[0] == 1 && shape[1] == 3)
            || (shape.len() == 3 && shape[0] == 3);

        if is_direct_image {
            return self.decode_direct_image(output, orig_w, orig_h);
        }

        // Check if output is a 2D coordinate grid (UVDoc displacement map)
        // Shapes: [1, 2, gh, gw] or [2, gh, gw] or [1, gh, gw, 2]
        let (grid_c, grid_h, grid_w, channels_first) = if shape.len() == 4 && shape[1] == 2 {
            (shape[1], shape[2], shape[3], true)
        } else if shape.len() == 3 && shape[0] == 2 {
            (shape[0], shape[1], shape[2], true)
        } else if shape.len() == 4 && shape[3] == 2 {
            (shape[3], shape[1], shape[2], false)
        } else {
            return Err(EngineError::OutputError(format!(
                "Unexpected unwarp model output shape: {:?}",
                shape
            )));
        };

        if grid_c != 2 {
            return Err(EngineError::OutputError(format!(
                "Expected 2 coordinate channels, got shape: {:?}",
                shape
            )));
        }

        // Resample image using the coordinate grid
        self.grid_sample_remap(src_image, &output, grid_h, grid_w, channels_first, orig_w, orig_h)
    }

    fn decode_direct_image(
        &self,
        output: ArrayD<f32>,
        orig_w: i32,
        orig_h: i32,
    ) -> Result<Mat, EngineError> {
        let shape = output.shape();
        let (out_h, out_w) = if shape.len() == 4 {
            (shape[2] as i32, shape[3] as i32)
        } else {
            (shape[1] as i32, shape[2] as i32)
        };

        #[cfg(feature = "use-opencv")]
        {
            use opencv::{core, imgproc};

            let mut unwarped_data = vec![0u8; (out_h * out_w * 3) as usize];
            let hw = (out_h * out_w) as usize;

            let c_offset = if shape.len() == 4 { 1 } else { 0 };
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

                    let idx = (y * out_w as usize + x) * 3;
                    unwarped_data[idx] = b;
                    unwarped_data[idx + 1] = g;
                    unwarped_data[idx + 2] = r;
                }
            }

            let unwarped = Mat::from_slice_rows_cols(&unwarped_data, out_h, out_w)?;

            if out_h != orig_h || out_w != orig_w {
                let mut resized = Mat::default();
                imgproc::resize(
                    &unwarped,
                    &mut resized,
                    core::Size::new(orig_w, orig_h),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;
                Ok(resized)
            } else {
                Ok(unwarped)
            }
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            use image::{ImageBuffer, Rgb};

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

            let dyn_img = image::DynamicImage::ImageRgb8(unwarped_img);
            let final_img = if out_h != orig_h || out_w != orig_w {
                dyn_img.resize_exact(orig_w as u32, orig_h as u32, image::imageops::FilterType::Triangle)
            } else {
                dyn_img
            };

            Ok(Mat::new(final_img))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn grid_sample_remap(
        &self,
        src_image: &Mat,
        grid: &ArrayD<f32>,
        gh: usize,
        gw: usize,
        channels_first: bool,
        orig_w: i32,
        orig_h: i32,
    ) -> Result<Mat, EngineError> {
        #[cfg(not(feature = "use-opencv"))]
        {
            use image::{ImageBuffer, Rgb};

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
                            let sx = if grid.ndim() == 4 { *grid.get([0, 0, r, c]).unwrap_or(&0.0) } else { *grid.get([0, r, c]).unwrap_or(&0.0) };
                            let sy = if grid.ndim() == 4 { *grid.get([0, 1, r, c]).unwrap_or(&0.0) } else { *grid.get([1, r, c]).unwrap_or(&0.0) };
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

                    // Support normalized [-1, 1], [0, 1], or absolute coordinates
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

                    // Bilinear interpolate from source Mat
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

            Ok(Mat::new(image::DynamicImage::ImageRgb8(out_img)))
        }

        #[cfg(feature = "use-opencv")]
        {
            // For OpenCV, fallback to cloned source if no direct grid mapping configured
            Ok(src_image.clone())
        }
    }
}
