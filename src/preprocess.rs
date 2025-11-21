use ndarray::Array4;

#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(feature = "use-opencv")]
use opencv::core::Mat;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::EngineError;

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

    pub fn run(&self, img: &Mat) -> Result<Array4<f32>, EngineError> {
        let resized = self.resize(img)?;
        self.normalize_and_permute(&resized)
    }

    fn resize(&self, img: &Mat) -> Result<Mat, EngineError> {
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
            return Err(EngineError::Preprocess("resize_h or resize_w <= 0".to_string()));
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
            crate::image_impl::resize(
                img,
                &mut d,
                crate::image_impl::Size::new(resize_w, resize_h),
                crate::image_impl::INTER_LINEAR,
            )?;
            d
        };

        Ok(dst)
    }

    #[cfg(feature = "use-opencv")]
    fn normalize_and_permute(&self, img: &Mat) -> Result<Array4<f32>, EngineError> {
        let size = img.size()?;
        let h = size.height as usize;
        let w = size.width as usize;

        let mut out = Array4::<f32>::zeros((1, 3, h, w));
        let scale = 1.0 / 255.0;

        for y in 0..h {
            for x in 0..w {
                let pix = img.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                let b = pix[0] as f32 * scale;
                let g = pix[1] as f32 * scale;
                let r = pix[2] as f32 * scale;

                out[[0, 0, y, x]] = (b - self.mean[0]) / self.std[0];
                out[[0, 1, y, x]] = (g - self.mean[1]) / self.std[1];
                out[[0, 2, y, x]] = (r - self.mean[2]) / self.std[2];
            }
        }

        Ok(out)
    }
    
    #[cfg(not(feature = "use-opencv"))]
    fn normalize_and_permute(&self, img: &Mat) -> Result<Array4<f32>, EngineError> {
        let size = img.size()?;
        let h = size.height as usize;
        let w = size.width as usize;

        let mut out = ndarray::Array4::<f32>::zeros((1, 3, h, w));
        let scale = 1.0 / 255.0;
        
        // Cache normalization parameters
        let mean_b = self.mean[0];
        let mean_g = self.mean[1];
        let mean_r = self.mean[2];
        let std_b = self.std[0];
        let std_g = self.std[1];
        let std_r = self.std[2];

        for y in 0..h {
            for x in 0..w {
                let pix = img.get_pixel(x as u32, y as u32);
                // CRITICAL: image crate loads as RGB, but OpenCV uses BGR
                // Model was trained on BGR, so we must convert RGB -> BGR
                let r = pix[0] as f32 * scale;  // Red channel
                let g = pix[1] as f32 * scale;  // Green channel
                let b = pix[2] as f32 * scale;  // Blue channel

                // Store in BGR order to match OpenCV
                out[[0, 0, y, x]] = (b - mean_b) / std_b;  // Blue
                out[[0, 1, y, x]] = (g - mean_g) / std_g;  // Green  
                out[[0, 2, y, x]] = (r - mean_r) / std_r;  // Red
            }
        }

        Ok(out)
    }
}

