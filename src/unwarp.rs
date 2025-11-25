use std::time::Instant;
use ndarray::{Array3, Array4};

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{EngineError, MnnSession};
use crate::types::UnwarpConfig;

pub struct UnwarpOutput {
    pub unwarped_image: Mat,
    pub elapse: f64,
}

pub struct DocUnwarper {
    session: MnnSession,
}

impl DocUnwarper {
    pub fn new(config: UnwarpConfig) -> Result<Self, EngineError> {
        let session = MnnSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session })
    }

    pub fn unwarp(&mut self, image: &Mat) -> Result<UnwarpOutput, EngineError> {
        let start = Instant::now();

        // Store original size
        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());
        
        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);

        // Preprocess image
        let input_tensor = self.preprocess(image)?;

        // Run inference - UVDoc outputs the unwarped image directly
        let output = self.session.run(input_tensor.into_dyn())?;

        // Postprocess - convert model output to BGR image
        let unwarped_image = self.postprocess(output, orig_w, orig_h)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(UnwarpOutput {
            unwarped_image,
            elapse,
        })
    }

    fn preprocess(&self, image: &Mat) -> Result<Array4<f32>, EngineError> {
        // UVDoc does NOT resize - processes at original size!
        // Only normalize to [0, 1] and convert to CHW format
        
        #[cfg(feature = "use-opencv")]
        {
            use opencv::core;
            
            // Get original image size
            let rows = image.rows() as usize;
            let cols = image.cols() as usize;
            
            let mut data = vec![0.0f32; 3 * rows * cols];
            let scale = 1.0 / 255.0;
            
            // Read uint8 BGR values and normalize - NO RESIZE
            for y in 0..rows {
                for x in 0..cols {
                    let pixel = image.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                    // pixel is BGR [B, G, R], normalize and store in CHW format
                    data[0 * rows * cols + y * cols + x] = pixel[0] as f32 * scale; // B
                    data[1 * rows * cols + y * cols + x] = pixel[1] as f32 * scale; // G
                    data[2 * rows * cols + y * cols + x] = pixel[2] as f32 * scale; // R
                }
            }

            let arr = Array3::from_shape_vec((3, rows, cols), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok(arr.insert_axis(ndarray::Axis(0)))
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            // Get original image size - NO RESIZE
            let size = image.size()?;
            let rows = size.height as usize;
            let cols = size.width as usize;
            
            let mut data = vec![0.0f32; 3 * rows * cols];
            
            for y in 0..rows {
                for x in 0..cols {
                    let pixel = image.get_pixel(x as u32, y as u32);
                    data[0 * (rows * cols) + y * cols + x] = pixel[0] as f32 / 255.0;
                    data[1 * (rows * cols) + y * cols + x] = pixel[1] as f32 / 255.0;
                    data[2 * (rows * cols) + y * cols + x] = pixel[2] as f32 / 255.0;
                }
            }

            let arr = Array3::from_shape_vec((3, rows, cols), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok(arr.insert_axis(ndarray::Axis(0)))
        }
    }

    fn postprocess(
        &self,
        output: ndarray::ArrayD<f32>,
        _orig_w: i32,
        _orig_h: i32,
    ) -> Result<Mat, EngineError> {
        // UVDoc outputs the unwarped image directly at ORIGINAL size
        // Output shape: [batch, 3, H, W] in [0, 1] range, RGB format
        // We need to: squeeze, transpose CHW->HWC, scale to [0,255], flip RGB->BGR, convert to uint8
        // NO RESIZE - output is already at original size!
        
        let shape = output.shape();
        if shape.len() != 4 {
            return Err(EngineError::OutputError(format!(
                "Expected 4D output [batch, channels, height, width], got shape: {:?}", shape
            )));
        }
        
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        
        if batch != 1 || channels != 3 {
            return Err(EngineError::OutputError(format!(
                "Expected [1, 3, H, W], got [{}, {}, {}, {}]", batch, channels, height, width
            )));
        }
        
        #[cfg(feature = "use-opencv")]
        {
            use opencv::core;
            
            // Convert from [1, 3, H, W] CHW RGB [0,1] to HWC BGR [0,255] uint8
            let mut unwarped_data = vec![0u8; (height * width * 3) as usize];
            
            for y in 0..height {
                for x in 0..width {
                    // Get RGB values from CHW format and scale to [0, 255]
                    let r = (*output.get([0, 0, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8;
                    let g = (*output.get([0, 1, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8;
                    let b = (*output.get([0, 2, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8;
                    
                    // Convert to BGR (flip RGB -> BGR) and write in HWC format
                    let idx = (y * width + x) * 3;
                    unwarped_data[idx as usize] = b;     // B
                    unwarped_data[idx as usize + 1] = g; // G
                    unwarped_data[idx as usize + 2] = r; // R
                }
            }
            
            // Create Mat from data - NO RESIZE, already at original size
            let unwarped = Mat::from_slice_rows_cols(
                &unwarped_data,
                height as i32,
                width as i32,
            )?;
            
            Ok(unwarped)
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            use image::{ImageBuffer, Rgb};
            
            // Convert from [1, 3, H, W] CHW RGB [0,1] to HWC RGB [0,255] uint8
            // NO RESIZE - output is already at original size!
            let mut unwarped_img = ImageBuffer::new(width as u32, height as u32);
            
            for y in 0..height {
                for x in 0..width {
                    // Get RGB values from CHW format and scale to [0, 255]
                    let r = (*output.get([0, 0, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8;
                    let g = (*output.get([0, 1, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8;
                    let b = (*output.get([0, 2, y, x]).unwrap() * 255.0).clamp(0.0, 255.0) as u8;
                    
                    // Store as RGB (model outputs RGB, image crate uses RGB)
                    unwarped_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
            
            Ok(Mat::new(image::DynamicImage::ImageRgb8(unwarped_img)))
        }
    }
}
