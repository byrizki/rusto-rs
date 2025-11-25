use std::time::Instant;
use ndarray::{Array3, Array4};

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{EngineError, MnnSession};
use crate::types::OrientConfig;

/// Orientation classification result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orientation {
    /// 0 degrees (normal)
    Normal = 0,
    /// 90 degrees clockwise
    Rotate90 = 1,
    /// 180 degrees
    Rotate180 = 2,
    /// 270 degrees clockwise (90 counter-clockwise)
    Rotate270 = 3,
}

impl Orientation {
    pub fn from_class(class_id: usize, num_classes: usize) -> Self {
        if num_classes == 2 {
            // Binary classification (0 vs 180) common in PPOCR
            match class_id {
                0 => Orientation::Normal,
                1 => Orientation::Rotate180,
                _ => Orientation::Normal,
            }
        } else {
            // 4-class classification (0, 90, 180, 270)
            match class_id {
                0 => Orientation::Normal,
                1 => Orientation::Rotate90,
                2 => Orientation::Rotate180,
                3 => Orientation::Rotate270,
                _ => Orientation::Normal,
            }
        }
    }

    pub fn degrees(&self) -> i32 {
        match self {
            Orientation::Normal => 0,
            Orientation::Rotate90 => -90,  // Counter-clockwise for image correction
            Orientation::Rotate180 => 180,
            Orientation::Rotate270 => -270,  // 90 clockwise for image correction
        }
    }
    
    /// Apply rotation to correct the image orientation
    #[cfg(feature = "use-opencv")]
    pub fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError> {
        use opencv::core::{Mat as CvMat, Point2f, BORDER_CONSTANT, Scalar};
        use opencv::imgproc::{get_rotation_matrix_2d, warp_affine, INTER_LINEAR};
        
        let degrees = self.degrees();
        if degrees == 0 {
            return Ok(img.clone());
        }
        
        let size = img.size()?;
        let center = Point2f::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
        let rot_mat = get_rotation_matrix_2d(center, degrees as f64, 1.0)?;
        let mut rotated = CvMat::default();
        warp_affine(
            img,
            &mut rotated,
            &rot_mat,
            size,
            INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar::all(255.0),
        )?;
        
        Ok(rotated)
    }
    
    /// Apply rotation to correct the image orientation (Pure Rust)
    #[cfg(not(feature = "use-opencv"))]
    pub fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError> {
        use crate::image_impl::{rotate_90, rotate_180, rotate_270};
        
        match self {
            Orientation::Normal => Ok(img.clone()),
            Orientation::Rotate90 => rotate_270(img)
                .map_err(|e| EngineError::Preprocess(e.to_string())),
            Orientation::Rotate180 => rotate_180(img)
                .map_err(|e| EngineError::Preprocess(e.to_string())),
            Orientation::Rotate270 => rotate_90(img)
                .map_err(|e| EngineError::Preprocess(e.to_string())),
        }
    }
}

pub struct OrientOutput {
    pub orientation: Orientation,
    pub confidence: f32,
    pub elapse: f64,
}

pub struct OrientClassifier {
    session: MnnSession,
    pub config: OrientConfig,  // Public to allow access to confidence_threshold
}

impl OrientClassifier {
    pub fn new(config: OrientConfig) -> Result<Self, EngineError> {
        let session = MnnSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    pub fn classify(&mut self, image: &Mat) -> Result<OrientOutput, EngineError> {
        let start = Instant::now();

        // Preprocess image
        let input_tensor = self.preprocess(image)?;

        // Run inference
        let output = self.session.run(input_tensor.into_dyn())?;

        // Postprocess
        let (orientation, confidence) = self.postprocess(output)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(OrientOutput {
            orientation,
            confidence,
            elapse,
        })
    }

    fn preprocess(&self, image: &Mat) -> Result<Array4<f32>, EngineError> {
        // Resize to model input size
        let [_, h, w] = self.config.orient_image_shape;
        
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

            // Convert to RGB float and normalize
            let mut float_img = Mat::default();
            resized.convert_to(&mut float_img, opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;

            // Convert to ndarray [1, 3, H, W]
            let channels = float_img.channels() as usize;
            let rows = float_img.rows() as usize;
            let cols = float_img.cols() as usize;
            
            let mut data = vec![0.0f32; channels * rows * cols];
            let mat_data = float_img.data_bytes()?;
            
            for i in 0..rows {
                for j in 0..cols {
                    for c in 0..channels {
                        let idx = (i * cols + j) * channels + c;
                        data[c * rows * cols + i * cols + j] = mat_data[idx * 4] as f32 / 255.0;
                    }
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
            let (_orig_h, _orig_w) = (image.size()?.height, image.size()?.width);
            
            // Convert to float and normalize [1, 3, H, W]
            let mut data = vec![0.0f32; 3 * h as usize * w as usize];
            
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    data[0 * (h as usize * w as usize) + y * w as usize + x] = pixel[0] as f32 / 255.0;
                    data[1 * (h as usize * w as usize) + y * w as usize + x] = pixel[1] as f32 / 255.0;
                    data[2 * (h as usize * w as usize) + y * w as usize + x] = pixel[2] as f32 / 255.0;
                }
            }

            let arr = Array3::from_shape_vec((3, h as usize, w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok(arr.insert_axis(ndarray::Axis(0)))
        }
    }

    fn postprocess(&self, output: ndarray::ArrayD<f32>) -> Result<(Orientation, f32), EngineError> {
        // Output shape can be [1, 2] (PPOCR) or [1, 4]
        let output_2d = output.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| EngineError::OutputError(format!("Invalid output shape: {}", e)))?;

        let logits = output_2d.row(0);
        let num_classes = logits.len();
        
        // Apply softmax and find max
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect();

        let (class_id, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| EngineError::OutputError("No class found".to_string()))?;

        let orientation = Orientation::from_class(class_id, num_classes);

        Ok((orientation, confidence))
    }
}
