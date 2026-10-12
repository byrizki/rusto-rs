use std::time::Instant;
use ndarray::{Array3, Array4};

#[cfg(feature = "use-opencv")]
use opencv::core::Point2f;
#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Point2f};

use crate::engine::{EngineError, EngineSession, InferenceSession};
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

    /// Rotate bounding box points from corrected image space back to original image space
    pub fn rotate_box_points_back(&self, box_points: &mut [Point2f; 4], corr_w: f32, corr_h: f32) {
        match self {
            Orientation::Normal => {}
            Orientation::Rotate90 => {
                for p in box_points.iter_mut() {
                    let old_x = p.x;
                    let old_y = p.y;
                    p.x = corr_h - 1.0 - old_y;
                    p.y = old_x;
                }
            }
            Orientation::Rotate180 => {
                for p in box_points.iter_mut() {
                    p.x = corr_w - 1.0 - p.x;
                    p.y = corr_h - 1.0 - p.y;
                }
            }
            Orientation::Rotate270 => {
                for p in box_points.iter_mut() {
                    let old_x = p.x;
                    let old_y = p.y;
                    p.x = old_y;
                    p.y = corr_w - 1.0 - old_x;
                }
            }
        }
    }
    
    /// Apply rotation to correct the image orientation
    #[cfg(feature = "use-opencv")]
    pub fn rotate_image(&self, img: &Mat) -> Result<Mat, EngineError> {
        use opencv::core::Mat as CvMat;

        match self {
            Orientation::Normal => Ok(img.clone()),
            Orientation::Rotate90 => {
                let mut rotated = CvMat::default();
                opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_90_COUNTERCLOCKWISE)?;
                Ok(rotated)
            }
            Orientation::Rotate180 => {
                let mut rotated = CvMat::default();
                opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_180)?;
                Ok(rotated)
            }
            Orientation::Rotate270 => {
                let mut rotated = CvMat::default();
                opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_90_CLOCKWISE)?;
                Ok(rotated)
            }
        }
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
    session: EngineSession,
    pub config: OrientConfig,  // Public to allow access to confidence_threshold
}

impl OrientClassifier {
    pub fn new(config: OrientConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    /// Construct orientation classifier with a custom inference session
    pub fn with_custom_session(config: OrientConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
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
            let (_orig_h, _orig_w) = (image.size()?.height, image.size()?.width);
            
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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| EngineError::OutputError("No class found".to_string()))?;

        let orientation = Orientation::from_class(class_id, num_classes);

        Ok((orientation, confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotate_box_points_back_90() {
        let mut pts = [
            Point2f::new(20.0, 10.0),
            Point2f::new(50.0, 10.0),
            Point2f::new(50.0, 30.0),
            Point2f::new(20.0, 30.0),
        ];
        // corrected width = 200, height = 100
        Orientation::Rotate90.rotate_box_points_back(&mut pts, 200.0, 100.0);
        // x_orig = corr_h - 1 - y = 100 - 1 - y = 99 - y
        // y_orig = x
        assert_eq!(pts[0].x, 89.0);
        assert_eq!(pts[0].y, 20.0);
        assert_eq!(pts[1].x, 89.0);
        assert_eq!(pts[1].y, 50.0);
    }

    #[test]
    fn test_rotate_box_points_back_180() {
        let mut pts = [
            Point2f::new(10.0, 20.0),
            Point2f::new(100.0, 20.0),
            Point2f::new(100.0, 50.0),
            Point2f::new(10.0, 50.0),
        ];
        Orientation::Rotate180.rotate_box_points_back(&mut pts, 200.0, 100.0);
        assert_eq!(pts[0].x, 189.0);
        assert_eq!(pts[0].y, 79.0);
    }

    #[test]
    fn test_rotate_box_points_back_270() {
        let mut pts = [
            Point2f::new(20.0, 10.0),
            Point2f::new(50.0, 10.0),
            Point2f::new(50.0, 30.0),
            Point2f::new(20.0, 30.0),
        ];
        // corr_w = 200, corr_h = 100
        // x_orig = y, y_orig = corr_w - 1 - x = 199 - x
        Orientation::Rotate270.rotate_box_points_back(&mut pts, 200.0, 100.0);
        assert_eq!(pts[0].x, 10.0);
        assert_eq!(pts[0].y, 179.0);
    }
}
