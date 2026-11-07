use std::time::Instant;
use crate::tensor::{Array3, Array4, ArrayD, Axis};
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::{self, Mat, Size, INTER_LINEAR, Point2f};
use crate::types::OrientConfig;

#[derive(Debug, Clone, nanoserde::DeJson, nanoserde::SerJson)]
pub struct OrientOutput {
    pub orientation: Orientation,
    pub confidence: f32,
    pub elapse: f64,
}

/// Orientation classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq, nanoserde::DeJson, nanoserde::SerJson)]
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
            match class_id {
                0 => Orientation::Normal,
                1 => Orientation::Rotate180,
                _ => Orientation::Normal,
            }
        } else {
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
            Orientation::Rotate90 => -90,
            Orientation::Rotate180 => 180,
            Orientation::Rotate270 => -270,
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

    /// Rotate an image to match orientation (pure-Rust).
    pub fn rotate_image(&self, img: &Mat) -> Result<Mat, CoreError> {
        match self {
            Orientation::Normal => Ok(img.clone()),
            Orientation::Rotate90 => image::rotate_270(img),
            Orientation::Rotate180 => image::rotate_180(img),
            Orientation::Rotate270 => image::rotate_90(img),
        }
    }
}

/// Resize and normalize an image for orientation classification (pure-Rust path).
pub fn preprocess_orient_image(
    image: &Mat,
    target_w: i32,
    target_h: i32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Array4<f32>, CoreError> {
    let mut resized = Mat::default();
    image::resize(image, &mut resized, Size::new(target_w, target_h), INTER_LINEAR)?;

    let hw = target_h as usize * target_w as usize;
    let mut data = vec![0.0f32; 3 * hw];

    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = y * target_w as usize + x;
            data[idx] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            data[hw + idx] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            data[2 * hw + idx] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let arr = Array3::from_shape_vec((3, target_h as usize, target_w as usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok(arr.insert_axis(Axis(0)))
}

/// Resize and convert to single-channel ink map for Indra orientation classification.
pub fn preprocess_indra_orient_image(
    image: &Mat,
    target_side: i32,
) -> Result<Array4<f32>, CoreError> {
    let side = target_side.max(1);
    let max_dim = (image.cols().max(image.rows())) as f32;
    let s = side as f32 / max_dim;
    let new_w = (image.cols() as f32 * s).round().max(1.0) as i32;
    let new_h = (image.rows() as f32 * s).round().max(1.0) as i32;

    let mut resized = Mat::default();
    image::resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;

    let side_usize = side as usize;
    let mut data = vec![0.0f32; side_usize * side_usize];

    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let px = resized.get_pixel(x as u32, y as u32);
            let gray = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            data[y * side_usize + x] = 1.0 - (gray / 255.0);
        }
    }

    let arr = Array3::from_shape_vec((1, side_usize, side_usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok(arr.insert_axis(Axis(0)))
}

/// Postprocess model output logits into an orientation and confidence score.
pub fn postprocess_orient_logits(logits: &[f32]) -> Result<(Orientation, f32), CoreError> {
    if logits.is_empty() {
        return Err(CoreError::Postprocess("No logits found".to_string()));
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect();

    let (class_id, &confidence) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| CoreError::Postprocess("No class found".to_string()))?;

    let orientation = Orientation::from_class(class_id, logits.len());
    Ok((orientation, confidence))
}

/// Postprocess model output dynamic tensor into an orientation and confidence score.
pub fn postprocess_orient_tensor(output: &ArrayD<f32>) -> Result<(Orientation, f32), CoreError> {
    let logits_slice = if output.shape().len() >= 2 {
        output.row(0)
    } else {
        output.as_slice()
    };
    if logits_slice.is_empty() {
        return Err(CoreError::Postprocess("Non-contiguous or empty logits buffer".to_string()));
    }
    postprocess_orient_logits(logits_slice)
}

/// Orientation classifier pipeline component.
pub struct OrientClassifier<S = Box<dyn InferenceSession>> {
    pub session: S,
    pub config: OrientConfig,
}

impl<S: InferenceSession> OrientClassifier<S> {
    pub fn with_session(config: OrientConfig, session: S) -> Self {
        Self { session, config }
    }

    pub fn classify(&mut self, image: &Mat) -> Result<OrientOutput, CoreError> {
        let start = Instant::now();
        let [_, h, w] = self.config.orient_image_shape;

        let input_tensor = preprocess_orient_image(image, w, h, self.config.mean, self.config.std)?;
        let output = self.session.run(input_tensor.into_dyn())?;
        let (orientation, confidence) = postprocess_orient_tensor(&output)?;

        Ok(OrientOutput {
            orientation,
            confidence,
            elapse: start.elapsed().as_secs_f64(),
        })
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
        Orientation::Rotate90.rotate_box_points_back(&mut pts, 200.0, 100.0);
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
        Orientation::Rotate270.rotate_box_points_back(&mut pts, 200.0, 100.0);
        assert_eq!(pts[0].x, 10.0);
        assert_eq!(pts[0].y, 179.0);
    }
}

