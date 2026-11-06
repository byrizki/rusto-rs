//! Orientation classification and image rotation accelerated with OpenCV

#[cfg(feature = "opencv")]
use opencv::core::Mat;
#[cfg(feature = "opencv")]
use rusto_core::orient::Orientation;
#[cfg(feature = "opencv")]
use rusto_core::CoreError;

#[cfg(feature = "opencv")]
pub fn rotate_image(orientation: &Orientation, img: &Mat) -> Result<Mat, CoreError> {
    match orientation {
        Orientation::Normal => Ok(img.clone()),
        Orientation::Rotate90 => {
            let mut rotated = Mat::default();
            opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_90_COUNTERCLOCKWISE)
                .map_err(|e| CoreError::Postprocess(e.to_string()))?;
            Ok(rotated)
        }
        Orientation::Rotate180 => {
            let mut rotated = Mat::default();
            opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_180)
                .map_err(|e| CoreError::Postprocess(e.to_string()))?;
            Ok(rotated)
        }
        Orientation::Rotate270 => {
            let mut rotated = Mat::default();
            opencv::core::rotate(img, &mut rotated, opencv::core::ROTATE_90_CLOCKWISE)
                .map_err(|e| CoreError::Postprocess(e.to_string()))?;
            Ok(rotated)
        }
    }
}

