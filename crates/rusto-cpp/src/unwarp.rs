//! Document unwarping operations accelerated with OpenCV

#[cfg(feature = "opencv")]
use ndarray::{Array3, Array4, ArrayD};
#[cfg(feature = "opencv")]
use opencv::{core, imgproc, prelude::*};
#[cfg(feature = "opencv")]
use opencv::core::Mat;
#[cfg(feature = "opencv")]
use rusto_core::CoreError;

#[cfg(feature = "opencv")]
pub struct UnwarpOutput {
    pub unwarped_image: Mat,
    pub elapse: f64,
}

#[cfg(feature = "opencv")]
pub fn preprocess_unwarp(image: &Mat, target_h: i32, target_w: i32) -> Result<Array4<f32>, CoreError> {
    let mut resized = Mat::default();
    imgproc::resize(
        image,
        &mut resized,
        core::Size::new(target_w, target_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    ).map_err(|e| CoreError::Preprocess(e.to_string()))?;

    let rows = resized.rows() as usize;
    let cols = resized.cols() as usize;
    let hw = rows * cols;

    let mut data = vec![0.0f32; 3 * hw];

    for i in 0..rows {
        for j in 0..cols {
            let pix = resized.at_2d::<core::Vec3b>(i as i32, j as i32)
                .map_err(|e| CoreError::Preprocess(e.to_string()))?;
            let idx = i * cols + j;
            data[idx] = pix[0] as f32 / 255.0;
            data[hw + idx] = pix[1] as f32 / 255.0;
            data[2 * hw + idx] = pix[2] as f32 / 255.0;
        }
    }

    let arr = Array3::from_shape_vec((3, target_h as usize, target_w as usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;

    Ok(arr.insert_axis(ndarray::Axis(0)))
}

#[cfg(feature = "opencv")]
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
        let (out_h, out_w) = if shape.len() == 4 {
            (shape[2] as i32, shape[3] as i32)
        } else {
            (shape[1] as i32, shape[2] as i32)
        };

        let mut unwarped_data = vec![0u8; (out_h * out_w * 3) as usize];

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

        let unwarped = Mat::from_slice_rows_cols(&unwarped_data, out_h, out_w)
            .map_err(|e| CoreError::Postprocess(e.to_string()))?;

        if out_h != orig_h || out_w != orig_w {
            let mut resized = Mat::default();
            imgproc::resize(
                &unwarped,
                &mut resized,
                core::Size::new(orig_w, orig_h),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            ).map_err(|e| CoreError::Postprocess(e.to_string()))?;
            Ok(resized)
        } else {
            Ok(unwarped)
        }
    } else {
        // Fallback to clone if grid mapping
        Ok(src_image.clone())
    }
}

