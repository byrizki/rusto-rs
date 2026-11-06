//! Preprocessing implementations accelerated with OpenCV

#[cfg(feature = "opencv")]
use ndarray::{Array3, Array4, ArrayD};
#[cfg(feature = "opencv")]
use opencv::{core, imgproc, prelude::*};
#[cfg(feature = "opencv")]
use opencv::core::Mat;
#[cfg(feature = "opencv")]
use rusto_core::CoreError;

#[cfg(feature = "opencv")]
pub struct DetPreProcess {
    pub limit_side_len: i32,
    pub limit_type: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

#[cfg(feature = "opencv")]
impl DetPreProcess {
    pub fn new(limit_side_len: i32, limit_type: String, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            limit_side_len,
            limit_type,
            mean,
            std,
        }
    }

    pub fn run(&self, img: &Mat) -> Result<Array4<f32>, CoreError> {
        let resized = self.resize(img)?;
        self.normalize_and_permute(&resized)
    }

    fn resize(&self, img: &Mat) -> Result<Mat, CoreError> {
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
            return Err(CoreError::Preprocess("resize_h or resize_w <= 0".to_string()));
        }

        let mut d = Mat::default();
        imgproc::resize(
            img,
            &mut d,
            core::Size::new(resize_w, resize_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        ).map_err(|e| CoreError::Preprocess(e.to_string()))?;

        Ok(d)
    }

    fn normalize_and_permute(&self, img: &Mat) -> Result<Array4<f32>, CoreError> {
        let size = img.size().map_err(|e| CoreError::Preprocess(e.to_string()))?;
        let h = size.height as usize;
        let w = size.width as usize;

        let mut out = Array4::<f32>::zeros((1, 3, h, w));
        let scale = 1.0 / 255.0;

        for y in 0..h {
            for x in 0..w {
                let pix = img.at_2d::<core::Vec3b>(y as i32, x as i32)
                    .map_err(|e| CoreError::Preprocess(e.to_string()))?;
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
}

/// Resize and normalize for orientation classification (OpenCV path).
#[cfg(feature = "opencv")]
pub fn preprocess_orient_image(
    image: &Mat,
    target_w: i32,
    target_h: i32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Array4<f32>, CoreError> {
    let mut resized = Mat::default();
    imgproc::resize(
        image, &mut resized, core::Size::new(target_w, target_h), 0.0, 0.0, imgproc::INTER_LINEAR,
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
            data[idx] = (pix[0] as f32 / 255.0 - mean[0]) / std[0];
            data[hw + idx] = (pix[1] as f32 / 255.0 - mean[1]) / std[1];
            data[2 * hw + idx] = (pix[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let arr = Array3::from_shape_vec((3, target_h as usize, target_w as usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok(arr.insert_axis(ndarray::Axis(0)))
}

/// Resize and normalize for PP-OCR text recognition (OpenCV path).
#[cfg(feature = "opencv")]
pub fn resize_norm_rec_img(
    img: &Mat,
    img_c: usize,
    img_h: usize,
    max_wh_ratio: f32,
) -> Result<Array3<f32>, CoreError> {
    let img_width = (img_h as f32 * max_wh_ratio).round() as i32;

    let h = img.rows();
    let w = img.cols();
    if h <= 0 || w <= 0 {
        return Err(CoreError::Preprocess("invalid image size".to_string()));
    }

    let ratio = w as f32 / h as f32;
    let resized_w = if ((img_h as f32) * ratio).ceil() as i32 > img_width {
        img_width
    } else {
        ((img_h as f32) * ratio).ceil() as i32
    };

    let mut resized = Mat::default();
    imgproc::resize(
        img, &mut resized, core::Size::new(resized_w, img_h as i32), 0.0, 0.0, imgproc::INTER_LINEAR,
    ).map_err(|e| CoreError::Preprocess(e.to_string()))?;

    let size = resized.size().map_err(|e| CoreError::Preprocess(e.to_string()))?;
    let h2 = size.height as usize;
    let w2 = size.width as usize;

    let mut out = Array3::<f32>::zeros((img_c, img_h, img_width as usize));
    for y in 0..h2 {
        for x in 0..w2.min(img_width as usize) {
            let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)
                .map_err(|e| CoreError::Preprocess(e.to_string()))?;
            out[[0, y, x]] = (pix[0] as f32 / 255.0 - 0.5) / 0.5;
            out[[1, y, x]] = (pix[1] as f32 / 255.0 - 0.5) / 0.5;
            out[[2, y, x]] = (pix[2] as f32 / 255.0 - 0.5) / 0.5;
        }
    }

    Ok(out)
}

/// Resize and normalize for layout detection (OpenCV path).
/// Returns `(tensor [1,3,new_h,new_w], scale_w, scale_h)`.
#[cfg(feature = "opencv")]
pub fn preprocess_layout_image(
    image: &Mat,
    new_w: i32,
    new_h: i32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(Array4<f32>, f32, f32), CoreError> {
    let (orig_h, orig_w) = (image.rows(), image.cols());
    let scale_w = new_w as f32 / orig_w as f32;
    let scale_h = new_h as f32 / orig_h as f32;

    let mut resized = Mat::default();
    imgproc::resize(
        image, &mut resized, core::Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR,
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
            for c in 0..3 {
                let val = pix[c] as f32 / 255.0;
                data[c * hw + idx] = (val - mean[c]) / std[c];
            }
        }
    }

    let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok((arr.insert_axis(ndarray::Axis(0)), scale_w, scale_h))
}

/// Resize and normalize for RT-DETR (640×640, ImageNet normalization, OpenCV path).
#[cfg(feature = "opencv")]
pub fn preprocess_rtdetr(
    image: &Mat,
) -> Result<(ArrayD<f32>, f32, f32, i32, i32), CoreError> {
    let (orig_h, orig_w) = (image.rows(), image.cols());
    let target = 640usize;
    let scale = target as f32 / orig_h.max(orig_w) as f32;
    let new_h = (orig_h as f32 * scale) as i32;
    let new_w = (orig_w as f32 * scale) as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        image, &mut resized, core::Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR,
    ).map_err(|e| CoreError::Preprocess(e.to_string()))?;

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let hw = target * target;
    let mut data = vec![0.0f32; 3 * hw];

    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)
                .map_err(|e| CoreError::Preprocess(e.to_string()))?;
            let idx = y * target + x;
            for c in 0..3 {
                let val = pix[c] as f32 / 255.0;
                data[c * hw + idx] = (val - mean[c]) / std[c];
            }
        }
    }

    let arr = Array3::from_shape_vec((3, target, target), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok((arr.insert_axis(ndarray::Axis(0)).into_dyn(), scale, scale, orig_w, orig_h))
}

/// Resize and normalize for SLANet (488×488, [0,1] range, OpenCV path).
#[cfg(feature = "opencv")]
pub fn preprocess_slanet(image: &Mat) -> Result<ArrayD<f32>, CoreError> {
    let (target_h, target_w) = (488i32, 488i32);
    let mut resized = Mat::default();
    imgproc::resize(
        image, &mut resized, core::Size::new(target_w, target_h), 0.0, 0.0, imgproc::INTER_LINEAR,
    ).map_err(|e| CoreError::Preprocess(e.to_string()))?;

    let hw = (target_h * target_w) as usize;
    let mut data = vec![0.0f32; 3 * hw];

    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)
                .map_err(|e| CoreError::Preprocess(e.to_string()))?;
            let idx = y * target_w as usize + x;
            for c in 0..3 {
                data[c * hw + idx] = pix[c] as f32 / 255.0;
            }
        }
    }

    let arr = Array3::from_shape_vec((3, target_h as usize, target_w as usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok(arr.insert_axis(ndarray::Axis(0)).into_dyn())
}

#[cfg(feature = "opencv")]
pub fn prepare_det_input(
    img: &Mat,
    _is_linenet: bool,
    params: &rusto_core::preprocess::ResolvedDetParams,
    _ori_w: i32,
    _ori_h: i32,
) -> Result<(Array4<f32>, f32, f32), CoreError> {
    let pre = DetPreProcess::new(
        params.limit_side_len,
        params.limit_type.clone(),
        params.mean,
        params.std,
    );
    let input = pre.run(img)?;
    Ok((input, 1.0, 1.0))
}

#[cfg(feature = "opencv")]
pub fn build_linenet_rec_batch(
    img_list: &[Mat],
    chunk: &[usize],
    target_widths: &[usize],
    rec_h: usize,
) -> Result<(Array4<f32>, Vec<i32>), CoreError> {
    let n = chunk.len();
    let max_w = chunk.iter().map(|&idx| target_widths[idx]).max().unwrap_or(rec_h);
    let mut batch_tensor = Array4::<f32>::zeros((n, 1, rec_h, max_w));
    let mut widths = Vec::with_capacity(n);

    for (local_i, &idx) in chunk.iter().enumerate() {
        let w = target_widths[idx];
        widths.push(w as i32);
        let img = &img_list[idx];
        let mut resized = Mat::default();
        imgproc::resize(
            img,
            &mut resized,
            core::Size::new(w as i32, rec_h as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        ).map_err(|e| CoreError::Preprocess(e.to_string()))?;
        for y in 0..rec_h {
            for x in 0..w {
                let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)
                    .map_err(|e| CoreError::Preprocess(e.to_string()))?;
                let gray = 0.299 * pix[0] as f32 + 0.587 * pix[1] as f32 + 0.114 * pix[2] as f32;
                let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                batch_tensor[[local_i, 0, y, x]] = ink;
            }
        }
    }

    Ok((batch_tensor, widths))
}

