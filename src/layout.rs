use std::time::Instant;
use ndarray::{Array3, Array4};

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{EngineError, MnnSession};
use crate::types::LayoutConfig;

#[derive(Debug, Clone)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

/// Layout element types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
    Text = 0,
    Title = 1,
    Figure = 2,
    FigureCaption = 3,
    Table = 4,
    TableCaption = 5,
    Header = 6,
    Footer = 7,
    Reference = 8,
    Equation = 9,
}

impl LayoutType {
    pub fn from_class(class_id: usize) -> Option<Self> {
        match class_id {
            0 => Some(LayoutType::Text),
            1 => Some(LayoutType::Title),
            2 => Some(LayoutType::Figure),
            3 => Some(LayoutType::FigureCaption),
            4 => Some(LayoutType::Table),
            5 => Some(LayoutType::TableCaption),
            6 => Some(LayoutType::Header),
            7 => Some(LayoutType::Footer),
            8 => Some(LayoutType::Reference),
            9 => Some(LayoutType::Equation),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LayoutType::Text => "text",
            LayoutType::Title => "title",
            LayoutType::Figure => "figure",
            LayoutType::FigureCaption => "figure_caption",
            LayoutType::Table => "table",
            LayoutType::TableCaption => "table_caption",
            LayoutType::Header => "header",
            LayoutType::Footer => "footer",
            LayoutType::Reference => "reference",
            LayoutType::Equation => "equation",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayoutRegion {
    pub bbox: BBox,
    pub layout_type: LayoutType,
    pub confidence: f32,
}

pub struct LayoutOutput {
    pub regions: Vec<LayoutRegion>,
    pub elapse: f64,
}

pub struct LayoutDetector {
    session: MnnSession,
    config: LayoutConfig,
}

impl LayoutDetector {
    pub fn new(config: LayoutConfig) -> Result<Self, EngineError> {
        let session = MnnSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    pub fn detect(&mut self, image: &Mat) -> Result<LayoutOutput, EngineError> {
        let start = Instant::now();

        // Store original size
        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());
        
        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);

        // Preprocess
        let (input_tensor, scale_w, scale_h) = self.preprocess(image)?;

        // Run inference
        let output = self.session.run(input_tensor.into_dyn())?;

        // Postprocess
        let regions = self.postprocess(output, orig_w, orig_h, scale_w, scale_h)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(LayoutOutput { regions, elapse })
    }

    fn preprocess(&self, image: &Mat) -> Result<(Array4<f32>, f32, f32), EngineError> {
        #[cfg(feature = "use-opencv")]
        let (h, w) = (image.rows(), image.cols());
        
        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (h, w) = (size.height, size.width);

        // Layout model expects fixed size 640x640
        let new_h = 640;
        let new_w = 640;
        let ratio = 640.0 / h.max(w) as f32;

        #[cfg(feature = "use-opencv")]
        {
            use opencv::{core, imgproc};
            
            let mut resized = Mat::default();
            imgproc::resize(
                image,
                &mut resized,
                core::Size::new(new_w, new_h),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            // Normalize with ImageNet mean/std
            let mean = self.config.mean;
            let std = self.config.std;

            let mut float_img = Mat::default();
            resized.convert_to(&mut float_img, opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;

            let channels = float_img.channels() as usize;
            let rows = float_img.rows() as usize;
            let cols = float_img.cols() as usize;
            
            let mut data = vec![0.0f32; channels * rows * cols];
            let mat_data = float_img.data_bytes()?;
            
            for i in 0..rows {
                for j in 0..cols {
                    for c in 0..channels {
                        let idx = (i * cols + j) * channels + c;
                        let val = mat_data[idx * 4] as f32 / 255.0;
                        data[c * rows * cols + i * cols + j] = (val - mean[c]) / std[c];
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok((arr.insert_axis(ndarray::Axis(0)), ratio, ratio))
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            use crate::image_impl::{resize, Size, INTER_LINEAR};
            
            let mut resized = Mat::default();
            resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;
            
            let mean = self.config.mean;
            let std = self.config.std;
            
            let mut data = vec![0.0f32; 3 * new_h as usize * new_w as usize];
            
            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        data[c * (new_h as usize * new_w as usize) + y * (new_w as usize) + x] = 
                            (val - mean[c]) / std[c];
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;
            
            Ok((arr.insert_axis(ndarray::Axis(0)), ratio, ratio))
        }
    }

    fn postprocess(
        &self,
        output: ndarray::ArrayD<f32>,
        orig_w: i32,
        orig_h: i32,
        scale_w: f32,
        scale_h: f32,
    ) -> Result<Vec<LayoutRegion>, EngineError> {
        // Placeholder implementation
        // Full implementation would parse detection output and apply NMS
        // Output format depends on the specific layout model used
        
        Ok(Vec::new())
    }
}
