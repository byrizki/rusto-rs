use ndarray::{Array2, Array3, Array4};
use std::time::Instant;

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

        // Prepare inputs
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("image".to_string(), input_tensor.into_dyn());

        // im_shape should be [H, W] in the resized space
        let target_size = self.config.target_size as f32;
        let im_shape = Array2::from_shape_vec((1, 2), vec![target_size, target_size])
            .map_err(|e| EngineError::ShapeError(e))?;
        inputs.insert("im_shape".to_string(), im_shape.into_dyn());

        let scale_factor = Array2::from_shape_vec((1, 2), vec![scale_h, scale_w])
            .map_err(|e| EngineError::ShapeError(e))?;
        inputs.insert("scale_factor".to_string(), scale_factor.into_dyn());

        // Run inference
        let outputs = self.session.run_with_inputs(inputs)?;

        // Postprocess
        let regions = self.postprocess(outputs, orig_w, orig_h, scale_w, scale_h)?;

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

        // Layout model expects fixed size (e.g. 640x640 or 800x800)
        let target_size = self.config.target_size;
        let new_h = target_size;
        let new_w = target_size;
        let ratio = target_size as f32 / h.max(w) as f32;

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
        outputs: std::collections::HashMap<String, ndarray::ArrayD<f32>>,
        _orig_w: i32,
        _orig_h: i32,
        scale_w: f32,
        scale_h: f32,
    ) -> Result<Vec<LayoutRegion>, EngineError> {
        // Find the output tensor containing boxes
        // Usually it's the one with shape [N, 6]
        let mut boxes_tensor = None;
        for (_name, tensor) in &outputs {
            let shape = tensor.shape();
            if shape.len() == 2 && shape[1] == 6 {
                boxes_tensor = Some(tensor);
                break;
            }
        }

        let boxes_tensor = match boxes_tensor {
            Some(t) => t,
            None => {
                return Err(EngineError::OutputError(
                    "No suitable output tensor found".to_string(),
                ))
            }
        };

        let mut raw_boxes = Vec::new();
        for row in boxes_tensor.outer_iter() {
            let class_id = row[0] as usize;
            let score = row[1];
            let x1 = row[2];
            let y1 = row[3];
            let x2 = row[4];
            let y2 = row[5];

            raw_boxes.push([class_id as f32, score, x1, y1, x2, y2]);
        }

        // Filter by confidence (handled by NMS or manual filter)
        // Assuming model output is already filtered or we filter here
        // NMS
        let indices = crate::geometry::nms(&raw_boxes, 0.6, 0.95);

        let mut regions = Vec::new();
        for idx in indices {
            let row = &raw_boxes[idx];
            let class_id = row[0] as usize;
            let score = row[1];
            let x1 = row[2];
            let y1 = row[3];
            let x2 = row[4];
            let y2 = row[5];

            // Map class ID to LayoutType
            // Mapping based on PP-DocLayout_plus-L
            // 0: Title, 1: Text, 2: Abandon, 3: Figure, 4: FigureCaption, 5: Table, 6: TableCaption,
            // 7: TableFootnote, 8: IsolateFormula, 9: FormulaCaption, 13: InlineFormula, 14: IsolatedFormula, 15: OcrText
            let layout_type = match class_id {
                0 => LayoutType::Title,
                1 | 15 => LayoutType::Text,
                2 => continue, // Abandon
                3 => LayoutType::Figure,
                4 => LayoutType::FigureCaption,
                5 => LayoutType::Table,
                6 => LayoutType::TableCaption,
                7 => LayoutType::Reference, // TableFootnote -> Reference?
                8 | 13 | 14 => LayoutType::Equation,
                9 => LayoutType::Text, // FormulaCaption -> Text
                _ => LayoutType::Text, // Default to Text
            };

            // Scale back to original image and clip to bounds
            let x_min = ((x1 / scale_w) as i32).max(0).min(_orig_w);
            let y_min = ((y1 / scale_h) as i32).max(0).min(_orig_h);
            let x_max = ((x2 / scale_w) as i32).max(0).min(_orig_w);
            let y_max = ((y2 / scale_h) as i32).max(0).min(_orig_h);

            // Skip invalid boxes
            if x_max <= x_min || y_max <= y_min {
                continue;
            }

            regions.push(LayoutRegion {
                bbox: BBox {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                },
                layout_type,
                confidence: score,
            });
        }

        Ok(regions)
    }
}
