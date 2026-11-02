use ndarray::{Array2, Array3, Array4};
use std::time::Instant;

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{EngineError, EngineSession, InferenceSession};
use crate::types::LayoutConfig;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

/// Layout element types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "text" | "paragraph" | "ocrtext" | "list" => Some(LayoutType::Text),
            "title" | "paragraph_title" => Some(LayoutType::Title),
            "figure" | "image" => Some(LayoutType::Figure),
            "figure_caption" => Some(LayoutType::FigureCaption),
            "table" => Some(LayoutType::Table),
            "table_caption" => Some(LayoutType::TableCaption),
            "header" => Some(LayoutType::Header),
            "footer" => Some(LayoutType::Footer),
            "reference" | "tablefootnote" => Some(LayoutType::Reference),
            "equation" | "formula" | "isolateformula" | "inlineformula" | "isolatedformula" => {
                Some(LayoutType::Equation)
            }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    session: EngineSession,
    pub config: LayoutConfig,
}

impl LayoutDetector {
    pub fn new(config: LayoutConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self { session, config })
    }

    /// Construct layout detector with a custom inference session
    pub fn with_custom_session(config: LayoutConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
        }
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

        let (new_h, new_w) = self.target_dimensions();

        // Preprocess
        let (input_tensor, scale_w, scale_h) = self.preprocess(image)?;

        // Prepare inputs
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("image".to_string(), input_tensor.into_dyn());

        let p_str = self.config.model_path.to_string_lossy();
        if p_str.contains("doclayout") {
            let im_shape = Array2::from_shape_vec((1, 2), vec![new_h as f32, new_w as f32])
                .map_err(EngineError::ShapeError)?;
            inputs.insert("im_shape".to_string(), im_shape.into_dyn());

            let scale_factor = Array2::from_shape_vec((1, 2), vec![scale_h, scale_w])
                .map_err(EngineError::ShapeError)?;
            inputs.insert("scale_factor".to_string(), scale_factor.into_dyn());
        }

        // Run inference
        let outputs = self.session.run_with_inputs(inputs)?;

        // Postprocess
        let regions = self.postprocess(outputs, orig_w, orig_h, new_w, new_h, scale_w, scale_h)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(LayoutOutput { regions, elapse })
    }

    fn target_dimensions(&self) -> (i32, i32) {
        match (self.config.target_height, self.config.target_width) {
            (Some(h), Some(w)) => (h, w),
            (Some(h), None) => (h, (h * 608) / 800),
            (None, Some(w)) => ((w * 800) / 608, w),
            (None, None) => {
                let p_str = self.config.model_path.to_string_lossy();
                if p_str.contains("doclayout_s") {
                    (480, 480)
                } else if p_str.contains("doclayout_l") || p_str.contains("doclayout_plus_l") {
                    (800, 800)
                } else if p_str.contains("yolo") {
                    (640, 640)
                } else if self.config.target_size > 0 && self.config.target_size != 800 {
                    (self.config.target_size, self.config.target_size)
                } else {
                    (800, 608)
                }
            }
        }
    }

    fn preprocess(&self, image: &Mat) -> Result<(Array4<f32>, f32, f32), EngineError> {
        #[cfg(feature = "use-opencv")]
        let (h, w) = (image.rows(), image.cols());

        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (h, w) = (size.height, size.width);

        let (new_h, new_w) = self.target_dimensions();
        let scale_w = new_w as f32 / w as f32;
        let scale_h = new_h as f32 / h as f32;

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

            let rows = resized.rows() as usize;
            let cols = resized.cols() as usize;
            let hw = rows * cols;

            let mut data = vec![0.0f32; 3 * hw];

            for i in 0..rows {
                for j in 0..cols {
                    let pix = resized.at_2d::<core::Vec3b>(i as i32, j as i32)?;
                    let idx = i * cols + j;
                    for c in 0..3 {
                        let val = pix[c] as f32 / 255.0;
                        data[c * hw + idx] = (val - mean[c]) / std[c];
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok((arr.insert_axis(ndarray::Axis(0)), scale_w, scale_h))
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            use crate::image_impl::{resize, Size, INTER_LINEAR};

            let mut resized = Mat::default();
            resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;

            let mean = self.config.mean;
            let std = self.config.std;

            let hw = new_h as usize * new_w as usize;
            let mut data = vec![0.0f32; 3 * hw];

            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let r = pixel[0] as f32 / 255.0;
                    let g = pixel[1] as f32 / 255.0;
                    let b = pixel[2] as f32 / 255.0;

                    let idx = y * (new_w as usize) + x;
                    // Store in BGR order matching OpenCV Vec3b and PP-Structure expectations
                    data[idx] = (b - mean[0]) / std[0];
                    data[hw + idx] = (g - mean[1]) / std[1];
                    data[2 * hw + idx] = (r - mean[2]) / std[2];
                }
            }

            let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok((arr.insert_axis(ndarray::Axis(0)), scale_w, scale_h))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn postprocess(
        &self,
        outputs: std::collections::HashMap<String, ndarray::ArrayD<f32>>,
        orig_w: i32,
        orig_h: i32,
        new_w: i32,
        new_h: i32,
        scale_w: f32,
        scale_h: f32,
    ) -> Result<Vec<LayoutRegion>, EngineError> {
        let mut raw_boxes = Vec::new();
        let mut found_direct = false;
        for tensor in outputs.values() {
            let shape = tensor.shape();
            if shape.len() == 2 && shape[1] == 6 {
                for row in tensor.outer_iter() {
                    raw_boxes.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
                }
                found_direct = true;
                break;
            } else if shape.len() == 3 && shape[2] == 6 {
                let batch0 = tensor.index_axis(ndarray::Axis(0), 0);
                for row in batch0.outer_iter() {
                    raw_boxes.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
                }
                found_direct = true;
                break;
            }
        }

        if found_direct {
            let indices = crate::geometry::nms(&raw_boxes, self.config.iou_thresh, self.config.conf_thresh);

            let mut regions = Vec::new();
            for idx in indices {
                let row = &raw_boxes[idx];
                let class_id = row[0] as usize;
                let score = row[1];
                let x1 = row[2];
                let y1 = row[3];
                let x2 = row[4];
                let y2 = row[5];

                let layout_type = match class_id {
                    0 => LayoutType::Title,
                    1 | 15 => LayoutType::Text,
                    2 => continue,
                    3 => LayoutType::Figure,
                    4 => LayoutType::FigureCaption,
                    5 => LayoutType::Table,
                    6 => LayoutType::TableCaption,
                    7 => LayoutType::Reference,
                    8 | 13 | 14 => LayoutType::Equation,
                    9 => LayoutType::Text,
                    _ => LayoutType::Text,
                };

                let x_min = ((x1 / scale_w) as i32).max(0).min(orig_w);
                let y_min = ((y1 / scale_h) as i32).max(0).min(orig_h);
                let x_max = ((x2 / scale_w) as i32).max(0).min(orig_w);
                let y_max = ((y2 / scale_h) as i32).max(0).min(orig_h);

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

            return Ok(regions);
        }

        self.postprocess_picodet(outputs, orig_w, orig_h, new_w, new_h, scale_w, scale_h)
    }

    #[allow(clippy::too_many_arguments)]
    fn postprocess_picodet(
        &self,
        outputs: std::collections::HashMap<String, ndarray::ArrayD<f32>>,
        orig_w: i32,
        orig_h: i32,
        new_w: i32,
        new_h: i32,
        scale_w: f32,
        scale_h: f32,
    ) -> Result<Vec<LayoutRegion>, EngineError> {
        let strides = [8usize, 16, 32, 64];

        struct StrideData<'a> {
            stride: usize,
            exp_points: usize,
            exp_w: usize,
            scores: std::borrow::Cow<'a, [f32]>,
            num_classes: usize,
            boxes: std::borrow::Cow<'a, [f32]>,
        }

        let mut stride_data_list = Vec::new();

        for &s in &strides {
            let exp_h = ((new_h as f32) / (s as f32)).ceil() as usize;
            let exp_w = ((new_w as f32) / (s as f32)).ceil() as usize;
            let exp_points = exp_h * exp_w;

            let mut matched_scores: Option<(std::borrow::Cow<'_, [f32]>, usize)> = None;
            let mut matched_boxes: Option<std::borrow::Cow<'_, [f32]>> = None;

            for tensor in outputs.values() {
                let shape = tensor.shape();
                let (points, last_dim) = match shape.len() {
                    2 => (shape[0], shape[1]),
                    3 if shape[0] == 1 => (shape[1], shape[2]),
                    _ => continue,
                };

                if points == exp_points {
                    if last_dim == 32 {
                        let cow = if let Some(slice) = tensor.as_slice() {
                            std::borrow::Cow::Borrowed(slice)
                        } else {
                            std::borrow::Cow::Owned(tensor.iter().copied().collect())
                        };
                        matched_boxes = Some(cow);
                    } else if last_dim != 32 {
                        let cow = if let Some(slice) = tensor.as_slice() {
                            std::borrow::Cow::Borrowed(slice)
                        } else {
                            std::borrow::Cow::Owned(tensor.iter().copied().collect())
                        };
                        matched_scores = Some((cow, last_dim));
                    }
                }
            }

            if let (Some((scores, num_classes)), Some(boxes)) = (matched_scores, matched_boxes) {
                stride_data_list.push(StrideData {
                    stride: s,
                    exp_points,
                    exp_w,
                    scores,
                    num_classes,
                    boxes,
                });
            }
        }

        if stride_data_list.is_empty() {
            let shapes: Vec<String> = outputs
                .iter()
                .map(|(k, v)| format!("{}: {:?}", k, v.shape()))
                .collect();
            return Err(EngineError::OutputError(format!(
                "No supported layout output tensors found (expected [N, 6] or PicoDet multi-scale transpose tensors). Outputs received: {}",
                shapes.join(", ")
            )));
        }

        let num_classes = stride_data_list[0].num_classes;

        struct Candidate {
            box_coords: [f32; 4],
            scores: Vec<f32>,
            max_score: f32,
        }

        let mut all_candidates: Vec<Candidate> = Vec::new();

        for data in &stride_data_list {
            let s = data.stride as f32;
            let mut stride_candidates: Vec<Candidate> = Vec::with_capacity(data.exp_points);

            for p in 0..data.exp_points {
                let r = (p / data.exp_w) as f32;
                let c = (p % data.exp_w) as f32;
                let cx = (c + 0.5) * s;
                let cy = (r + 0.5) * s;

                // Softmax DFL for 4 coordinates (left, top, right, bottom)
                let mut d = [0.0f32; 4];
                for (side, d_side) in d.iter_mut().enumerate() {
                    let base = p * 32 + side * 8;
                    let mut max_val = f32::NEG_INFINITY;
                    for k in 0..8 {
                        let v = data.boxes[base + k];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    let mut exp_vals = [0.0f32; 8];
                    let mut sum_exp = 0.0f32;
                    for (k, exp_val) in exp_vals.iter_mut().enumerate() {
                        let ev = (data.boxes[base + k] - max_val).exp();
                        *exp_val = ev;
                        sum_exp += ev;
                    }
                    let inv_sum = 1.0 / (sum_exp + 1e-9);
                    let mut expected = 0.0f32;
                    for (k, &ev) in exp_vals.iter().enumerate() {
                        expected += (k as f32) * (ev * inv_sum);
                    }
                    *d_side = expected * s;
                }

                let x1 = cx - d[0];
                let y1 = cy - d[1];
                let x2 = cx + d[2];
                let y2 = cy + d[3];

                let s_base = p * data.num_classes;
                let mut p_scores = Vec::with_capacity(data.num_classes);
                let mut max_score = 0.0f32;
                for ci in 0..data.num_classes {
                    let sc = data.scores[s_base + ci];
                    if sc > max_score {
                        max_score = sc;
                    }
                    p_scores.push(sc);
                }

                stride_candidates.push(Candidate {
                    box_coords: [x1, y1, x2, y2],
                    scores: p_scores,
                    max_score,
                });
            }

            stride_candidates.sort_by(|a, b| {
                b.max_score
                    .partial_cmp(&a.max_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if stride_candidates.len() > 1000 {
                stride_candidates.truncate(1000);
            }
            all_candidates.extend(stride_candidates);
        }

        let mut regions = Vec::new();
        let chars_opt = self.session.get_character_list("character");

        for class_idx in 0..num_classes {
            let mut candidates_for_class: Vec<usize> = (0..all_candidates.len())
                .filter(|&i| all_candidates[i].scores[class_idx] > self.config.conf_thresh)
                .collect();

            if candidates_for_class.is_empty() {
                continue;
            }

            candidates_for_class.sort_by(|&a, &b| {
                all_candidates[b].scores[class_idx]
                    .partial_cmp(&all_candidates[a].scores[class_idx])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut kept: Vec<usize> = Vec::new();
            for &idx in &candidates_for_class {
                let b1 = &all_candidates[idx].box_coords;
                let mut suppressed = false;
                for &k in &kept {
                    let b2 = &all_candidates[k].box_coords;
                    if calculate_box_iou(b1, b2) > self.config.iou_thresh {
                        suppressed = true;
                        break;
                    }
                }
                if !suppressed {
                    kept.push(idx);
                    if kept.len() >= 100 {
                        break;
                    }
                }
            }

            for idx in kept {
                let cand = &all_candidates[idx];
                let score = cand.scores[class_idx];
                let b = cand.box_coords;

                let cx1 = b[0].max(0.0).min(new_w as f32);
                let cy1 = b[1].max(0.0).min(new_h as f32);
                let cx2 = b[2].max(0.0).min(new_w as f32);
                let cy2 = b[3].max(0.0).min(new_h as f32);

                let orig_x1 = cx1 / scale_w;
                let orig_y1 = cy1 / scale_h;
                let orig_x2 = cx2 / scale_w;
                let orig_y2 = cy2 / scale_h;

                let x_min = (orig_x1.round() as i32).max(0).min(orig_w);
                let y_min = (orig_y1.round() as i32).max(0).min(orig_h);
                let x_max = (orig_x2.round() as i32).max(0).min(orig_w);
                let y_max = (orig_y2.round() as i32).max(0).min(orig_h);

                if x_max <= x_min || y_max <= y_min {
                    continue;
                }

                let layout_type = if let Some(chars) = &chars_opt {
                    if class_idx < chars.len() {
                        LayoutType::from_name(&chars[class_idx]).unwrap_or(LayoutType::Text)
                    } else {
                        LayoutType::from_class(class_idx).unwrap_or(LayoutType::Text)
                    }
                } else if num_classes == 10 {
                    match class_idx {
                        0 => LayoutType::Text,
                        1 => LayoutType::Title,
                        2 => LayoutType::Figure,
                        3 => LayoutType::FigureCaption,
                        4 => LayoutType::Table,
                        5 => LayoutType::TableCaption,
                        6 => LayoutType::Header,
                        7 => LayoutType::Footer,
                        8 => LayoutType::Reference,
                        9 => LayoutType::Equation,
                        _ => LayoutType::Text,
                    }
                } else if num_classes == 5 {
                    match class_idx {
                        0 => LayoutType::Text,
                        1 => LayoutType::Title,
                        2 => LayoutType::Text,
                        3 => LayoutType::Table,
                        4 => LayoutType::Figure,
                        _ => LayoutType::Text,
                    }
                } else {
                    LayoutType::from_class(class_idx).unwrap_or(LayoutType::Text)
                };

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
        }

        Ok(regions)
    }
}

fn calculate_box_iou(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let x1 = b1[0].max(b2[0]);
    let y1 = b1[1].max(b2[1]);
    let x2 = b1[2].min(b2[2]);
    let y2 = b1[3].min(b2[3]);
    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;
    let a1 = (b1[2] - b1[0]).max(0.0) * (b1[3] - b1[1]).max(0.0);
    let a2 = (b2[2] - b2[0]).max(0.0) * (b2[3] - b2[1]).max(0.0);
    let union = a1 + a2 - inter;
    if union > 0.0 {
        inter / (union + 1e-6)
    } else {
        0.0
    }
}
