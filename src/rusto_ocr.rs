use std::path::Path;

#[cfg(feature = "use-opencv")]
use opencv::{
    core::{Mat, Point2f},
    prelude::MatTraitConst,
};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Point2f};

use crate::cal_rec_boxes::CalRecBoxes;
use crate::config::RustOConfig;
use crate::det::TextDetector;
use crate::engine::EngineError;
use crate::geometry::{
    apply_vertical_padding, get_rotate_crop_image, map_boxes_to_original,
    resize_image_within_bounds, OpRecord,
};
use crate::orient::{OrientClassifier, Orientation};
use crate::rec::{TextRecOutput, TextRecognizer};
use crate::types::GlobalConfig;

pub struct RustOOutput {
    pub boxes: Vec<[Point2f; 4]>,
    pub txts: Vec<String>,
    pub scores: Vec<f32>,
    pub word_results: Vec<Vec<(String, f32, [Point2f; 4])>>,
    pub orientation: Option<Orientation>,
    pub elapse_det: f64,
    pub elapse_rec: f64,
    pub elapse_orient: f64,
    /// Debug: Orientation-corrected image (if orientation was detected)
    pub debug_oriented_image: Option<Mat>,
}

pub struct RustO {
    pub det: TextDetector,
    pub rec: TextRecognizer,
    pub global: GlobalConfig,
    pub cal_rec_boxes: CalRecBoxes,
    pub orient: Option<OrientClassifier>,
    pub cls: Option<OrientClassifier>,
}

impl RustO {
    /// Create a new OCR engine with the given configuration
    pub fn new(config: RustOConfig) -> Result<Self, EngineError> {
        let det = TextDetector::new(config.det.clone())?;
        let rec = TextRecognizer::new(config.rec.clone())?;
        let cal_rec_boxes = CalRecBoxes::new();

        // Initialize orient classifier if provided
        let orient = if let Some(orient_cfg) = config.orient {
            Some(OrientClassifier::new(orient_cfg)?)
        } else {
            None
        };

        // Initialize CLS (text line orientation) if provided
        let cls = if let Some(cls_cfg) = config.cls {
            // Convert ClsConfig to OrientConfig for reuse
            let orient_cfg = crate::types::OrientConfig {
                engine_type: cls_cfg.engine_type,
                model_type: cls_cfg.model_type,
                task_type: cls_cfg.task_type,
                model_path: cls_cfg.model_path,
                orient_image_shape: cls_cfg.cls_image_shape,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                confidence_threshold: cls_cfg.cls_thresh,
                orient_batch_num: cls_cfg.cls_batch_num,
                orient_thresh: cls_cfg.cls_thresh,
                engine_cfg: cls_cfg.engine_cfg,
            };
            Some(OrientClassifier::new(orient_cfg)?)
        } else {
            None
        };

        Ok(Self {
            det,
            rec,
            global: config.global,
            cal_rec_boxes,
            orient,
            cls,
        })
    }

    /// Convenience constructor for PPOCRv5 models with minimal configuration
    pub fn new_ppv5<P: AsRef<Path>>(
        det_model: P,
        rec_model: P,
        dict_path: P,
    ) -> Result<Self, EngineError> {
        let config = RustOConfig::new_ppv5(
            det_model.as_ref().to_path_buf(),
            rec_model.as_ref().to_path_buf(),
            dict_path.as_ref().to_path_buf(),
        );
        Self::new(config)
    }

    /// Run OCR on an image file (convenience wrapper for run_on_mat)
    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<RustOOutput, EngineError> {
        use crate::image_impl::imread;
        let img = imread(image_path)?;
        self.run_on_mat(&img)
    }

    pub fn run_on_mat(&mut self, img: &Mat) -> Result<RustOOutput, EngineError> {
        let size = img.size()?;
        let ori_h = size.height;
        let ori_w = size.width;

        let mut elapse_orient = 0.0;
        let mut orientation = None;
        let mut debug_oriented_image = None;

        // Step 1: Orientation classification and correction (if enabled)
        // Apply to ENTIRE image before detection
        let mut working_img = img.clone();
        if self.global.use_orient && self.orient.is_some() {
            if let Some(orient_classifier) = &mut self.orient {
                let orient_result = orient_classifier.classify(img)?;
                elapse_orient = orient_result.elapse;

                // Apply rotation for internal processing if orientation detected
                if orient_result.orientation.degrees() != 0 {
                    let rotated = orient_result.orientation.rotate_image(&working_img)?;
                    if self.global.debug_images {
                        debug_oriented_image = Some(rotated.clone());
                    }
                    working_img = rotated;

                    // Only report orientation if confidence meets threshold
                    if orient_result.confidence >= orient_classifier.config.confidence_threshold {
                        orientation = Some(orient_result.orientation);
                    }
                } else {
                    orientation = Some(orient_result.orientation);
                }
            }
        }

        let mut op_record: OpRecord = OpRecord::new();

        // Step 2: Global resize within bounds (use corrected image)
        let (resized, ratio_h, ratio_w) = resize_image_within_bounds(
            &working_img,
            self.global.min_side_len,
            self.global.max_side_len,
        )?;
        let mut m = std::collections::BTreeMap::new();
        m.insert("ratio_h".to_string(), ratio_h);
        m.insert("ratio_w".to_string(), ratio_w);
        op_record.insert("preprocess".to_string(), m);

        // Vertical padding
        let (padded, op_record) = apply_vertical_padding(
            &resized,
            op_record,
            self.global.width_height_ratio,
            self.global.min_height,
        )?;

        // Detection (boxes are in padded-image coordinates here)
        // IMPORTANT: Pass padded image dimensions, not original!
        let det_res = self.det.run(&padded)?;
        let padded_boxes = match det_res.boxes {
            Some(b) if !b.is_empty() => b,
            _ => {
                return Ok(RustOOutput {
                    boxes: Vec::new(),
                    txts: Vec::new(),
                    scores: Vec::new(),
                    word_results: Vec::new(),
                    orientation,
                    elapse_det: det_res.elapse,
                    elapse_rec: 0.0,
                    elapse_orient,
                    debug_oriented_image,
                })
            }
        };

        // Step 4: Crop text regions from padded image
        let mut crop_imgs: Vec<Mat> = Vec::with_capacity(padded_boxes.len());
        for b in &padded_boxes {
            let crop = get_rotate_crop_image(&padded, b)?;
            crop_imgs.push(crop);
        }

        // Step 4.5: Text Line Orientation Classification (CLS) on cropped images
        // If enabled, classify each crop and rotate if needed (0 vs 180 degrees)
        if self.global.use_cls && self.cls.is_some() {
            if let Some(cls_classifier) = &mut self.cls {
                for crop in &mut crop_imgs {
                    if let Ok(cls_result) = cls_classifier.classify(crop) {
                        // Only rotate if orientation is 180 degrees and confidence is high
                        if cls_result.orientation == Orientation::Rotate180
                            && cls_result.confidence >= cls_classifier.config.confidence_threshold
                        {
                            // Rotate crop 180 degrees
                            if let Ok(rotated) = cls_result.orientation.rotate_image(crop) {
                                *crop = rotated;
                            }
                        }
                    }
                }
            }
        }

        // Map boxes back to original image coords for final output and word boxes
        let mut boxes = padded_boxes.clone();
        map_boxes_to_original(&mut boxes, &op_record, ori_h, ori_w);

        // Recognition
        let rec_res: TextRecOutput = self.rec.run(&crop_imgs, self.global.return_word_box)?;

        // Optional word boxes (computed before we move fields out of rec_res)
        let word_results_all: Vec<Vec<(String, f32, [Point2f; 4])>> = if self.global.return_word_box
        {
            self.cal_rec_boxes
                .calc_word_boxes(&boxes, &rec_res, self.global.return_single_char_box)
        } else {
            vec![Vec::new(); boxes.len()]
        };

        let mut txts = rec_res.txts;
        let mut scores = rec_res.scores;

        // Filter by text_score
        let mut f_boxes = Vec::new();
        let mut f_txts = Vec::new();
        let mut f_scores = Vec::new();
        let mut f_word_results: Vec<Vec<(String, f32, [Point2f; 4])>> = Vec::new();

        for (idx, (b, (t, s))) in boxes
            .into_iter()
            .zip(txts.drain(..).zip(scores.drain(..)))
            .enumerate()
        {
            if s < self.global.text_score {
                continue;
            }
            f_boxes.push(b);
            f_txts.push(t);
            f_scores.push(s);

            if idx < word_results_all.len() {
                f_word_results.push(word_results_all[idx].clone());
            } else {
                f_word_results.push(Vec::new());
            }
        }

        Ok(RustOOutput {
            boxes: f_boxes,
            txts: f_txts,
            scores: f_scores,
            word_results: f_word_results,
            orientation,
            elapse_det: det_res.elapse,
            elapse_rec: rec_res.elapse,
            elapse_orient,
            debug_oriented_image,
        })
    }

    /// Set optional orient classifier
    pub fn with_orient(mut self, orient: OrientClassifier) -> Self {
        self.orient = Some(orient);
        self
    }
}

impl RustOOutput {
    /// Export raw OCR results as ASCII format
    /// Format: [x,y] confidence% text
    /// X,Y are from top-left point of bounding box
    /// Sorted by Y position first, then X position (follows raw_to_csv.py)
    pub fn to_raw(&self) -> String {
        if self.boxes.is_empty() {
            return String::new();
        }

        // Collect all entries with coordinates
        let mut entries: Vec<(f32, f32, String, f32)> = self
            .boxes
            .iter()
            .zip(self.txts.iter())
            .zip(self.scores.iter())
            .map(|((bbox, text), &score)| {
                let x = bbox[0].x;
                let y = bbox[0].y;
                (y, x, text.clone(), score)
            })
            .collect();

        // Sort by Y first, then by X (like raw_to_csv.py)
        entries.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });

        let mut result = String::new();
        for (y, x, text, score) in entries {
            result.push_str(&format!(
                "[{:.0},{:.0}] {:.2}% {}\n",
                x,
                y,
                score * 100.0,
                text
            ));
        }
        result
    }

    /// Export OCR results as CSV format
    /// Format: line_id,column_id,text
    /// Groups text by Y position into lines, then by X gaps into columns
    pub fn to_csv(&self) -> String {
        if self.boxes.is_empty() {
            return "line_id,column_id,text\n".to_string();
        }

        #[derive(Debug, Clone)]
        struct Token {
            x: f32,
            y: f32,
            text: String,
        }

        // Create tokens from boxes (use top-left point)
        let mut tokens: Vec<Token> = self
            .boxes
            .iter()
            .zip(self.txts.iter())
            .map(|(bbox, text)| Token {
                x: bbox[0].x,
                y: bbox[0].y,
                text: text.clone(),
            })
            .collect();

        // Sort by Y first
        tokens.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());

        // Calculate typical Y diff for line grouping
        let y_diffs: Vec<f32> = tokens
            .windows(2)
            .map(|w| w[1].y - w[0].y)
            .filter(|&d| d > 0.0)
            .collect();
        let typical_y_diff = if !y_diffs.is_empty() {
            let mut sorted = y_diffs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        } else {
            10.0
        };
        let y_tolerance = typical_y_diff * 0.6;

        // Group into lines
        let mut lines: Vec<Vec<Token>> = Vec::new();
        let mut current_line: Vec<Token> = Vec::new();
        let mut current_y: Option<f32> = None;

        for token in tokens {
            if let Some(cy) = current_y {
                if (token.y - cy).abs() <= y_tolerance {
                    current_line.push(token.clone());
                    current_y = Some(
                        (cy * (current_line.len() - 1) as f32 + token.y)
                            / current_line.len() as f32,
                    );
                } else {
                    lines.push(current_line.clone());
                    current_line = vec![token.clone()];
                    current_y = Some(token.y);
                }
            } else {
                current_line = vec![token.clone()];
                current_y = Some(token.y);
            }
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }

        // Sort each line by X and segment into columns
        let mut csv_rows = Vec::new();
        for (line_id, mut line) in lines.into_iter().enumerate() {
            line.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

            if line.len() == 1 {
                csv_rows.push((line_id, 0, line[0].text.clone()));
                continue;
            }

            // Calculate X gaps for column segmentation
            let x_gaps: Vec<f32> = line.windows(2).map(|w| w[1].x - w[0].x).collect();
            let median_gap = if !x_gaps.is_empty() {
                let mut sorted = x_gaps.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2].max(1.0)
            } else {
                1.0
            };
            let gap_threshold = median_gap * 1.3;

            // Segment into columns
            let mut columns: Vec<Vec<Token>> = Vec::new();
            let mut current_col = vec![line[0].clone()];
            for i in 1..line.len() {
                let gap = line[i].x - line[i - 1].x;
                if gap > gap_threshold {
                    columns.push(current_col.clone());
                    current_col = vec![line[i].clone()];
                } else {
                    current_col.push(line[i].clone());
                }
            }
            columns.push(current_col);

            // Add to CSV rows
            for (col_id, col) in columns.into_iter().enumerate() {
                let text = col
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                csv_rows.push((line_id, col_id, text));
            }
        }

        // Format as CSV
        let mut result = String::from("line_id,column_id,text\n");
        for (line_id, col_id, text) in csv_rows {
            let safe_text = text.replace('"', "\"\"");
            result.push_str(&format!("{},{},\"{}\"\n", line_id, col_id, safe_text));
        }
        result
    }

    /// Export results as plain text with position info
    pub fn to_text_with_position(&self) -> String {
        let mut result = String::new();

        // Sort by position
        let mut indexed: Vec<(&[Point2f; 4], &String, f32)> = self
            .boxes
            .iter()
            .zip(self.txts.iter())
            .zip(self.scores.iter())
            .map(|((b, t), &s)| (b, t, s))
            .collect();

        indexed.sort_by(|a, b| {
            let ay = (a.0[0].y + a.0[2].y) / 2.0;
            let by = (b.0[0].y + b.0[2].y) / 2.0;
            let ax = (a.0[0].x + a.0[2].x) / 2.0;
            let bx = (b.0[0].x + b.0[2].x) / 2.0;

            if (ay - by).abs() < 20.0 {
                ax.partial_cmp(&bx).unwrap()
            } else {
                ay.partial_cmp(&by).unwrap()
            }
        });

        for (bbox, text, score) in &indexed {
            let x = (bbox[0].x + bbox[2].x) / 2.0;
            let y = (bbox[0].y + bbox[2].y) / 2.0;
            result.push_str(&format!(
                "[{:.0},{:.0}] {:.2}% {}\n",
                x,
                y,
                score * 100.0,
                text
            ));
        }

        result
    }
}
