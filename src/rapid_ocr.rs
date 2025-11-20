use std::path::Path;

#[cfg(feature = "use-opencv")]
use opencv::{core::{Mat, Point2f}, prelude::MatTraitConst};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Point2f};

use crate::cal_rec_boxes::CalRecBoxes;
use crate::det::TextDetector;
use crate::engine::EngineError;
use crate::geometry::{apply_vertical_padding, map_boxes_to_original, resize_image_within_bounds, get_rotate_crop_image, OpRecord};
use crate::rec::{TextRecOutput, TextRecognizer};
use crate::types::{DetConfig, GlobalConfig, RecConfig};

pub struct RapidOcrOutput {
    pub boxes: Vec<[Point2f; 4]>,
    pub txts: Vec<String>,
    pub scores: Vec<f32>,
    pub word_results: Vec<Vec<(String, f32, [Point2f; 4])>>,
    pub elapse_det: f64,
    pub elapse_rec: f64,
}

pub struct RapidOcr {
    pub det: TextDetector,
    pub rec: TextRecognizer,
    pub global: GlobalConfig,
    pub cal_rec_boxes: CalRecBoxes,
}

impl RapidOcr {
    pub fn new_ppv5<P: AsRef<Path>>(det_model: P, rec_model: P, dict_path: P) -> Result<Self, EngineError> {
        let det_cfg = DetConfig::ppv5(det_model.as_ref().to_path_buf());
        let mut rec_cfg = RecConfig::ppv5(rec_model.as_ref().to_path_buf());
        rec_cfg.rec_keys_path = Some(dict_path.as_ref().to_path_buf());

        let global = GlobalConfig {
            use_cls: false,
            ..GlobalConfig::default()
        };

        let det = TextDetector::new(det_cfg.clone())?;
        let rec = TextRecognizer::new(rec_cfg.clone())?;
        let cal_rec_boxes = CalRecBoxes::new();

        Ok(Self { det, rec, global, cal_rec_boxes })
    }

    /// Run OCR on an image file (convenience wrapper for run_on_mat)
    pub fn run<P: AsRef<Path>>(&self, image_path: P) -> Result<RapidOcrOutput, EngineError> {
        use crate::image_impl::imread;
        let img = imread(image_path)?;
        self.run_on_mat(&img)
    }

    pub fn run_on_mat(&self, img: &Mat) -> Result<RapidOcrOutput, EngineError> {
        let size = img.size()?;
        let ori_h = size.height;
        let ori_w = size.width;

        let mut op_record: OpRecord = OpRecord::new();

        // Global resize within bounds
        let (resized, ratio_h, ratio_w) = resize_image_within_bounds(
            img,
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
                return Ok(RapidOcrOutput {
                    boxes: Vec::new(),
                    txts: Vec::new(),
                    scores: Vec::new(),
                    word_results: Vec::new(),
                    elapse_det: det_res.elapse,
                    elapse_rec: 0.0,
                })
            }
        };

        // Crop text regions from padded image using padded-space boxes
        let mut crop_imgs: Vec<Mat> = Vec::with_capacity(padded_boxes.len());
        for b in &padded_boxes {
            let crop = get_rotate_crop_image(&padded, b)?;
            crop_imgs.push(crop);
        }

        // Map boxes back to original image coords for final output and word boxes
        let mut boxes = padded_boxes.clone();
        map_boxes_to_original(&mut boxes, &op_record, ori_h, ori_w);

        // Recognition
        let rec_res: TextRecOutput = self.rec.run(&crop_imgs, self.global.return_word_box)?;

        // Optional word boxes (computed before we move fields out of rec_res)
        let word_results_all: Vec<Vec<(String, f32, [Point2f; 4])>> = if self.global.return_word_box {
            self
                .cal_rec_boxes
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

        eprintln!("[RapidOCR] Filtering {} boxes by text_score threshold {}", boxes.len(), self.global.text_score);
        for (idx, (b, (t, s))) in boxes
            .into_iter()
            .zip(txts.drain(..).zip(scores.drain(..)))
            .enumerate()
        {
            if s < self.global.text_score {
                eprintln!("[RapidOCR] Box {} rejected: rec_score={:.3} < {}, text=\"{}\"", idx, s, self.global.text_score, t);
                continue;
            }
            eprintln!("[RapidOCR] Box {} ACCEPTED: rec_score={:.3}, text=\"{}\"", idx, s, t);
            f_boxes.push(b);
            f_txts.push(t);
            f_scores.push(s);

            if idx < word_results_all.len() {
                f_word_results.push(word_results_all[idx].clone());
            } else {
                f_word_results.push(Vec::new());
            }
        }

        Ok(RapidOcrOutput {
            boxes: f_boxes,
            txts: f_txts,
            scores: f_scores,
            word_results: f_word_results,
            elapse_det: det_res.elapse,
            elapse_rec: rec_res.elapse,
        })
    }
}

