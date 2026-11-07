//! End-to-end OCR pipeline coordinator in pure Rust.

use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array4;

use crate::cal_rec_boxes::CalRecBoxes;
use crate::calibration::apply_calibration;
use crate::ctc::{scatter_linenet_results, scatter_rec_results, CtcDecoder, WordInfo};
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::geometry::{
    apply_vertical_padding, calculate_crop_box, get_rotate_crop_image, map_boxes_to_original,
    resize_image_within_bounds, OpRecord,
};
use crate::image::{Mat, Point2f};
use crate::layout::{
    build_layout_model_inputs, order_text_results_by_layout, postprocess_direct_layout_boxes,
    postprocess_picodet, preprocess_layout_image, resolve_layout_target_dimensions, LayoutRegion,
    PicoDetInput,
};
use crate::orient::{postprocess_orient_tensor, preprocess_orient_image, Orientation};
use crate::postprocess::{decode_det_predictions, sort_det_boxes};
use crate::preprocess::{
    calculate_linenet_rec_target_widths, plan_rec_batches, prepare_det_input,
    resize_norm_rec_img, resolve_det_params, stack_rec_batch,
};
use crate::spatial::{build_text_results, build_word_results, group_words, to_spatial_text};
use crate::types::{
    DetConfig, DetectTextResult, DetectionRunOptions, GlobalConfig, LayoutConfig, OcrRunOptions,
    OutputGranularity, PostprocessRunOptions, TextResult,
};
use crate::unwarp::{postprocess_unwarp, preprocess_unwarp};

#[derive(Clone, Debug)]
pub struct PipelineOutput {
    pub boxes: Vec<[Point2f; 4]>,
    pub txts: Vec<String>,
    pub scores: Vec<f32>,
    pub word_results: Vec<Vec<(String, f32, [Point2f; 4])>>,
    pub orientation: Option<Orientation>,
    pub elapse_det: f64,
    pub elapse_rec: f64,
    pub elapse_orient: f64,
    pub elapse_unwarp: f64,
    pub debug_oriented_image: Option<Mat>,
    pub debug_unwarped_image: Option<Mat>,
    pub layout_regions: Option<Vec<LayoutRegion>>,
    pub y_threshold_multiplier: Option<f32>,
    pub x_threshold_multiplier: Option<f32>,
    pub img_w: f32,
    pub img_h: f32,
}

impl PipelineOutput {
    pub fn to_text_results(&self) -> Vec<TextResult> {
        build_text_results(&self.boxes, &self.txts, &self.scores)
    }

    pub fn to_text_results_with_options(&self, options: &OcrRunOptions) -> Vec<TextResult> {
        let (line_y, word_x) = options.pre_recognition.as_ref().map_or((0.5, 0.4), |pr| {
            (pr.line_y_threshold.unwrap_or(0.5), pr.word_x_threshold.unwrap_or(0.4))
        });
        if options.output == OutputGranularity::Words {
            let word_lines = build_word_results(&self.word_results);
            let grouped = group_words(
                word_lines,
                line_y,
                word_x,
            );
            if let Some(regions) = &self.layout_regions {
                if !regions.is_empty() {
                    return order_text_results_by_layout(&grouped, regions, self.img_w);
                }
            }
            grouped
        } else {
            let entries = self.to_text_results();
            let segments = crate::spatial::cluster_text_results_into_lines(
                entries,
                line_y,
                word_x,
            );
            if let Some(regions) = &self.layout_regions {
                if !regions.is_empty() {
                    return order_text_results_by_layout(&segments, regions, self.img_w);
                }
            }
            segments
        }
    }

    pub fn to_spatial_text(
        &self,
        line_y_threshold: Option<f32>,
        word_x_threshold: Option<f32>,
    ) -> String {
        let y_thresh = line_y_threshold.or(self.y_threshold_multiplier);
        let x_thresh = word_x_threshold.or(self.x_threshold_multiplier);
        to_spatial_text(&self.boxes, &self.txts, y_thresh, x_thresh)
    }

    pub fn to_raw(&self) -> String {
        crate::spatial::to_raw(&self.boxes, &self.txts, &self.scores)
    }

    pub fn to_tsv(&self) -> String {
        crate::spatial::to_tsv(&self.boxes, &self.txts)
    }

    pub fn to_text_with_position(&self) -> String {
        crate::spatial::to_text_with_position(&self.boxes, &self.txts, &self.scores)
    }
}

pub type RustOOutput = PipelineOutput;

/// Pure-Rust OCR Pipeline executing detection, recognition, orientation, unwarping, and layout.
pub struct OcrPipeline {
    pub det: Box<dyn InferenceSession>,
    pub rec: Box<dyn InferenceSession>,
    pub decoder: CtcDecoder,
    pub global: GlobalConfig,
    pub rec_batch_num: usize,
    pub orient: Option<Box<dyn InferenceSession>>,
    pub cls: Option<Box<dyn InferenceSession>>,
    pub unwarp: Option<Box<dyn InferenceSession>>,
    pub layout: Option<Box<dyn InferenceSession>>,
    pub layout_config: Option<LayoutConfig>,
    pub cal_rec_boxes: CalRecBoxes,
}

pub type RustO = OcrPipeline;

impl OcrPipeline {
    pub fn new(
        det: Box<dyn InferenceSession>,
        rec: Box<dyn InferenceSession>,
        decoder: CtcDecoder,
        global: GlobalConfig,
    ) -> Self {
        Self {
            det,
            rec,
            decoder,
            global,
            rec_batch_num: 6,
            orient: None,
            cls: None,
            unwarp: None,
            layout: None,
            layout_config: None,
            cal_rec_boxes: CalRecBoxes::new(),
        }
    }

    pub fn with_orient(mut self, orient: Box<dyn InferenceSession>) -> Self {
        self.orient = Some(orient);
        self
    }

    pub fn with_cls(mut self, cls: Box<dyn InferenceSession>) -> Self {
        self.cls = Some(cls);
        self
    }

    pub fn with_unwarp(mut self, unwarp: Box<dyn InferenceSession>) -> Self {
        self.unwarp = Some(unwarp);
        self
    }

    pub fn with_layout(mut self, layout: Box<dyn InferenceSession>) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_layout_config(mut self, config: LayoutConfig) -> Self {
        self.layout_config = Some(config);
        self
    }

    pub fn run_with_options(
        &mut self,
        img: &Mat,
        options: &OcrRunOptions,
    ) -> Result<PipelineOutput, CoreError> {
        let mut effective = self.global.clone();
        effective.return_word_box = options.output == OutputGranularity::Words;
        effective.return_single_char_box = false;
        effective.use_cls = options.line_classification.unwrap_or(self.global.use_cls) && self.cls.is_some();
        effective.use_orient = options.orientation_detection.unwrap_or(self.global.use_orient) && self.orient.is_some();
        effective.use_unwarp = options.auto_unwarp.unwrap_or(self.global.use_unwarp) && self.unwarp.is_some();
        effective.use_layout = options.layout_reading.unwrap_or(self.global.use_layout) && self.layout.is_some();

        if let Some(post_rec) = &options.post_recognition {
            if let Some(value) = post_rec.score_threshold {
                effective.text_score = value;
            }
        }
        if let Some(post_det) = &options.post_detection {
            if let Some(value) = post_det.min_height { effective.min_height = value; }
            if let Some(value) = post_det.max_side_len { effective.max_side_len = value; }
            if let Some(value) = post_det.min_side_len { effective.min_side_len = value; }
            if let Some(value) = post_det.width_height_ratio { effective.width_height_ratio = value; }
        }
        if let Some(pre_det) = &options.pre_detection {
            if let Some(calib) = pre_det.to_calibration_options() {
                effective.calibration = Some(calib);
            }
            if let Some(opt) = pre_det.to_optimization_options() {
                effective.optimization = Some(opt);
            }
        }

        let orig_rec_batch_num = self.rec_batch_num;
        if let Some(rec) = &options.recognition {
            if let Some(batch_num) = rec.batch_size {
                self.rec_batch_num = batch_num.max(1) as usize;
            }
        }
        let postprocess_options = options.post_detection.as_ref().map(|p| p.to_postprocess_run_options());
        let result = self.run_pipeline_internal(
            img,
            &effective,
            options.detection.as_ref(),
            postprocess_options.as_ref(),
        );
        self.rec_batch_num = orig_rec_batch_num;
        result
    }

    pub fn run_pipeline(
        &mut self,
        img: &Mat,
        detection: Option<&DetectionRunOptions>,
        postprocess: Option<&PostprocessRunOptions>,
    ) -> Result<PipelineOutput, CoreError> {
        let global = self.global.clone();
        self.run_pipeline_internal(img, &global, detection, postprocess)
    }

    fn run_pipeline_internal(
        &mut self,
        img: &Mat,
        global: &GlobalConfig,
        detection: Option<&DetectionRunOptions>,
        postprocess: Option<&PostprocessRunOptions>,
    ) -> Result<PipelineOutput, CoreError> {
        let mut elapse_orient = 0.0;
        let mut elapse_unwarp = 0.0;
        let mut orientation = None;
        let mut debug_oriented_image = None;
        let mut debug_unwarped_image = None;
        let mut layout_regions = None;

        // Step 0: Image calibration (if configured)
        let calibrated_img = if let Some(calib) = &global.calibration {
            if calib.enabled.unwrap_or(true) {
                apply_calibration(img, calib)?
            } else {
                img.clone()
            }
        } else {
            img.clone()
        };

        // Step 1: Orientation classification & rotation
        let mut working_img = calibrated_img;
        let mut active_orientation = None;
        if global.use_orient && self.orient.is_some() {
            if let Some(orient_session) = &mut self.orient {
                let start_orient = Instant::now();
                let [_, h, w] = [3, 48, 192];
                let input_tensor = preprocess_orient_image(&working_img, w, h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])?;
                let output = orient_session.run(input_tensor.into_dyn())?;
                let (det_orientation, conf) = postprocess_orient_tensor(&output)?;
                elapse_orient = start_orient.elapsed().as_secs_f64();

                if conf >= 0.5 {
                    orientation = Some(det_orientation);
                    if det_orientation.degrees() != 0 {
                        let rotated = det_orientation.rotate_image(&working_img)?;
                        if global.debug_images {
                            debug_oriented_image = Some(rotated.clone());
                        }
                        working_img = rotated;
                        active_orientation = Some(det_orientation);
                    }
                } else if det_orientation.degrees() == 0 {
                    orientation = Some(det_orientation);
                }
            }
        }

        // Step 2: Document unwarping (if configured)
        if global.use_unwarp && self.unwarp.is_some() {
            if let Some(unwarp_session) = &mut self.unwarp {
                let start_unwarp = Instant::now();
                let (orig_h, orig_w) = (working_img.rows(), working_img.cols());
                let input_tensor = preprocess_unwarp(&working_img, 512, 512)?;
                let output = unwarp_session.run(input_tensor.into_dyn())?;
                working_img = postprocess_unwarp(&working_img, &output, orig_w, orig_h)?;
                if global.debug_images {
                    debug_unwarped_image = Some(working_img.clone());
                }
                elapse_unwarp = start_unwarp.elapsed().as_secs_f64();
            }
        }

        // Step 3: Document layout detection (if configured)
        if global.use_layout && self.layout.is_some() {
            if let Some(layout_session) = &mut self.layout {
                let layout_cfg = self.layout_config.clone().unwrap_or_else(|| {
                    LayoutConfig::default(PathBuf::new())
                });
                let (orig_h, orig_w) = (working_img.rows(), working_img.cols());
                let (new_h, new_w) = resolve_layout_target_dimensions(
                    &layout_cfg.model_path,
                    layout_cfg.target_height,
                    layout_cfg.target_width,
                    layout_cfg.target_size,
                );
                let (input_tensor, scale_w, scale_h) = preprocess_layout_image(
                    &working_img,
                    new_w,
                    new_h,
                    layout_cfg.mean,
                    layout_cfg.std,
                )?;
                let inputs = build_layout_model_inputs(
                    input_tensor,
                    &layout_cfg.model_path,
                    new_h,
                    new_w,
                    scale_h,
                    scale_w,
                )?;
                let outputs = layout_session.run_with_inputs(inputs)?;
                let regions = if let Some(regions) = postprocess_direct_layout_boxes(
                    &outputs,
                    orig_w,
                    orig_h,
                    scale_w,
                    scale_h,
                    layout_cfg.conf_thresh,
                    layout_cfg.iou_thresh,
                ) {
                    regions
                } else {
                    let chars_opt = layout_session.get_character_list("character");
                    postprocess_picodet(PicoDetInput {
                        outputs: &outputs,
                        orig_w,
                        orig_h,
                        new_w,
                        new_h,
                        scale_w,
                        scale_h,
                        conf_thresh: layout_cfg.conf_thresh,
                        iou_thresh: layout_cfg.iou_thresh,
                        labels: chars_opt.as_deref(),
                    })?
                };
                layout_regions = Some(regions);
            }
        }

        let mut op_record = OpRecord::new();
        let effective_max_side = if let Some(opt) = &global.optimization {
            if opt.enabled.unwrap_or(true) {
                if let Some(target) = opt.target_max_side {
                    (target as f32).min(global.max_side_len)
                } else {
                    global.max_side_len
                }
            } else {
                global.max_side_len
            }
        } else {
            global.max_side_len
        };

        let (resized, ratio_h, ratio_w) =
            resize_image_within_bounds(&working_img, global.min_side_len, effective_max_side)?;
        let mut m = std::collections::BTreeMap::new();
        m.insert("ratio_h".to_string(), ratio_h);
        m.insert("ratio_w".to_string(), ratio_w);
        op_record.insert("preprocess".to_string(), m);

        let (padded, op_record) = apply_vertical_padding(
            &resized,
            op_record,
            global.width_height_ratio,
            global.min_height,
        )?;

        let working_w_f32 = working_img.cols() as f32;
        let working_h_f32 = working_img.rows() as f32;

        // Step 4: Text Detection
        let start_det = Instant::now();
        let (ori_h, ori_w) = (padded.rows(), padded.cols());
        let params = resolve_det_params(
            &DetConfig::default(PathBuf::new()),
            detection,
            postprocess,
            ori_w,
            ori_h,
        );

        let is_linenet_det = self.det.is_linenet();
        let (input, scale_w, scale_h) =
            prepare_det_input(&padded, is_linenet_det, &params, ori_w, ori_h)?;
        let preds_dyn = self.det.run(input.into_dyn())?;
        let preds: Array4<f32> = preds_dyn
            .into_dimensionality()
            .map_err(|_| CoreError::InvalidShape("detection output must be 4D".into()))?;

        let (mut boxes, _scores) = decode_det_predictions(
            &preds,
            is_linenet_det,
            ori_w,
            ori_h,
            scale_w,
            scale_h,
            &params,
        )?;

        let elapse_det = start_det.elapsed().as_secs_f64();

        if boxes.is_empty() {
            return Ok(PipelineOutput {
                boxes: Vec::new(),
                txts: Vec::new(),
                scores: Vec::new(),
                word_results: Vec::new(),
                orientation,
                elapse_det,
                elapse_rec: 0.0,
                elapse_orient,
                elapse_unwarp,
                debug_oriented_image,
                debug_unwarped_image,
                layout_regions,
                y_threshold_multiplier: global.y_threshold_multiplier,
                x_threshold_multiplier: global.x_threshold_multiplier,
                img_w: working_w_f32,
                img_h: working_h_f32,
            });
        }

        map_boxes_to_original(&mut boxes, &op_record, working_img.rows(), working_img.cols());
        sort_det_boxes(&mut boxes);

        // Step 5: Crop text regions
        let is_linenet_rec = self.rec.is_linenet();
        let (pad_x, pad_y) = if let Some(opt) = &global.optimization {
            if opt.enabled.unwrap_or(true) {
                (opt.crop_padding_x.unwrap_or(0.0), opt.crop_padding_y.unwrap_or(0.0))
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        let (pad_x_factor, pad_y_factor) = if let Some(opt) = &global.optimization {
            (opt.crop_padding_x.unwrap_or(0.3), opt.crop_padding_y.unwrap_or(1.6))
        } else {
            (0.3, 1.6)
        };

        let mut crop_imgs = Vec::with_capacity(boxes.len());
        for b in &boxes {
            let crop_box = calculate_crop_box(
                b,
                is_linenet_rec,
                pad_x,
                pad_y,
                pad_x_factor,
                pad_y_factor,
                working_w_f32,
                working_h_f32,
            );
            crop_imgs.push(get_rotate_crop_image(&working_img, &crop_box)?);
        }

        // Step 5.5: Text Line Orientation Classification (CLS) on cropped images
        if global.use_cls && self.cls.is_some() {
            if let Some(cls_session) = &mut self.cls {
                for crop in &mut crop_imgs {
                    let [_, h, w] = [3, 48, 192];
                    if let Ok(input_tensor) = preprocess_orient_image(crop, w, h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) {
                        if let Ok(output) = cls_session.run(input_tensor.into_dyn()) {
                            if let Ok((cls_orient, conf)) = postprocess_orient_tensor(&output) {
                                if cls_orient == Orientation::Rotate180 && conf >= 0.9 {
                                    if let Ok(rotated) = cls_orient.rotate_image(crop) {
                                        *crop = rotated;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 6: Text Recognition
        let start_rec = Instant::now();
        let img_num = crop_imgs.len();
        let mut all_texts: Vec<(String, f32)> = vec![(String::new(), 0.0); img_num];
        let mut all_word_infos: Vec<Option<WordInfo>> = vec![None; img_num];

        if is_linenet_rec {
            let rec_h = 96usize;
            let sizes: Vec<(f32, f32)> = crop_imgs
                .iter()
                .map(|img: &Mat| (img.cols() as f32, img.rows() as f32))
                .collect();
            let target_widths = calculate_linenet_rec_target_widths(&sizes, rec_h);
            let batch_num = self.rec_batch_num.max(1);

            for chunk in crate::preprocess::plan_linenet_rec_batches(&target_widths, batch_num) {
                let (batch_tensor, widths) =
                    crate::preprocess::build_linenet_rec_batch(&crop_imgs, &chunk, &target_widths, rec_h)?;
                let (batch_ids, batch_confs) = self.rec.run_linenet_rec(batch_tensor, widths)?;
                let decoded = self.decoder.decode_linenet_batch(batch_ids, batch_confs);
                scatter_linenet_results(&chunk, decoded, &mut all_texts);
            }
        } else {
            let (img_c, img_h, img_w) = (3usize, 48usize, 320usize);
            let sizes: Vec<(f32, f32)> = crop_imgs
                .iter()
                .map(|img: &Mat| (img.cols() as f32, img.rows() as f32))
                .collect();
            let default_wh_ratio = img_w as f32 / img_h as f32;
            let plan = plan_rec_batches(&sizes, self.rec_batch_num.max(1), default_wh_ratio);

            for batch_item in plan.batches {
                let mut norm_batch: Vec<ndarray::Array3<f32>> = Vec::with_capacity(batch_item.indices.len());
                for &idx in &batch_item.indices {
                    norm_batch.push(resize_norm_rec_img(&crop_imgs[idx], img_c, img_h, batch_item.max_wh_ratio)?);
                }

                let batch_img_width = (img_h as f32 * batch_item.max_wh_ratio).round() as usize;
                let batch = stack_rec_batch(&norm_batch, img_c, img_h, batch_img_width);
                let preds_dyn = self.rec.run(batch.into_dyn())?;
                let preds: ndarray::Array3<f32> = preds_dyn
                    .into_dimensionality()
                    .map_err(|_| CoreError::InvalidShape("recognition output must be 3D".into()))?;

                let (line_results, batch_word_infos) = self.decoder.decode(
                    preds,
                    global.return_word_box,
                    &batch_item.wh_ratio_list,
                    batch_item.max_wh_ratio,
                );

                let word_infos = if global.return_word_box {
                    Some(batch_word_infos)
                } else {
                    None
                };
                scatter_rec_results(
                    &batch_item.indices,
                    line_results,
                    word_infos,
                    &mut all_texts,
                    &mut all_word_infos,
                );
            }
        }

        let elapse_rec = start_rec.elapsed().as_secs_f64();
        let (txts, scores): (Vec<String>, Vec<f32>) = all_texts.into_iter().unzip();

        // Step 7: Word boxes alignment
        let mut word_results_all: Vec<Vec<(String, f32, [Point2f; 4])>> = if global.return_word_box {
            self.cal_rec_boxes.calc_word_boxes(
                &boxes,
                &txts,
                &all_word_infos,
                global.return_single_char_box,
            )
        } else {
            vec![Vec::new(); boxes.len()]
        };

        // Step 8: Rotate back if orientation was applied
        if let Some(orient) = active_orientation {
            for b in &mut boxes {
                orient.rotate_box_points_back(b, working_w_f32, working_h_f32);
            }
            for line_words in &mut word_results_all {
                for (_, _, word_box) in line_words.iter_mut() {
                    orient.rotate_box_points_back(word_box, working_w_f32, working_h_f32);
                }
            }
        }

        // Step 9: Filter by text_score
        let mut f_boxes = Vec::new();
        let mut f_txts = Vec::new();
        let mut f_scores = Vec::new();
        let mut f_word_results = Vec::new();

        for (idx, (b, (t, s))) in boxes.into_iter().zip(txts.into_iter().zip(scores.into_iter())).enumerate() {
            if s < global.text_score {
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

        Ok(PipelineOutput {
            boxes: f_boxes,
            txts: f_txts,
            scores: f_scores,
            word_results: f_word_results,
            orientation,
            elapse_det,
            elapse_rec,
            elapse_orient,
            elapse_unwarp,
            debug_oriented_image,
            debug_unwarped_image,
            layout_regions,
            y_threshold_multiplier: global.y_threshold_multiplier,
            x_threshold_multiplier: global.x_threshold_multiplier,
            img_w: working_w_f32,
            img_h: working_h_f32,
        })
    }

    pub fn detect_text(
        &mut self,
        img: &Mat,
        options: &OcrRunOptions,
    ) -> Result<DetectTextResult, CoreError> {
        let output = self.run_with_options(img, options)?;
        match options.output {
            OutputGranularity::Spatial => {
                let (line_y, word_x) = options.pre_recognition.as_ref().map_or((None, None), |pr| {
                    (pr.line_y_threshold, pr.word_x_threshold)
                });
                Ok(DetectTextResult::Spatial(
                    output.to_spatial_text(line_y, word_x),
                ))
            }
            OutputGranularity::Tsv => {
                Ok(DetectTextResult::Spatial(output.to_tsv()))
            }
            OutputGranularity::Lines | OutputGranularity::Words => Ok(
                DetectTextResult::Structured(output.to_text_results_with_options(options)),
            ),
        }
    }

    pub fn detect_text_rgb(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        options: &OcrRunOptions,
    ) -> Result<DetectTextResult, CoreError> {
        let mat = Mat::from_rgb_stride(width, height, stride, pixels)?;
        self.detect_text(&mat, options)
    }
}
