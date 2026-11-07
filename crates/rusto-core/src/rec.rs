use std::time::Instant;

use ndarray::{Array3, Ix3};

pub use crate::ctc::{scatter_linenet_results, scatter_rec_results, CtcDecoder, WordInfo, WordType};
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::Mat;
pub use crate::preprocess::{
    build_linenet_rec_batch, calculate_linenet_rec_target_widths, plan_linenet_rec_batches,
    plan_rec_batches, prepare_linenet_rec_crop, resize_norm_rec_img, stack_rec_batch,
    RecBatchItem, RecBatchPlan,
};
use crate::types::RecConfig;

/// Output of text recognition execution.
pub struct TextRecOutput {
    pub imgs: Vec<Mat>,
    pub txts: Vec<String>,
    pub scores: Vec<f32>,
    pub word_infos: Vec<Option<WordInfo>>,
    pub elapse: f64,
}

pub fn decoder_from_cfg(
    cfg: &RecConfig,
    session: &impl InferenceSession,
) -> Result<CtcDecoder, CoreError> {
    if let Some(list) = session.get_character_list("character") {
        return Ok(CtcDecoder::from_model_character_list(list));
    }
    if let Some(path) = &cfg.rec_keys_path {
        return CtcDecoder::from_keys_file(path);
    }
    Err(CoreError::Preprocess("no character list found for recognizer".to_string()))
}

/// Text recognition pipeline component executing CTC recognition.
pub struct TextRecognizer<S = Box<dyn InferenceSession>> {
    pub cfg: RecConfig,
    pub session: S,
    pub decoder: CtcDecoder,
}

impl<S: InferenceSession> TextRecognizer<S> {
    pub fn with_session(cfg: RecConfig, session: S, decoder: CtcDecoder) -> Self {
        Self { cfg, session, decoder }
    }

    pub fn is_linenet(&self) -> bool {
        self.session.input_channels() == 1
    }

    pub fn run(&mut self, imgs: &[Mat], return_word_box: bool) -> Result<TextRecOutput, CoreError> {
        let start = Instant::now();

        if imgs.is_empty() {
            return Ok(TextRecOutput {
                imgs: Vec::new(),
                txts: Vec::new(),
                scores: Vec::new(),
                word_infos: Vec::new(),
                elapse: 0.0,
            });
        }

        let img_list: Vec<Mat> = imgs.to_vec();
        let img_num = img_list.len();
        let mut all_texts: Vec<(String, f32)> = vec![(String::new(), 0.0); img_num];
        let mut all_word_infos: Vec<Option<WordInfo>> = vec![None; img_num];

        if self.is_linenet() {
            let rec_h = 96usize;
            let sizes: Vec<(f32, f32)> = img_list
                .iter()
                .map(|img| (img.cols() as f32, img.rows() as f32))
                .collect();
            let target_widths = calculate_linenet_rec_target_widths(&sizes, rec_h);
            let batch_num = (self.cfg.rec_batch_num as usize).max(1);

            for chunk in plan_linenet_rec_batches(&target_widths, batch_num) {
                let (batch_tensor, widths) =
                    build_linenet_rec_batch(&img_list, &chunk, &target_widths, rec_h)?;
                let (batch_ids, batch_confs) = self.session.run_linenet_rec(batch_tensor, widths)?;
                let decoded = self.decoder.decode_linenet_batch(batch_ids, batch_confs);
                scatter_linenet_results(&chunk, decoded, &mut all_texts);
            }
        } else {
            let (img_c, img_h, img_w) = (
                self.cfg.rec_img_shape[0] as usize,
                self.cfg.rec_img_shape[1] as usize,
                self.cfg.rec_img_shape[2] as usize,
            );
            let sizes: Vec<(f32, f32)> = img_list
                .iter()
                .map(|img| (img.cols() as f32, img.rows() as f32))
                .collect();
            let default_wh_ratio = img_w as f32 / img_h as f32;
            let plan = plan_rec_batches(&sizes, self.cfg.rec_batch_num as usize, default_wh_ratio);

            for batch_item in plan.batches {
                let mut norm_batch: Vec<Array3<f32>> = Vec::with_capacity(batch_item.indices.len());
                for &idx in &batch_item.indices {
                    norm_batch.push(resize_norm_rec_img(&img_list[idx], img_c, img_h, batch_item.max_wh_ratio)?);
                }

                let batch_img_width = (img_h as f32 * batch_item.max_wh_ratio).round() as usize;
                let batch = stack_rec_batch(&norm_batch, img_c, img_h, batch_img_width);

                let preds_dyn = self.session.run(batch.into_dyn())?;
                let preds: Array3<f32> = preds_dyn
                    .into_dimensionality::<Ix3>()
                    .map_err(|_| CoreError::InvalidShape("recognition output must be 3D".into()))?;

                let (line_results, batch_word_infos) =
                    self.decoder.decode(preds, return_word_box, &batch_item.wh_ratio_list, batch_item.max_wh_ratio);

                let word_infos = if return_word_box { Some(batch_word_infos) } else { None };
                scatter_rec_results(
                    &batch_item.indices,
                    line_results,
                    word_infos,
                    &mut all_texts,
                    &mut all_word_infos,
                );
            }
        }

        let (txts, scores): (Vec<String>, Vec<f32>) = all_texts.into_iter().unzip();
        let elapse = start.elapsed().as_secs_f64();

        Ok(TextRecOutput {
            imgs: img_list,
            txts,
            scores,
            word_infos: all_word_infos,
            elapse,
        })
    }
}
