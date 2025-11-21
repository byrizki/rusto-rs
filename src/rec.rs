use std::io::{BufRead, BufReader};
use std::fs::File;
use std::time::Instant;

use ndarray::{Array3, Array4, Ix3};

#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(feature = "use-opencv")]
use opencv::core::Mat;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Size, INTER_LINEAR};

use crate::engine::{EngineError, MnnSession};
use crate::types::RecConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordType {
    Cn,
    EnNum,
}

#[derive(Clone, Debug, Default)]
pub struct WordInfo {
    pub words: Vec<Vec<String>>,
    pub word_cols: Vec<Vec<usize>>,
    pub word_types: Vec<WordType>,
    pub line_txt_len: f32,
    pub confs: Vec<f32>,
}

pub struct TextRecOutput {
    pub imgs: Vec<Mat>,
    pub txts: Vec<String>,
    pub scores: Vec<f32>,
    pub word_infos: Vec<Option<WordInfo>>,
    pub elapse: f64,
}

struct CtcDecoder {
    chars: Vec<String>,
}

impl CtcDecoder {
    fn from_cfg(cfg: &RecConfig, session: &MnnSession) -> Result<Self, EngineError> {
        let mut chars: Option<Vec<String>> = None;

        if session.have_key("character") {
            if let Some(list) = session.get_character_list("character") {
                chars = Some(list);
            }
        }

        if chars.is_none() {
            if let Some(path) = &cfg.rec_keys_path {
                let file = File::open(path).map_err(|e| {
                    EngineError::Preprocess(format!("failed to open rec_keys_path: {e}"))
                })?;
                let reader = BufReader::new(file);
                let mut list = Vec::new();
                for line in reader.lines() {
                    let l = line.map_err(|e| {
                        EngineError::Preprocess(format!("failed to read rec_keys_path: {e}"))
                    })?;
                    list.push(l);
                }
                chars = Some(list);
            }
        }

        let mut character_list = chars.ok_or_else(|| {
            EngineError::Preprocess("no character list found for recognizer".to_string())
        })?;

        character_list.push(" ".to_string());
        character_list.insert(0, "blank".to_string());

        Ok(Self { chars: character_list })
    }

    fn decode(
        &self,
        preds: Array3<f32>,
        return_word_box: bool,
        wh_ratio_list: &[f32],
        max_wh_ratio: f32,
    ) -> (Vec<(String, f32)>, Vec<WordInfo>) {
        let (n, t, c) = preds.dim();
        let mut line_results = Vec::with_capacity(n);
        let mut word_infos = Vec::with_capacity(if return_word_box { n } else { 0 });

        for b in 0..n {
            let mut token_indices = Vec::with_capacity(t);
            let mut token_probs = Vec::with_capacity(t);

            for ti in 0..t {
                let batch_view = preds.index_axis(ndarray::Axis(0), b);
                let row_view = batch_view.index_axis(ndarray::Axis(0), ti);

                let mut best_idx = 0usize;
                let mut best_val = f32::MIN;
                for ci in 0..c {
                    let v = row_view[ci];
                    if v > best_val {
                        best_val = v;
                        best_idx = ci;
                    }
                }

                token_indices.push(best_idx);
                token_probs.push(best_val);
            }

            let ignored_tokens = self.get_ignored_tokens();
            let mut selection = vec![true; token_indices.len()];

            if !token_indices.is_empty() {
                for i in 1..token_indices.len() {
                    if token_indices[i] == token_indices[i - 1] {
                        selection[i] = false;
                    }
                }
            }

            for &ignored in &ignored_tokens {
                for (i, sel) in selection.iter_mut().enumerate() {
                    if token_indices[i] == ignored {
                        *sel = false;
                    }
                }
            }

            // Pre-allocate conf_list with estimated size
            let est_size = selection.iter().filter(|&&s| s).count().max(1);
            let mut conf_list = Vec::with_capacity(est_size);
            for (i, &sel) in selection.iter().enumerate() {
                if sel {
                    let mut v = token_probs[i];
                    v = (v * 1e5).round() / 1e5;
                    conf_list.push(v);
                }
            }

            if conf_list.is_empty() {
                conf_list.push(0.0);
            }

            // Pre-allocate chars vector
            let mut chars = Vec::with_capacity(est_size);
            for (i, &sel) in selection.iter().enumerate() {
                if sel {
                    if let Some(ch) = self.chars.get(token_indices[i]) {
                        chars.push(ch.as_str());
                    }
                }
            }

            let text = chars.concat();
            let mean_score: f32 = conf_list.iter().copied().sum::<f32>() / (conf_list.len() as f32);

            line_results.push((text.clone(), mean_score));

            if return_word_box {
                let mut info = self.get_word_info(&text, &selection);
                if !token_indices.is_empty()
                    && b < wh_ratio_list.len()
                    && max_wh_ratio > 0.0
                {
                    let len_tokens = token_indices.len() as f32;
                    info.line_txt_len = len_tokens * wh_ratio_list[b] / max_wh_ratio;
                }
                info.confs = conf_list.clone();
                word_infos.push(info);
            }
        }

        (line_results, word_infos)
    }

    fn get_ignored_tokens(&self) -> Vec<usize> {
        vec![0]
    }

    fn get_word_info(&self, text: &str, selection: &[bool]) -> WordInfo {
        let mut word_list: Vec<Vec<String>> = Vec::new();
        let mut word_col_list: Vec<Vec<usize>> = Vec::new();
        let mut state_list: Vec<WordType> = Vec::new();

        let mut word_content: Vec<String> = Vec::new();
        let mut word_col_content: Vec<usize> = Vec::new();

        let mut valid_col: Vec<usize> = Vec::new();
        for (i, &sel) in selection.iter().enumerate() {
            if sel {
                valid_col.push(i);
            }
        }

        if valid_col.is_empty() {
            return WordInfo::default();
        }

        let mut col_width = vec![0i32; valid_col.len()];
        for i in 1..valid_col.len() {
            col_width[i] = (valid_col[i] as i32) - (valid_col[i - 1] as i32);
        }

        let first_char = text.chars().next().unwrap_or(' ');
        let first_width = if has_chinese_char(&first_char.to_string()) {
            3
        } else {
            2
        };
        let first_col = valid_col[0] as i32;
        col_width[0] = first_width.min(first_col);

        let mut state: Option<WordType> = None;
        for (c_i, ch) in text.chars().enumerate() {
            if ch.is_whitespace() {
                if !word_content.is_empty() {
                    word_list.push(word_content.clone());
                    word_col_list.push(word_col_content.clone());
                    if let Some(s) = state {
                        state_list.push(s);
                    }
                    word_content.clear();
                    word_col_content.clear();
                }
                continue;
            }

            let c_state = if has_chinese_char(&ch.to_string()) {
                WordType::Cn
            } else {
                WordType::EnNum
            };

            if state.is_none() {
                state = Some(c_state);
            }

            if state != Some(c_state) || (c_i < col_width.len() && col_width[c_i] > 5) {
                if !word_content.is_empty() {
                    word_list.push(word_content.clone());
                    word_col_list.push(word_col_content.clone());
                    if let Some(s) = state {
                        state_list.push(s);
                    }
                    word_content.clear();
                    word_col_content.clear();
                }
                state = Some(c_state);
            }

            word_content.push(ch.to_string());
            if c_i < valid_col.len() {
                word_col_content.push(valid_col[c_i]);
            }
        }

        if !word_content.is_empty() {
            word_list.push(word_content);
            word_col_list.push(word_col_content);
            if let Some(s) = state {
                state_list.push(s);
            }
        }

        WordInfo {
            words: word_list,
            word_cols: word_col_list,
            word_types: state_list,
            line_txt_len: 0.0,
            confs: Vec::new(),
        }
    }
}

pub struct TextRecognizer {
    pub cfg: RecConfig,
    pub session: MnnSession,
    decoder: CtcDecoder,
}

impl TextRecognizer {
    pub fn new(cfg: RecConfig) -> Result<Self, EngineError> {
        let session = MnnSession::from_rec_config(&cfg)?;
        let decoder = CtcDecoder::from_cfg(&cfg, &session)?;
        Ok(Self { cfg, session, decoder })
    }

    pub fn run(&mut self, imgs: &[Mat], return_word_box: bool) -> Result<TextRecOutput, EngineError> {
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

        let mut width_list = Vec::with_capacity(img_list.len());
        for img in &img_list {
            let h = img.rows();
            let w = img.cols();
            width_list.push(w as f32 / h.max(1) as f32);
        }

        let mut indices: Vec<usize> = (0..img_list.len()).collect();
        indices.sort_by(|&a, &b| width_list[a].partial_cmp(&width_list[b]).unwrap_or(std::cmp::Ordering::Equal));

        let img_num = img_list.len();
        let batch_num = self.cfg.rec_batch_num as usize;

        let mut all_texts: Vec<(String, f32)> = vec![(String::new(), 0.0); img_num];
        let mut all_word_infos: Vec<Option<WordInfo>> = vec![None; img_num];

        let (img_c, img_h, img_w) = (
            self.cfg.rec_img_shape[0] as usize,
            self.cfg.rec_img_shape[1] as usize,
            self.cfg.rec_img_shape[2] as usize,
        );

        let mut beg = 0usize;
        while beg < img_num {
            let end = (beg + batch_num).min(img_num);

            let mut max_wh_ratio = img_w as f32 / img_h as f32;
            let mut wh_ratio_list: Vec<f32> = Vec::with_capacity(end - beg);
            for &idx in &indices[beg..end] {
                let h = img_list[idx].rows() as f32;
                let w = img_list[idx].cols() as f32;
                let wh_ratio = if h > 0.0 { w / h } else { 1.0 };
                if wh_ratio > max_wh_ratio {
                    max_wh_ratio = wh_ratio;
                }
                wh_ratio_list.push(wh_ratio);
            }

            let mut norm_batch: Vec<Array3<f32>> = Vec::with_capacity(end - beg);
            for &idx in &indices[beg..end] {
                let norm = self.resize_norm_img(&img_list[idx], img_c, img_h, img_w, max_wh_ratio)?;
                norm_batch.push(norm);
            }

            let n = norm_batch.len();
            // Use calculated batch_img_width based on max_wh_ratio, not configured img_w
            let batch_img_width = (img_h as f32 * max_wh_ratio).round() as usize;
            let mut batch = Array4::<f32>::zeros((n, img_c, img_h, batch_img_width));
            for (i, arr) in norm_batch.into_iter().enumerate() {
                batch.slice_mut(ndarray::s![i, .., .., ..]).assign(&arr);
            }

            let preds_dyn = self.session.run(batch.into_dyn())?;
            let preds: Array3<f32> = preds_dyn
                .into_dimensionality::<Ix3>()
                .map_err(|_| EngineError::InvalidInputShape)?;

            let (line_results, batch_word_infos) =
                self.decoder.decode(preds, return_word_box, &wh_ratio_list, max_wh_ratio);

            if return_word_box {
                for (local_idx, ((text, score), info)) in line_results
                    .into_iter()
                    .zip(batch_word_infos.into_iter())
                    .enumerate()
                {
                    let actual_idx = indices[beg + local_idx];
                    all_texts[actual_idx] = (text, score);
                    all_word_infos[actual_idx] = Some(info);
                }
            } else {
                for (local_idx, (text, score)) in line_results.into_iter().enumerate() {
                    let actual_idx = indices[beg + local_idx];
                    all_texts[actual_idx] = (text, score);
                }
            }

            beg = end;
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

    #[cfg(feature = "use-opencv")]
    fn resize_norm_img(
        &self,
        img: &Mat,
        img_c: usize,
        img_h: usize,
        _img_w: usize,
        max_wh_ratio: f32,
    ) -> Result<Array3<f32>, EngineError> {
        let img_width = (img_h as f32 * max_wh_ratio).round() as i32;

        let h = img.rows();
        let w = img.cols();
        if h <= 0 || w <= 0 {
            return Err(EngineError::Preprocess("invalid image size".to_string()));
        }

        let ratio = w as f32 / h as f32;
        let resized_w = if ((img_h as f32) * ratio).ceil() as i32 > img_width {
            img_width
        } else {
            ((img_h as f32) * ratio).ceil() as i32
        };

        let mut resized = Mat::default();
        imgproc::resize(
            img,
            &mut resized,
            core::Size::new(resized_w, img_h as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let size = resized.size()?;
        let h2 = size.height as usize;
        let w2 = size.width as usize;

        // Create zero-padded array like Python: padding_im = np.zeros((img_channel, img_height, img_width))
        // IMPORTANT: Use calculated img_width (from max_wh_ratio), not configured img_w!
        let mut out = Array3::<f32>::zeros((img_c, img_h, img_width as usize));

        // Only fill the resized portion: padding_im[:, :, 0:resized_w] = resized_image
        // The rest remains zeros (padding on the right)
        for y in 0..h2 {
            for x in 0..w2.min(img_width as usize) {  // Ensure we don't exceed img_width
                let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                let b = pix[0] as f32 / 255.0;
                let g = pix[1] as f32 / 255.0;
                let r = pix[2] as f32 / 255.0;

                out[[0, y, x]] = (b - 0.5) / 0.5;
                out[[1, y, x]] = (g - 0.5) / 0.5;
                out[[2, y, x]] = (r - 0.5) / 0.5;
            }
        }

        Ok(out)
    }

    #[cfg(not(feature = "use-opencv"))]
    fn resize_norm_img(
        &self,
        img: &Mat,
        img_c: usize,
        img_h: usize,
        _img_w: usize,
        max_wh_ratio: f32,
    ) -> Result<Array3<f32>, EngineError> {
        let img_width = (img_h as f32 * max_wh_ratio).round() as i32;

        let h = img.rows();
        let w = img.cols();
        if h <= 0 || w <= 0 {
            return Err(EngineError::Preprocess("invalid image size".to_string()));
        }

        let ratio = w as f32 / h as f32;
        let resized_w = if ((img_h as f32) * ratio).ceil() as i32 > img_width {
            img_width
        } else {
            ((img_h as f32) * ratio).ceil() as i32
        };

        let mut resized = Mat::default();
        crate::image_impl::resize(
            img,
            &mut resized,
            Size::new(resized_w, img_h as i32),
            INTER_LINEAR,
        )?;

        let size = resized.size()?;
        let h2 = size.height as usize;
        let w2 = size.width as usize;

        let mut out = Array3::<f32>::zeros((img_c, img_h, img_width as usize));

        for y in 0..h2 {
            for x in 0..w2.min(img_width as usize) {
                let pix = resized.get_pixel(x as u32, y as u32);
                let b = pix[0] as f32 / 255.0;
                let g = pix[1] as f32 / 255.0;
                let r = pix[2] as f32 / 255.0;

                out[[0, y, x]] = (b - 0.5) / 0.5;
                out[[1, y, x]] = (g - 0.5) / 0.5;
                out[[2, y, x]] = (r - 0.5) / 0.5;
            }
        }

        Ok(out)
    }
}

fn has_chinese_char(text: &str) -> bool {
    for ch in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            return true;
        }
    }
    false
}
