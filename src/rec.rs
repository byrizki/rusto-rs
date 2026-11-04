use std::time::Instant;

use ndarray::{Array3, Array4, Ix3};
use rayon::prelude::*;

#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(feature = "use-opencv")]
use opencv::core::Mat;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Size, INTER_LINEAR};

use crate::engine::{EngineError, EngineSession, InferenceSession};
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

pub struct CtcDecoder {
    pub chars: Vec<String>,
}

impl CtcDecoder {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, EngineError> {
        let path_ref = path.as_ref();
        let content = std::fs::read_to_string(path_ref).map_err(|e| {
            EngineError::Preprocess(format!("failed to open rec_keys_path: {e}"))
        })?;

        if path_ref.extension().and_then(|s| s.to_str()) == Some("json")
            || content.trim_start().starts_with('{')
        {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                let chars_val = if let Some(c) = val.get("chars") {
                    Some(c.clone())
                } else if let Some(v) = val.get("vocab") {
                    if let Some(s) = v.as_str() {
                        serde_json::from_str::<serde_json::Value>(s)
                            .ok()
                            .and_then(|pv| pv.get("chars").cloned())
                    } else {
                        v.get("chars").cloned()
                    }
                } else {
                    None
                };
                if let Some(chars) = chars_val.and_then(|c| c.as_array().cloned()) {
                    let mut list = Vec::with_capacity(chars.len() + 1);
                    list.push("blank".to_string());
                    for c in chars {
                        if let Some(s) = c.as_str() {
                            list.push(s.to_string());
                        }
                    }
                    return Ok(Self { chars: list });
                }
            }
        }

        let mut list = Vec::new();
        for line in content.lines() {
            list.push(line.to_string());
        }
        list.push(" ".to_string());
        list.insert(0, "blank".to_string());
        Ok(Self { chars: list })
    }

    fn from_cfg(cfg: &RecConfig, session: &impl InferenceSession) -> Result<Self, EngineError> {
        if session.have_key("character") {
            if let Some(mut list) = session.get_character_list("character") {
                list.push(" ".to_string());
                list.insert(0, "blank".to_string());
                return Ok(Self { chars: list });
            }
        }

        if let Some(path) = &cfg.rec_keys_path {
            return Self::from_file(path);
        }

        Err(EngineError::Preprocess("no character list found for recognizer".to_string()))
    }

    pub fn decode(
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

            let batch_view = preds.index_axis(ndarray::Axis(0), b);
            for ti in 0..t {
                let row_view = batch_view.index_axis(ndarray::Axis(0), ti);
                let (best_idx, best_val) = if let Some(row_slice) = row_view.as_slice() {
                    let mut b_idx = 0usize;
                    let mut b_val = f32::MIN;
                    for (ci, &v) in row_slice.iter().enumerate() {
                        if v > b_val {
                            b_val = v;
                            b_idx = ci;
                        }
                    }
                    (b_idx, b_val)
                } else {
                    let mut b_idx = 0usize;
                    let mut b_val = f32::MIN;
                    for ci in 0..c {
                        let v = row_view[ci];
                        if v > b_val {
                            b_val = v;
                            b_idx = ci;
                        }
                    }
                    (b_idx, b_val)
                };

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
    pub session: EngineSession,
    decoder: CtcDecoder,
}

impl TextRecognizer {
    pub fn new(cfg: RecConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_rec_config(&cfg)?;
        let decoder = CtcDecoder::from_cfg(&cfg, &session)?;
        Ok(Self { cfg, session, decoder })
    }

    /// Construct recognizer with an EngineSession
    pub fn with_engine_session(
        cfg: RecConfig,
        session: EngineSession,
    ) -> Result<Self, EngineError> {
        let decoder = CtcDecoder::from_cfg(&cfg, &session)?;
        Ok(Self { cfg, session, decoder })
    }

    /// Construct recognizer with a custom inference session implementation
    pub fn with_custom_session(
        cfg: RecConfig,
        session: impl InferenceSession + 'static,
    ) -> Result<Self, EngineError> {
        let session = EngineSession::from_custom(session);
        let decoder = CtcDecoder::from_cfg(&cfg, &session)?;
        Ok(Self { cfg, session, decoder })
    }

    pub fn is_linenet(&self) -> bool {
        self.session.input_channels() == 1
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

        if self.is_linenet() {
            let img_num = img_list.len();
            let batch_num = (self.cfg.rec_batch_num as usize).max(1);
            let mut all_texts: Vec<(String, f32)> = vec![(String::new(), 0.0); img_num];
            let all_word_infos: Vec<Option<WordInfo>> = vec![None; img_num];

            let rec_h = 96usize;
            let mut beg = 0usize;
            while beg < img_num {
                let end = (beg + batch_num).min(img_num);
                let current_batch = &img_list[beg..end];

                let mut target_widths = Vec::with_capacity(current_batch.len());
                for img in current_batch {
                    let h = img.rows() as f32;
                    let w = img.cols() as f32;
                    let target_w = if h > 0.0 {
                        (96.0 * (w / h)).round() as usize
                    } else {
                        96
                    };
                    let aligned_w = (target_w.div_ceil(8) * 8).clamp(8, 4096);
                    target_widths.push(aligned_w);
                }

                let max_w = *target_widths.iter().max().unwrap_or(&96);
                let n = current_batch.len();
                let mut batch_tensor = Array4::<f32>::zeros((n, 1, rec_h, max_w));
                let mut widths = Vec::with_capacity(n);

                for (i, img) in current_batch.iter().enumerate() {
                    let w = target_widths[i];
                    widths.push(w as i32);
                    let ink_crop = prepare_linenet_rec_crop(img, rec_h, w)?;
                    for y in 0..rec_h {
                        for x in 0..w {
                            batch_tensor[[i, 0, y, x]] = ink_crop[[y, x]];
                        }
                    }
                }

                let (batch_ids, batch_confs) = self.session.run_linenet_rec(batch_tensor, widths)?;
                for (local_idx, (tokens, confs)) in batch_ids.into_iter().zip(batch_confs.into_iter()).enumerate() {
                    let actual_idx = beg + local_idx;
                    let mut chars = Vec::new();
                    let mut scores = Vec::new();
                    for (token_id, conf) in tokens.into_iter().zip(confs.into_iter()) {
                        let idx = token_id as usize;
                        if idx > 0 && idx < self.decoder.chars.len() {
                            chars.push(self.decoder.chars[idx].clone());
                            scores.push(conf);
                        }
                    }
                    let text = chars.concat();
                    let score = if scores.is_empty() {
                        0.0
                    } else {
                        scores.iter().sum::<f32>() / scores.len() as f32
                    };
                    all_texts[actual_idx] = (text, score);
                }

                beg = end;
            }

            let (txts, scores): (Vec<String>, Vec<f32>) = all_texts.into_iter().unzip();
            let elapse = start.elapsed().as_secs_f64();
            return Ok(TextRecOutput {
                imgs: img_list,
                txts,
                scores,
                word_infos: all_word_infos,
                elapse,
            });
        }

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

            #[cfg(not(feature = "use-opencv"))]
            let norm_batch: Vec<Array3<f32>> = indices[beg..end]
                .par_iter()
                .map(|&idx| resize_norm_rec_img(&img_list[idx], img_c, img_h, max_wh_ratio))
                .collect::<Result<Vec<_>, _>>()?;

            #[cfg(feature = "use-opencv")]
            let mut norm_batch: Vec<Array3<f32>> = Vec::with_capacity(end - beg);
            #[cfg(feature = "use-opencv")]
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

    #[allow(dead_code)]
    #[cfg(not(feature = "use-opencv"))]
    fn resize_norm_img(
        &self,
        img: &Mat,
        img_c: usize,
        img_h: usize,
        _img_w: usize,
        max_wh_ratio: f32,
    ) -> Result<Array3<f32>, EngineError> {
        resize_norm_rec_img(img, img_c, img_h, max_wh_ratio)
    }
}

static NORM_LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0f32; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        *item = (i as f32 / 255.0 - 0.5) / 0.5;
    }
    lut
});

#[cfg(not(feature = "use-opencv"))]
pub fn resize_norm_rec_img(
    img: &Mat,
    img_c: usize,
    img_h: usize,
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

    let target_w = img_width as usize;
    let plane_size = img_h * target_w;
    let valid_w = w2.min(target_w);
    let valid_h = h2.min(img_h);

    let mut out = Array3::<f32>::zeros((img_c, img_h, target_w));

    if img_c == 3 && resized.as_dynamic().as_rgb8().is_some() {
        let rgb = resized.as_dynamic().as_rgb8().unwrap();
        let raw: &[u8] = rgb.as_raw();
        let stride = w2 * 3;
        let out_slice = out.as_slice_mut().expect("contiguous array");
        let (b_plane, rest) = out_slice.split_at_mut(plane_size);
        let (g_plane, r_plane) = rest.split_at_mut(plane_size);

        for y in 0..valid_h {
            let row = &raw[y * stride..y * stride + valid_w * 3];
            let b_row = &mut b_plane[y * target_w..y * target_w + valid_w];
            let g_row = &mut g_plane[y * target_w..y * target_w + valid_w];
            let r_row = &mut r_plane[y * target_w..y * target_w + valid_w];

            for x in 0..valid_w {
                b_row[x] = NORM_LUT[row[x * 3 + 2] as usize];
                g_row[x] = NORM_LUT[row[x * 3 + 1] as usize];
                r_row[x] = NORM_LUT[row[x * 3] as usize];
            }
        }
    } else {
        for y in 0..valid_h {
            for x in 0..valid_w {
                let pix = resized.get_pixel(x as u32, y as u32);
                out[[0, y, x]] = NORM_LUT[pix[2] as usize];
                out[[1, y, x]] = NORM_LUT[pix[1] as usize];
                out[[2, y, x]] = NORM_LUT[pix[0] as usize];
            }
        }
    }

    Ok(out)
}

fn has_chinese_char(text: &str) -> bool {
    for ch in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            return true;
        }
    }
    false
}

fn prepare_linenet_rec_crop(
    img: &Mat,
    target_h: usize,
    target_w: usize,
) -> Result<ndarray::Array2<f32>, EngineError> {
    let mut out = ndarray::Array2::<f32>::zeros((target_h, target_w));
    let orig_w = img.cols() as f32;
    let orig_h = img.rows() as f32;
    if orig_w <= 0.0 || orig_h <= 0.0 {
        return Ok(out);
    }
    let scale_w = target_w as f32 / orig_w;
    let scale_h = target_h as f32 / orig_h;

    #[cfg(feature = "use-opencv")]
    {
        for ty in 0..target_h {
            let sy = ((ty as f32 + 0.5) / scale_h - 0.5).clamp(0.0, orig_h - 1.0);
            let y0 = sy.floor() as i32;
            let y1 = (y0 + 1).min(img.rows() - 1);
            let fy = sy - y0 as f32;

            for tx in 0..target_w {
                let sx = ((tx as f32 + 0.5) / scale_w - 0.5).clamp(0.0, orig_w - 1.0);
                let x0 = sx.floor() as i32;
                let x1 = (x0 + 1).min(img.cols() - 1);
                let fx = sx - x0 as f32;

                let p00 = img.at_2d::<opencv::core::Vec3b>(y0, x0)?;
                let p10 = img.at_2d::<opencv::core::Vec3b>(y0, x1)?;
                let p01 = img.at_2d::<opencv::core::Vec3b>(y1, x0)?;
                let p11 = img.at_2d::<opencv::core::Vec3b>(y1, x1)?;

                let b = (p00[0] as f32 * (1.0 - fx) + p10[0] as f32 * fx) * (1.0 - fy)
                    + (p01[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
                let g = (p00[1] as f32 * (1.0 - fx) + p10[1] as f32 * fx) * (1.0 - fy)
                    + (p01[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
                let r = (p00[2] as f32 * (1.0 - fx) + p10[2] as f32 * fx) * (1.0 - fy)
                    + (p01[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                out[[ty, tx]] = ink;
            }
        }
    }

    #[cfg(not(feature = "use-opencv"))]
    {
        for ty in 0..target_h {
            let sy = ((ty as f32 + 0.5) / scale_h - 0.5).clamp(0.0, orig_h - 1.0);
            let y0 = sy.floor() as u32;
            let y1 = (y0 + 1).min(img.rows() as u32 - 1);
            let fy = sy - y0 as f32;

            for tx in 0..target_w {
                let sx = ((tx as f32 + 0.5) / scale_w - 0.5).clamp(0.0, orig_w - 1.0);
                let x0 = sx.floor() as u32;
                let x1 = (x0 + 1).min(img.cols() as u32 - 1);
                let fx = sx - x0 as f32;

                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x1, y0);
                let p01 = img.get_pixel(x0, y1);
                let p11 = img.get_pixel(x1, y1);

                let r = (p00[0] as f32 * (1.0 - fx) + p10[0] as f32 * fx) * (1.0 - fy)
                    + (p01[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
                let g = (p00[1] as f32 * (1.0 - fx) + p10[1] as f32 * fx) * (1.0 - fy)
                    + (p01[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
                let b = (p00[2] as f32 * (1.0 - fx) + p10[2] as f32 * fx) * (1.0 - fy)
                    + (p01[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                out[[ty, tx]] = ink;
            }
        }
    }

    Ok(out)
}
