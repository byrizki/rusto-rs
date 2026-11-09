use crate::tensor::Array3;

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

#[derive(Clone, Debug)]
pub struct CtcDecoder {
    pub chars: Vec<String>,
}

impl CtcDecoder {
    pub fn new(chars: Vec<String>) -> Self {
        Self { chars }
    }

    pub fn from_model_character_list(mut list: Vec<String>) -> Self {
        list.push(" ".to_string());
        list.insert(0, "blank".to_string());
        Self::new(list)
    }

    pub fn from_keys_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, crate::error::CoreError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| crate::error::CoreError::Preprocess(format!("failed to open rec_keys_path: {e}")))?;
        Ok(Self::from_string(&content))
    }

    pub fn from_string(content: &str) -> Self {
        #[derive(nanoserde::DeJson, Default)]
        struct MetaFile {
            vocab: Option<String>,
            chars: Option<Vec<String>>,
        }
        #[derive(nanoserde::DeJson, Default)]
        struct VocabFile {
            chars: Option<Vec<String>>,
        }

        let trimmed = content.trim_start();
        if trimmed.starts_with('{') {
            if let Ok(meta) = <MetaFile as nanoserde::DeJson>::deserialize_json(content) {
                let chars_opt = if let Some(c) = meta.chars {
                    Some(c)
                } else if let Some(v_str) = meta.vocab {
                    <VocabFile as nanoserde::DeJson>::deserialize_json(&v_str)
                        .ok()
                        .and_then(|v| v.chars)
                } else {
                    None
                };

                if let Some(chars) = chars_opt {
                    let mut list = Vec::with_capacity(chars.len() + 1);
                    list.push("blank".to_string());
                    for c in chars {
                        list.push(c);
                    }
                    return Self { chars: list };
                }
            }
        } else if trimmed.starts_with('[') {
            if let Ok(tokens) = <Vec<String> as nanoserde::DeJson>::deserialize_json(content) {
                return Self::from_chars(tokens);
            }
        }

        let mut list = Vec::new();
        for line in content.lines() {
            list.push(line.to_string());
        }
        list.push(" ".to_string());
        list.insert(0, "blank".to_string());
        Self { chars: list }
    }

    pub fn decode_linenet_tokens(&self, ids: &[i32], confs: &[f32], max_len: usize) -> Option<(String, f32)> {
        self.decode_linenet_tokens_with_positions(ids, confs, max_len)
            .map(|(t, s, _)| (t, s))
    }

    pub fn decode_linenet_tokens_with_positions(
        &self,
        ids: &[i32],
        confs: &[f32],
        max_len: usize,
    ) -> Option<(String, f32, Vec<f32>)> {
        let actual_len = max_len.min(ids.len());
        if actual_len == 0 {
            return None;
        }

        let mut kept: Vec<(usize, f32, &str)> = Vec::new();
        let mut last_id = 0i32;

        let total_t = actual_len as f32;
        for step in 0..actual_len {
            let id = ids[step];
            if id != 0 && (step == 0 || id != last_id) {
                if let Some(ch) = self.chars.get(id as usize) {
                    let score = confs.get(step).copied().unwrap_or(0.0);
                    kept.push((step, score, ch.as_str()));
                }
            }
            last_id = id;
        }

        if kept.is_empty() {
            return None;
        }

        let avg_conf = kept.iter().map(|&(_, s, _)| s).sum::<f32>() / kept.len() as f32;

        let mut text = String::new();
        let mut raw_positions = Vec::new();
        for (k, &(step, _, ch)) in kept.iter().enumerate() {
            text.push_str(ch);
            let pos = (step as f32 + 0.5) / total_t;
            for _ in ch.chars() {
                raw_positions.push(pos);
            }
            if ch.len() > 2 && ch.starts_with('\\') && ch[1..].chars().all(|c| c.is_alphabetic()) {
                if let Some(&(_, _, nxt)) = kept.get(k + 1) {
                    if nxt.chars().next().is_some_and(|c| c.is_alphabetic()) {
                        text.push(' ');
                        raw_positions.push(pos);
                    }
                }
            }
        }

        let leading_spaces = text.chars().take_while(|c| c.is_whitespace()).count();
        let trailing_spaces = text.chars().rev().take_while(|c| c.is_whitespace()).count();
        let total_chars = text.chars().count();

        if leading_spaces >= total_chars {
            return None;
        }

        let trimmed = text.trim().to_string();
        let end_idx = total_chars.saturating_sub(trailing_spaces);
        let start_idx = leading_spaces.min(end_idx);

        let positions = if start_idx <= end_idx && end_idx <= raw_positions.len() {
            raw_positions[start_idx..end_idx].to_vec()
        } else {
            raw_positions
        };

        Some((trimmed, avg_conf, positions))
    }

    pub fn decode_linenet_batch(
        &self,
        batch_ids: Vec<Vec<i32>>,
        batch_confs: Vec<Vec<f32>>,
    ) -> Vec<(String, f32)> {
        batch_ids
            .into_iter()
            .zip(batch_confs.into_iter())
            .map(|(tokens, confs)| {
                let len = tokens.len();
                self.decode_linenet_tokens(&tokens, &confs, len)
                    .unwrap_or_else(|| (String::new(), 0.0))
            })
            .collect()
    }

    pub fn decode_linenet_batch_with_positions(
        &self,
        batch_ids: Vec<Vec<i32>>,
        batch_confs: Vec<Vec<f32>>,
    ) -> Vec<Option<(String, f32, Vec<f32>)>> {
        batch_ids
            .into_iter()
            .zip(batch_confs.into_iter())
            .map(|(tokens, confs)| {
                let len = tokens.len();
                self.decode_linenet_tokens_with_positions(&tokens, &confs, len)
            })
            .collect()
    }

    pub fn from_chars(mut chars: Vec<String>) -> Self {
        if !chars.is_empty() && chars[0] != "blank" && !chars.contains(&"blank".to_string()) {
            chars.insert(0, "blank".to_string());
        }
        if !chars.contains(&" ".to_string()) {
            chars.push(" ".to_string());
        }
        Self { chars }
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
        let preds_slice = preds.as_slice();

        for b in 0..n {
            let mut token_indices = Vec::with_capacity(t);
            let mut token_probs = Vec::with_capacity(t);
            let b_offset = b * t * c;

            for ti in 0..t {
                let row_start = b_offset + ti * c;
                let row_slice = &preds_slice[row_start..row_start + c];
                let mut b_idx = 0usize;
                let mut b_val = f32::MIN;
                for (ci, &v) in row_slice.iter().enumerate() {
                    if v > b_val {
                        b_val = v;
                        b_idx = ci;
                    }
                }

                token_indices.push(b_idx);
                token_probs.push(b_val);
            }

            let ignored = self.get_ignored_tokens();
            let mut selection = vec![true; t];

            for i in 0..t {
                if ignored.contains(&token_indices[i]) {
                    selection[i] = false;
                }
                if i > 0 && token_indices[i - 1] == token_indices[i] {
                    selection[i] = false;
                }
            }

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

    pub fn decode_single_crop(
        &self,
        rec_data: &[f32],
        seq_len: usize,
        num_classes: usize,
    ) -> Option<(String, f32)> {
        let mut text = String::new();
        let mut char_scores = Vec::new();
        let mut last_idx = 0usize;

        for step in 0..seq_len {
            let step_offset = step * num_classes;
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;

            for c in 0..num_classes {
                let val = rec_data[step_offset + c];
                if val > best_val {
                    best_val = val;
                    best_idx = c;
                }
            }

            if best_idx != 0 && (step == 0 || best_idx != last_idx) {
                if let Some(ch) = self.chars.get(best_idx) {
                    text.push_str(ch);
                    char_scores.push(best_val);
                }
            }
            last_idx = best_idx;
        }

        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        let avg_score = if char_scores.is_empty() {
            0.0
        } else {
            char_scores.iter().sum::<f32>() / char_scores.len() as f32
        };

        Some((trimmed.to_string(), avg_score))
    }

    fn get_ignored_tokens(&self) -> Vec<usize> {
        vec![0]
    }

    fn get_word_info(&self, text: &str, selection: &[bool]) -> WordInfo {
        let mut words = Vec::new();
        let mut word_cols = Vec::new();
        let mut word_types = Vec::new();

        let mut current_word = Vec::new();
        let mut current_cols = Vec::new();
        let mut current_type: Option<WordType> = None;

        let mut sel_indices = Vec::new();
        for (i, &sel) in selection.iter().enumerate() {
            if sel {
                sel_indices.push(i);
            }
        }

        let chars_vec: Vec<char> = text.chars().collect();
        for (idx, &ch) in chars_vec.iter().enumerate() {
            let col = if idx < sel_indices.len() {
                sel_indices[idx]
            } else {
                0
            };

            let char_type = if ch.is_ascii_alphanumeric() || ch.is_ascii_punctuation() {
                WordType::EnNum
            } else {
                WordType::Cn
            };

            if let Some(ct) = current_type {
                if ct == char_type && ct == WordType::EnNum {
                    current_word.push(ch.to_string());
                    current_cols.push(col);
                } else {
                    if !current_word.is_empty() {
                        words.push(current_word);
                        word_cols.push(current_cols);
                        word_types.push(ct);
                    }
                    current_word = vec![ch.to_string()];
                    current_cols = vec![col];
                    current_type = Some(char_type);
                }
            } else {
                current_word.push(ch.to_string());
                current_cols.push(col);
                current_type = Some(char_type);
            }
        }

        if !current_word.is_empty() {
            if let Some(ct) = current_type {
                words.push(current_word);
                word_cols.push(current_cols);
                word_types.push(ct);
            }
        }

        WordInfo {
            words,
            word_cols,
            word_types,
            line_txt_len: 0.0,
            confs: Vec::new(),
        }
    }
}


pub fn composite_ocr_score(det_score: f32, rec_score: f32) -> f32 {
    let raw = det_score * 0.3 + 0.7 * rec_score.clamp(0.0, 1.0);
    ((raw * 1000.0).round() / 1000.0).clamp(0.0, 1.0)
}

pub fn scatter_rec_results(
    indices: &[usize],
    line_results: Vec<(String, f32)>,
    word_infos: Option<Vec<WordInfo>>,
    all_texts: &mut [(String, f32)],
    all_word_infos: &mut [Option<WordInfo>],
) {
    let return_word_box = word_infos.is_some();
    let word_infos = word_infos.unwrap_or_default();
    for (local_idx, (text, score)) in line_results.into_iter().enumerate() {
        if local_idx < indices.len() {
            let actual_idx = indices[local_idx];
            all_texts[actual_idx] = (text, score);
            if return_word_box && local_idx < word_infos.len() {
                all_word_infos[actual_idx] = Some(word_infos[local_idx].clone());
            }
        }
    }
}

pub fn scatter_linenet_results(
    chunk: &[usize],
    decoded: Vec<(String, f32)>,
    all_texts: &mut [(String, f32)],
) {
    for (local_idx, res) in decoded.into_iter().enumerate() {
        if local_idx < chunk.len() {
            all_texts[chunk[local_idx]] = res;
        }
    }
}

pub fn scatter_linenet_results_with_positions(
    chunk: &[usize],
    decoded: Vec<Option<(String, f32, Vec<f32>)>>,
    all_texts: &mut [(String, f32)],
    all_positions: &mut [Vec<f32>],
) {
    for (local_idx, res) in decoded.into_iter().enumerate() {
        if local_idx < chunk.len() {
            let actual_idx = chunk[local_idx];
            if let Some((text, score, positions)) = res {
                all_texts[actual_idx] = (text, score);
                all_positions[actual_idx] = positions;
            } else {
                all_texts[actual_idx] = (String::new(), 0.0);
                all_positions[actual_idx] = Vec::new();
            }
        }
    }
}

#[cfg(test)]
mod linenet_tests {
    use super::*;

    fn decoder() -> CtcDecoder {
        CtcDecoder::new(
            ["blank", "a", "S", "\\in", "x", " "].iter().map(|s| s.to_string()).collect(),
        )
    }

    #[test]
    fn test_confidence_is_averaged_per_ctc_step() {
        let ids = [3, 3, 0, 1, 0];
        let confs = [0.2, 0.9, 0.0, 1.0, 0.0];
        let (text, conf, positions) =
            decoder().decode_linenet_tokens_with_positions(&ids, &confs, ids.len()).unwrap();
        assert_eq!(text, "\\in a");
        assert!((conf - 0.6).abs() < 1e-6);
        assert_eq!(positions.len(), text.chars().count());
    }

    #[test]
    fn test_latex_command_gets_space_before_letter() {
        let ids = [3, 2, 0, 3, 5, 4];
        let confs = [1.0; 6];
        let (text, _, positions) =
            decoder().decode_linenet_tokens_with_positions(&ids, &confs, ids.len()).unwrap();
        assert_eq!(text, "\\in S\\in x");
        assert_eq!(positions.len(), text.chars().count());
    }
}
