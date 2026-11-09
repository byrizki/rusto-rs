//! Reading order sequencing and line feature extraction.
//!
//! Provides coordinate and text feature extraction alongside pairwise logit
//! decoding for neural reading order models (e.g. Indra OCR OrderNet).

use std::cmp::Ordering;
use std::collections::HashMap;
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::order_blocks::decode_blocks_order;
use crate::tensor::ArrayD;

const N_GEOM: usize = 10;
const N_TEXT: usize = 6;
pub const N_ORDER_FEATS: usize = N_GEOM + N_TEXT;

fn is_wide_char(c: char) -> bool {
    matches!(c,
        '\u{1100}'..='\u{115F}'
        | '\u{2E80}'..='\u{A4CF}'
        | '\u{AC00}'..='\u{D7A3}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{FE10}'..='\u{FE19}'
        | '\u{FE30}'..='\u{FE6F}'
        | '\u{FF01}'..='\u{FF60}'
        | '\u{FFE0}'..='\u{FFE6}'
        | '\u{20000}'..='\u{323AF}'
    )
}

fn display_len(t: &str) -> usize {
    t.chars().map(|c| if is_wide_char(c) { 2 } else { 1 }).sum()
}

/// Computes a flat `[N, 16]` feature matrix for `N` text lines.
pub fn compute_line_features(
    boxes: &[[f32; 4]],
    page_w: f32,
    page_h: f32,
    texts: &[String],
) -> Vec<f32> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }
    let safe_w = page_w.max(1.0);
    let safe_h = page_h.max(1.0);

    let mut heights = Vec::with_capacity(n);
    for b in boxes {
        let h = (b[3] - b[1]).max(1e-3);
        heights.push(h);
    }
    let mut sorted_h = heights.clone();
    sorted_h.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let med_h = if n % 2 == 1 {
        sorted_h[n / 2]
    } else {
        0.5 * (sorted_h[n / 2 - 1] + sorted_h[n / 2])
    };

    let mut feats = vec![0.0f32; n * N_ORDER_FEATS];
    let sentence_ends = ['.', '!', '?', ':', ';', '。', '！', '？', '：', '；', '．'];

    for i in 0..n {
        let b = boxes[i];
        let w = (b[2] - b[0]).max(1e-3);
        let h = (b[3] - b[1]).max(1e-3);
        let base = i * N_ORDER_FEATS;

        // 10 geometric features
        feats[base + 0] = b[0] / safe_w;
        feats[base + 1] = b[1] / safe_h;
        feats[base + 2] = b[2] / safe_w;
        feats[base + 3] = b[3] / safe_h;
        feats[base + 4] = (b[0] + b[2]) / (2.0 * safe_w);
        feats[base + 5] = (b[1] + b[3]) / (2.0 * safe_h);
        feats[base + 6] = w / safe_w;
        feats[base + 7] = (h / med_h).ln();
        feats[base + 8] = (w / h).ln() / 4.0;
        feats[base + 9] = (med_h / safe_h) * 20.0;

        // 6 text features
        if i < texts.len() {
            let t = texts[i].trim();
            if !t.is_empty() {
                let d_len = display_len(t);
                feats[base + 10] = (1.0 + d_len as f32).ln() / 5.0;

                let char_count = t.chars().count();
                let digit_count = t.chars().filter(|c| c.is_ascii_digit()).count();
                feats[base + 11] = digit_count as f32 / char_count.max(1) as f32;

                let first_char = t.chars().next().unwrap();
                let is_bullet_char = matches!(first_char, '•' | '-' | '–' | '▪' | '◦' | '*' | '·' | '→');
                let is_num_bullet = char_count > 3 && {
                    let prefix: String = t.chars().take(3).collect();
                    let prefix_trimmed = prefix.trim_end_matches(['.', ')']);
                    !prefix_trimmed.is_empty() && prefix_trimmed.chars().all(|c| c.is_ascii_digit())
                };
                feats[base + 12] = if is_bullet_char || is_num_bullet { 1.0 } else { 0.0 };

                feats[base + 13] = if t.ends_with('-') { 1.0 } else { 0.0 };

                let last_char = t.chars().last().unwrap();
                feats[base + 14] = if sentence_ends.contains(&last_char) { 1.0 } else { 0.0 };

                feats[base + 15] = if first_char.is_uppercase() { 1.0 } else { 0.0 };
            }
        }
    }

    feats
}

/// Decodes pairwise relation logits `[N, N]` into sequential reading order indices.
pub fn decode_order_logits(a: &[f32], n: usize, sweeps: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }

    let mut lp = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let val = a[i * n + j];
            lp[i * n + j] = if val >= 0.0 {
                -(1.0 + (-val).exp()).ln()
            } else {
                val - (1.0 + val.exp()).ln()
            };
        }
    }

    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let row_sum: f32 = (0..n).map(|j| lp[i * n + j]).sum();
            let col_sum: f32 = (0..n).map(|j| lp[j * n + i]).sum();
            (row_sum - col_sum, i)
        })
        .collect();

    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    let mut order: Vec<usize> = scores.into_iter().map(|s| s.1).collect();

    for _ in 0..sweeps {
        let mut improved = false;
        let current_order = order.clone();

        for &x in &current_order {
            let rest: Vec<usize> = order.iter().copied().filter(|&v| v != x).collect();
            let m = rest.len();

            let mut before = Vec::with_capacity(m);
            let mut after = Vec::with_capacity(m);
            for &r in &rest {
                before.push(lp[r * n + x]);
                after.push(lp[x * n + r]);
            }

            let mut cum_before = vec![0.0f32; m + 1];
            for j in 0..m {
                cum_before[j + 1] = cum_before[j] + before[j];
            }
            let mut cum_after_rev = vec![0.0f32; m + 1];
            for j in (0..m).rev() {
                cum_after_rev[j] = cum_after_rev[j + 1] + after[j];
            }

            let mut best_gain = f32::NEG_INFINITY;
            let mut best_k = 0;
            for k in 0..=m {
                let gain = cum_before[k] + cum_after_rev[k];
                if gain > best_gain {
                    best_gain = gain;
                    best_k = k;
                }
            }

            let cur_k = order.iter().position(|&v| v == x).unwrap();
            let cur_gain = cum_before[cur_k.min(m)] + cum_after_rev[cur_k.min(m)];

            if best_gain > cur_gain + 1e-6 {
                let mut new_order = rest;
                new_order.insert(best_k, x);
                order = new_order;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    order
}

/// Executes neural reading order prediction on extracted text lines.
pub fn predict_reading_order(
    session: &mut dyn InferenceSession,
    boxes: &[[f32; 4]],
    page_w: f32,
    page_h: f32,
    texts: &[String],
) -> Result<Vec<usize>, CoreError> {
    let n = boxes.len();
    if n <= 1 {
        return Ok((0..n).collect());
    }

    const MAX_ORDER_N: usize = 1024;
    if n <= MAX_ORDER_N {
        predict_reading_order_block(session, boxes, page_w, page_h, texts)
    } else {
        predict_reading_order_banded(session, boxes, page_w, page_h, texts, MAX_ORDER_N)
    }
}

fn predict_reading_order_block(
    session: &mut dyn InferenceSession,
    boxes: &[[f32; 4]],
    page_w: f32,
    page_h: f32,
    texts: &[String],
) -> Result<Vec<usize>, CoreError> {
    let n = boxes.len();
    let feats = compute_line_features(boxes, page_w, page_h, texts);
    let feats_arr = ArrayD::from_shape_vec(vec![1, n, N_ORDER_FEATS], feats)
        .map_err(|e| CoreError::InvalidShape(e.to_string()))?;
    let mask_vec = vec![1.0f32; n];
    let mask_arr = ArrayD::from_shape_vec(vec![1, n], mask_vec)
        .map_err(|e| CoreError::InvalidShape(e.to_string()))?;

    let mut inputs = HashMap::new();
    inputs.insert("feats".to_string(), feats_arr);
    inputs.insert("mask".to_string(), mask_arr);

    let outputs = session.run_with_inputs(inputs)?;
    let a_tensor = outputs
        .get("a")
        .or_else(|| outputs.get("output"))
        .or_else(|| outputs.values().next())
        .ok_or_else(|| CoreError::Model("Reading order model produced no output logits 'a'".into()))?;

    let a_slice = a_tensor.as_slice();
    if a_slice.len() < n * n {
        return Err(CoreError::InvalidShape(format!(
            "Reading order logits output length {} < N*N {}",
            a_slice.len(),
            n * n
        )));
    }

    if let (Some(same), Some(cls)) = (outputs.get("same"), outputs.get("cls")) {
        let same_slice = same.as_slice();
        if same_slice.len() >= n * n {
            return Ok(decode_blocks_order(
                &a_slice[..n * n],
                &same_slice[..n * n],
                boxes,
                Some(cls.as_slice()),
            ));
        }
    }

    Ok(decode_order_logits(a_slice, n, 4))
}

fn predict_reading_order_banded(
    session: &mut dyn InferenceSession,
    boxes: &[[f32; 4]],
    page_w: f32,
    page_h: f32,
    texts: &[String],
    max_chunk: usize,
) -> Result<Vec<usize>, CoreError> {
    let n = boxes.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&i, &j| boxes[i][1].partial_cmp(&boxes[j][1]).unwrap_or(Ordering::Equal));

    let mut final_order = Vec::with_capacity(n);
    for chunk in sorted_indices.chunks(max_chunk) {
        let chunk_boxes: Vec<[f32; 4]> = chunk.iter().map(|&idx| boxes[idx]).collect();
        let chunk_texts: Vec<String> = chunk
            .iter()
            .map(|&idx| texts.get(idx).cloned().unwrap_or_default())
            .collect();
        let local_order =
            predict_reading_order_block(session, &chunk_boxes, page_w, page_h, &chunk_texts)?;
        for local_idx in local_order {
            final_order.push(chunk[local_idx]);
        }
    }
    Ok(final_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_line_features_dimensions() {
        let boxes = vec![
            [10.0, 10.0, 100.0, 30.0],
            [10.0, 40.0, 120.0, 60.0],
            [10.0, 70.0, 150.0, 90.0],
        ];
        let texts = vec![
            "Hello world.".to_string(),
            "123 items-".to_string(),
            "Goòt APRON GINGHAM. 5c".to_string(),
        ];
        let feats = compute_line_features(&boxes, 500.0, 500.0, &texts);
        assert_eq!(feats.len(), 3 * N_ORDER_FEATS);
        assert!(feats[0] > 0.0);
        assert_eq!(feats[14], 1.0); // ends with sentence dot
        assert_eq!(feats[16 + 13], 1.0); // ends with hyphen
    }

    #[test]
    fn test_decode_order_logits_inversion() {
        // Line 0 before 1: logit a[0, 1] = 5.0, a[1, 0] = -5.0
        let a = vec![
            0.0, 5.0,
            -5.0, 0.0,
        ];
        let order = decode_order_logits(&a, 2, 4);
        assert_eq!(order, vec![0, 1]);

        // Line 1 before 0: logit a[0, 1] = -5.0, a[1, 0] = 5.0
        let a_rev = vec![
            0.0, -5.0,
            5.0, 0.0,
        ];
        let order_rev = decode_order_logits(&a_rev, 2, 4);
        assert_eq!(order_rev, vec![1, 0]);
    }
}
