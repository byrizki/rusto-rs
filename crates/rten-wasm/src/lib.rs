use wasm_bindgen::prelude::*;
use rten::{Dimension, Model, NodeId, Value};
use rten_tensor::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
pub struct InferenceOutput {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[wasm_bindgen]
impl InferenceOutput {
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
}

#[wasm_bindgen]
pub struct RtenModel {
    model: Model,
}

#[wasm_bindgen]
impl RtenModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: Vec<u8>) -> Result<RtenModel, String> {
        let model = Model::load(model_data).map_err(|e| e.to_string())?;
        Ok(RtenModel { model })
    }

    #[wasm_bindgen(js_name = findNode)]
    pub fn find_node(&self, name: &str) -> Option<u32> {
        self.model.find_node(name).map(|id| id.as_u32())
    }

    #[wasm_bindgen(js_name = inputIds)]
    pub fn input_ids(&self) -> Vec<u32> {
        self.model.input_ids().iter().map(|id| id.as_u32()).collect()
    }

    #[wasm_bindgen(js_name = outputIds)]
    pub fn output_ids(&self) -> Vec<u32> {
        self.model.output_ids().iter().map(|id| id.as_u32()).collect()
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
    ) -> Result<InferenceOutput, String> {
        let input_id = self.model.input_ids().first().copied()
            .ok_or_else(|| "No input node found in model".to_string())?;
        let output_id = self.model.output_ids().first().copied()
            .ok_or_else(|| "No output node found in model".to_string())?;

        let tensor = rten_tensor::Tensor::from_data(&input_shape, input_data);
        let inputs = vec![(input_id, tensor.view().into())];
        let outputs = vec![output_id];
        let results = self.model.run(inputs, &outputs, None).map_err(|e| format!("{:?}", e))?;
        if let Some(first) = results.into_iter().next() {
            if let Value::FloatTensor(t) = first {
                return Ok(InferenceOutput {
                    shape: t.shape().to_vec(),
                    data: t.to_vec(),
                });
            }
        }
        Err("Expected float tensor output from model".to_string())
    }

    #[wasm_bindgen(js_name = runWithIds)]
    pub fn run_with_ids(
        &self,
        input_id: u32,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
        output_id: u32,
    ) -> Result<InferenceOutput, String> {
        let tensor = rten_tensor::Tensor::from_data(&input_shape, input_data);
        let inputs = vec![(NodeId::from_u32(input_id), tensor.view().into())];
        let outputs = vec![NodeId::from_u32(output_id)];
        let results = self.model.run(inputs, &outputs, None).map_err(|e| format!("{:?}", e))?;
        if let Some(first) = results.into_iter().next() {
            if let Value::FloatTensor(t) = first {
                return Ok(InferenceOutput {
                    shape: t.shape().to_vec(),
                    data: t.to_vec(),
                });
            }
        }
        Err("Expected float tensor output from model".to_string())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextResultWasm {
    pub text: String,
    pub score: f32,
    pub box_points: Vec<[f32; 2]>,
    pub frame: FrameWasm,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FrameWasm {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

#[wasm_bindgen]
pub struct RtenOcrPipeline {
    det_model: Model,
    rec_model: Model,
    dictionary: Vec<String>,
}

/// Bilinear interpolation sampler from an RGBA byte buffer
fn sample_bilinear(rgba: &[u8], w: usize, h: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let idx00 = (y0 * w + x0) * 4;
    let idx10 = (y0 * w + x1) * 4;
    let idx01 = (y1 * w + x0) * 4;
    let idx11 = (y1 * w + x1) * 4;

    let r0 = rgba[idx00] as f32 * (1.0 - fx) + rgba[idx10] as f32 * fx;
    let r1 = rgba[idx01] as f32 * (1.0 - fx) + rgba[idx11] as f32 * fx;
    let r = r0 * (1.0 - fy) + r1 * fy;

    let g0 = rgba[idx00 + 1] as f32 * (1.0 - fx) + rgba[idx10 + 1] as f32 * fx;
    let g1 = rgba[idx01 + 1] as f32 * (1.0 - fx) + rgba[idx11 + 1] as f32 * fx;
    let g = g0 * (1.0 - fy) + g1 * fy;

    let b0 = rgba[idx00 + 2] as f32 * (1.0 - fx) + rgba[idx10 + 2] as f32 * fx;
    let b1 = rgba[idx01 + 2] as f32 * (1.0 - fx) + rgba[idx11 + 2] as f32 * fx;
    let b = b0 * (1.0 - fy) + b1 * fy;

    (r, g, b)
}

/// Attenuate chromatic background patterns, security guilloche lines, and paper tint
/// across diverse document types (ID cards, banknotes, checks, certificates, forms).
///
/// Dark ink (low luminance) is preserved, while chromatic tint (color saturation) on mid-to-bright
/// background regions is neutralized toward the paper level.
fn descreen_background(rgba: &mut [u8], strength: f32) {
    if strength <= 0.0 {
        return;
    }

    for chunk in rgba.chunks_exact_mut(4) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;

        let lum = 0.299 * r + 0.587 * g + 0.114 * b;
        if lum < 70.0 {
            continue;
        }

        let max_c = r.max(g.max(b));
        let min_c = r.min(g.min(b));
        let chroma = max_c - min_c;
        if chroma < 4.0 {
            continue;
        }

        let weight = ((lum - 70.0) / 50.0).clamp(0.0, 1.0);
        let factor = (strength * weight).clamp(0.0, 2.0);

        let r_lift = (max_c - r) * factor;
        let g_lift = (max_c - g) * factor;
        let b_lift = (max_c - b) * factor;

        let white_lift = (255.0 - max_c) * (chroma / 255.0) * (factor * 0.25);

        chunk[0] = (r + r_lift + white_lift).clamp(0.0, 255.0) as u8;
        chunk[1] = (g + g_lift + white_lift).clamp(0.0, 255.0) as u8;
        chunk[2] = (b + b_lift + white_lift).clamp(0.0, 255.0) as u8;
    }
}

#[wasm_bindgen]
impl RtenOcrPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new(
        det_data: Vec<u8>,
        rec_data: Vec<u8>,
        dict_text: &str,
    ) -> Result<RtenOcrPipeline, String> {
        let det_model = Model::load(det_data).map_err(|e| format!("Det model load error: {:?}", e))?;
        let rec_model = Model::load(rec_data).map_err(|e| format!("Rec model load error: {:?}", e))?;

        let dictionary: Vec<String> = if dict_text.trim_start().starts_with('{') {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(dict_text) {
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
                    list
                } else {
                    let mut d: Vec<String> = dict_text.lines().map(|s| s.to_string()).collect();
                    d.push(" ".to_string());
                    d.insert(0, "blank".to_string());
                    d
                }
            } else {
                let mut d: Vec<String> = dict_text.lines().map(|s| s.to_string()).collect();
                d.push(" ".to_string());
                d.insert(0, "blank".to_string());
                d
            }
        } else {
            // Standard PaddleOCR CTC dictionary structure:
            // index 0: blank token
            // index 1..=N: characters from dict.txt
            // index N+1: ' ' (space character)
            let mut d: Vec<String> = dict_text.lines().map(|s| s.to_string()).collect();
            d.push(" ".to_string());
            d.insert(0, "blank".to_string());
            d
        };

        Ok(RtenOcrPipeline {
            det_model,
            rec_model,
            dictionary,
        })
    }

    fn detect_boxes(
        &self,
        processed_rgba: &[u8],
        orig_w: usize,
        orig_h: usize,
        det_thresh: f32,
        box_thresh: f32,
        unclip_ratio: f32,
        max_side_len: Option<u32>,
    ) -> Result<Vec<([f32; 4], f32)>, String> {
        let max_side = (max_side_len.unwrap_or(960) as f32).clamp(320.0, 2560.0);
        let ratio = (max_side / (orig_w.max(orig_h) as f32)).min(1.0);
        let target_w = ((orig_w as f32 * ratio / 32.0).round() as usize * 32).max(32);
        let target_h = ((orig_h as f32 * ratio / 32.0).round() as usize * 32).max(32);

        let scale_w = target_w as f32 / orig_w as f32;
        let scale_h = target_h as f32 / orig_h as f32;

        let det_in_channels = self.det_model.input_shape(0)
            .and_then(|sh| sh.get(1).cloned())
            .map(|d| match d {
                Dimension::Fixed(c) => c,
                _ => 3,
            })
            .unwrap_or(3);

        let det_in_id = self.det_model.input_ids().first().copied()
            .ok_or_else(|| "Det model has no inputs".to_string())?;
        let det_out_id = self.det_model.output_ids().first().copied()
            .ok_or_else(|| "Det model has no outputs".to_string())?;

        let mut raw_boxes: Vec<([f32; 4], f32)> = Vec::new();

        if det_in_channels == 1 {
            // LineNet 1-channel grayscale ink-map detection (e.g. Indra OCR)
            let mut det_input = vec![0.0f32; target_w * target_h];
            for ty in 0..target_h {
                let sy = (ty as f32 + 0.5) / scale_h - 0.5;
                for tx in 0..target_w {
                    let sx = (tx as f32 + 0.5) / scale_w - 0.5;
                    let (r, g, b) = sample_bilinear(processed_rgba, orig_w, orig_h, sx, sy);
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                    det_input[ty * target_w + tx] = ink;
                }
            }

            let det_tensor = rten_tensor::Tensor::from_data(&[1, 1, target_h, target_w], det_input);
            let det_outputs = self.det_model.run(
                vec![(det_in_id, det_tensor.view().into())],
                &[det_out_id],
                None,
            ).map_err(|e| format!("Det inference error: {:?}", e))?;

            let (sh, det_data) = match det_outputs.into_iter().next() {
                Some(Value::FloatTensor(t)) => (t.shape().to_vec(), t.to_vec()),
                _ => return Err("Invalid detection output tensor".to_string()),
            };

            if sh.len() >= 4 && sh[1] >= 3 {
                let out_h = sh[2];
                let out_w = sh[3];
                let plane = out_h * out_w;

                let mut prob = vec![0.0f32; plane];
                for i in 0..plane {
                    let logit = det_data[i];
                    prob[i] = 1.0 / (1.0 + (-logit).exp());
                }

                let mut visited = vec![false; plane];
                for y in 0..out_h {
                    for x in 0..out_w {
                        let idx = y * out_w + x;
                        if visited[idx] || prob[idx] < det_thresh {
                            continue;
                        }
                        let mut queue = std::collections::VecDeque::new();
                        queue.push_back((x, y));
                        visited[idx] = true;

                        let mut min_x = x;
                        let mut max_x = x;
                        let mut sum_top = 0.0f32;
                        let mut sum_bot = 0.0f32;
                        let mut sum_prob = 0.0f32;
                        let mut count = 0usize;

                        while let Some((cx, cy)) = queue.pop_front() {
                            let cidx = cy * out_w + cx;
                            let ys = (cy as f32 + 0.5) * 2.0;
                            sum_top += ys - det_data[plane + cidx];
                            sum_bot += ys + det_data[2 * plane + cidx];
                            sum_prob += prob[cidx];
                            count += 1;

                            if cx < min_x { min_x = cx; }
                            if cx > max_x { max_x = cx; }

                            for (nx, ny) in [
                                (cx.wrapping_sub(1), cy),
                                (cx + 1, cy),
                                (cx, cy.wrapping_sub(1)),
                                (cx, cy + 1),
                            ] {
                                if nx < out_w && ny < out_h {
                                    let nidx = ny * out_w + nx;
                                    if !visited[nidx] && prob[nidx] >= det_thresh {
                                        visited[nidx] = true;
                                        queue.push_back((nx, ny));
                                    }
                                }
                            }
                        }

                        if count < 2 { continue; }
                        let avg_prob = sum_prob / count as f32;
                        if avg_prob < box_thresh { continue; }

                        let y0_in = sum_top / count as f32;
                        let y1_in = sum_bot / count as f32;
                        let lh = (y1_in - y0_in).max(1.0);
                        let ex = (0.15 * lh).min(0.5 * lh);
                        let x0_in = (min_x as f32 * 2.0) - ex;
                        let x1_in = ((max_x + 1) as f32 * 2.0) + ex;

                        let x0 = (x0_in / scale_w).clamp(0.0, orig_w as f32);
                        let y0 = (y0_in / scale_h).clamp(0.0, orig_h as f32);
                        let x1 = (x1_in / scale_w).clamp(0.0, orig_w as f32);
                        let y1 = (y1_in / scale_h).clamp(0.0, orig_h as f32);

                        if (x1 - x0) > 4.0 && (y1 - y0) > 4.0 {
                            raw_boxes.push(([x0, y0, x1, y1], avg_prob));
                        }
                    }
                }
            }
        } else {
            // Standard PP-OCR detection normalization in BGR order
            let mean_b = 0.485f32;
            let mean_g = 0.456f32;
            let mean_r = 0.406f32;
            let std_b = 0.229f32;
            let std_g = 0.224f32;
            let std_r = 0.225f32;

            let mut det_input = vec![0.0f32; target_w * target_h * 3];

            for ty in 0..target_h {
                let sy = (ty as f32 + 0.5) / scale_h - 0.5;
                for tx in 0..target_w {
                    let sx = (tx as f32 + 0.5) / scale_w - 0.5;
                    let (r, g, b) = sample_bilinear(processed_rgba, orig_w, orig_h, sx, sy);

                    let bn = (b / 255.0 - mean_b) / std_b;
                    let gn = (g / 255.0 - mean_g) / std_g;
                    let rn = (r / 255.0 - mean_r) / std_r;

                    let pixel_idx = ty * target_w + tx;
                    det_input[pixel_idx] = bn;
                    det_input[target_h * target_w + pixel_idx] = gn;
                    det_input[2 * target_h * target_w + pixel_idx] = rn;
                }
            }

            let det_tensor = rten_tensor::Tensor::from_data(&[1, 3, target_h, target_w], det_input);
            let det_outputs = self.det_model.run(
                vec![(det_in_id, det_tensor.view().into())],
                &[det_out_id],
                None,
            ).map_err(|e| format!("Det inference error: {:?}", e))?;

            let heat_map = match det_outputs.into_iter().next() {
                Some(Value::FloatTensor(t)) => t.to_vec(),
                _ => return Err("Invalid detection output tensor".to_string()),
            };

            let mut visited = vec![false; target_w * target_h];

            for y in 0..target_h {
                for x in 0..target_w {
                    let idx = y * target_w + x;
                    if visited[idx] || heat_map[idx] < det_thresh {
                        continue;
                    }

                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((x, y));
                    visited[idx] = true;

                    let mut min_x = x;
                    let mut max_x = x;
                    let mut min_y = y;
                    let mut max_y = y;
                    let mut sum_prob = 0.0f32;
                    let mut count = 0usize;

                    while let Some((cx, cy)) = queue.pop_front() {
                        let cidx = cy * target_w + cx;
                        sum_prob += heat_map[cidx];
                        count += 1;

                        if cx < min_x { min_x = cx; }
                        if cx > max_x { max_x = cx; }
                        if cy < min_y { min_y = cy; }
                        if cy > max_y { max_y = cy; }

                        for (nx, ny) in [
                            (cx.wrapping_sub(1), cy),
                            (cx + 1, cy),
                            (cx, cy.wrapping_sub(1)),
                            (cx, cy + 1),
                        ] {
                            if nx < target_w && ny < target_h {
                                let nidx = ny * target_w + nx;
                                if !visited[nidx] && heat_map[nidx] >= det_thresh {
                                    visited[nidx] = true;
                                    queue.push_back((nx, ny));
                                }
                            }
                        }
                    }

                    if count < 16 || (max_x - min_x) < 4 || (max_y - min_y) < 3 {
                        continue;
                    }

                    let avg_score = sum_prob / (count as f32);
                    if avg_score < box_thresh {
                        continue;
                    }

                    let bw = (max_x - min_x + 1) as f32;
                    let bh = (max_y - min_y + 1) as f32;
                    let peri = 2.0 * (bw + bh);
                    let area = bw * bh;
                    let dist = (area * unclip_ratio) / peri.max(1.0);

                    let x0 = (min_x as f32 - dist).max(0.0) / scale_w;
                    let y0 = (min_y as f32 - dist).max(0.0) / scale_h;
                    let x1 = ((max_x as f32 + dist).min(target_w as f32)) / scale_w;
                    let y1 = ((max_y as f32 + dist).min(target_h as f32)) / scale_h;

                    let clamped_x0 = x0.clamp(0.0, orig_w as f32);
                    let clamped_y0 = y0.clamp(0.0, orig_h as f32);
                    let clamped_x1 = x1.clamp(0.0, orig_w as f32);
                    let clamped_y1 = y1.clamp(0.0, orig_h as f32);

                    if (clamped_x1 - clamped_x0) > 4.0 && (clamped_y1 - clamped_y0) > 4.0 {
                        raw_boxes.push(([clamped_x0, clamped_y0, clamped_x1, clamped_y1], avg_score));
                    }
                }
            }
        }

        raw_boxes.sort_by(|a, b| {
            a.0[1].total_cmp(&b.0[1]).then_with(|| a.0[0].total_cmp(&b.0[0]))
        });

        Ok(raw_boxes)
    }

    fn recognize_box(
        &self,
        processed_rgba: &[u8],
        orig_w: usize,
        orig_h: usize,
        box_coords: [f32; 4],
        det_score: f32,
    ) -> Result<Option<TextResultWasm>, String> {
        let [x0, y0, x1, y1] = box_coords;
        let box_w = x1 - x0;
        let box_h = y1 - y0;

        let rec_in_shape = self.rec_model.input_shape(0);
        let rec_in_channels = rec_in_shape.as_ref()
            .and_then(|sh| sh.get(1).cloned())
            .map(|d| match d { Dimension::Fixed(c) => c, _ => 3 })
            .unwrap_or(3);

        if rec_in_channels == 1 {
            // LineNet recognition (e.g. Indra OCR)
            let exp_x0 = (x0 - 0.3 * box_h).max(0.0);
            let exp_x1 = (x1 + 0.3 * box_h).min(orig_w as f32);
            let exp_y0 = (y0 - 1.6 * box_h).max(0.0);
            let exp_y1 = (y1 + 1.6 * box_h).min(orig_h as f32);
            let crop_w = exp_x1 - exp_x0;
            let crop_h = exp_y1 - exp_y0;
            if crop_w < 4.0 || crop_h < 4.0 {
                return Ok(None);
            }

            let rec_h = 96usize;
            let target_w = (96.0 * (crop_w / crop_h)).round() as usize;
            let rec_w = ((target_w + 7) / 8 * 8).clamp(8, 4096);

            let mut rec_input = vec![0.0f32; rec_h * rec_w];
            for ry in 0..rec_h {
                let sy = exp_y0 + (ry as f32 + 0.5) * (crop_h / rec_h as f32) - 0.5;
                for rx in 0..rec_w {
                    let sx = exp_x0 + (rx as f32 + 0.5) * (crop_w / rec_w as f32) - 0.5;
                    let (r, g, b) = sample_bilinear(processed_rgba, orig_w, orig_h, sx, sy);
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                    rec_input[ry * rec_w + rx] = ink;
                }
            }

            let rec_in_ids = self.rec_model.input_ids();
            let x_tensor = rten_tensor::Tensor::from_data(&[1, 1, rec_h, rec_w], rec_input);
            let widths_tensor = rten_tensor::Tensor::from_data(&[1], vec![rec_w as i32]);

            let rec_outputs = self.rec_model.run(
                vec![
                    (rec_in_ids[0], x_tensor.view().into()),
                    (rec_in_ids[1], widths_tensor.view().into()),
                ],
                self.rec_model.output_ids(),
                None,
            ).map_err(|e| format!("Rec inference error: {:?}", e))?;

            let mut it = rec_outputs.into_iter();
            let ids_vec = match it.next() {
                Some(Value::Int32Tensor(t)) => t.to_vec(),
                _ => return Ok(None),
            };
            let conf_vec = match it.next() {
                Some(Value::FloatTensor(t)) => t.to_vec(),
                _ => return Ok(None),
            };
            let len_val = match it.next() {
                Some(Value::Int32Tensor(t)) => t.to_vec().first().copied().unwrap_or(0) as usize,
                _ => return Ok(None),
            };

            let mut text = String::new();
            let mut char_scores = Vec::new();
            let mut last_id = 0i32;

            for step in 0..len_val.min(ids_vec.len()) {
                let id = ids_vec[step];
                if id != 0 && (step == 0 || id != last_id) {
                    if let Some(ch) = self.dictionary.get(id as usize) {
                        text.push_str(ch);
                        if let Some(&score) = conf_vec.get(step) {
                            char_scores.push(score);
                        }
                    }
                }
                last_id = id;
            }

            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }

            let avg_conf = if char_scores.is_empty() {
                det_score
            } else {
                char_scores.iter().sum::<f32>() / char_scores.len() as f32
            };
            let text_score = (det_score * 0.3 + 0.7 * avg_conf.clamp(0.0, 1.0)).clamp(0.0, 1.0);

            Ok(Some(TextResultWasm {
                text: trimmed.to_string(),
                score: (text_score * 1000.0).round() / 1000.0,
                box_points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                frame: FrameWasm {
                    left: (x0 * 100.0).round() / 100.0,
                    top: (y0 * 100.0).round() / 100.0,
                    width: (box_w * 100.0).round() / 100.0,
                    height: (box_h * 100.0).round() / 100.0,
                },
            }))
        } else {
            // Standard PP-OCR SVTR recognition
            let pad_x_ratio = 0.035f32;
            let pad_y_ratio = 0.045f32;

            let exp_x0 = (x0 - box_w * pad_x_ratio).max(0.0);
            let exp_x1 = (x1 + box_w * pad_x_ratio).min(orig_w as f32);
            let exp_y0 = (y0 - box_h * pad_y_ratio).max(0.0);
            let exp_y1 = (y1 + box_h * pad_y_ratio).min(orig_h as f32);

            let crop_w = exp_x1 - exp_x0;
            let crop_h = exp_y1 - exp_y0;
            if crop_w < 4.0 || crop_h < 4.0 {
                return Ok(None);
            }

            let rec_h = 48usize;
            let target_w = (48.0 * (crop_w / crop_h)).ceil() as usize;
            let rec_w = ((target_w as f32 / 32.0).ceil() as usize * 32).clamp(64, 960);

            let mut rec_input = vec![0.0f32; rec_h * rec_w * 3];

            for ry in 0..rec_h {
                let sy = exp_y0 + (ry as f32 + 0.5) * (crop_h / rec_h as f32) - 0.5;
                for rx in 0..rec_w {
                    let sx = exp_x0 + (rx as f32 + 0.5) * (crop_w / rec_w as f32) - 0.5;
                    let (r, g, b) = sample_bilinear(processed_rgba, orig_w, orig_h, sx, sy);

                    let bn = (b / 255.0 - 0.5) / 0.5;
                    let gn = (g / 255.0 - 0.5) / 0.5;
                    let rn = (r / 255.0 - 0.5) / 0.5;

                    let pixel_idx = ry * rec_w + rx;
                    rec_input[pixel_idx] = bn;
                    rec_input[rec_h * rec_w + pixel_idx] = gn;
                    rec_input[2 * rec_h * rec_w + pixel_idx] = rn;
                }
            }

            let rec_in_id = self.rec_model.input_ids().first().copied()
                .ok_or_else(|| "Rec model has no inputs".to_string())?;
            let rec_out_id = self.rec_model.output_ids().first().copied()
                .ok_or_else(|| "Rec model has no outputs".to_string())?;

            let rec_tensor = rten_tensor::Tensor::from_data(&[1, 3, rec_h, rec_w], rec_input);
            let rec_outputs = self.rec_model.run(
                vec![(rec_in_id, rec_tensor.view().into())],
                &[rec_out_id],
                None,
            ).map_err(|e| format!("Rec inference error: {:?}", e))?;

            let (seq_len, num_classes, rec_data) = match rec_outputs.into_iter().next() {
                Some(Value::FloatTensor(t)) => {
                    let sh = t.shape().to_vec();
                    if sh.len() >= 3 {
                        (sh[1], sh[2], t.to_vec())
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            };

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
                    if let Some(ch) = self.dictionary.get(best_idx) {
                        text.push_str(ch);
                        char_scores.push(best_val);
                    }
                }
                last_idx = best_idx;
            }

            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }

            let text_score = if char_scores.is_empty() {
                det_score
            } else {
                let avg_conf = char_scores.iter().sum::<f32>() / char_scores.len() as f32;
                (det_score * 0.3 + 0.7 * avg_conf.clamp(0.0, 1.0)).clamp(0.0, 1.0)
            };

            Ok(Some(TextResultWasm {
                text: trimmed.to_string(),
                score: (text_score * 1000.0).round() / 1000.0,
                box_points: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                frame: FrameWasm {
                    left: (x0 * 100.0).round() / 100.0,
                    top: (y0 * 100.0).round() / 100.0,
                    width: (box_w * 100.0).round() / 100.0,
                    height: (box_h * 100.0).round() / 100.0,
                },
            }))
        }
    }

    /// Run full OCR pipeline directly in WebAssembly with calibrated image optimization
    #[wasm_bindgen(js_name = detectAndRecognize)]
    pub fn detect_and_recognize(
        &self,
        image_rgba: &[u8],
        width: u32,
        height: u32,
        det_thresh: f32,
        box_thresh: f32,
        unclip_ratio: f32,
        apply_calibration: bool,
        max_side_len: Option<u32>,
        descreen_strength: Option<f32>,
    ) -> Result<String, String> {
        if width == 0 || height == 0 || image_rgba.len() < (width * height * 4) as usize {
            return Err("Invalid image dimensions or buffer size".to_string());
        }

        let orig_w = width as usize;
        let orig_h = height as usize;

        // Apply background descreening if requested
        let mut processed_rgba = image_rgba[..orig_w * orig_h * 4].to_vec();
        if apply_calibration {
            let strength = descreen_strength.unwrap_or(1.15);
            descreen_background(&mut processed_rgba, strength);
        }

        let raw_boxes = self.detect_boxes(
            &processed_rgba,
            orig_w,
            orig_h,
            det_thresh,
            box_thresh,
            unclip_ratio,
            max_side_len,
        )?;

        let mut results = Vec::new();
        for (box_coords, det_score) in raw_boxes {
            if let Some(res) = self.recognize_box(&processed_rgba, orig_w, orig_h, box_coords, det_score)? {
                results.push(res);
            }
        }

        serde_json::to_string(&results).map_err(|e| format!("Serialization error: {:?}", e))
    }
}

