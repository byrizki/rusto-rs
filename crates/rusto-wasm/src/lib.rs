mod backend;

use wasm_bindgen::prelude::*;
use backend::WasmBackendModel;
use rusto_core::{
    calibration::descreen_background_rgba,
    ctc::{composite_ocr_score, CtcDecoder},
    image::Point2f,
    layout::sort_lines_reading_order,
    postprocess::{decode_dbnet_boxes_fast, decode_linenet_det_boxes_slice},
    preprocess::{
        compute_linenet_rec_crop_dims, compute_ppocr_rec_crop_dims, plan_linenet_rec_batches,
        prepare_linenet_rec_crop_rgba, prepare_ppocr_rec_crop_rgba, preprocess_linenet_det_rgba,
        preprocess_rgba_raw,
    },
    spatial::{cluster_text_results_into_lines, to_spatial_text, to_tsv},
    types::{Frame, OcrRunOptions, OutputGranularity, TextResult},
};

/// Returns the engine library version string.
#[wasm_bindgen(js_name = version)]
pub fn wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Returns whether WebAssembly inference engine is ready.
#[wasm_bindgen(js_name = isReady)]
pub fn wasm_is_ready() -> bool {
    console_error_panic_hook::set_once();
    true
}

/// Returns the active inference engine backend identifier ('rusto-nn' | 'rusto-rten' | 'rusto-webgpu').
#[wasm_bindgen(js_name = engineBackend)]
pub fn wasm_engine_backend() -> String {
    #[cfg(feature = "rusto-webgpu")]
    {
        "rusto-webgpu".to_string()
    }
    #[cfg(feature = "rusto-rten")]
    {
        "rusto-rten".to_string()
    }
    #[cfg(not(any(feature = "rusto-webgpu", feature = "rusto-rten")))]
    {
        "rusto-nn".to_string()
    }
}

/// Initializes WebGPU compute acceleration if supported by this WASM build.
#[wasm_bindgen(js_name = initWebGpu)]
pub async fn wasm_init_webgpu() -> Result<bool, String> {
    console_error_panic_hook::set_once();
    #[cfg(feature = "rusto-webgpu")]
    {
        if !rusto_nn_webgpu::WebGpuBackend::is_webgpu_supported() {
            return Ok(false);
        }
        match rusto_nn_webgpu::WebGpuBackend::init_async().await {
            Ok(backend) => {
                backend.register();
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
    #[cfg(not(feature = "rusto-webgpu"))]
    {
        Ok(false)
    }
}

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
    inner: WasmBackendModel,
}

pub type RustoModel = RtenModel;

#[wasm_bindgen]
impl RtenModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: Vec<u8>) -> Result<RtenModel, String> {
        console_error_panic_hook::set_once();
        let inner = WasmBackendModel::from_bytes(model_data)?;
        Ok(RtenModel { inner })
    }

    #[wasm_bindgen(js_name = findNode)]
    pub fn find_node(&self, name: &str) -> Option<u32> {
        self.inner.find_node(name)
    }

    #[wasm_bindgen(js_name = inputIds)]
    pub fn input_ids(&self) -> Vec<u32> {
        self.inner.input_ids()
    }

    #[wasm_bindgen(js_name = outputIds)]
    pub fn output_ids(&self) -> Vec<u32> {
        self.inner.output_ids()
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
    ) -> Result<InferenceOutput, String> {
        let (shape, data) = self.inner.run_single(&input_shape, &input_data)?;
        Ok(InferenceOutput { shape, data })
    }

    #[wasm_bindgen(js_name = runWithIds)]
    pub fn run_with_ids(
        &self,
        input_id: u32,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
        output_id: u32,
    ) -> Result<InferenceOutput, String> {
        let (shape, data) =
            self.inner
                .run_with_ids(input_id, &input_shape, &input_data, output_id)?;
        Ok(InferenceOutput { shape, data })
    }

    #[wasm_bindgen(js_name = runOrder)]
    pub fn run_order(
        &self,
        num_items: usize,
        feats_data: Vec<f32>,
    ) -> Result<InferenceOutput, String> {
        let (shape, data) = self.inner.run_order(num_items, feats_data)?;
        Ok(InferenceOutput { shape, data })
    }
}

pub type TextResultWasm = TextResult;
pub type FrameWasm = Frame;

#[wasm_bindgen]
pub struct RtenOcrPipeline {
    det_model: WasmBackendModel,
    rec_model: WasmBackendModel,
    decoder: CtcDecoder,
    layout_model: Option<WasmBackendModel>,
}

pub type RustoOcrPipeline = RtenOcrPipeline;

#[wasm_bindgen]
impl RtenOcrPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new(
        det_data: Vec<u8>,
        rec_data: Vec<u8>,
        dict_text: &str,
    ) -> Result<RtenOcrPipeline, String> {
        console_error_panic_hook::set_once();
        let det_model = WasmBackendModel::from_bytes(det_data)?;
        let rec_model = WasmBackendModel::from_bytes(rec_data)?;
        let decoder = CtcDecoder::from_string(dict_text);

        Ok(RtenOcrPipeline {
            det_model,
            rec_model,
            decoder,
            layout_model: None,
        })
    }

    #[wasm_bindgen(js_name = newWithModels)]
    pub fn new_with_models(
        det_data: Vec<u8>,
        rec_data: Vec<u8>,
        dict_text: &str,
        layout_data: Option<Vec<u8>>,
    ) -> Result<RtenOcrPipeline, String> {
        console_error_panic_hook::set_once();
        let det_model = WasmBackendModel::from_bytes(det_data)?;
        let rec_model = WasmBackendModel::from_bytes(rec_data)?;
        let decoder = CtcDecoder::from_string(dict_text);
        let layout_model = match layout_data {
            Some(d) if !d.is_empty() => Some(WasmBackendModel::from_bytes(d)?),
            _ => None,
        };

        Ok(RtenOcrPipeline {
            det_model,
            rec_model,
            decoder,
            layout_model,
        })
    }

    #[wasm_bindgen(js_name = setLayoutModel)]
    pub fn set_layout_model(&mut self, layout_data: Vec<u8>) -> Result<(), String> {
        let model = WasmBackendModel::from_bytes(layout_data)?;
        self.layout_model = Some(model);
        Ok(())
    }

    /// Returns the engine library version string.
    #[wasm_bindgen(js_name = version)]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Returns whether the pipeline is initialized and ready.
    #[wasm_bindgen(js_name = isReady)]
    pub fn is_ready(&self) -> bool {
        true
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
        let det_in_channels = self.det_model.input_channels();

        if det_in_channels == 1 {
            // LineNet 1-channel grayscale ink-map detection (e.g. Indra OCR)
            let limit = max_side_len.unwrap_or(960) as i32;
            let (det_input, target_w, target_h, scale_w, scale_h) =
                preprocess_linenet_det_rgba(processed_rgba, orig_w, orig_h, limit, "max");

            let (sh, det_slice) = self
                .det_model
                .run_single(&[1, 1, target_h, target_w], &det_input)?;

            if sh.len() >= 4 && sh[1] >= 3 {
                Ok(decode_linenet_det_boxes_slice(
                    &det_slice,
                    sh[2],
                    sh[3],
                    orig_w as f32,
                    orig_h as f32,
                    scale_w,
                    scale_h,
                    det_thresh,
                    box_thresh,
                ))
            } else {
                Ok(Vec::new())
            }
        } else {
            // Standard PP-OCR detection normalization in BGR order
            let (det_input, target_w, target_h, scale_w, scale_h) = preprocess_rgba_raw(
                processed_rgba,
                orig_w,
                orig_h,
                max_side_len.unwrap_or(960) as usize,
                "max",
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            );

            let (_sh, heat_map) = self
                .det_model
                .run_single(&[1, 3, target_h, target_w], &det_input)?;

            Ok(decode_dbnet_boxes_fast(
                &heat_map,
                target_w,
                target_h,
                orig_w,
                orig_h,
                scale_w,
                scale_h,
                det_thresh,
                box_thresh,
                unclip_ratio,
            ))
        }
    }

    fn recognize_boxes_batched(
        &self,
        processed_rgba: &[u8],
        orig_w: usize,
        orig_h: usize,
        raw_boxes: &[([f32; 4], f32)],
        batch_size: usize,
    ) -> Result<Vec<TextResult>, String> {
        if raw_boxes.is_empty() {
            return Ok(Vec::new());
        }

        let batch_cap = batch_size.max(1);
        let mut slot_results: Vec<Option<TextResult>> = vec![None; raw_boxes.len()];
        let rec_in_channels = self.rec_model.input_channels();

        struct CropMeta {
            orig_idx: usize,
            box_coords: [f32; 4],
            det_score: f32,
            rec_w: usize,
            rec_h: usize,
        }

        if rec_in_channels == 1 {
            let mut crops = Vec::with_capacity(raw_boxes.len());
            for (orig_idx, &(box_coords, det_score)) in raw_boxes.iter().enumerate() {
                if let Some((rec_w, rec_h)) =
                    compute_linenet_rec_crop_dims(orig_w, orig_h, box_coords)
                {
                    crops.push(CropMeta {
                        orig_idx,
                        box_coords,
                        det_score,
                        rec_w,
                        rec_h,
                    });
                }
            }

            let target_widths: Vec<usize> = crops.iter().map(|c| c.rec_w).collect();
            let batch_groups = plan_linenet_rec_batches(&target_widths, batch_cap);

            for group in batch_groups {
                if group.is_empty() {
                    continue;
                }
                let b = group.len();
                let rec_h = crops[group[0]].rec_h;
                let max_w = group.iter().map(|&ci| crops[ci].rec_w).max().unwrap_or(8);
                let mut widths = Vec::with_capacity(b);
                let mut batch_data = Vec::with_capacity(b * rec_h * max_w);

                for &ci in &group {
                    widths.push(max_w as i32);
                    if let Some((crop_input, _w, _h)) =
                        prepare_linenet_rec_crop_rgba(processed_rgba, orig_w, orig_h, crops[ci].box_coords)
                    {
                        batch_data.extend_from_slice(&crop_input);
                    } else {
                        batch_data.resize(batch_data.len() + rec_h * max_w, 0.0);
                    }
                }

                let (ids_vec, conf_vec, lens_vec, max_len) = self.rec_model.run_linenet_rec(
                    b,
                    rec_h,
                    max_w,
                    batch_data,
                    widths,
                )?;

                for (local_idx, &ci) in group.iter().enumerate() {
                    let item = &crops[ci];
                    let row_start = local_idx * max_len;
                    let row_end = (row_start + max_len).min(ids_vec.len());
                    if row_start >= row_end {
                        continue;
                    }
                    let ids_row = &ids_vec[row_start..row_end];
                    let conf_row = &conf_vec[row_start..row_end.min(conf_vec.len())];
                    let valid_len = lens_vec
                        .get(local_idx)
                        .map(|&l| (l.max(0) as usize).min(ids_row.len()))
                        .unwrap_or(ids_row.len());

                    if let Some((trimmed, avg_conf)) =
                        self.decoder.decode_linenet_tokens(ids_row, conf_row, valid_len)
                    {
                        let [x0, y0, x1, y1] = item.box_coords;
                        let text_score = composite_ocr_score(item.det_score, avg_conf);
                        slot_results[item.orig_idx] =
                            Some(TextResult::from_coords(x0, y0, x1, y1, trimmed, text_score));
                    }
                }
            }
        } else {
            let mut crops = Vec::with_capacity(raw_boxes.len());
            for (orig_idx, &(box_coords, det_score)) in raw_boxes.iter().enumerate() {
                if let Some((rec_w, rec_h)) =
                    compute_ppocr_rec_crop_dims(orig_w, orig_h, box_coords)
                {
                    crops.push(CropMeta {
                        orig_idx,
                        box_coords,
                        det_score,
                        rec_w,
                        rec_h,
                    });
                }
            }

            crops.sort_by_key(|c| (c.rec_w, c.orig_idx));

            let mut i = 0;
            while i < crops.len() {
                let rec_h = crops[i].rec_h;
                let rec_w = crops[i].rec_w;
                let mut end = i + 1;
                while end < crops.len()
                    && end - i < batch_cap
                    && crops[end].rec_w == rec_w
                {
                    end += 1;
                }

                let b = end - i;
                let mut batch_data = Vec::with_capacity(b * 3 * rec_h * rec_w);

                for item in &crops[i..end] {
                    if let Some((crop_data, _w, _h)) =
                        prepare_ppocr_rec_crop_rgba(processed_rgba, orig_w, orig_h, item.box_coords)
                    {
                        batch_data.extend_from_slice(&crop_data);
                    } else {
                        batch_data.resize(batch_data.len() + 3 * rec_h * rec_w, 0.0);
                    }
                }

                let (sh, rec_data) = self
                    .rec_model
                    .run_single(&[b, 3, rec_h, rec_w], &batch_data)?;

                if sh.len() >= 3 {
                    let (seq_len, num_classes) = (sh[1], sh[2]);
                    let stride = seq_len * num_classes;

                    for (local_idx, item) in crops[i..end].iter().enumerate() {
                        let start = local_idx * stride;
                        let stop = start + stride;
                        if stop > rec_data.len() {
                            break;
                        }
                        if let Some((trimmed, avg_conf)) = self.decoder.decode_single_crop(
                            &rec_data[start..stop],
                            seq_len,
                            num_classes,
                        ) {
                            let [x0, y0, x1, y1] = item.box_coords;
                            let text_score = composite_ocr_score(item.det_score, avg_conf);
                            slot_results[item.orig_idx] =
                                Some(TextResult::from_coords(x0, y0, x1, y1, trimmed, text_score));
                        }
                    }
                }

                i = end;
            }
        }

        Ok(slot_results.into_iter().flatten().collect())
    }

    fn apply_reading_order(
        &self,
        items: &mut Vec<TextResult>,
        layout_mode: Option<&str>,
        layout_reading: bool,
        orig_w: usize,
        orig_h: usize,
    ) {
        if items.len() <= 1 {
            return;
        }

        let mode = layout_mode.unwrap_or("heuristic");

        // If explicitly disabled or none, maintain raw raster scan order
        if !layout_reading || mode == "none" {
            items.sort_by(|a, b| {
                a.frame
                    .top
                    .partial_cmp(&b.frame.top)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.frame
                            .left
                            .partial_cmp(&b.frame.left)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            return;
        }

        // Neural reading layout direction model (e.g. Indra or Paddle layout models)
        if mode != "heuristic" {
            if let Some(layout_model) = &self.layout_model {
                let boxes: Vec<[f32; 4]> = items
                    .iter()
                    .map(|it| {
                        [
                            it.frame.left,
                            it.frame.top,
                            it.frame.left + it.frame.width,
                            it.frame.top + it.frame.height,
                        ]
                    })
                    .collect();
                let texts: Vec<String> = items.iter().map(|it| it.text.clone()).collect();
                let feats = rusto_core::order::compute_line_features(
                    &boxes,
                    orig_w as f32,
                    orig_h as f32,
                    &texts,
                );

                if let Ok((_sh, out_data)) = layout_model.run_order(items.len(), feats) {
                    let order_indices =
                        rusto_core::order::decode_order_logits(&out_data, items.len(), 4);
                    if order_indices.len() == items.len() {
                        let original = items.clone();
                        for (new_idx, &orig_idx) in order_indices.iter().enumerate() {
                            if orig_idx < original.len() {
                                items[new_idx] = original[orig_idx].clone();
                            }
                        }
                        return;
                    }
                }
            }
        }

        // Default / heuristic column flow
        sort_lines_reading_order(items);
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

        let mut processed_rgba = image_rgba[..orig_w * orig_h * 4].to_vec();
        if apply_calibration {
            let strength = descreen_strength.unwrap_or(1.15);
            descreen_background_rgba(&mut processed_rgba, strength);
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

        let results = self.recognize_boxes_batched(&processed_rgba, orig_w, orig_h, &raw_boxes, 12)?;

        Ok(nanoserde::SerJson::serialize_json(&results))
    }

    /// Run full OCR pipeline directly in WebAssembly accepting RGBA or RGB buffers with runtime options.
    #[wasm_bindgen(js_name = detectText)]
    pub fn detect_text(
        &self,
        image_bytes: &[u8],
        width: u32,
        height: u32,
        channels: u32,
        stride: Option<u32>,
        options_json: Option<String>,
    ) -> Result<String, String> {
        if width == 0 || height == 0 {
            return Err("Invalid image dimensions: width and height must be > 0".to_string());
        }

        let options: OcrRunOptions = match &options_json {
            Some(json) if !json.trim().is_empty() => {
                nanoserde::DeJson::deserialize_json(json).map_err(|e| format!("Invalid options JSON: {e}"))?
            }
            _ => OcrRunOptions::default(),
        };

        let orig_w = width as usize;
        let orig_h = height as usize;

        let mut processed_rgba = match channels {
            4 => {
                let min_len = orig_w * orig_h * 4;
                if image_bytes.len() < min_len {
                    return Err(format!(
                        "Buffer size too small for RGBA: expected {min_len}, got {}",
                        image_bytes.len()
                    ));
                }
                image_bytes[..min_len].to_vec()
            }
            3 => {
                let eff_stride = stride.unwrap_or(width.saturating_mul(3)) as usize;
                let min_len = orig_h.saturating_sub(1) * eff_stride + orig_w * 3;
                if image_bytes.len() < min_len {
                    return Err(format!(
                        "Buffer size too small for RGB: expected {min_len}, got {}",
                        image_bytes.len()
                    ));
                }
                let mut rgba = Vec::with_capacity(orig_w * orig_h * 4);
                for y in 0..orig_h {
                    let row_start = y * eff_stride;
                    for x in 0..orig_w {
                        let px = row_start + x * 3;
                        rgba.push(image_bytes[px]);
                        rgba.push(image_bytes[px + 1]);
                        rgba.push(image_bytes[px + 2]);
                        rgba.push(255);
                    }
                }
                rgba
            }
            _ => {
                return Err(format!(
                    "Unsupported channel count: {channels}, expected 3 (RGB) or 4 (RGBA)"
                ))
            }
        };

        let apply_calibration = options
            .pre_detection
            .as_ref()
            .and_then(|p| p.descreen_strength)
            .map(|s| s > 0.0)
            .unwrap_or(true);
        if apply_calibration {
            let strength = options
                .pre_detection
                .as_ref()
                .and_then(|p| p.descreen_strength)
                .unwrap_or(1.15);
            descreen_background_rgba(&mut processed_rgba, strength);
        }

        let det_thresh = options
            .post_detection
            .as_ref()
            .and_then(|p| p.threshold)
            .unwrap_or(0.3);
        let box_thresh = options
            .post_detection
            .as_ref()
            .and_then(|p| p.box_threshold)
            .unwrap_or(0.5);
        let unclip_ratio = options
            .post_detection
            .as_ref()
            .and_then(|p| p.unclip_ratio)
            .unwrap_or(1.8);
        let max_side_len = options
            .detection
            .as_ref()
            .and_then(|d| d.limit_side_len.map(|l| l as u32))
            .or_else(|| {
                options
                    .post_detection
                    .as_ref()
                    .and_then(|p| p.max_side_len.map(|l| l as u32))
            })
            .unwrap_or(960);

        let raw_boxes = self.detect_boxes(
            &processed_rgba,
            orig_w,
            orig_h,
            det_thresh,
            box_thresh,
            unclip_ratio,
            Some(max_side_len),
        )?;

        let score_thresh = options
            .post_recognition
            .as_ref()
            .and_then(|p| p.score_threshold)
            .unwrap_or(0.5);

        let batch_size = options
            .recognition
            .as_ref()
            .and_then(|r| r.batch_size)
            .unwrap_or(6)
            .max(1) as usize;

        let recognized: Vec<TextResult> = self
            .recognize_boxes_batched(&processed_rgba, orig_w, orig_h, &raw_boxes, batch_size)?
            .into_iter()
            .filter(|res| res.score >= score_thresh)
            .collect();

        let line_y = options.pre_recognition.as_ref().and_then(|pr| pr.line_y_threshold);
        let word_x = options.pre_recognition.as_ref().and_then(|pr| pr.word_x_threshold);

        match options.output {
            OutputGranularity::Spatial => {
                let boxes: Vec<[Point2f; 4]> = recognized
                    .iter()
                    .map(|c| {
                        [
                            Point2f::new(c.box_points[0].0, c.box_points[0].1),
                            Point2f::new(c.box_points[1].0, c.box_points[1].1),
                            Point2f::new(c.box_points[2].0, c.box_points[2].1),
                            Point2f::new(c.box_points[3].0, c.box_points[3].1),
                        ]
                    })
                    .collect();
                let texts: Vec<String> = recognized.iter().map(|c| c.text.clone()).collect();
                let spatial_text = to_spatial_text(&boxes, &texts, line_y, word_x);
                Ok(nanoserde::SerJson::serialize_json(&spatial_text))
            }
            OutputGranularity::Tsv => {
                let boxes: Vec<[Point2f; 4]> = recognized
                    .iter()
                    .map(|c| {
                        [
                            Point2f::new(c.box_points[0].0, c.box_points[0].1),
                            Point2f::new(c.box_points[1].0, c.box_points[1].1),
                            Point2f::new(c.box_points[2].0, c.box_points[2].1),
                            Point2f::new(c.box_points[3].0, c.box_points[3].1),
                        ]
                    })
                    .collect();
                let texts: Vec<String> = recognized.iter().map(|c| c.text.clone()).collect();
                let tsv_text = to_tsv(&boxes, &texts);
                Ok(nanoserde::SerJson::serialize_json(&tsv_text))
            }
            OutputGranularity::Words => {
                let mut words: Vec<TextResult> = Vec::new();
                for cand in &recognized {
                    let parts: Vec<&str> = cand.text.split_whitespace().collect();
                    if parts.len() <= 1 {
                        words.push(cand.clone());
                    } else {
                        let char_w = cand.frame.width / (cand.text.len().max(1) as f32);
                        let mut curr_x = cand.frame.left;
                        for part in parts {
                            let part_w = (part.len() as f32 * char_w).max(1.0);
                            let part_pts = [
                                (curr_x, cand.frame.top),
                                (curr_x + part_w, cand.frame.top),
                                (curr_x + part_w, cand.frame.top + cand.frame.height),
                                (curr_x, cand.frame.top + cand.frame.height),
                            ];
                            words.push(TextResult {
                                text: part.to_string(),
                                score: cand.score,
                                frame: Frame::from_points(&part_pts),
                                box_points: part_pts,
                                region_id: cand.region_id,
                                layout_type: cand.layout_type.clone(),
                            });
                            curr_x += part_w + char_w;
                        }
                    }
                }
                let layout_mode = options.layout_model.as_deref();
                let layout_reading = options.layout_reading.unwrap_or(true);
                self.apply_reading_order(&mut words, layout_mode, layout_reading, orig_w, orig_h);
                Ok(nanoserde::SerJson::serialize_json(&words))
            }
            OutputGranularity::Lines => {
                let mut segments = cluster_text_results_into_lines(
                    recognized,
                    line_y.unwrap_or(0.5),
                    word_x.unwrap_or(0.4),
                );
                let layout_mode = options.layout_model.as_deref();
                let layout_reading = options.layout_reading.unwrap_or(true);
                self.apply_reading_order(&mut segments, layout_mode, layout_reading, orig_w, orig_h);
                Ok(nanoserde::SerJson::serialize_json(&segments))
            }
        }
    }

    /// Run full OCR pipeline on raw RGB pixels directly in WebAssembly.
    #[wasm_bindgen(js_name = detectTextRgb)]
    pub fn detect_text_rgb(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
        stride: Option<u32>,
        options_json: Option<String>,
    ) -> Result<String, String> {
        self.detect_text(image_rgb, width, height, 3, stride, options_json)
    }
}
