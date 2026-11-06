use std::sync::Arc;
use wasm_bindgen::prelude::*;
use rusto_nn::{Model, Session, Tensor, TensorLayout};
use rusto_core::{
    calibration::descreen_background_rgba,
    ctc::{composite_ocr_score, CtcDecoder},
    postprocess::{decode_dbnet_boxes_fast, decode_linenet_det_boxes_slice},
    preprocess::{
        prepare_linenet_rec_crop_rgba, prepare_ppocr_rec_crop_rgba,
        preprocess_linenet_det_rgba, preprocess_rgba_raw,
    },
    types::{Frame, TextResult},
};

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
    model: Arc<Model>,
    session: Session<'static>,
}

pub type RustoModel = RtenModel;

#[wasm_bindgen]
impl RtenModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: Vec<u8>) -> Result<RtenModel, String> {
        let model = Arc::new(Model::from_bytes(model_data).map_err(|e| e.to_string())?);
        let session = Session::from_arc(Arc::clone(&model)).map_err(|e| e.to_string())?;
        Ok(RtenModel { model, session })
    }

    #[wasm_bindgen(js_name = findNode)]
    pub fn find_node(&self, name: &str) -> Option<u32> {
        self.model.tensor_names().iter().position(|n| n == name).map(|id| id as u32)
    }

    #[wasm_bindgen(js_name = inputIds)]
    pub fn input_ids(&self) -> Vec<u32> {
        self.model.input_names().iter().filter_map(|name| {
            self.model.tensor_names().iter().position(|n| n == name).map(|id| id as u32)
        }).collect()
    }

    #[wasm_bindgen(js_name = outputIds)]
    pub fn output_ids(&self) -> Vec<u32> {
        self.model.output_names().iter().filter_map(|name| {
            self.model.tensor_names().iter().position(|n| n == name).map(|id| id as u32)
        }).collect()
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
    ) -> Result<InferenceOutput, String> {
        let tensor = Tensor::from_floats(&input_shape, input_data, TensorLayout::NCHW);
        let out = self.session.run_single(&tensor).map_err(|e| e.to_string())?;
        Ok(InferenceOutput {
            shape: out.shape().to_vec(),
            data: out.to_f32_vec(),
        })
    }

    #[wasm_bindgen(js_name = runWithIds)]
    pub fn run_with_ids(
        &self,
        input_id: u32,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
        output_id: u32,
    ) -> Result<InferenceOutput, String> {
        let in_name = self.model.tensor_names().get(input_id as usize)
            .cloned()
            .or_else(|| self.model.input_names().first().cloned())
            .ok_or_else(|| "Input tensor not found".to_string())?;
        let out_name = self.model.tensor_names().get(output_id as usize)
            .cloned()
            .or_else(|| self.model.output_names().first().cloned())
            .ok_or_else(|| "Output tensor not found".to_string())?;

        let tensor = Tensor::from_floats(&input_shape, input_data, TensorLayout::NCHW);
        let outputs = self.session.run(&[(&in_name, &tensor)]).map_err(|e| e.to_string())?;
        let out = outputs.get(&out_name)
            .or_else(|| outputs.values().next())
            .ok_or_else(|| format!("Output tensor '{}' not produced", out_name))?;
        Ok(InferenceOutput {
            shape: out.shape().to_vec(),
            data: out.to_f32_vec(),
        })
    }

    #[wasm_bindgen(js_name = runOrder)]
    pub fn run_order(
        &self,
        num_items: usize,
        feats_data: Vec<f32>,
    ) -> Result<InferenceOutput, String> {
        let in_names = self.model.input_names();
        let feats_name = if in_names.iter().any(|n| n == "feats") {
            "feats"
        } else {
            in_names.first().map(|s| s.as_str()).unwrap_or("feats")
        };
        let mask_name = if in_names.iter().any(|n| n == "mask") {
            "mask"
        } else {
            in_names.get(1).map(|s| s.as_str()).unwrap_or("mask")
        };

        let feats_tensor = Tensor::from_floats(&[1, num_items, 16], feats_data, TensorLayout::NCHW);
        let mask_tensor = Tensor::from_ints(&[1, num_items], vec![1i32; num_items]);

        let results = self.session.run(&[
            (feats_name, &feats_tensor),
            (mask_name, &mask_tensor),
        ]).map_err(|e| format!("Order model run error: {:?}", e))?;

        let out = results.get("a")
            .or_else(|| results.get("logits"))
            .or_else(|| results.values().next())
            .ok_or_else(|| "Expected float tensor output from order model".to_string())?;

        Ok(InferenceOutput {
            shape: out.shape().to_vec(),
            data: out.to_f32_vec(),
        })
    }
}

pub type TextResultWasm = TextResult;
pub type FrameWasm = Frame;

#[wasm_bindgen]
pub struct RtenOcrPipeline {
    det_model: Arc<Model>,
    det_session: Session<'static>,
    rec_model: Arc<Model>,
    rec_session: Session<'static>,
    decoder: CtcDecoder,
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
        let det_model = Arc::new(Model::from_bytes(det_data).map_err(|e| format!("Det model load error: {:?}", e))?);
        let det_session = Session::from_arc(Arc::clone(&det_model)).map_err(|e| format!("Det session error: {:?}", e))?;

        let rec_model = Arc::new(Model::from_bytes(rec_data).map_err(|e| format!("Rec model load error: {:?}", e))?);
        let rec_session = Session::from_arc(Arc::clone(&rec_model)).map_err(|e| format!("Rec session error: {:?}", e))?;

        let decoder = CtcDecoder::from_string(dict_text);

        Ok(RtenOcrPipeline {
            det_model,
            det_session,
            rec_model,
            rec_session,
            decoder,
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
        let det_in_channels = self.det_model.input_channels();

        if det_in_channels == 1 {
            // LineNet 1-channel grayscale ink-map detection (e.g. Indra OCR)
            let limit = max_side_len.unwrap_or(960) as i32;
            let (det_input, target_w, target_h, scale_w, scale_h) =
                preprocess_linenet_det_rgba(processed_rgba, orig_w, orig_h, limit, "max");

            let det_tensor = Tensor::from_floats(&[1, 1, target_h, target_w], det_input, TensorLayout::NCHW);
            let det_outputs = self.det_session.run_single(&det_tensor)
                .map_err(|e| format!("Det inference error: {:?}", e))?;

            let sh = det_outputs.shape().to_vec();
            if sh.len() >= 4 && sh[1] >= 3 {
                Ok(decode_linenet_det_boxes_slice(
                    &det_outputs.to_f32_vec(),
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

            let det_tensor = Tensor::from_floats(&[1, 3, target_h, target_w], det_input, TensorLayout::NCHW);
            let det_outputs = self.det_session.run_single(&det_tensor)
                .map_err(|e| format!("Det inference error: {:?}", e))?;

            let heat_map = det_outputs.to_f32_vec();
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

    fn recognize_box(
        &self,
        processed_rgba: &[u8],
        orig_w: usize,
        orig_h: usize,
        box_coords: [f32; 4],
        det_score: f32,
    ) -> Result<Option<TextResult>, String> {
        let [x0, y0, x1, y1] = box_coords;

        let rec_in_channels = self.rec_model.input_channels();

        if rec_in_channels == 1 {
            // LineNet recognition (e.g. Indra OCR)
            let (rec_input, rec_w, rec_h) = match prepare_linenet_rec_crop_rgba(processed_rgba, orig_w, orig_h, box_coords) {
                Some(crop) => crop,
                None => return Ok(None),
            };

            let in_names = self.rec_model.input_names();
            let in_name0 = in_names.first().map(|s| s.as_str()).unwrap_or("x");
            let in_name1 = in_names.get(1).map(|s| s.as_str()).unwrap_or("widths");

            let x_tensor = Tensor::from_floats(&[1, 1, rec_h, rec_w], rec_input, TensorLayout::NCHW);
            let widths_tensor = Tensor::from_ints(&[1], vec![rec_w as i32]);

            let rec_outputs = self.rec_session.run(&[
                (in_name0, &x_tensor),
                (in_name1, &widths_tensor),
            ]).map_err(|e| format!("Rec inference error: {:?}", e))?;

            let ids_vec = rec_outputs.get("ids")
                .map(|t| t.to_i32_vec())
                .or_else(|| rec_outputs.values().next().map(|t| t.to_i32_vec()))
                .unwrap_or_default();
            let conf_vec = rec_outputs.get("conf")
                .map(|t| t.to_f32_vec())
                .or_else(|| rec_outputs.values().nth(1).map(|t| t.to_f32_vec()))
                .unwrap_or_default();
            let len_val = rec_outputs.get("len")
                .or_else(|| rec_outputs.get("lengths"))
                .and_then(|t| t.to_i32_vec().first().copied())
                .map(|l| l as usize)
                .or_else(|| rec_outputs.values().nth(2).and_then(|t| t.to_i32_vec().first().copied().map(|l| l as usize)))
                .unwrap_or(ids_vec.len());

            let (trimmed, avg_conf) = match self.decoder.decode_linenet_tokens(&ids_vec, &conf_vec, len_val) {
                Some(res) => res,
                None => return Ok(None),
            };
            let text_score = composite_ocr_score(det_score, avg_conf);

            Ok(Some(TextResult::from_coords(x0, y0, x1, y1, trimmed, text_score)))
        } else {
            // Standard PP-OCR SVTR recognition
            let (rec_input, rec_w, rec_h) = match prepare_ppocr_rec_crop_rgba(processed_rgba, orig_w, orig_h, box_coords) {
                Some(crop) => crop,
                None => return Ok(None),
            };

            let rec_tensor = Tensor::from_floats(&[1, 3, rec_h, rec_w], rec_input, TensorLayout::NCHW);
            let rec_outputs = self.rec_session.run_single(&rec_tensor)
                .map_err(|e| format!("Rec inference error: {:?}", e))?;

            let sh = rec_outputs.shape().to_vec();
            if sh.len() < 3 {
                return Ok(None);
            }
            let (seq_len, num_classes) = (sh[1], sh[2]);
            let rec_data = rec_outputs.to_f32_vec();

            let (trimmed, avg_conf) = match self.decoder.decode_single_crop(&rec_data, seq_len, num_classes) {
                Some(res) => res,
                None => return Ok(None),
            };
            let text_score = composite_ocr_score(det_score, avg_conf);

            Ok(Some(TextResult::from_coords(x0, y0, x1, y1, trimmed, text_score)))
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

        let mut results = Vec::new();
        for (box_coords, det_score) in raw_boxes {
            if let Some(res) = self.recognize_box(&processed_rgba, orig_w, orig_h, box_coords, det_score)? {
                results.push(res);
            }
        }

        serde_json::to_string(&results).map_err(|e| format!("Serialization error: {:?}", e))
    }
}
