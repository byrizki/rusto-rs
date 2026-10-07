use ndarray::{Array3, Array4, Ix3, Ix4};
use ort::session::Session as OrtSession;
use ort::value::Tensor as OrtTensor;
use rten::Model as RtenModel;
use rten_tensor::Tensor as RtenTensor;
use rten_tensor::{AsView, Layout};
use rusto::config::InitializeConfig;
use rusto::det::sort_det_boxes;
use rusto::engine::EngineSession;
use rusto::types::EngineConfig;
use rusto::geometry::{
    apply_vertical_padding, get_rotate_crop_image, map_boxes_to_original,
    resize_image_within_bounds, OpRecord,
};
use rusto::image_impl::{imread, Mat, Point2f};
use rusto::postprocess::DBPostProcess;
use rusto::preprocess::DetPreProcess;
use rusto::rec::{resize_norm_rec_img, CtcDecoder};
use rusto::{DetectTextResult, ImageSource, OcrRunOptions, RustO};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Benchmark timing breakdown for each stage of RustO base pipeline
#[derive(Debug, Clone, Default)]
struct PipelineTimings {
    det_pre_ms: f64,
    det_infer_ms: f64,
    det_post_ms: f64,
    crop_unwarp_ms: f64,
    rec_pre_ms: f64,
    rec_infer_ms: f64,
    rec_decode_ms: f64,
    total_pipeline_ms: f64,
}

#[derive(Debug, Clone)]
struct PipelineResult {
    lines: Vec<(String, f32, [Point2f; 4])>,
    timings: PipelineTimings,
}

trait OcrEngineBackend {
    fn name(&self) -> &'static str;
    fn infer_det(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, Box<dyn std::error::Error>>;
    fn infer_rec(&mut self, input: &Array4<f32>) -> Result<Array3<f32>, Box<dyn std::error::Error>>;
}

// ----------------------------------------------------------------------------
// 1. EngineSession Backend (Native RustO RTen Engine)
// ----------------------------------------------------------------------------
struct EngineSessionBackend {
    det: EngineSession,
    rec: EngineSession,
}

impl EngineSessionBackend {
    fn new(det_path: &Path, rec_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let engine_cfg = EngineConfig::default();
        let det = EngineSession::from_path(det_path, &engine_cfg)?;
        let rec = EngineSession::from_path(rec_path, &engine_cfg)?;
        Ok(Self { det, rec })
    }
}

impl OcrEngineBackend for EngineSessionBackend {
    fn name(&self) -> &'static str {
        "EngineSession (Native rusto)"
    }

    fn infer_det(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let preds_dyn = self.det.run(input.clone().into_dyn())?;
        let preds: Array4<f32> = preds_dyn.into_dimensionality::<Ix4>()?;
        Ok(preds)
    }

    fn infer_rec(&mut self, input: &Array4<f32>) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let preds_dyn = self.rec.run(input.clone().into_dyn())?;
        let preds: Array3<f32> = preds_dyn.into_dimensionality::<Ix3>()?;
        Ok(preds)
    }
}

// ----------------------------------------------------------------------------
// 2. Rust ort Backend
// ----------------------------------------------------------------------------
struct OrtBackend {
    det_session: OrtSession,
    rec_session: OrtSession,
}

impl OrtBackend {
    fn new(det_path: &Path, rec_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let det_session = OrtSession::builder()?.commit_from_file(det_path)?;
        let rec_session = OrtSession::builder()?.commit_from_file(rec_path)?;
        Ok(Self {
            det_session,
            rec_session,
        })
    }
}

impl OcrEngineBackend for OrtBackend {
    fn name(&self) -> &'static str {
        "Rust ort (ONNX)"
    }

    fn infer_det(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let s = input.shape();
        let input_slice = input.as_slice().ok_or("det tensor not contiguous")?;
        let ort_tensor = OrtTensor::<f32>::from_array(([s[0], s[1], s[2], s[3]], input_slice.to_vec()))?;
        let outputs = self.det_session.run(ort::inputs![ort_tensor])?;
        let out_view = outputs[0].try_extract_array::<f32>()?;
        let shape = out_view.shape();
        let data: Vec<f32> = out_view.iter().copied().collect();
        let preds = Array4::from_shape_vec(
            (shape[0], shape[1], shape[2], shape[3]),
            data,
        )?;
        Ok(preds)
    }

    fn infer_rec(&mut self, input: &Array4<f32>) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let s = input.shape();
        let input_slice = input.as_slice().ok_or("rec tensor not contiguous")?;
        let ort_tensor = OrtTensor::<f32>::from_array(([s[0], s[1], s[2], s[3]], input_slice.to_vec()))?;
        let outputs = self.rec_session.run(ort::inputs![ort_tensor])?;
        let out_view = outputs[0].try_extract_array::<f32>()?;
        let shape = out_view.shape();
        let data: Vec<f32> = out_view.iter().copied().collect();
        let preds = Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            data,
        )?;
        Ok(preds)
    }
}

// ----------------------------------------------------------------------------
// 3. RTen Backend (Generic for .onnx and .rten formats)
// ----------------------------------------------------------------------------
struct RtenBackend {
    name: &'static str,
    det_model: RtenModel,
    rec_model: RtenModel,
}

impl RtenBackend {
    fn new(name: &'static str, det_path: &Path, rec_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let det_model = RtenModel::load_file(det_path)?;
        let rec_model = RtenModel::load_file(rec_path)?;
        Ok(Self {
            name,
            det_model,
            rec_model,
        })
    }
}

impl OcrEngineBackend for RtenBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn infer_det(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let s = input.shape();
        let input_slice = input.as_slice().ok_or("det tensor not contiguous")?;
        let tensor = RtenTensor::<f32>::from_data(&[s[0], s[1], s[2], s[3]], input_slice.to_vec());
        let in_id = self.det_model.input_ids()[0];
        let outputs = self.det_model.run(
            vec![(in_id, tensor.view().into())],
            &self.det_model.output_ids(),
            None,
        )?;
        let out_float = match &outputs[0] {
            rten::Value::FloatTensor(t) => t,
            _ => return Err("Expected FloatTensor from RTen det model".into()),
        };
        let out_shape = out_float.shape();
        let data: Vec<f32> = out_float.iter().copied().collect();
        let preds = Array4::from_shape_vec(
            (out_shape[0], out_shape[1], out_shape[2], out_shape[3]),
            data,
        )?;
        Ok(preds)
    }

    fn infer_rec(&mut self, input: &Array4<f32>) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let s = input.shape();
        let input_slice = input.as_slice().ok_or("rec tensor not contiguous")?;
        let tensor = RtenTensor::<f32>::from_data(&[s[0], s[1], s[2], s[3]], input_slice.to_vec());
        let in_id = self.rec_model.input_ids()[0];
        let outputs = self.rec_model.run(
            vec![(in_id, tensor.view().into())],
            &self.rec_model.output_ids(),
            None,
        )?;
        let out_float = match &outputs[0] {
            rten::Value::FloatTensor(t) => t,
            _ => return Err("Expected FloatTensor from RTen rec model".into()),
        };
        let out_shape = out_float.shape();
        let data: Vec<f32> = out_float.iter().copied().collect();
        let preds = Array3::from_shape_vec(
            (out_shape[0], out_shape[1], out_shape[2]),
            data,
        )?;
        Ok(preds)
    }
}

// ----------------------------------------------------------------------------
// Authentic RustO Base Pipeline Runner
// ----------------------------------------------------------------------------
fn run_rusto_base_pipeline<E: OcrEngineBackend>(
    engine: &mut E,
    raw_img: &Mat,
    decoder: &CtcDecoder,
) -> Result<PipelineResult, Box<dyn std::error::Error>> {
    let t_pipeline_start = Instant::now();

    // --------------------------------------------------------
    // Step 1: Detection Preprocessing
    // --------------------------------------------------------
    let t_det_pre0 = Instant::now();
    let mut op_record: OpRecord = OpRecord::new();
    let (resized, ratio_h, ratio_w) =
        resize_image_within_bounds(raw_img, 30.0, 2000.0)?;
    let mut m = std::collections::BTreeMap::new();
    m.insert("ratio_h".to_string(), ratio_h);
    m.insert("ratio_w".to_string(), ratio_w);
    op_record.insert("preprocess".to_string(), m);

    let (padded, op_record) = apply_vertical_padding(
        &resized,
        op_record,
        8.0,
        30.0,
    )?;

    // PP-OCRv6 Det configuration: min side 736, BGR [0.5, 0.5, 0.5] mean & std
    let pre = DetPreProcess::new(
        736,
        "min".to_string(),
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    );
    let det_input = pre.run(&padded)?;
    let det_pre_ms = t_det_pre0.elapsed().as_secs_f64() * 1000.0;

    // --------------------------------------------------------
    // Step 2: Detection Neural Inference
    // --------------------------------------------------------
    let t_det_infer0 = Instant::now();
    let det_preds = engine.infer_det(&det_input)?;
    let det_infer_ms = t_det_infer0.elapsed().as_secs_f64() * 1000.0;

    // --------------------------------------------------------
    // Step 3: Detection Postprocessing (DBNet binarization & unclip)
    // --------------------------------------------------------
    let t_det_post0 = Instant::now();
    let post = DBPostProcess::new(0.3, 0.6, 1000, 2.0, true);
    let (mut padded_boxes, _) = post.process(&det_preds, padded.rows(), padded.cols())?;
    sort_det_boxes(&mut padded_boxes);
    let det_post_ms = t_det_post0.elapsed().as_secs_f64() * 1000.0;

    // --------------------------------------------------------
    // Step 4: Text Line Crop & Perspective Unwarp
    // --------------------------------------------------------
    let t_crop0 = Instant::now();
    let mut crop_imgs = Vec::with_capacity(padded_boxes.len());
    for b in &padded_boxes {
        let crop = get_rotate_crop_image(&padded, b)?;
        crop_imgs.push(crop);
    }
    let mut orig_boxes = padded_boxes.clone();
    map_boxes_to_original(&mut orig_boxes, &op_record, raw_img.rows(), raw_img.cols());
    let crop_unwarp_ms = t_crop0.elapsed().as_secs_f64() * 1000.0;

    // --------------------------------------------------------
    // Step 5: Recognition Preprocessing & Batch Construction
    // --------------------------------------------------------
    let t_rec_pre0 = Instant::now();
    let img_num = crop_imgs.len();
    let batch_num = 6usize; // PP-OCRv6 default batch size

    let mut width_list = Vec::with_capacity(img_num);
    for img in &crop_imgs {
        width_list.push(img.cols() as f32 / img.rows().max(1) as f32);
    }
    let mut indices: Vec<usize> = (0..img_num).collect();
    indices.sort_by(|&a, &b| {
        width_list[a]
            .partial_cmp(&width_list[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut all_batches = Vec::new();
    let mut beg = 0usize;
    while beg < img_num {
        let end = (beg + batch_num).min(img_num);
        let mut max_wh_ratio = 320.0 / 48.0;
        let mut wh_ratio_list = Vec::with_capacity(end - beg);
        for &idx in &indices[beg..end] {
            let h = crop_imgs[idx].rows() as f32;
            let w = crop_imgs[idx].cols() as f32;
            let wh = if h > 0.0 { w / h } else { 1.0 };
            if wh > max_wh_ratio {
                max_wh_ratio = wh;
            }
            wh_ratio_list.push(wh);
        }

        let mut norm_batch = Vec::with_capacity(end - beg);
        for &idx in &indices[beg..end] {
            let norm = resize_norm_rec_img(&crop_imgs[idx], 3, 48, max_wh_ratio)?;
            norm_batch.push(norm);
        }

        let n = norm_batch.len();
        let batch_img_width = (48.0 * max_wh_ratio).round() as usize;
        let mut batch_tensor = Array4::<f32>::zeros((n, 3, 48, batch_img_width));
        for (i, arr) in norm_batch.into_iter().enumerate() {
            batch_tensor.slice_mut(ndarray::s![i, .., .., ..]).assign(&arr);
        }
        all_batches.push((beg, end, batch_tensor, wh_ratio_list, max_wh_ratio));
        beg = end;
    }
    let rec_pre_ms = t_rec_pre0.elapsed().as_secs_f64() * 1000.0;

    // --------------------------------------------------------
    // Step 6: Recognition Neural Inference & Step 7: CTC Decoding
    // --------------------------------------------------------
    let mut rec_infer_ms = 0.0;
    let mut rec_decode_ms = 0.0;
    let mut all_texts = vec![(String::new(), 0.0f32); img_num];

    for (beg, _end, batch_tensor, wh_ratio_list, max_wh_ratio) in all_batches {
        let t_infer = Instant::now();
        let preds = engine.infer_rec(&batch_tensor)?;
        rec_infer_ms += t_infer.elapsed().as_secs_f64() * 1000.0;

        let t_dec = Instant::now();
        let (lines, _) = decoder.decode(preds, false, &wh_ratio_list, max_wh_ratio);
        for (local_idx, (txt, score)) in lines.into_iter().enumerate() {
            let actual_idx = indices[beg + local_idx];
            all_texts[actual_idx] = (txt, score);
        }
        rec_decode_ms += t_dec.elapsed().as_secs_f64() * 1000.0;
    }

    let total_pipeline_ms = t_pipeline_start.elapsed().as_secs_f64() * 1000.0;

    // Filter by text_score threshold (>= 0.5)
    let mut final_results = Vec::new();
    for (box_pts, (text, score)) in orig_boxes.into_iter().zip(all_texts.into_iter()) {
        if score >= 0.5 {
            final_results.push((text, score, box_pts));
        }
    }

    Ok(PipelineResult {
        lines: final_results,
        timings: PipelineTimings {
            det_pre_ms,
            det_infer_ms,
            det_post_ms,
            crop_unwarp_ms,
            rec_pre_ms,
            rec_infer_ms,
            rec_decode_ms,
            total_pipeline_ms,
        },
    })
}

#[allow(dead_code)]
struct BenchmarkStats {
    name: &'static str,
    runs: usize,
    avg: PipelineTimings,
    min_total_ms: f64,
    max_total_ms: f64,
    detected_count: usize,
    sample_lines: Vec<(String, f32)>,
}

fn benchmark_engine<E: OcrEngineBackend>(
    mut engine: E,
    raw_img: &Mat,
    decoder: &CtcDecoder,
    warmup_runs: usize,
    bench_runs: usize,
) -> Result<BenchmarkStats, Box<dyn std::error::Error>> {
    let name = engine.name();
    print!("  Benchmarking {} (warmup={} runs={})...", name, warmup_runs, bench_runs);
    std::io::stdout().flush()?;

    // Warmup
    for _ in 0..warmup_runs {
        let _ = run_rusto_base_pipeline(&mut engine, raw_img, decoder)?;
    }

    let mut timings_list = Vec::with_capacity(bench_runs);
    let mut last_result = None;

    for _ in 0..bench_runs {
        let res = run_rusto_base_pipeline(&mut engine, raw_img, decoder)?;
        timings_list.push(res.timings.clone());
        last_result = Some(res);
    }

    let n = bench_runs as f64;
    let mut avg = PipelineTimings::default();
    let mut min_total_ms = f64::MAX;
    let mut max_total_ms = f64::MIN;

    for t in &timings_list {
        avg.det_pre_ms += t.det_pre_ms / n;
        avg.det_infer_ms += t.det_infer_ms / n;
        avg.det_post_ms += t.det_post_ms / n;
        avg.crop_unwarp_ms += t.crop_unwarp_ms / n;
        avg.rec_pre_ms += t.rec_pre_ms / n;
        avg.rec_infer_ms += t.rec_infer_ms / n;
        avg.rec_decode_ms += t.rec_decode_ms / n;
        avg.total_pipeline_ms += t.total_pipeline_ms / n;

        if t.total_pipeline_ms < min_total_ms {
            min_total_ms = t.total_pipeline_ms;
        }
        if t.total_pipeline_ms > max_total_ms {
            max_total_ms = t.total_pipeline_ms;
        }
    }

    let res = last_result.unwrap();
    let sample_lines = res.lines.iter().map(|(t, s, _)| (t.clone(), *s)).collect();
    println!(" Done! Avg Total: {:.2} ms (Det Infer: {:.2} ms, Rec Infer: {:.2} ms)",
        avg.total_pipeline_ms, avg.det_infer_ms, avg.rec_infer_ms);

    Ok(BenchmarkStats {
        name,
        runs: bench_runs,
        avg,
        min_total_ms,
        max_total_ms,
        detected_count: res.lines.len(),
        sample_lines,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  RUSTO BASE PIPELINE: INFERENCE & STAGE TIMING COMPARISON (PP-OCRv6 Tiny)");
    println!("  Target: ktp.jpeg (1600x993) | Preprocessing til Finished");
    println!("================================================================================\n");

    let repo_root = Path::new("../../");
    let ktp_path = repo_root.join("ktp.jpeg");
    let models_dir = repo_root.join("models/PPOCR_v6_tiny");
    let dict_path = models_dir.join("dict.txt");

    let raw_img = imread(&ktp_path)?;
    let decoder = CtcDecoder::from_file(&dict_path)?;
    println!("Loaded image ktp.jpeg ({}x{}) and dictionary ({} tokens).\n",
        raw_img.cols(), raw_img.rows(), decoder.chars.len());

    let warmup = 2;
    let runs = 5;

    // ------------------------------------------------------------
    // 0. Verify Native RustO::detect_text baseline
    // ------------------------------------------------------------
    println!("--- [0] Native RustO::initialize / detect_text Baseline ---");
    let ocr_config = InitializeConfig::ppv6(
        models_dir.join("det.onnx"),
        models_dir.join("rec.onnx"),
        dict_path.clone(),
    );
    let mut native_ocr = RustO::initialize(ocr_config)?;
    let t_native0 = Instant::now();
    let native_res = native_ocr.detect_text(
        &ImageSource::Path(ktp_path.clone()),
        &OcrRunOptions::default(),
    )?;
    let native_warmup_ms = t_native0.elapsed().as_secs_f64() * 1000.0;
    let mut native_count = 0;
    if let DetectTextResult::Structured(items) = native_res {
        native_count = items.len();
    }
    println!("  Native RustO::detect_text warm run: {:.2} ms ({} lines detected)\n",
        native_warmup_ms, native_count);

    // ------------------------------------------------------------
    // Initialize All 4 Engines
    // ------------------------------------------------------------
    println!("--- Initializing Inference Engines ---");
    let engine_backend = EngineSessionBackend::new(&models_dir.join("det.onnx"), &models_dir.join("rec.onnx"))?;
    println!("  [✓] EngineSession backend initialized");

    let ort_backend = OrtBackend::new(&models_dir.join("det.onnx"), &models_dir.join("rec.onnx"))?;
    println!("  [✓] Rust ort backend initialized");

    let rten_onnx_backend = RtenBackend::new(
        "RTen (.onnx)",
        &models_dir.join("det.onnx"),
        &models_dir.join("rec.onnx"),
    )?;
    println!("  [✓] RTen (.onnx) backend initialized");

    let rten_rten_backend = RtenBackend::new(
        "RTen (.rten)",
        &models_dir.join("det.rten"),
        &models_dir.join("rec.rten"),
    )?;
    println!("  [✓] RTen (.rten) backend initialized\n");

    // ------------------------------------------------------------
    // Execute Benchmarks
    // ------------------------------------------------------------
    println!("--- Running Identical Base Pipeline Across All Engines ---");
    let stats_engine = benchmark_engine(engine_backend, &raw_img, &decoder, warmup, runs)?;
    let stats_ort = benchmark_engine(ort_backend, &raw_img, &decoder, warmup, runs)?;
    let stats_rten_onnx = benchmark_engine(rten_onnx_backend, &raw_img, &decoder, warmup, runs)?;
    let stats_rten_rten = benchmark_engine(rten_rten_backend, &raw_img, &decoder, warmup, runs)?;

    let all_stats = vec![&stats_engine, &stats_ort, &stats_rten_onnx, &stats_rten_rten];

    // ------------------------------------------------------------
    // Print Summary Console Table
    // ------------------------------------------------------------
    println!("\n=======================================================================================================================");
    println!("                                   STAGE-BY-STAGE LATENCY BREAKDOWN (AVG ms, N={})", runs);
    println!("=======================================================================================================================");
    println!("{:<17} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<12}",
        "Engine", "Det Pre", "Det Infer", "Det Post", "Crop/Warp", "Rec Pre", "Rec Infer", "Rec Dec", "TOTAL PIPELINE");
    println!("------------------+------------+------------+------------+------------+------------+------------+------------+--------------");
    for s in &all_stats {
        println!("{:<17} | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>10.2} ms",
            s.name,
            s.avg.det_pre_ms,
            s.avg.det_infer_ms,
            s.avg.det_post_ms,
            s.avg.crop_unwarp_ms,
            s.avg.rec_pre_ms,
            s.avg.rec_infer_ms,
            s.avg.rec_decode_ms,
            s.avg.total_pipeline_ms,
        );
    }
    println!("=======================================================================================================================");

    // ------------------------------------------------------------
    // Generate Comprehensive Markdown Report
    // ------------------------------------------------------------
    let report_path = repo_root.join("benchmarks/BASE_PIPELINE_COMPARISON.md");
    let mut f = File::create(&report_path)?;

    writeln!(f, "# RustO Base Pipeline End-to-End Latency Benchmark")?;
    writeln!(f, "")?;
    writeln!(f, "> **Benchmark Target**: `ktp.jpeg` (1600x993 pixels)")?;
    writeln!(f, "> **Model Architecture**: PP-OCRv6 Tiny (`det` ~1.8 MB, `rec` ~4.3 MB)")?;
    writeln!(f, "> **Execution Pipeline**: Native RustO base pipeline (DetPreProcess -> Det Inference -> DBPostProcess -> 4-point Perspective Crop -> RecPreProcess -> Rec Inference -> CTC Decode)")?;
    writeln!(f, "> **Iterations**: {} warmup runs, {} timed benchmark runs", warmup, runs)?;
    writeln!(f, "")?;
    writeln!(f, "---")?;
    writeln!(f, "")?;
    writeln!(f, "## 1. Executive Summary & Total Pipeline Latency")?;
    writeln!(f, "")?;
    writeln!(f, "| Inference Engine | Total Pipeline Latency (ms) | Min (ms) | Max (ms) | Neural Infer Only (ms) | Image & Postproc (ms) | Text Lines Detected |")?;
    writeln!(f, "|---|---|---|---|---|---|---|")?;

    for s in &all_stats {
        let neural_ms = s.avg.det_infer_ms + s.avg.rec_infer_ms;
        let non_neural_ms = s.avg.total_pipeline_ms - neural_ms;
        writeln!(f, "| **{}** | **{:.2} ms** | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {}/16 lines |",
            s.name, s.avg.total_pipeline_ms, s.min_total_ms, s.max_total_ms, neural_ms, non_neural_ms, s.detected_count)?;
    }

    writeln!(f, "")?;
    writeln!(f, "---")?;
    writeln!(f, "")?;
    writeln!(f, "## 2. Detailed Stage-by-Stage Breakdown (ms)")?;
    writeln!(f, "")?;
    writeln!(f, "Every millisecond from raw image preprocessing til final transcription decoded:")?;
    writeln!(f, "")?;
    writeln!(f, "| Pipeline Stage | Description | EngineSession (Native) | Rust ort | RTen (.onnx) | RTen (.rten) |")?;
    writeln!(f, "|---|---|---|---|---|---|")?;
    writeln!(f, "| **1. Det Preprocess** | Bounds resize, vertical pad, 736 min-side resize, ImageNet BGR norm | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.det_pre_ms, stats_ort.avg.det_pre_ms, stats_rten_onnx.avg.det_pre_ms, stats_rten_rten.avg.det_pre_ms)?;
    writeln!(f, "| **2. Det Inference** | Neural detection model inference on `[1, 3, 736, 1184]` | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.det_infer_ms, stats_ort.avg.det_infer_ms, stats_rten_onnx.avg.det_infer_ms, stats_rten_rten.avg.det_infer_ms)?;
    writeln!(f, "| **3. Det Postprocess** | DBNet binary thresholding (0.3), contour extraction, unclip (2.0), box sorting | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.det_post_ms, stats_ort.avg.det_post_ms, stats_rten_onnx.avg.det_post_ms, stats_rten_rten.avg.det_post_ms)?;
    writeln!(f, "| **4. Crop & Unwarp** | 4-point perspective warp & 90° orientation rotation for all 16 boxes | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.crop_unwarp_ms, stats_ort.avg.crop_unwarp_ms, stats_rten_onnx.avg.crop_unwarp_ms, stats_rten_rten.avg.crop_unwarp_ms)?;
    writeln!(f, "| **5. Rec Preprocess** | Aspect ratio sorting, height-48 scaling, zero padding, batch construction | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.rec_pre_ms, stats_ort.avg.rec_pre_ms, stats_rten_onnx.avg.rec_pre_ms, stats_rten_rten.avg.rec_pre_ms)?;
    writeln!(f, "| **6. Rec Inference** | Neural text recognition across all 16 crops (batches of 6) | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.rec_infer_ms, stats_ort.avg.rec_infer_ms, stats_rten_onnx.avg.rec_infer_ms, stats_rten_rten.avg.rec_infer_ms)?;
    writeln!(f, "| **7. CTC Decode** | Greedy token decoding & vocabulary mapping to UTF-8 characters | {:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms |",
        stats_engine.avg.rec_decode_ms, stats_ort.avg.rec_decode_ms, stats_rten_onnx.avg.rec_decode_ms, stats_rten_rten.avg.rec_decode_ms)?;
    writeln!(f, "| **TOTAL PIPELINE** | **Preprocessing Til Finished** | **{:.2} ms** | **{:.2} ms** | **{:.2} ms** | **{:.2} ms** |",
        stats_engine.avg.total_pipeline_ms, stats_ort.avg.total_pipeline_ms, stats_rten_onnx.avg.total_pipeline_ms, stats_rten_rten.avg.total_pipeline_ms)?;

    writeln!(f, "")?;
    writeln!(f, "---")?;
    writeln!(f, "")?;
    writeln!(f, "## 3. Transcription Parity Verification on ktp.jpeg")?;
    writeln!(f, "")?;
    writeln!(f, "| Line # | EngineSession (Native rusto) | Rust ort | RTen (.onnx) | RTen (.rten) |")?;
    writeln!(f, "|---|---|---|---|---|")?;

    let max_lines = stats_engine.sample_lines.len()
        .max(stats_ort.sample_lines.len())
        .max(stats_rten_onnx.sample_lines.len())
        .max(stats_rten_rten.sample_lines.len());

    for i in 0..max_lines {
        let l_engine = stats_engine.sample_lines.get(i).map(|(t, s)| format!("{} ({:.2})", t, s)).unwrap_or_else(|| "-".into());
        let l_ort = stats_ort.sample_lines.get(i).map(|(t, s)| format!("{} ({:.2})", t, s)).unwrap_or_else(|| "-".into());
        let l_ro = stats_rten_onnx.sample_lines.get(i).map(|(t, s)| format!("{} ({:.2})", t, s)).unwrap_or_else(|| "-".into());
        let l_rr = stats_rten_rten.sample_lines.get(i).map(|(t, s)| format!("{} ({:.2})", t, s)).unwrap_or_else(|| "-".into());
        writeln!(f, "| Line {:02} | {} | {} | {} | {} |", i + 1, l_engine, l_ort, l_ro, l_rr)?;
    }

    writeln!(f, "")?;
    println!("\nReport successfully generated at: {}", report_path.display());
    Ok(())
}
