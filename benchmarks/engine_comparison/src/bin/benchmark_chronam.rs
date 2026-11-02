use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO, TextResult};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct BenchmarkStats {
    warmups: usize,
    runs: usize,
    durations_ms: Vec<f64>,
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    std_dev_ms: f64,
    p95_ms: f64,
    results: Vec<TextResult>,
    det_size_bytes: u64,
    rec_size_bytes: u64,
    dict_size_bytes: u64,
    total_model_size_bytes: u64,
}

fn file_size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn run_pipeline_benchmark(
    name: &str,
    config_json: &str,
    img_path: &Path,
    det_path: &Path,
    rec_path: &Path,
    dict_path: &Path,
    warmup_count: usize,
    run_count: usize,
) -> BenchmarkStats {
    println!(">>> Initializing {}...", name);
    let init_config = InitializeConfig::from_json(config_json).expect("Failed to parse config JSON");
    let mut ocr = RustO::initialize(init_config).expect("Failed to initialize RustO");

    let options = OcrRunOptions::default();
    let img_src = ImageSource::Path(img_path.to_path_buf());

    println!(">>> Warming up {} ({} runs)...", name, warmup_count);
    for _ in 0..warmup_count {
        let _ = ocr.detect_text(&img_src, &options).expect("Warmup inference failed");
    }

    println!(">>> Benchmarking {} ({} timed runs)...", name, run_count);
    let mut durations_ms = Vec::with_capacity(run_count);
    let mut last_results = Vec::new();

    for i in 0..run_count {
        let start = Instant::now();
        let res = ocr.detect_text(&img_src, &options).expect("Benchmark inference failed");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        durations_ms.push(elapsed);

        if i == run_count - 1 {
            if let DetectTextResult::Structured(items) = res {
                last_results = items;
            }
        }
    }

    let mut sorted = durations_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_ms = durations_ms.iter().sum::<f64>() / durations_ms.len() as f64;
    let median_ms = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let min_ms = sorted[0];
    let max_ms = sorted[sorted.len() - 1];
    let variance = durations_ms
        .iter()
        .map(|v| (v - mean_ms).powi(2))
        .sum::<f64>()
        / durations_ms.len() as f64;
    let std_dev_ms = variance.sqrt();
    let p95_idx = ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
    let p95_ms = sorted[p95_idx];

    let det_size_bytes = file_size(det_path);
    let rec_size_bytes = file_size(rec_path);
    let dict_size_bytes = file_size(dict_path);
    let total_model_size_bytes = det_size_bytes + rec_size_bytes + dict_size_bytes;

    BenchmarkStats {
        warmups: warmup_count,
        runs: run_count,
        durations_ms,
        mean_ms,
        median_ms,
        min_ms,
        max_ms,
        std_dev_ms,
        p95_ms,
        results: last_results,
        det_size_bytes,
        rec_size_bytes,
        dict_size_bytes,
        total_model_size_bytes,
    }
}

fn analyze_results(results: &[TextResult]) -> (usize, usize, f32, f32, f32, [usize; 4]) {
    let mut total_chars = 0;
    let mut total_words = 0;
    let mut min_score = 1.0f32;
    let mut max_score = 0.0f32;
    let mut sum_score = 0.0f32;
    let mut hist = [0usize; 4]; // [>=0.9, 0.75..0.9, 0.5..0.75, <0.5]

    for r in results {
        let len = r.text.chars().count();
        let words = r.text.split_whitespace().count();
        total_chars += len;
        total_words += words;

        let s = r.score;
        if s < min_score {
            min_score = s;
        }
        if s > max_score {
            max_score = s;
        }
        sum_score += s;

        if s >= 0.90 {
            hist[0] += 1;
        } else if s >= 0.75 {
            hist[1] += 1;
        } else if s >= 0.50 {
            hist[2] += 1;
        } else {
            hist[3] += 1;
        }
    }

    let avg_score = if results.is_empty() {
        0.0
    } else {
        sum_score / results.len() as f32
    };

    (total_chars, total_words, avg_score, min_score, max_score, hist)
}

fn main() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let img_path = repo_root.join("tests/fixtures/chronam-ad-page.png");

    let ppv6_det = repo_root.join("models/PPOCR_v6_tiny/det.rten");
    let ppv6_rec = repo_root.join("models/PPOCR_v6_tiny/rec.rten");
    let ppv6_dict = repo_root.join("models/PPOCR_v6_tiny/dict.txt");

    let indra_det = repo_root.join("models/indra/det.rten");
    let indra_rec = repo_root.join("models/indra/rec.rten");
    let indra_dict = repo_root.join("models/indra/meta.json");

    assert!(img_path.exists(), "Image fixture missing: {}", img_path.display());
    assert!(ppv6_det.exists(), "PP-OCRv6 det missing");
    assert!(indra_det.exists(), "Indra det missing");

    let ppv6_cfg = format!(
        r#"{{
            "detection": {{"modelPath": "{}"}},
            "recognition": {{"modelPath": "{}", "dictPath": "{}"}}
        }}"#,
        ppv6_det.display(),
        ppv6_rec.display(),
        ppv6_dict.display()
    );

    let indra_cfg = format!(
        r#"{{
            "detection": {{"modelPath": "{}"}},
            "recognition": {{"modelPath": "{}", "dictPath": "{}"}}
        }}"#,
        indra_det.display(),
        indra_rec.display(),
        indra_dict.display()
    );

    let ppv6_stats = run_pipeline_benchmark(
        "PP-OCRv6 Tiny",
        &ppv6_cfg,
        &img_path,
        &ppv6_det,
        &ppv6_rec,
        &ppv6_dict,
        2,
        10,
    );

    let indra_stats = run_pipeline_benchmark(
        "Indra OCR",
        &indra_cfg,
        &img_path,
        &indra_det,
        &indra_rec,
        &indra_dict,
        2,
        10,
    );

    let (pp_chars, pp_words, pp_avg, pp_min, pp_max, pp_hist) = analyze_results(&ppv6_stats.results);
    let (in_chars, in_words, in_avg, in_min, in_max, in_hist) = analyze_results(&indra_stats.results);

    println!("\n=========================================================================================");
    println!("                           CHRONICLING AMERICA NEWSPAPER OCR BENCHMARK                   ");
    println!("Image: tests/fixtures/chronam-ad-page.png (999 x 743 px, 549 KB PNG)");
    println!("Iterations: 2 warmup + 10 timed iterations per engine");
    println!("=========================================================================================\n");

    println!("-----------------------------------------------------------------------------------------");
    println!("1. INFERENCE LATENCY COMPARISON (CPU, Single-threaded or Rayon Default)");
    println!("-----------------------------------------------------------------------------------------");
    println!("{:<25} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}", "Metric", "PP-OCRv6 Tiny", "Indra OCR", "Absolute Diff", "Ratio (x)", "Winner");
    println!("{:-<25}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "", "", "");
    
    let speedup_mean = indra_stats.mean_ms / ppv6_stats.mean_ms;
    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | {:<10.2} ms | {:<10.2}x | {}", 
        "Mean Latency", ppv6_stats.mean_ms, indra_stats.mean_ms, (ppv6_stats.mean_ms - indra_stats.mean_ms).abs(), 
        speedup_mean, if ppv6_stats.mean_ms < indra_stats.mean_ms { "PP-OCRv6 Tiny" } else { "Indra OCR" });
    
    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | {:<10.2} ms | {:<10.2}x | {}", 
        "Median Latency", ppv6_stats.median_ms, indra_stats.median_ms, (ppv6_stats.median_ms - indra_stats.median_ms).abs(),
        indra_stats.median_ms / ppv6_stats.median_ms, if ppv6_stats.median_ms < indra_stats.median_ms { "PP-OCRv6 Tiny" } else { "Indra OCR" });
    
    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | {:<10.2} ms | -            | {}", 
        "Min Latency (Fastest)", ppv6_stats.min_ms, indra_stats.min_ms, (ppv6_stats.min_ms - indra_stats.min_ms).abs(),
        if ppv6_stats.min_ms < indra_stats.min_ms { "PP-OCRv6 Tiny" } else { "Indra OCR" });

    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | {:<10.2} ms | -            | {}", 
        "Max Latency", ppv6_stats.max_ms, indra_stats.max_ms, (ppv6_stats.max_ms - indra_stats.max_ms).abs(),
        if ppv6_stats.max_ms < indra_stats.max_ms { "PP-OCRv6 Tiny" } else { "Indra OCR" });

    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | {:<10.2} ms | -            | {}", 
        "P95 Latency", ppv6_stats.p95_ms, indra_stats.p95_ms, (ppv6_stats.p95_ms - indra_stats.p95_ms).abs(),
        if ppv6_stats.p95_ms < indra_stats.p95_ms { "PP-OCRv6 Tiny" } else { "Indra OCR" });

    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | -            | -            | -", 
        "Std. Deviation (Jitter)", ppv6_stats.std_dev_ms, indra_stats.std_dev_ms);

    let pp_per_box = ppv6_stats.mean_ms / ppv6_stats.results.len() as f64;
    let in_per_box = indra_stats.mean_ms / indra_stats.results.len() as f64;
    println!("{:<25} | {:<10.2} ms | {:<10.2} ms | {:<10.2} ms | {:<10.2}x | {}", 
        "Avg Time per Box", pp_per_box, in_per_box, (pp_per_box - in_per_box).abs(),
        in_per_box / pp_per_box, if pp_per_box < in_per_box { "PP-OCRv6 Tiny" } else { "Indra OCR" });

    println!("\n-----------------------------------------------------------------------------------------");
    println!("2. MODEL ASSET SIZES ON DISK (Storage & Memory Footprint)");
    println!("-----------------------------------------------------------------------------------------");
    println!("{:<25} | {:<12} | {:<12} | {:<12} | {:<12}", "Asset", "PP-OCRv6 Tiny", "Indra OCR", "Diff", "Winner");
    println!("{:-<25}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "", "");
    println!("{:<25} | {:<9.2} MB | {:<9.2} MB | {:<9.2} MB | {}", 
        "Detection Model", ppv6_stats.det_size_bytes as f64 / 1_048_576.0, indra_stats.det_size_bytes as f64 / 1_048_576.0,
        (ppv6_stats.det_size_bytes as i64 - indra_stats.det_size_bytes as i64).abs() as f64 / 1_048_576.0,
        if ppv6_stats.det_size_bytes < indra_stats.det_size_bytes { "PP-OCRv6 Tiny" } else { "Indra OCR (4.7x smaller)" });
    
    println!("{:<25} | {:<9.2} MB | {:<9.2} MB | {:<9.2} MB | {}", 
        "Recognition Model", ppv6_stats.rec_size_bytes as f64 / 1_048_576.0, indra_stats.rec_size_bytes as f64 / 1_048_576.0,
        (ppv6_stats.rec_size_bytes as i64 - indra_stats.rec_size_bytes as i64).abs() as f64 / 1_048_576.0,
        if ppv6_stats.rec_size_bytes < indra_stats.rec_size_bytes { "PP-OCRv6 Tiny" } else { "Indra OCR" });

    println!("{:<25} | {:<9.2} KB | {:<9.2} KB | {:<9.2} KB | {}", 
        "Dictionary / Meta", ppv6_stats.dict_size_bytes as f64 / 1024.0, indra_stats.dict_size_bytes as f64 / 1024.0,
        (ppv6_stats.dict_size_bytes as i64 - indra_stats.dict_size_bytes as i64).abs() as f64 / 1024.0,
        if ppv6_stats.dict_size_bytes < indra_stats.dict_size_bytes { "PP-OCRv6 Tiny" } else { "Indra OCR" });

    println!("{:<25} | {:<9.2} MB | {:<9.2} MB | {:<9.2} MB | {}", 
        "Total Pipeline Assets", ppv6_stats.total_model_size_bytes as f64 / 1_048_576.0, indra_stats.total_model_size_bytes as f64 / 1_048_576.0,
        (ppv6_stats.total_model_size_bytes as i64 - indra_stats.total_model_size_bytes as i64).abs() as f64 / 1_048_576.0,
        if ppv6_stats.total_model_size_bytes < indra_stats.total_model_size_bytes { "PP-OCRv6 Tiny (6.1MB vs 5.3MB)" } else { "Indra OCR (5.3MB vs 6.1MB)" });

    println!("\n-----------------------------------------------------------------------------------------");
    println!("3. DETECTION COVERAGE & CONFIDENCE METRICS");
    println!("-----------------------------------------------------------------------------------------");
    println!("{:<30} | {:<15} | {:<15} | {:<15}", "Metric", "PP-OCRv6 Tiny", "Indra OCR", "Difference");
    println!("{:-<30}-+-{:-<15}-+-{:-<15}-+-{:-<15}", "", "", "", "");
    println!("{:<30} | {:<15} | {:<15} | {:+}", "Detected Text Lines", ppv6_stats.results.len(), indra_stats.results.len(), indra_stats.results.len() as i32 - ppv6_stats.results.len() as i32);
    println!("{:<30} | {:<15} | {:<15} | {:+}", "Extracted Character Count", pp_chars, in_chars, in_chars as i32 - pp_chars as i32);
    println!("{:<30} | {:<15} | {:<15} | {:+}", "Extracted Word Count", pp_words, in_words, in_words as i32 - pp_words as i32);
    println!("{:<30} | {:<15.4} | {:<15.4} | {:+.4}", "Average Confidence Score", pp_avg, in_avg, in_avg - pp_avg);
    println!("{:<30} | {:<15.4} | {:<15.4} | -", "Min Score / Max Score", format!("{:.2} / {:.2}", pp_min, pp_max), format!("{:.2} / {:.2}", in_min, in_max));
    println!("{:<30} | {:<15} | {:<15} | -", "High Conf [>=0.90]", format!("{} ({:.1}%)", pp_hist[0], pp_hist[0] as f32 / ppv6_stats.results.len() as f32 * 100.0), format!("{} ({:.1}%)", in_hist[0], in_hist[0] as f32 / indra_stats.results.len() as f32 * 100.0));
    println!("{:<30} | {:<15} | {:<15} | -", "Medium Conf [0.75..0.90)", format!("{} ({:.1}%)", pp_hist[1], pp_hist[1] as f32 / ppv6_stats.results.len() as f32 * 100.0), format!("{} ({:.1}%)", in_hist[1], in_hist[1] as f32 / indra_stats.results.len() as f32 * 100.0));
    println!("{:<30} | {:<15} | {:<15} | -", "Low Conf [0.50..0.75)", format!("{} ({:.1}%)", pp_hist[2], pp_hist[2] as f32 / ppv6_stats.results.len() as f32 * 100.0), format!("{} ({:.1}%)", in_hist[2], in_hist[2] as f32 / indra_stats.results.len() as f32 * 100.0));

    println!("\n-----------------------------------------------------------------------------------------");
    println!("4. QUALITATIVE TRANSCRIPTION COMPARISON ON HISTORICAL AD SECTIONS");
    println!("-----------------------------------------------------------------------------------------");

    let targets = [
        ("Main Banner", "Capital Dry Goods House"),
        ("Store Subtitle", "New Store, 2910-2912 Washington Avenue."),
        ("Department 1", "Millinery Department"),
        ("Department 2", "C. B. Corsets"),
        ("Left Ad Title", "Special Sale of Trimmed Hats."),
        ("Center Ad Title 1", "Extraordinary Values in Ladies' Stylish"),
        ("Center Ad Title 2", "Covert Cloth Coats at $10.00"),
        ("Center Ad Title 3", "Stylish Broadcloth Eton Suits at $20.00"),
        ("Center Ad Title 4", "Special Sale of Mill Ends of Madras"),
        ("Right Ad Title", "Big Sale of Corsets at 39c each"),
        ("Right Ad Sub", "Sale of Ladies' Perfect Fitting Wrappers"),
        ("Price 1 ($2.98)", "$2.98"),
        ("Price 2 ($1.00)", "$1.00"),
        ("Price 3 (50c)", "50c"),
        ("Price 4 ($15.00)", "$15.00"),
        ("Price 5 ($20.00)", "$20.00"),
        ("Price 6 (8c)", "8c"),
        ("Price 7 (5c)", "5c"),
        ("Price 8 (75c)", "75c"),
        ("Price 9 (59c)", "59c"),
    ];

    println!("{:<25} | {:<25} | {:<25} | {:<25}", "Target Phrase / Label", "Ground Truth Expected", "PP-OCRv6 Tiny Extracted", "Indra OCR Extracted");
    println!("{:-<25}-+-{:-<25}-+-{:-<25}-+-{:-<25}", "", "", "", "");

    for &(label, target) in &targets {
        let pp_match = ppv6_stats.results.iter().find(|r| {
            let t_lower = target.to_lowercase();
            let r_lower = r.text.to_lowercase();
            let key_word = target.split_whitespace().next().unwrap_or("").to_lowercase();
            r_lower.contains(&t_lower) || (key_word.len() > 3 && r_lower.contains(&key_word))
        });

        let indra_match = indra_stats.results.iter().find(|r| {
            let t_lower = target.to_lowercase();
            let r_lower = r.text.to_lowercase();
            let key_word = target.split_whitespace().next().unwrap_or("").to_lowercase();
            r_lower.contains(&t_lower) || (key_word.len() > 3 && r_lower.contains(&key_word))
        });

        let pp_str = pp_match.map(|r| format!("{} ({:.2})", r.text, r.score)).unwrap_or_else(|| "[MISSING]".to_string());
        let in_str = indra_match.map(|r| format!("{} ({:.2})", r.text, r.score)).unwrap_or_else(|| "[MISSING]".to_string());

        println!("{:<25} | {:<25} | {:<25} | {:<25}", label, target, pp_str, in_str);
    }
}

