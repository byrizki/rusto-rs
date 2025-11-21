use rusto::{RustO, RustOConfig};
use std::time::Instant;

fn main() {
    println!("=== RustO Performance Test ===\n");
    
    let config = RustOConfig {
        det_model_path: "models/PPOCR_v5/det.onnx".to_string(),
        rec_model_path: "models/PPOCR_v5/rec.onnx".to_string(),
        dict_path: "models/PPOCR_v5/dict.txt".to_string(),
    };
    
    println!("Initializing OCR engine...");
    let init_start = Instant::now();
    let mut ocr = RustO::new(config).expect("Failed to create OCR");
    println!("Initialization took: {:?}\n", init_start.elapsed());
    
    let test_images = vec![
        ("KTP", "models/images/ktp-teng.jpg"),
        ("Example1", "models/test_images/example1.png"),
    ];
    
    for (name, path) in &test_images {
        if !std::path::Path::new(path).exists() {
            println!("⚠ Skipping {} - file not found: {}\n", name, path);
            continue;
        }
        
        println!("Testing {} ({}):", name, path);
        
        // Warmup run
        println!("  Warmup run...");
        let _ = ocr.ocr(path);
        
        // Timed runs
        let num_runs = 5;
        let mut times = Vec::new();
        
        for i in 1..=num_runs {
            let start = Instant::now();
            let results = ocr.ocr(path).expect("OCR failed");
            let elapsed = start.elapsed();
            times.push(elapsed);
            
            println!("  Run {}: {:?} - {} text regions detected", i, elapsed, results.len());
            if i == 1 && !results.is_empty() {
                println!("    Sample result: {} (score: {:.3})", 
                         results[0].text, results[0].score);
            }
        }
        
        // Calculate statistics
        let total: std::time::Duration = times.iter().sum();
        let avg = total / num_runs as u32;
        let min = times.iter().min().unwrap();
        let max = times.iter().max().unwrap();
        
        println!("\n  Statistics:");
        println!("    Average: {:?}", avg);
        println!("    Min: {:?}", min);
        println!("    Max: {:?}", max);
        println!("    Throughput: {:.2} images/sec\n", 1.0 / avg.as_secs_f64());
    }
}
