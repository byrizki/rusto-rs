use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::time::Instant;

fn main() {
    println!("=== RustO Performance Test ===\n");
    
    let config = InitializeConfig::ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/dict.txt"
    );
    
    println!("Initializing OCR engine...");
    let init_start = Instant::now();
    let mut ocr = RustO::initialize(config).expect("Failed to create OCR");
    println!("Initialization took: {:?}\n", init_start.elapsed());
    
    let test_images = vec![
        ("example1", "models/test_images/example1.png"),
    ];
    
    for (name, path) in &test_images {
        if !std::path::Path::new(path).exists() {
            println!("⚠ Skipping {} - file not found: {}\n", name, path);
            continue;
        }
        
        println!("Testing {} ({}):", name, path);
        
        // Warmup run
        println!("  Warmup run...");
        let _ = ocr.detect_text(&ImageSource::Path(path.into()), &OcrRunOptions::default());
        
        // Timed runs
        let num_runs = 5;
        let mut times = Vec::new();
        
        for i in 1..=num_runs {
            let start = Instant::now();
            let results = ocr.detect_text(&ImageSource::Path(path.into()), &OcrRunOptions::default()).expect("OCR failed");
            let elapsed = start.elapsed();
            times.push(elapsed);
            
            match results {
                DetectTextResult::Structured(items) => {
                    println!("  Run {}: {:?} - {} text regions detected", i, elapsed, items.len());
                    if i == 1 && !items.is_empty() {
                        println!("    Sample result: {} (score: {:.3})", items[0].text, items[0].score);
                    }
                }
                DetectTextResult::Spatial(_) => unreachable!("default output is structured"),
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
