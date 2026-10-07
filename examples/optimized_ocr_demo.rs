use rusto::{
    CalibrationOptions, DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions,
    OptimizationOptions, RustO,
};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img_path = Path::new("ktp.jpeg");
    let models_dir = Path::new("models/PPOCR_v6_tiny");

    if !img_path.exists() || !models_dir.exists() {
        eprintln!("ktp.jpeg or models/PPOCR_v6_tiny not found.");
        return Ok(());
    }

    println!("================================================================================");
    println!("      DEMO: PP-OCRv6 TINY WITH BUILT-IN CALIBRATION & OPTIMIZATION");
    println!("================================================================================\n");

    // 1. Initialize with built-in calibration & optimization presets
    let config = InitializeConfig::ppv6(
        models_dir.join("det.rten"),
        models_dir.join("rec.rten"),
        models_dir.join("dict.txt"),
    )
    .with_calibration(CalibrationOptions::for_id_card())
    .with_optimization(OptimizationOptions::for_id_card());

    let mut ocr = RustO::initialize(config)?;

    // 2. Run OCR (calibration and optimization execute transparently in the pipeline)
    let t0 = Instant::now();
    let result = ocr.detect_text(
        &ImageSource::Path(img_path.to_path_buf()),
        &OcrRunOptions::default(),
    )?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Inference finished in {:.2} ms\n", latency_ms);
    println!("{:<4} | {:<40} | {:<6}", "No", "Recognized Text", "Score");
    println!("-----+------------------------------------------+-------");

    if let DetectTextResult::Structured(items) = result {
        for (i, item) in items.iter().enumerate() {
            println!("{:<4} | {:<40} | {:.3}", i + 1, item.text, item.score);
        }
    }
    println!("================================================================================\n");

    Ok(())
}
