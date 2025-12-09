/// Spatial Text Export Example
/// Demonstrates extracting OCR results while preserving spatial layout
/// Outputs to data/output/ folder
use rusto::rusto_ocr::RustO as RapidOcr;
use std::error::Error;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Spatial Text Export Example ===\n");

    // Create output directory
    let output_dir = Path::new("data/output");
    fs::create_dir_all(output_dir)?;
    println!("Output directory: {}\n", output_dir.display());

    // Initialize OCR engine with PPOCRv5 models
    // Note: This assumes you are running from the project root
    let mut ocr = RapidOcr::new_ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/dict.txt",
    )?;

    // Run OCR on test image
    let test_image = "models/images/ktp.jpg";
    if !Path::new(test_image).exists() {
        println!("Warning: {} not found in project root. Please ensure it exists.", test_image);
        return Ok(());
    }
    
    println!("Processing: {}\n", test_image);
    let result = ocr.run(test_image)?;

    // 0. Debug Boxes
    println!("========== Debug Boxes ==========\n");
    let debug_boxes = result.to_text_with_position();
    println!("{}", debug_boxes);

    // 1. Default Spatial Export
    // Uses default multipliers (Y: 0.6, X: 1.3)
    // Good for most general documents
    println!("========== Default Spatial Layout ==========\n");
    let spatial_default = result.to_spatial_text(None, None);
    println!("{}", spatial_default);

    // 2. Custom Spatial Export (Strict)
    // Demonstrates strict line separation (separate rows for even small vertical offsets)
    // Y multiplier 0.2 -> Enhanced line separation
    println!("\n========== Custom Spatial Layout (Strict) ==========\n");
    let spatial_custom = result.to_spatial_text(Some(0.2), None);
    println!("{}", spatial_custom);

    // Save outputs to files
    let default_file = output_dir.join("spatial_default.txt");
    let custom_file = output_dir.join("spatial_strict.txt");

    std::fs::write(&default_file, &spatial_default)?;
    std::fs::write(&custom_file, &spatial_custom)?;

    println!("\n✓ Outputs saved:");
    println!("  - {} (Default parameters)", default_file.display());
    println!("  - {} (Custom parameters: y=0.2, strict separation)", custom_file.display());

    // 3. Custom Spatial Export (Tuned X/Y)
    // Demonstrates X threshold for horizontal compaction
    // X multiplier 2.0 -> Gaps smaller than 2 chars use 1 space (compact words)
    // Gap larger than 2 chars use spatial spacing (columns)
    println!("\n========== Custom Spatial Layout (Tuned X/Y) ==========\n");
    let spatial_tuned = result.to_spatial_text(Some(0.5), Some(2.0));
    println!("{}", spatial_tuned);

    let tuned_file = output_dir.join("spatial_tuned.txt");
    std::fs::write(&tuned_file, &spatial_tuned)?;
    println!("  - {} (Custom parameters: y=0.5, x=2.0)", tuned_file.display());

    Ok(())
}
