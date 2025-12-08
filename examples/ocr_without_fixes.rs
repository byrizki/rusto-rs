/// OCR Without Image Fixes Example
/// Demonstrates basic OCR without orientation or unwarping
/// Uses example image from models/test_images/
/// Outputs to data/output/ folder
use rusto::rusto_ocr::RustO as RapidOcr;
use std::error::Error;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== OCR Without Image Fixes ===\n");

    // Create output directory
    let output_dir = Path::new("data/output");
    fs::create_dir_all(output_dir)?;
    println!("Output directory: {}\n", output_dir.display());

    // Initialize basic OCR engine (no orientation, no unwarping)
    let mut ocr = RapidOcr::new_ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/dict.txt",
    )?;

    // Test image from models/test_images
    // let test_image = "models/test_images/example1.png";
    let test_image = "models/images/ktp-teng.jpg";
    println!("Processing: {}\n", test_image);

    // Run OCR
    let result = ocr.run(test_image)?;

    println!("Detected {} text regions", result.boxes.len());
    println!("\nPerformance:");
    println!("  Detection: {:.3}s", result.elapse_det);
    println!("  Recognition: {:.3}s", result.elapse_rec);
    println!("  Total: {:.3}s\n", result.elapse_det + result.elapse_rec);

    // RAW ASCII output
    println!("========== RAW ASCII Output ==========\n");
    let raw = result.to_raw();
    println!("{}", raw);

    // CSV output
    println!("\n========== CSV Output ==========\n");
    let csv = result.to_csv();
    println!("{}", csv);

    // Save to files
    let raw_file = output_dir.join("example1_raw.txt");
    let csv_file = output_dir.join("example1.csv");
    std::fs::write(&raw_file, &raw)?;
    std::fs::write(&csv_file, &csv)?;

    println!("\n✓ Outputs saved:");
    println!("  - {}", raw_file.display());
    println!("  - {}", csv_file.display());

    Ok(())
}
