/// Raw OCR Example
/// Demonstrates extracting OCR results in RAW ASCII and CSV formats
/// Outputs to data/output/ folder
use rusto::rusto_ocr::RustO as RapidOcr;
use std::error::Error;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Raw OCR Export Example ===\n");

    // Create output directory
    let output_dir = Path::new("data/output");
    fs::create_dir_all(output_dir)?;
    println!("Output directory: {}\n", output_dir.display());

    // Initialize OCR engine
    let mut ocr = RapidOcr::new_ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/ppocr_keys_v1.txt",
    )?;

    // Run OCR
    let test_image = "test.jpg";
    println!("Processing: {}\n", test_image);
    let result = ocr.run(test_image)?;

    // Export as RAW ASCII format
    println!("========== RAW ASCII Output ==========\n");
    let raw = result.to_raw();
    println!("{}", raw);

    // Export as CSV format
    println!("\n========== CSV Output ==========\n");
    let csv = result.to_csv();
    println!("{}", csv);

    // Save to files
    let raw_file = output_dir.join("raw.txt");
    let csv_file = output_dir.join("output.csv");
    std::fs::write(&raw_file, &raw)?;
    std::fs::write(&csv_file, &csv)?;

    println!("\n✓ Outputs saved:");
    println!("  - {} (RAW ASCII format)", raw_file.display());
    println!("  - {} (CSV format)", csv_file.display());

    Ok(())
}
