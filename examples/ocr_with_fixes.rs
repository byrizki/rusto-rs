use rusto::{RustO, RustOConfig};
/// OCR with Image Fixes Example
/// Demonstrates OCR with orientation classification and text unwarping
/// Uses test images from models/test_images/ directory
/// Outputs to data/output/ folder with debug images
use std::error::Error;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== OCR with Image Fixes ===\n");

    // Create output directory
    let output_dir = Path::new("data/output");
    fs::create_dir_all(output_dir)?;
    println!("Output directory: {}\n", output_dir.display());

    // Initialize OCR engine with orientation and unwarping
    let config = RustOConfig::new_ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/dict.txt",
    )
    .with_cls("models/PPOCR_v5/lcnet-text.mnn") // Text line orientation
    .with_orientation("models/PPOCR_v5/lcnet.mnn") // Page orientation (optional)
    .with_unwarp("models/PPOCR_v5/uvdoc.mnn") // Document unwarping (optional)
    .with_debug_images(true); // Enable debug images

    let mut ocr = RustO::new(config)?;

    // Test images directory
    let test_dir = "models/test_images";

    // Get all test images
    let mut test_images: Vec<String> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ["jpg", "png"].contains(&ext.to_lowercase().as_str()) {
                    path.to_str().map(|s| s.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    test_images.sort();

    println!("Found {} test images\n", test_images.len());

    // Process each test image
    for image_path in &test_images {
        let filename = std::path::Path::new(image_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        println!("====================");
        println!("Processing: {}", filename);
        println!("====================");

        match ocr.run(image_path) {
            Ok(result) => {
                println!("Detected {} text regions", result.boxes.len());

                if let Some(orientation) = &result.orientation {
                    println!("Orientation: {} degrees", orientation.degrees());
                }

                println!("\nRAW Output:");
                let raw = result.to_raw();
                println!("{}", raw.lines().take(5).collect::<Vec<_>>().join("\n"));
                if raw.lines().count() > 5 {
                    println!("... ({} more lines)", raw.lines().count() - 5);
                }

                // Save output
                let output_name = filename.replace(".jpg", "").replace(".png", "");
                let raw_file = output_dir.join(format!("{}_raw.txt", output_name));
                let csv_file = output_dir.join(format!("{}.csv", output_name));
                std::fs::write(&raw_file, &raw)?;
                std::fs::write(&csv_file, result.to_csv())?;

                // Save debug images (both OpenCV and Pure Rust)
                #[cfg(feature = "use-opencv")]
                {
                    use opencv::imgcodecs::imwrite;

                    if let Some(oriented) = &result.debug_oriented_image {
                        let debug_file = output_dir.join(format!("{}_oriented.jpg", output_name));
                        imwrite(
                            &debug_file.to_string_lossy(),
                            oriented,
                            &opencv::core::Vector::new(),
                        )?;
                        println!(
                            "✓ Saved orientation-corrected image: {}",
                            debug_file.display()
                        );
                    }

                    if let Some(unwarped) = &result.debug_unwarped_image {
                        let debug_file = output_dir.join(format!("{}_unwarped.jpg", output_name));
                        imwrite(
                            &debug_file.to_string_lossy(),
                            unwarped,
                            &opencv::core::Vector::new(),
                        )?;
                        println!("✓ Saved unwarped image: {}", debug_file.display());
                    }
                }

                #[cfg(not(feature = "use-opencv"))]
                {
                    if let Some(oriented) = &result.debug_oriented_image {
                        let debug_file = output_dir.join(format!("{}_oriented.png", output_name));
                        oriented.as_dynamic().save(&debug_file)?;
                        println!(
                            "✓ Saved orientation-corrected image: {}",
                            debug_file.display()
                        );
                    }
                }

                println!(
                    "✓ Saved to {} and {}",
                    raw_file.display(),
                    csv_file.display()
                );
            }
            Err(e) => {
                eprintln!("✗ Error: {}", e);
            }
        }
        println!();
    }

    println!("\n=== Processing Complete ===");

    Ok(())
}
