/// Spatial text detection example.
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, OutputGranularity, RustO};
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let test_image = "models/images/ktp.jpg";
    if !Path::new(test_image).exists() {
        println!("Warning: {test_image} not found.");
        return Ok(());
    }
    let mut ocr = RustO::initialize(InitializeConfig::ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/dict.txt",
    ))?;
    let result = ocr.detect_text(
        &ImageSource::Path(test_image.into()),
        &OcrRunOptions {
            output: OutputGranularity::Spatial,
            line_y_threshold: Some(0.5),
            word_x_threshold: Some(0.4),
            ..Default::default()
        },
    )?;
    if let DetectTextResult::Spatial(text) = result {
        println!("{text}");
    }
    Ok(())
}
