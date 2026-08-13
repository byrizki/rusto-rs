/// Structured text detection example.
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut ocr = RustO::initialize(InitializeConfig::ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/ppocr_keys_v1.txt",
    ))?;
    let result = ocr.detect_text(
        &ImageSource::Path("test.jpg".into()),
        &OcrRunOptions::default(),
    )?;
    if let DetectTextResult::Structured(results) = result {
        for result in results {
            println!("{:.3}\t{}", result.score, result.text);
        }
    }
    Ok(())
}
