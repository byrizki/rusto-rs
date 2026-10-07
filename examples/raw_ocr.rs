/// Structured text detection example.
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut ocr = RustO::initialize(InitializeConfig::ppv5(
        "models/PPOCR_v5/det.onnx",
        "models/PPOCR_v5/rec.onnx",
        "models/PPOCR_v5/dict.txt",
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
