/// Basic OCR example.
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut ocr = RustO::initialize(InitializeConfig::ppv5(
        "models/PPOCR_v5/det.mnn",
        "models/PPOCR_v5/rec.mnn",
        "models/PPOCR_v5/dict.txt",
    ))?;
    let result = ocr.detect_text(
        &ImageSource::Path("models/images/ktp-teng.jpg".into()),
        &OcrRunOptions::default(),
    )?;
    if let DetectTextResult::Structured(results) = result {
        for result in results {
            println!("{} ({:.3})", result.text, result.score);
        }
    }
    Ok(())
}
