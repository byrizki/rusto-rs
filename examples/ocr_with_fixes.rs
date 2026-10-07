/// OCR example with prepared orientation and classification models.
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = InitializeConfig::ppv5(
        "models/PPOCR_v5/det.onnx",
        "models/PPOCR_v5/rec.onnx",
        "models/PPOCR_v5/dict.txt",
    )
    .with_cls("models/PPOCR_v5/cls.onnx")
    .with_orientation("models/PPOCR_v5/cls.onnx");
    let mut ocr = RustO::initialize(config)?;
    let result = ocr.detect_text(
        &ImageSource::Path("models/test_images/example1.png".into()),
        &OcrRunOptions {
            classification: Some(true),
            orientation: Some(true),
            ..Default::default()
        },
    )?;
    if let DetectTextResult::Structured(results) = result {
        for result in results {
            println!("{} ({:.3})", result.text, result.score);
        }
    }
    Ok(())
}
