use clap::{Parser, ValueEnum};
use rusto::{RapidOCR, RapidOCRConfig};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rusto")]
#[command(about = "RustO! - Pure Rust OCR based on RapidOCR with PaddleOCR engine", long_about = None)]
struct Cli {
    /// Path to detection model (ONNX)
    #[arg(long)]
    det_model: PathBuf,

    /// Path to recognition model (ONNX)
    #[arg(long)]
    rec_model: PathBuf,

    /// Path to dictionary file
    #[arg(long)]
    dict: PathBuf,

    /// Input image path
    image: PathBuf,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum OutputFormat {
    /// JSON output with full details
    Json,
    /// Plain text, one line per detected text
    Text,
    /// TSV format: text\tscore\tx1,y1,x2,y2,x3,y3,x4,y4
    Tsv,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize OCR
    let config = RapidOCRConfig {
        det_model_path: cli.det_model.to_str().unwrap().to_string(),
        rec_model_path: cli.rec_model.to_str().unwrap().to_string(),
        dict_path: cli.dict.to_str().unwrap().to_string(),
    };

    let ocr = RapidOCR::new(config)?;

    // Load image
    let image_path = cli.image.to_str().unwrap();
    
    // Run OCR
    let results = ocr.ocr(image_path)?;

    // Output results
    match cli.format {
        OutputFormat::Json => {
            let json_output = serde_json::json!({
                "boxes": results.iter().map(|r| vec![
                    serde_json::json!({"x": r.box_points[0].0, "y": r.box_points[0].1}),
                    serde_json::json!({"x": r.box_points[1].0, "y": r.box_points[1].1}),
                    serde_json::json!({"x": r.box_points[2].0, "y": r.box_points[2].1}),
                    serde_json::json!({"x": r.box_points[3].0, "y": r.box_points[3].1}),
                ]).collect::<Vec<_>>(),
                "txts": results.iter().map(|r| &r.text).collect::<Vec<_>>(),
                "scores": results.iter().map(|r| r.score).collect::<Vec<_>>(),
                "word_results": results.iter().map(|_| Vec::<String>::new()).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        OutputFormat::Text => {
            for result in &results {
                println!("{}", result.text);
            }
        }
        OutputFormat::Tsv => {
            for result in &results {
                let box_str = format!(
                    "{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    result.box_points[0].0, result.box_points[0].1,
                    result.box_points[1].0, result.box_points[1].1,
                    result.box_points[2].0, result.box_points[2].1,
                    result.box_points[3].0, result.box_points[3].1,
                );
                println!("{}\t{:.3}\t{}", result.text, result.score, box_str);
            }
        }
    }

    Ok(())
}
