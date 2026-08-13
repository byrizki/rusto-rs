use clap::{Parser, ValueEnum};
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rusto")]
#[command(about = "RustO! - Pure Rust OCR powered by PaddleOCR engine", long_about = None)]
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

    /// Enable document orientation classification
    #[arg(long)]
    enable_orient: bool,

    /// Path to orientation classification model (lcnet.mnn)
    #[arg(long)]
    orient_model: Option<PathBuf>,

    /// Enable text unwarping
    #[arg(long)]
    enable_unwarp: bool,

    /// Path to text unwarping model (uvdoc.mnn)
    #[arg(long)]
    unwarp_model: Option<PathBuf>,
    // Layout detection removed
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum OutputFormat {
    /// JSON output with full details
    Json,
    /// Plain text, one line per detected text
    Text,
    /// TSV format: text\tscore\tx1,y1,x2,y2,x3,y3,x4,y4
    Tsv,
    /// Markdown format with metadata and ordering
    Markdown,
    /// Plain text ordered by position
    TextOrdered,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize base OCR using the advanced API
    let mut ocr = RustO::initialize(InitializeConfig::ppv5(&cli.det_model, &cli.rec_model, &cli.dict))?;

    if cli.enable_orient || cli.orient_model.is_some() || cli.enable_unwarp || cli.unwarp_model.is_some() {
        eprintln!("Warning: CLI optional stages are not configured by this canonical API command.");
    }

    // Layout detection removed

    // Run OCR
    eprintln!("Processing image...");
    let detected = ocr.detect_text(
        &ImageSource::Path(cli.image.clone()),
        &OcrRunOptions::default(),
    )?;
    let DetectTextResult::Structured(results) = detected else {
        unreachable!("default OCR output is structured lines");
    };
    eprintln!("✓ OCR completed");

    // Output projected text results.
    match cli.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&results)?),
        OutputFormat::Text | OutputFormat::TextOrdered => {
            for result in &results {
                println!("{}", result.text);
            }
        }
        OutputFormat::Tsv => {
            for result in &results {
                let box_points = result.box_points;
                println!(
                    "{}\t{:.3}\t{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    result.text,
                    result.score,
                    box_points[0].0,
                    box_points[0].1,
                    box_points[1].0,
                    box_points[1].1,
                    box_points[2].0,
                    box_points[2].1,
                    box_points[3].0,
                    box_points[3].1,
                );
            }
        }
        OutputFormat::Markdown => {
            println!("```");
            for result in &results {
                println!("{}", result.text);
            }
            println!("```");
        }
    }

    Ok(())
}
