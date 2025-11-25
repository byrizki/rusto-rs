use clap::{Parser, ValueEnum};
use rusto::rusto_ocr::RustO as RustOCR;
use rusto::{OrientClassifier, OrientConfig, DocUnwarper, UnwarpConfig};
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
    let mut ocr = RustOCR::new_ppv5(
        &cli.det_model,
        &cli.rec_model,
        &cli.dict,
    )?;

    // Add optional features
    if cli.enable_orient {
        if let Some(orient_path) = cli.orient_model {
            eprintln!("Loading orientation classifier...");
            let orient_config = OrientConfig::default(orient_path);
            let orient = OrientClassifier::new(orient_config)?;
            ocr.global.use_orient = true;
            ocr = ocr.with_orient(orient);
            eprintln!("✓ Orientation classifier enabled");
        } else {
            eprintln!("Warning: --enable-orient specified but --orient-model not provided");
        }
    }

    if cli.enable_unwarp {
        if let Some(unwarp_path) = cli.unwarp_model {
            eprintln!("Loading text unwarper...");
            let unwarp_config = UnwarpConfig::default(unwarp_path);
            let unwarper = DocUnwarper::new(unwarp_config)?;
            ocr.global.use_unwarp = true;
            ocr = ocr.with_unwarp(unwarper);
            eprintln!("✓ Text unwarper enabled");
        } else {
            eprintln!("Warning: --enable-unwarp specified but --unwarp-model not provided");
        }
    }

    // Layout detection removed

    // Run OCR
    eprintln!("Processing image...");
    let result = ocr.run(&cli.image)?;
    eprintln!("✓ OCR completed");

    // Output results based on format
    match cli.format {
        OutputFormat::Json => {
            let json_output = serde_json::json!({
                "boxes": result.boxes.iter().map(|b| vec![
                    vec![b[0].x, b[0].y],
                    vec![b[1].x, b[1].y],
                    vec![b[2].x, b[2].y],
                    vec![b[3].x, b[3].y],
                ]).collect::<Vec<_>>(),
                "txts": result.txts,
                "scores": result.scores,
                "orientation": result.orientation.as_ref().map(|o| o.degrees()),
                // "layout_regions" removed
                "elapse": {
                    "detection": result.elapse_det,
                    "recognition": result.elapse_rec,
                    "orientation": result.elapse_orient,
                    "unwarping": result.elapse_unwarp,
                    // "layout" removed
                },
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        OutputFormat::Text => {
            for text in &result.txts {
                println!("{}", text);
            }
        }
        OutputFormat::Tsv => {
            for (i, text) in result.txts.iter().enumerate() {
                let bbox = &result.boxes[i];
                let score = result.scores[i];
                let box_str = format!(
                    "{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
                    bbox[0].x, bbox[0].y,
                    bbox[1].x, bbox[1].y,
                    bbox[2].x, bbox[2].y,
                    bbox[3].x, bbox[3].y,
                );
                println!("{}\t{:.3}\t{}", text, score, box_str);
            }
        }
        OutputFormat::Markdown => {
            // Markdown format removed - use plain text with code block
            println!("```");
            println!("{}", result.to_raw());
            println!("```");
        }
        OutputFormat::TextOrdered => {
            println!("{}", result.to_raw());
        }
    }

    Ok(())
}
