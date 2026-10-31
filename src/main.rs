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

    /// Enable text line orientation classifier
    #[arg(long)]
    enable_cls: bool,

    /// Path to text line orientation classifier model
    #[arg(long)]
    cls_model: Option<PathBuf>,

    /// Enable document orientation classification
    #[arg(long)]
    enable_orient: bool,

    /// Path to orientation classification model (lcnet.rten)
    #[arg(long)]
    orient_model: Option<PathBuf>,

    /// Enable text unwarping
    #[arg(long)]
    enable_unwarp: bool,

    /// Path to text unwarping model (uvdoc.rten)
    #[arg(long)]
    unwarp_model: Option<PathBuf>,

    /// Enable document layout detection
    #[arg(long)]
    enable_layout: bool,

    /// Path to document layout detection model
    #[arg(long)]
    layout_model: Option<PathBuf>,
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

    // Initialize base OCR configuration
    let mut config = InitializeConfig::ppv5(&cli.det_model, &cli.rec_model, &cli.dict);

    if let Some(cls_model) = cli.cls_model {
        config = config.with_cls(cls_model);
    } else if cli.enable_cls {
        config.global.use_cls = true;
    }

    if let Some(orient_model) = cli.orient_model {
        config = config.with_orientation(orient_model);
    } else if cli.enable_orient {
        config.global.use_orient = true;
    }

    if let Some(unwarp_model) = cli.unwarp_model {
        config = config.with_unwarp(unwarp_model);
    } else if cli.enable_unwarp {
        config.global.use_unwarp = true;
    }

    if let Some(layout_model) = cli.layout_model {
        config = config.with_layout(layout_model);
    } else if cli.enable_layout {
        config.global.use_layout = true;
    }

    let mut ocr = RustO::initialize(config)?;

    let mut options = OcrRunOptions::default();
    if cli.enable_cls {
        options.classification = Some(true);
    }
    if cli.enable_orient {
        options.orientation = Some(true);
    }
    if cli.enable_unwarp {
        options.unwarp = Some(true);
    }
    if cli.enable_layout {
        options.layout = Some(true);
    }

    // Run OCR
    eprintln!("Processing image...");
    let detected = ocr.detect_text(
        &ImageSource::Path(cli.image.clone()),
        &options,
    )?;
    let DetectTextResult::Structured(results) = detected else {
        unreachable!("default OCR output is structured lines");
    };
    eprintln!("✓ OCR completed");

    let layout_regions = if cli.enable_layout && ocr.layout.is_some() {
        match ocr.detect_layout(&ImageSource::Path(cli.image.clone())) {
            Ok(regions) => Some(regions),
            Err(e) => {
                eprintln!("Warning: Layout detection failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Output projected text results.
    match cli.format {
        OutputFormat::Json => {
            if let Some(regions) = &layout_regions {
                #[derive(serde::Serialize)]
                struct OutputWithLayout<'a> {
                    text_results: &'a [rusto::TextResult],
                    layout_regions: &'a [rusto::LayoutRegion],
                }
                println!("{}", serde_json::to_string_pretty(&OutputWithLayout {
                    text_results: &results,
                    layout_regions: regions,
                })?);
            } else {
                println!("{}", serde_json::to_string_pretty(&results)?);
            }
        }
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
