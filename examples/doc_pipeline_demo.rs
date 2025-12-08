use clap::Parser;
use rusto::{
    DocPipeline, DocPipelineConfig, LayoutConfig, RustOConfig, TableDetectorConfig,
    TableModelType, TableStructureConfig,
};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image: PathBuf,

    #[arg(long, default_value = "models/DocOCR/layout.mnn")]
    layout_model: PathBuf,

    #[arg(long, default_value = "models/PPOCR_v5/det.mnn")]
    det_model: PathBuf,

    #[arg(long, default_value = "models/PPOCR_v5/rec.mnn")]
    rec_model: PathBuf,

    #[arg(long, default_value = "models/PPOCR_v5/dict.txt")]
    keys_path: PathBuf,

    #[arg(long, short = 'o')]
    output: Option<PathBuf>,

    /// Enable table recognition (requires rtdetr and slanext models in the same dir as layout model)
    #[arg(long, default_value_t = true)]
    table: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let layout_config = LayoutConfig::default(args.layout_model.clone());
    let ocr_config = RustOConfig::new_ppv5(args.det_model, args.rec_model, args.keys_path);

    // Configure table recognition if enabled
    let (table_detector, table_recognizer) = if args.table {
        let model_dir = args
            .layout_model
            .parent()
            .expect("Layout model path has no parent");

        (
            Some(TableDetectorConfig {
                model_path: model_dir.join("rtdetr-wired.mnn"),
                conf_threshold: 0.5,
                iou_threshold: 0.5,
                model_type: TableModelType::Wired,
            }),
            Some(TableStructureConfig {
                model_path: model_dir.join("slanext-wired.mnn"),
                model_type: TableModelType::Wired,
            }),
        )
    } else {
        (None, None)
    };

    let config = DocPipelineConfig {
        layout: layout_config,
        ocr: ocr_config,
        table_detector,
        table_recognizer,
    };

    let mut pipeline = DocPipeline::new(config)?;
    let result = pipeline.run(&args.image)?;

    let markdown = result.to_markdown();

    // Print to stdout
    println!("{}", markdown);

    // Write to file if output path is specified
    if let Some(output_path) = args.output {
        fs::write(&output_path, markdown)?;
        eprintln!("\nMarkdown saved to: {}", output_path.display());
    } else {
        // Auto-generate output filename based on input
        let input_stem = args
            .image
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let output_path = PathBuf::from(format!("{}.md", input_stem));
        fs::write(&output_path, markdown)?;
        eprintln!("\nMarkdown saved to: {}", output_path.display());
    }

    Ok(())
}
