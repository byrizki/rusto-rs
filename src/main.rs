use clap::{Args, Parser, Subcommand, ValueEnum};
use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rusto")]
#[command(about = "RustO! - Pure Rust OCR powered by PaddleOCR and Indra engines", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input image path
    #[arg(value_name = "IMAGE")]
    image: Option<PathBuf>,

    /// OCR model pipeline: "indraocr" (default), "ppocrv6-tiny", "ppocrv6-small", "ppocrv5", "ppocrv4", etc.
    #[arg(short, long, default_value = "indraocr")]
    model: String,

    /// Base model directory containing det, rec, and dict files (overridden by --det-model, --rec-model, --dict)
    #[arg(long, alias = "model-dir")]
    model_path: Option<PathBuf>,

    /// Target language: "latin" (default), "ja", "ko", "ar", "cyrillic", "devanagari", "chinese-cht", etc.
    #[arg(short, long)]
    lang: Option<String>,

    /// Download model(s) to cache without running inference
    #[arg(long)]
    download_only: bool,

    /// Path to detection model (.rten / .onnx). Overrides model auto-download.
    #[arg(long)]
    det_model: Option<PathBuf>,

    /// Path to recognition model (.rten / .onnx). Overrides model auto-download.
    #[arg(long)]
    rec_model: Option<PathBuf>,

    /// Path to dictionary / metadata file (.txt / .json). Overrides model auto-download.
    #[arg(long)]
    dict: Option<PathBuf>,

    /// Custom model cache directory
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Enable neural reading order recovery
    #[arg(long)]
    reading_order: bool,

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

    /// Path to orientation classification model
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

    /// Document layout variant: "cdla" (default 10-class) or "publaynet" (5-class)
    #[arg(long)]
    layout_variant: Option<String>,

    /// Path to document layout detection model
    #[arg(long)]
    layout_model: Option<PathBuf>,

    /// Maximum detection image dimension (default: configured in model, typically 960 or 736)
    #[arg(long)]
    det_limit: Option<i32>,

    /// Text recognition batch size (default: 6)
    #[arg(long)]
    rec_batch_num: Option<i32>,

    /// Maximum image side length for preprocessing
    #[arg(long)]
    max_side_len: Option<f32>,
}

#[derive(Subcommand)]
enum Commands {
    /// Download OCR models to the cache directory without running inference
    Download(DownloadArgs),
}

#[derive(Args)]
struct DownloadArgs {
    /// OCR model pipeline to download: "indraocr" (default), "ppocrv6-tiny", "ppocrv6-small", "ppocrv5", "ppocrv4", etc.
    #[arg(short, long, default_value = "indraocr")]
    model: String,

    /// Target language: "latin" (default), "ja", "ko", "ar", "cyrillic", "devanagari", "chinese-cht", etc.
    #[arg(short, long)]
    lang: Option<String>,

    /// Download all available pre-trained models
    #[arg(long)]
    all: bool,

    /// Download text-line orientation classifier model (ppocr_cls.rten)
    #[arg(long)]
    cls: bool,

    /// Download document orientation classifier model (ppocr_orient.rten / indra_orient.rten)
    #[arg(long)]
    orient: bool,

    /// Download document unwarping model (uvdoc.rten)
    #[arg(long)]
    unwarp: bool,

    /// Download document layout detection model (layout_cdla.rten or layout_publaynet.rten)
    #[arg(long)]
    layout: bool,

    /// Document layout variant: "cdla" (default 10-class) or "publaynet" (5-class)
    #[arg(long)]
    layout_variant: Option<String>,

    /// Download neural reading order model (indra_order.rten)
    #[arg(long)]
    reading_order: bool,

    /// Custom model cache directory
    #[arg(long)]
    cache_dir: Option<PathBuf>,
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

    if let Some(Commands::Download(args)) = cli.command {
        let cache_dir = rusto::model_manager::get_cache_dir(args.cache_dir.as_deref());
        if args.all {
            eprintln!("Downloading all available models to {}...", cache_dir.display());
            let downloaded = rusto::model_manager::download_all_models(&cache_dir)?;
            eprintln!("✓ Successfully verified/downloaded {} model asset(s).", downloaded.len());
        } else {
            eprintln!("Downloading {} pipeline to {}...", args.model, cache_dir.display());
            let downloaded = rusto::model_manager::download_model_pipeline(
                Some(&args.model),
                args.lang.as_deref(),
                args.cls,
                args.orient,
                args.unwarp,
                if args.layout { Some(args.layout_variant.as_deref().unwrap_or("cdla")) } else { None },
                args.reading_order,
                &cache_dir,
            )?;
            eprintln!("✓ Successfully verified/downloaded {} model asset(s).", downloaded.len());
        }
        return Ok(());
    }

    if cli.download_only {
        let cache_dir = rusto::model_manager::get_cache_dir(cli.cache_dir.as_deref());
        eprintln!("Downloading {} pipeline to {}...", cli.model, cache_dir.display());
        let downloaded = rusto::model_manager::download_model_pipeline(
            Some(&cli.model),
            cli.lang.as_deref(),
            cli.enable_cls || cli.cls_model.is_some(),
            cli.enable_orient || cli.orient_model.is_some(),
            cli.enable_unwarp || cli.unwarp_model.is_some(),
            if cli.enable_layout || cli.layout_model.is_some() || cli.layout_variant.is_some() {
                Some(cli.layout_variant.as_deref().unwrap_or("cdla"))
            } else {
                None
            },
            cli.reading_order,
            &cache_dir,
        )?;
        eprintln!("✓ Successfully verified/downloaded {} model asset(s).", downloaded.len());
        return Ok(());
    }

    let image_path = match cli.image {
        Some(path) => path,
        None => {
            eprintln!("Error: Please provide an image path to run OCR, or use 'rusto download' to download models.");
            std::process::exit(1);
        }
    };

    let effective_dir = cli.model_path.as_deref().or_else(|| {
        let p = std::path::Path::new(&cli.model);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    });

    let cache_dir = rusto::model_manager::get_cache_dir(cli.cache_dir.as_deref());
    let (det_path, rec_path, dict_path) = rusto::model_manager::resolve_base_pipeline(
        Some(&cli.model),
        effective_dir,
        cli.lang.as_deref(),
        cli.det_model.as_deref(),
        cli.rec_model.as_deref(),
        cli.dict.as_deref(),
        &cache_dir,
    )?;

    let model_lower = cli.model.to_ascii_lowercase();
    let preset = if model_lower.contains("v5") || model_lower.contains("ppocrv5") {
        rusto::PPV5_MODEL_CONFIG
    } else if model_lower.contains("v4") || model_lower.contains("ppocrv4") {
        rusto::PPV4_MODEL_CONFIG
    } else if model_lower.contains("v3") || model_lower.contains("ppocrv3") {
        rusto::PPV3_MODEL_CONFIG
    } else {
        rusto::PPV6_MODEL_CONFIG
    };

    let mut config = InitializeConfig::from_preset(preset, det_path, rec_path, dict_path);

    if cli.enable_cls || cli.cls_model.is_some() {
        let cls_path = rusto::model_manager::resolve_asset_with_dir(cli.cls_model.as_deref(), effective_dir, "cls", "ppocr_cls.rten", &cache_dir)?;
        config = config.with_cls(cls_path);
    }

    if cli.enable_orient || cli.orient_model.is_some() {
        let orient_file = if cli.model.eq_ignore_ascii_case("indraocr") || cli.model.eq_ignore_ascii_case("indra") {
            "indra_orient.rten"
        } else {
            "ppocr_orient.rten"
        };
        let orient_path = rusto::model_manager::resolve_asset_with_dir(cli.orient_model.as_deref(), effective_dir, "orient", orient_file, &cache_dir)?;
        config = config.with_orientation(orient_path);
    }

    if cli.enable_unwarp || cli.unwarp_model.is_some() {
        let unwarp_path = rusto::model_manager::resolve_asset_with_dir(cli.unwarp_model.as_deref(), effective_dir, "unwarp", "uvdoc.rten", &cache_dir)?;
        config = config.with_unwarp(unwarp_path);
    }

    if cli.enable_layout || cli.layout_model.is_some() || cli.layout_variant.is_some() {
        let layout_file = if cli.layout_variant.as_deref() == Some("publaynet") { "layout_publaynet.rten" } else { "layout_cdla.rten" };
        let layout_path = rusto::model_manager::resolve_asset_with_dir(cli.layout_model.as_deref(), effective_dir, "layout", layout_file, &cache_dir)?;
        config = config.with_layout(layout_path);
    }

    if cli.reading_order {
        let order_path = rusto::model_manager::resolve_asset_with_dir(None, effective_dir, "order", "indra_order.rten", &cache_dir)?;
        config.global.reading_order = true;
        config.global.reading_order_model = Some(order_path);
    }

    if let Some(batch_num) = cli.rec_batch_num {
        config = config.with_rec_batch_num(batch_num);
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
    if let Some(det_limit) = cli.det_limit {
        let det_opts = options.detection.get_or_insert_with(Default::default);
        det_opts.limit_side_len = Some(det_limit);
        det_opts.limit_type = Some("max".to_string());
    }
    if let Some(max_side) = cli.max_side_len {
        options.max_side_len = Some(max_side);
    }

    // Run OCR
    eprintln!("Processing image...");
    let detected = ocr.detect_text(
        &ImageSource::Path(image_path.clone()),
        &options,
    )?;
    let DetectTextResult::Structured(results) = detected else {
        unreachable!("default OCR output is structured lines");
    };
    eprintln!("✓ OCR completed");

    let layout_regions = if cli.enable_layout && ocr.layout.is_some() {
        match ocr.detect_layout(&ImageSource::Path(image_path.clone())) {
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
    let effective_results = if let Some(regions) = &layout_regions {
        if !regions.is_empty() {
            let img_w = results
                .iter()
                .map(|r| r.frame.left + r.frame.width)
                .chain(regions.iter().map(|r| r.bbox.x_max as f32))
                .fold(1.0f32, f32::max);
            order_text_results_by_layout(&results, regions, img_w)
        } else {
            results
        }
    } else {
        results
    };

    match cli.format {
        OutputFormat::Json => {
            if let Some(regions) = &layout_regions {
                #[derive(serde::Serialize)]
                struct OutputWithLayout<'a> {
                    text_results: &'a [rusto::TextResult],
                    layout_regions: &'a [rusto::LayoutRegion],
                }
                println!("{}", serde_json::to_string_pretty(&OutputWithLayout {
                    text_results: &effective_results,
                    layout_regions: regions,
                })?);
            } else {
                println!("{}", serde_json::to_string_pretty(&effective_results)?);
            }
        }
        OutputFormat::Text | OutputFormat::TextOrdered => {
            for result in &effective_results {
                println!("{}", result.text);
            }
        }
        OutputFormat::Tsv => {
            for result in &effective_results {
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
            for result in &effective_results {
                println!("{}", result.text);
            }
            println!("```");
        }
    }

    Ok(())
}

fn order_text_results_by_layout(
    results: &[rusto::TextResult],
    regions: &[rusto::LayoutRegion],
    w: f32,
) -> Vec<rusto::TextResult> {
    if regions.is_empty() || results.is_empty() {
        return results.to_vec();
    }

    let mut sorted_regions = regions.to_vec();
    sorted_regions.sort_by(|a, b| {
        a.bbox
            .y_min
            .cmp(&b.bbox.y_min)
            .then_with(|| a.bbox.x_min.cmp(&b.bbox.x_min))
    });

    let mut new_res = Vec::new();
    let mut res_left = Vec::new();
    let mut res_right = Vec::new();

    for reg in sorted_regions {
        let x1 = reg.bbox.x_min as f32;
        let x2 = reg.bbox.x_max as f32;

        if x1 < w / 4.0 && x2 < 3.0 * w / 5.0 {
            res_left.push(reg);
        } else if x1 > 2.0 * w / 5.0 {
            res_right.push(reg);
        } else {
            new_res.append(&mut res_left);
            new_res.append(&mut res_right);
            new_res.push(reg);
        }
    }
    new_res.append(&mut res_left);
    new_res.append(&mut res_right);

    let mut region_buckets: Vec<Vec<rusto::TextResult>> = vec![Vec::new(); new_res.len()];
    let mut unassigned: Vec<rusto::TextResult> = Vec::new();

    for res in results {
        let cx = res.frame.left + res.frame.width / 2.0;
        let cy = res.frame.top + res.frame.height / 2.0;

        let mut matched = false;
        for (i, reg) in new_res.iter().enumerate() {
            if cx >= reg.bbox.x_min as f32
                && cx <= reg.bbox.x_max as f32
                && cy >= reg.bbox.y_min as f32
                && cy <= reg.bbox.y_max as f32
            {
                region_buckets[i].push(res.clone());
                matched = true;
                break;
            }
        }
        if !matched {
            unassigned.push(res.clone());
        }
    }

    let mut ordered = Vec::with_capacity(results.len());
    for mut bucket in region_buckets {
        bucket.sort_by(|a, b| {
            let y_diff = (a.frame.top - b.frame.top).abs();
            if y_diff < 10.0 {
                a.frame.left
                    .partial_cmp(&b.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.frame.top
                    .partial_cmp(&b.frame.top)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        ordered.extend(bucket);
    }

    unassigned.sort_by(|a, b| {
        let y_diff = (a.frame.top - b.frame.top).abs();
        if y_diff < 10.0 {
            a.frame.left
                .partial_cmp(&b.frame.left)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.frame.top
                .partial_cmp(&b.frame.top)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    ordered.extend(unassigned);

    ordered
}
