use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use crate::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, OutputGranularity, RustO};
use crate::download::download_model_pipeline;
use rusto_core::models::{
    get_cache_dir, resolve_asset_local, resolve_base_pipeline,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// Pure-Rust MNN runtime (rusto-nn) - safe, fast, zero-dependency default
    RustoNn,
    /// Pure-Rust RTen FlatBuffers runtime (rusto-rten)
    Rten,
    /// Alibaba MNN C++ runtime (rusto-mnn)
    MnnCpp,
}

#[derive(Parser)]
#[command(version)]
#[command(about = "RustO! - Fast, memory-safe OCR powered by PaddleOCR and Indra OCR engines", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[command(flatten)]
    pub ocr: OcrArgs,
}

#[derive(Args, Clone)]
pub struct OcrArgs {
    /// Input image path
    #[arg(value_name = "IMAGE")]
    pub image: Option<PathBuf>,

    /// OCR model pipeline: "ppocrv6-tiny" (default), "ppocrv6-tiny-fp16", "ppocrv6-tiny-int8", "ppocrv6-small", "indraocr", "ppocrv5", "ppocrv4", etc.
    #[arg(short, long, default_value = "ppocrv6-tiny")]
    pub model: String,

    /// Base model directory containing det, rec, and dict files (overridden by --det-model, --rec-model, --dict)
    #[arg(long, alias = "model-dir")]
    pub model_path: Option<PathBuf>,

    /// Target language: "latin" (default), "ja", "ko", "ar", "cyrillic", "devanagari", "chinese-cht", etc.
    #[arg(short, long)]
    pub lang: Option<String>,

    /// Download model(s) to cache without running inference
    #[arg(long)]
    pub download_only: bool,

    /// Path to detection model (.mnn / .rten / .onnx). Overrides model auto-download.
    #[arg(long)]
    pub det_model: Option<PathBuf>,

    /// Path to recognition model (.mnn / .rten / .onnx). Overrides model auto-download.
    #[arg(long)]
    pub rec_model: Option<PathBuf>,

    /// Path to dictionary / metadata file (.txt / .json). Overrides model auto-download.
    #[arg(long)]
    pub dict: Option<PathBuf>,

    /// Custom model cache directory
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Enable neural reading order recovery
    #[arg(long)]
    pub reading_order: bool,

    /// Enable pure-Rust algorithmic reading order without neural tensor models
    #[arg(long, alias = "reading-order-heuristic", alias = "pure-rust-order", alias = "heuristic-order")]
    pub heuristic_reading_order: bool,

    /// Path to neural reading order model (order.mnn / indra_order.rten)
    #[arg(long)]
    pub order_model: Option<PathBuf>,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Json)]
    pub format: OutputFormat,

    /// Enable text line orientation classifier
    #[arg(long)]
    pub enable_cls: bool,

    /// Path to text line orientation classifier model
    #[arg(long)]
    pub cls_model: Option<PathBuf>,

    /// Enable document orientation classification
    #[arg(long)]
    pub enable_orient: bool,

    /// Path to orientation classification model
    #[arg(long)]
    pub orient_model: Option<PathBuf>,

    /// Enable text unwarping
    #[arg(long)]
    pub enable_unwarp: bool,

    /// Path to text unwarping model (uvdoc.rten / uvdoc.mnn)
    #[arg(long)]
    pub unwarp_model: Option<PathBuf>,

    /// Enable document layout detection
    #[arg(long)]
    pub enable_layout: bool,

    /// Document layout variant: "cdla" (default 10-class) or "publaynet" (5-class)
    #[arg(long)]
    pub layout_variant: Option<String>,

    /// Path to document layout detection model
    #[arg(long)]
    pub layout_model: Option<PathBuf>,

    /// Maximum detection image dimension (default: configured in model, typically 960 or 736)
    #[arg(long)]
    pub det_limit: Option<i32>,

    /// Text recognition batch size (default: 6, or 1 for Indra OCR)
    #[arg(long, alias = "batch-num")]
    pub rec_batch_num: Option<i32>,

    /// Minimum recognition confidence score threshold [0.0 - 1.0]
    #[arg(long, alias = "text-score")]
    pub score_threshold: Option<f32>,

    /// Maximum image side length for preprocessing
    #[arg(long)]
    pub max_side_len: Option<f32>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run OCR on an image file
    Ocr(OcrArgs),
    /// Download OCR models to the cache directory without running inference
    Download(DownloadArgs),
}

#[derive(Args)]
pub struct DownloadArgs {
    /// OCR model pipeline to download: "ppocrv6-tiny" (default), "ppocrv6-tiny-fp16", "ppocrv6-tiny-int8", "ppocrv6-small", "indraocr", "ppocrv5", "ppocrv4", etc.
    #[arg(short, long, default_value = "ppocrv6-tiny")]
    pub model: String,

    /// Target language: "latin" (default), "ja", "ko", "ar", "cyrillic", "devanagari", "chinese-cht", etc.
    #[arg(short, long)]
    pub lang: Option<String>,

    /// Download text-line orientation classifier model
    #[arg(long)]
    pub cls: bool,

    /// Download document orientation classifier model
    #[arg(long)]
    pub orient: bool,

    /// Download document unwarping model
    #[arg(long)]
    pub unwarp: bool,

    /// Download document layout detection model (layout_cdla or layout_publaynet)
    #[arg(long)]
    pub layout: bool,

    /// Document layout variant: "cdla" (default 10-class) or "publaynet" (5-class)
    #[arg(long)]
    pub layout_variant: Option<String>,

    /// Download neural reading order model
    #[arg(long)]
    pub reading_order: bool,

    /// Custom model cache directory
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OutputFormat {
    /// JSON output with full details
    Json,
    /// Plain text, one line per detected text
    Text,
    /// TSV format: text\tscore\tx1,y1,x2,y2,x3,y3,x4,y4
    Tsv,
    /// Markdown format with metadata and ordering
    Markdown,
    /// Plain text ordered by vertical and horizontal position
    TextOrdered,
    /// Reconstructed 2D spatial text layout preserving columns and alignment
    Spatial,
}

pub fn run_cli(engine: EngineKind) -> Result<(), Box<dyn std::error::Error>> {
    let bin_name = match engine {
        EngineKind::RustoNn => "rusto",
        EngineKind::Rten => {
            crate::ensure_rten_registered();
            "rusto-rten"
        }
        EngineKind::MnnCpp => {
            crate::ensure_mnn_cpp_registered();
            "rusto-mnn"
        }
    };

    let parsed = Cli::parse();

    let cli = match parsed.command {
        Some(Commands::Download(args)) => {
            let cache_dir = get_cache_dir(args.cache_dir.as_deref());
            eprintln!("Downloading {} pipeline to {}...", args.model, cache_dir.display());
            let downloaded = download_model_pipeline(
                Some(&args.model),
                args.lang.as_deref(),
                args.cls,
                args.orient,
                args.unwarp,
                if args.layout {
                    Some(args.layout_variant.as_deref().unwrap_or("cdla"))
                } else {
                    None
                },
                args.reading_order,
                &cache_dir,
            )?;
            eprintln!("✓ Successfully verified/downloaded {} model asset(s).", downloaded.len());
            return Ok(());
        }
        Some(Commands::Ocr(ocr_args)) => ocr_args,
        None => parsed.ocr,
    };

    if cli.download_only {
        let cache_dir = get_cache_dir(cli.cache_dir.as_deref());
        eprintln!("Downloading {} pipeline to {}...", cli.model, cache_dir.display());
        let downloaded = download_model_pipeline(
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
            eprintln!("Error: Please provide an image path to run OCR (`{bin_name} <IMAGE>` or `{bin_name} ocr <IMAGE>`), or use `{bin_name} download` to download models.");
            std::process::exit(1);
        }
    };

    let effective_dir = cli.model_path.as_deref().or_else(|| {
        let p = Path::new(&cli.model);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    });

    let (orient_file, unwarp_file, cls_file, layout_cdla_file, layout_publaynet_file, order_file) = match engine {
        EngineKind::Rten => (
            if cli.model.eq_ignore_ascii_case("indraocr") || cli.model.eq_ignore_ascii_case("indra") {
                "indra_orient.rten"
            } else {
                "ppocr_orient.rten"
            },
            "uvdoc.rten",
            "ppocr_cls.rten",
            "layout_cdla.rten",
            "layout_publaynet.rten",
            "indra_order.rten",
        ),
        _ => (
            if cli.model.eq_ignore_ascii_case("indraocr") || cli.model.eq_ignore_ascii_case("indra") {
                "indra_orient.mnn"
            } else {
                "ppocr_orient.mnn"
            },
            "uvdoc.mnn",
            "ppocr_cls.mnn",
            "layout_cdla.mnn",
            "layout_publaynet.mnn",
            "indra_order.mnn",
        ),
    };

    let cache_dir = get_cache_dir(cli.cache_dir.as_deref());
    if effective_dir.is_none() {
        let needs_download = resolve_base_pipeline(
            Some(&cli.model),
            None,
            cli.lang.as_deref(),
            cli.det_model.as_deref(),
            cli.rec_model.as_deref(),
            cli.dict.as_deref(),
            &cache_dir,
        )
        .is_err()
            || (cli.enable_cls && cli.cls_model.is_none() && resolve_asset_local(None, None, "cls", cls_file, &cache_dir).is_err())
            || (cli.enable_orient && cli.orient_model.is_none() && resolve_asset_local(None, None, "orient", orient_file, &cache_dir).is_err())
            || (cli.enable_unwarp && cli.unwarp_model.is_none() && resolve_asset_local(None, None, "unwarp", unwarp_file, &cache_dir).is_err())
            || ((cli.enable_layout || cli.layout_variant.is_some()) && cli.layout_model.is_none() && {
                let layout_file = if cli.layout_variant.as_deref() == Some("publaynet") {
                    layout_publaynet_file
                } else {
                    layout_cdla_file
                };
                resolve_asset_local(None, None, "layout", layout_file, &cache_dir).is_err()
            })
            || (cli.reading_order && cli.order_model.is_none() && resolve_asset_local(None, None, "order", order_file, &cache_dir).is_err());

        if needs_download {
            eprintln!("Downloading missing {} model assets to {}...", cli.model, cache_dir.display());
            download_model_pipeline(
                Some(&cli.model),
                cli.lang.as_deref(),
                cli.enable_cls && cli.cls_model.is_none(),
                cli.enable_orient && cli.orient_model.is_none(),
                cli.enable_unwarp && cli.unwarp_model.is_none(),
                if (cli.enable_layout || cli.layout_variant.is_some()) && cli.layout_model.is_none() {
                    Some(cli.layout_variant.as_deref().unwrap_or("cdla"))
                } else {
                    None
                },
                cli.reading_order && cli.order_model.is_none(),
                &cache_dir,
            )?;
        }
    }

    let (det_path, rec_path, dict_path) = resolve_base_pipeline(
        Some(&cli.model),
        effective_dir,
        cli.lang.as_deref(),
        cli.det_model.as_deref(),
        cli.rec_model.as_deref(),
        cli.dict.as_deref(),
        &cache_dir,
    )?;

    let det_str = det_path.to_string_lossy().to_ascii_lowercase();
    let rec_str = rec_path.to_string_lossy().to_ascii_lowercase();
    let dict_str = dict_path.to_string_lossy().to_ascii_lowercase();
    let is_dict_txt = dict_path.extension().and_then(|s| s.to_str()).map_or(false, |e| e.eq_ignore_ascii_case("txt"));
    let is_indra_asset = dict_str.contains("meta.json") || det_str.contains("indra") || rec_str.contains("indra") || det_str.contains("linenet");
    let is_ppocr_asset = is_dict_txt || det_str.contains("ppocr") || rec_str.contains("ppocr");

    let model_lower = cli.model.to_ascii_lowercase();
    let preset = if is_indra_asset {
        crate::INDRA_MODEL_CONFIG
    } else if is_ppocr_asset || (!model_lower.contains("indra") && !model_lower.contains("linenet")) {
        if det_str.contains("v5") || rec_str.contains("v5") || model_lower.contains("v5") {
            crate::PPV5_MODEL_CONFIG
        } else if det_str.contains("v4") || rec_str.contains("v4") || model_lower.contains("v4") {
            crate::PPV4_MODEL_CONFIG
        } else if det_str.contains("v3") || rec_str.contains("v3") || model_lower.contains("v3") {
            crate::PPV3_MODEL_CONFIG
        } else {
            crate::PPV6_MODEL_CONFIG
        }
    } else if model_lower.contains("indra") || model_lower.contains("linenet") {
        crate::INDRA_MODEL_CONFIG
    } else {
        crate::PPV6_MODEL_CONFIG
    };

    let mut config = InitializeConfig::from_preset(preset, det_path, rec_path, dict_path);

    if cli.enable_cls || cli.cls_model.is_some() {
        let cls_path = resolve_asset_local(
            cli.cls_model.as_deref(),
            effective_dir,
            "cls",
            cls_file,
            &cache_dir,
        )?;
        config = config.with_cls(cls_path);
    }

    if cli.enable_orient || cli.orient_model.is_some() {
        let orient_path = resolve_asset_local(
            cli.orient_model.as_deref(),
            effective_dir,
            "orient",
            orient_file,
            &cache_dir,
        )?;
        config = config.with_orientation(orient_path);
    }

    if cli.enable_unwarp || cli.unwarp_model.is_some() {
        let unwarp_path = resolve_asset_local(
            cli.unwarp_model.as_deref(),
            effective_dir,
            "unwarp",
            unwarp_file,
            &cache_dir,
        )?;
        config = config.with_unwarp(unwarp_path);
    }

    if cli.enable_layout || cli.layout_model.is_some() || cli.layout_variant.is_some() {
        let layout_file = if cli.layout_variant.as_deref() == Some("publaynet") {
            layout_publaynet_file
        } else {
            layout_cdla_file
        };
        let layout_path = resolve_asset_local(
            cli.layout_model.as_deref(),
            effective_dir,
            "layout",
            layout_file,
            &cache_dir,
        )?;
        config = config.with_layout(layout_path);
    }

    if !cli.heuristic_reading_order && (cli.reading_order || cli.order_model.is_some()) {
        let order_path = resolve_asset_local(
            cli.order_model.as_deref(),
            effective_dir,
            "order",
            order_file,
            &cache_dir,
        )?;
        config.global.reading_order = true;
        config.global.reading_order_model = Some(order_path);
    }

    if let Some(batch_num) = cli.rec_batch_num {
        config = config.with_rec_batch_num(batch_num);
    }

    let mut ocr = RustO::initialize(config)?;

    let mut options = OcrRunOptions::default();
    if cli.format == OutputFormat::Spatial {
        options.output = OutputGranularity::Spatial;
    }
    if cli.enable_cls {
        options.line_classification = Some(true);
    }
    if cli.enable_orient {
        options.orientation_detection = Some(true);
    }
    if cli.enable_unwarp {
        options.auto_unwarp = Some(true);
    }
    let layout_active = cli.enable_layout || cli.layout_model.is_some() || cli.layout_variant.is_some();
    if layout_active || cli.reading_order || cli.order_model.is_some() || cli.heuristic_reading_order {
        options.layout_reading = Some(true);
    }
    if let Some(det_limit) = cli.det_limit {
        let det_opts = options.detection.get_or_insert_with(Default::default);
        det_opts.limit_side_len = Some(det_limit);
        det_opts.limit_type = Some("max".to_string());
    }
    if let Some(batch_num) = cli.rec_batch_num {
        let rec_opts = options.recognition.get_or_insert_with(Default::default);
        rec_opts.batch_size = Some(batch_num);
    }
    if let Some(score_thresh) = cli.score_threshold {
        let post_rec = options.post_recognition.get_or_insert_with(Default::default);
        post_rec.score_threshold = Some(score_thresh);
    }
    if let Some(max_side) = cli.max_side_len {
        let pre_opts = options.pre_detection.get_or_insert_with(Default::default);
        pre_opts.target_max_side = Some(max_side as u32);
    }

    let result = ocr.detect_text(&ImageSource::Path(image_path), &options)?;

    let mut effective_results = match result {
        DetectTextResult::Structured(results) => results,
        DetectTextResult::Spatial(text) => {
            println!("{}", text);
            return Ok(());
        }
    };

    let has_model_order = cli.reading_order || cli.order_model.is_some() || layout_active;
    if cli.format == OutputFormat::TextOrdered && !has_model_order {
        effective_results = rusto_core::layout::cluster_and_sort_reading_order(effective_results, None, None);
    }

    match cli.format {
        OutputFormat::Json => {
            println!("{}", nanoserde::SerJson::serialize_json(&effective_results));
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
        OutputFormat::Spatial => unreachable!("Spatial output returned early"),
    }

    Ok(())
}
