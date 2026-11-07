use std::path::PathBuf;
use clap::Parser;
use rusto_core::det::TextDetector;
use rusto_core::image::imread;
use rusto_core::rec::{decoder_from_cfg, TextRecognizer};
use rusto_core::types::{DetConfig, RecConfig};
use rusto_rten::RtenSession;

#[derive(Parser, Debug)]
#[command(name = "rten-direct", about = "Direct RTen OCR Inference Engine (without RustO facade)")]
struct Args {
    #[arg(long)]
    det_model: PathBuf,
    #[arg(long)]
    rec_model: PathBuf,
    #[arg(long)]
    dict: Option<PathBuf>,
    #[arg(long, default_value = "960")]
    det_limit: usize,
    #[arg(long, default_value = "text")]
    format: String,
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let img = imread(&args.input)?;
    let mut det_cfg = DetConfig::default(args.det_model.clone());
    det_cfg.limit_side_len = args.det_limit as i32;
    let mut rec_cfg = RecConfig::default(args.rec_model.clone());
    rec_cfg.rec_keys_path = args.dict;

    let det_session = RtenSession::from_file(&args.det_model, 4)?;
    let rec_session = RtenSession::from_file(&args.rec_model, 4)?;
    let decoder = decoder_from_cfg(&rec_cfg, &rec_session)?;

    let mut detector = TextDetector::with_session(det_cfg, det_session);
    let mut recognizer = TextRecognizer::with_session(rec_cfg, rec_session, decoder);

    let det_out = detector.run(&img)?;
    let boxes = det_out.boxes.unwrap_or_default();
    let mut crops = Vec::with_capacity(boxes.len());
    for b in &boxes {
        let crop = rusto_core::geometry::get_rotate_crop_image(&img, b)?;
        crops.push(crop);
    }
    let rec_out = recognizer.run(&crops, false)?;
    for txt in rec_out.txts {
        if !txt.is_empty() {
            println!("{}", txt);
        }
    }
    Ok(())
}
