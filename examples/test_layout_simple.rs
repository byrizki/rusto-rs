use clap::Parser;
use rusto::{LayoutConfig, LayoutDetector};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "models/DocOCR/layout.mnn")]
    model: PathBuf,
    
    #[arg(long, default_value = "models/test_images/page3.png")]
    image: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("Loading model: {:?}", args.model);
    
    let config = LayoutConfig::default(args.model);
    let mut detector = LayoutDetector::new(config)?;
    
    println!("Loading image: {:?}", args.image);
    let img = image::open(&args.image)?;
    let img_rgb = img.to_rgb8();
    
    // Convert to Mat
    #[cfg(not(feature = "use-opencv"))]
    let mat = rusto::image_impl::Mat::from_rgb8(
        img_rgb.width(), 
        img_rgb.height(), 
        img_rgb.into_raw()
    )?;
    
    println!("Running detection...");
    let output = detector.detect(&mat)?;
    
    println!("Success! Detected {} regions in {:.3}s:", output.regions.len(), output.elapse);
    for (i, region) in output.regions.iter().take(5).enumerate() {
        println!("  {}: {:?} @ ({},{}) - ({},{}) conf={:.3}", 
                 i, region.layout_type,
                 region.bbox.x_min, region.bbox.y_min,
                 region.bbox.x_max, region.bbox.y_max,
                 region.confidence);
    }
    
    Ok(())
}
