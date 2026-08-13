use std::path::Path;

#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{self, Mat, Rect};

use crate::config::InitializeConfig;
use crate::engine::EngineError;
use crate::layout::{LayoutDetector, LayoutRegion, LayoutType};
use crate::rusto_ocr::RustO;
use crate::table::{TableDetector, TableDetectorConfig, TableStructureConfig, TableStructureRecognizer};
use crate::types::LayoutConfig;

pub struct DocPipelineConfig {
    pub layout: LayoutConfig,
    pub ocr: InitializeConfig,
    pub table_detector: Option<TableDetectorConfig>,
    pub table_recognizer: Option<TableStructureConfig>,
}

pub struct DocPipeline {
    layout: LayoutDetector,
    ocr: RustO,
    table_detector: Option<TableDetector>,
    table_recognizer: Option<TableStructureRecognizer>,
}

#[derive(Debug, Clone)]
pub struct DocBlock {
    pub region: LayoutRegion,
    pub text: Option<String>,
    pub html: Option<String>, // For tables: HTML markup
}

pub struct DocResult {
    pub blocks: Vec<DocBlock>,
}

impl DocPipeline {
    pub fn new(config: DocPipelineConfig) -> Result<Self, EngineError> {
        let layout = LayoutDetector::new(config.layout)?;
        let ocr = RustO::initialize(config.ocr)?;
        
        // Initialize table recognition components if configured
        let table_detector = if let Some(detector_config) = config.table_detector {
            Some(TableDetector::new(detector_config)?)
        } else {
            None
        };
        
        let table_recognizer = if let Some(recognizer_config) = config.table_recognizer {
            Some(TableStructureRecognizer::new(recognizer_config)?)
        } else {
            None
        };
        
        Ok(Self { 
            layout, 
            ocr,
            table_detector,
            table_recognizer,
        })
    }

    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<DocResult, EngineError> {
        #[cfg(feature = "use-opencv")]
        use crate::image_impl::imread;
        #[cfg(not(feature = "use-opencv"))]
        use crate::image_impl::imread;

        let img = imread(image_path)?;

        // 1. Layout Analysis
        let layout_output = self.layout.detect(&img)?;
        let mut regions = layout_output.regions;

        // Sort regions: top-to-bottom, left-to-right
        regions.sort_by(|a, b| {
            let y_diff = (a.bbox.y_min - b.bbox.y_min).abs();
            if y_diff < 20 {
                // Tolerance for same line
                a.bbox.x_min.cmp(&b.bbox.x_min)
            } else {
                a.bbox.y_min.cmp(&b.bbox.y_min)
            }
        });

        let mut blocks = Vec::new();

        // 2. Process each region
        for region in regions {
            match region.layout_type {
                LayoutType::Text
                | LayoutType::Title
                | LayoutType::FigureCaption
                | LayoutType::TableCaption
                | LayoutType::Header
                | LayoutType::Footer
                | LayoutType::Reference
                | LayoutType::Equation => {
                    // Crop and OCR
                    let crop = self.crop_image(&img, &region)?;
                    let ocr_res = self.ocr.run_on_mat_with_options(
                        &crop,
                        &crate::OcrRunOptions::default(),
                    )?;

                    // Combine text lines
                    let text = ocr_res.txts.join(" ");
                    blocks.push(DocBlock {
                        region,
                        text: Some(text),
                        html: None,
                    });
                }
                LayoutType::Figure => {
                    // Figures - just store region
                    blocks.push(DocBlock { 
                        region, 
                        text: None,
                        html: None,
                    });
                }
                LayoutType::Table => {
                    // Tables - run recognition if configured
                    if self.table_detector.is_some() && self.table_recognizer.is_some() {
                        match self.process_table(&img, &region) {
                            Ok(html) => {
                                blocks.push(DocBlock {
                                    region,
                                    text: None,
                                    html: Some(html),
                                });
                            }
                            Err(e) => {
                                eprintln!("Table recognition failed: {}", e);
                                blocks.push(DocBlock { region, text: None, html: None });
                            }
                        }
                    } else {
                        // No table recognition configured
                        blocks.push(DocBlock { region, text: None, html: None });
                    }
                }
            }
        }

        Ok(DocResult { blocks })
    }

    /// Process a table region: recognize structure and OCR cells
    fn process_table(&mut self, img: &Mat, region: &LayoutRegion) -> Result<String, EngineError> {
        // 1. Crop table region
        let table_crop = self.crop_image(img, region)?;
        
        // 2. Recognize table structure
        let mut structure = self.table_recognizer.as_mut().unwrap()
            .recognize(&table_crop)?;
        
        // 3. OCR each cell
        #[cfg(feature = "use-opencv")]
        let (img_h, img_w) = (table_crop.rows(), table_crop.cols());
        #[cfg(not(feature = "use-opencv"))]
        let size = table_crop.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (img_h, img_w) = (size.height as i32, size.width as i32);
        
        for row in &mut structure.rows {
            for cell in &mut row.cells {
                // Scale cell bbox to actual image size
                let cell_bbox = &cell.bbox;
                let x_min = (cell_bbox.x_min as f32 * img_w as f32 / 100.0) as i32;
                let y_min = (cell_bbox.y_min as f32 * img_h as f32 / 100.0) as i32;
                let x_max = (cell_bbox.x_max as f32 * img_w as f32 / 100.0) as i32;
                let y_max = (cell_bbox.y_max as f32 * img_h as f32 / 100.0) as i32;
                
                // Create a region for the cell
                let cell_region = LayoutRegion {
                    bbox: crate::layout::BBox { x_min, y_min, x_max, y_max },
                    layout_type: LayoutType::Text,
                    confidence: 1.0,
                };
                
                // Crop and OCR the cell
                match self.crop_image(&table_crop, &cell_region) {
                    Ok(cell_crop) => {
                        match self.ocr.run_on_mat_with_options(
                            &cell_crop,
                            &crate::OcrRunOptions::default(),
                        ) {
                            Ok(ocr_res) => {
                                cell.text = ocr_res.txts.join(" ");
                            }
                            Err(_) => {
                                cell.text = String::new();
                            }
                        }
                    }
                    Err(_) => {
                        cell.text = String::new();
                    }
                }
            }
        }
        
        // 4. Generate HTML
        Ok(structure.to_html())
    }

    #[cfg(feature = "use-opencv")]
    fn crop_image(&self, img: &Mat, region: &LayoutRegion) -> Result<Mat, EngineError> {
        let rect = core::Rect::new(
            region.bbox.x_min,
            region.bbox.y_min,
            region.bbox.x_max - region.bbox.x_min,
            region.bbox.y_max - region.bbox.y_min,
        );

        // Ensure rect is within image bounds
        let img_w = img.cols();
        let img_h = img.rows();
        let x = rect.x.max(0).min(img_w - 1);
        let y = rect.y.max(0).min(img_h - 1);
        let w = rect.width.min(img_w - x);
        let h = rect.height.min(img_h - y);

        if w <= 0 || h <= 0 {
            return Err(EngineError::ImageError("Invalid crop region".to_string()));
        }

        let valid_rect = core::Rect::new(x, y, w, h);
        let cropped = Mat::roi(img, valid_rect)?;
        let mut dst = Mat::default();
        cropped.copy_to(&mut dst)?;
        Ok(dst)
    }

    #[cfg(not(feature = "use-opencv"))]
    fn crop_image(&self, img: &Mat, region: &LayoutRegion) -> Result<Mat, EngineError> {
        let rect = Rect {
            x: region.bbox.x_min,
            y: region.bbox.y_min,
            width: (region.bbox.x_max - region.bbox.x_min) as u32,
            height: (region.bbox.y_max - region.bbox.y_min) as u32,
        };

        // Ensure rect is within image bounds
        let size = img.size()?;
        let img_w = size.width;
        let img_h = size.height;

        let x = rect.x.max(0).min(img_w as i32 - 1);
        let y = rect.y.max(0).min(img_h as i32 - 1);
        let w = rect.width.min((img_w as i32 - x) as u32);
        let h = rect.height.min((img_h as i32 - y) as u32);

        if w <= 0 || h <= 0 {
            return Err(EngineError::ImageError("Invalid crop region".to_string()));
        }

        let valid_rect = Rect {
            x,
            y,
            width: w,
            height: h,
        };
        image_impl::crop(img, valid_rect).map_err(|e| EngineError::ImageError(e.to_string()))
    }
}

impl DocResult {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        let mut last_was_caption = false;

        for (idx, block) in self.blocks.iter().enumerate() {
            if let Some(text) = &block.text {
                if text.trim().is_empty() {
                    continue;
                }

                match block.region.layout_type {
                    LayoutType::Title => {
                        // Detect title level based on font size heuristics
                        // For now, use bbox height as proxy for font size
                        let height = block.region.bbox.y_max - block.region.bbox.y_min;
                        let level = if height > 100 {
                            1 // # Main title
                        } else if height > 60 {
                            2 // ## Section
                        } else {
                            3 // ### Subsection
                        };
                        let prefix = "#".repeat(level);
                        md.push_str(&format!("{} {}\n\n", prefix, text.trim()));
                    }
                    LayoutType::Header => {
                        // Headers are typically page headers - can be omitted or included
                        // For RapidDoc compatibility, include them but don't make them prominent
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Footer => {
                        // Footers (page numbers, etc.) - typically skipped in content
                        // but included for completeness
                        md.push_str(&format!("---\n{}\n\n", text.trim()));
                    }
                    LayoutType::FigureCaption => {
                        // Figure captions should be in italics and on separate line
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::TableCaption => {
                        // Table captions should be in italics
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::Equation => {
                        // Equations - treat as regular text for now (LaTeX formatting ignored)
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Reference => {
                        // References - keep as plain text but maybe add indicator
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Text => {
                        // Check if this looks like a list
                        if Self::is_list_line(text) {
                            // Format as markdown list
                            md.push_str(&format!("{}\n", text.trim()));
                        } else {
                            // Regular paragraph
                            md.push_str(&format!("{}\n\n", text.trim()));
                            last_was_caption = false;
                        }
                    }
                    _ => {
                        md.push_str(&format!("{}\n\n", text.trim()));
                        last_was_caption = false;
                    }
                }
            } else {
                // Blocks without text (figures, tables)
                match block.region.layout_type {
                    LayoutType::Figure => {
                        // Check if next block is a figure caption
                        let has_caption = idx + 1 < self.blocks.len() 
                            && matches!(self.blocks[idx + 1].region.layout_type, LayoutType::FigureCaption);
                        
                        if !has_caption && !last_was_caption {
                            // No caption, show placeholder
                            md.push_str("![Figure]\n\n");
                        } else if !last_was_caption {
                            // Caption comes after, just show image
                            md.push_str("![Figure]\n");
                        }
                        // If caption was before, it's already added
                        last_was_caption = false;
                    }
                    LayoutType::Table => {
                        // Check if we have HTML table markup (from table recognition)
                        if let Some(html) = &block.html {
                            // Insert HTML table directly (RapidDoc style)
                            md.push_str(&format!("\n{}\n\n", html));
                        } else {
                            // No table recognition - show placeholder
                            md.push_str("| Table | (Structure recognition not yet implemented) |\n");
                            md.push_str("|-------|---------------------------------------------|\n\n");
                        }
                        last_was_caption = false;
                    }
                    _ => {}
                }
            }
        }

        md
    }

    /// Detect if a line looks like a list item
    fn is_list_line(text: &str) -> bool {
        let trimmed = text.trim();
        // Check for common list patterns
        trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("• ")
            || (trimmed.len() > 2 
                && trimmed.chars().next().unwrap().is_ascii_digit()
                && (trimmed.chars().nth(1) == Some('.') || trimmed.chars().nth(1) == Some(')')))
    }
}
