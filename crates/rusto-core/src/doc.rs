use std::path::Path;

use crate::config::InitializeConfig;
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::{crop_bbox, imread, Mat};
use crate::layout::{sort_layout_regions_top_down, LayoutDetector, LayoutRegion, LayoutType};
use crate::pipeline::OcrPipeline;
use crate::table::{TableDetector, TableDetectorConfig, TableStructureConfig, TableStructureRecognizer};
use crate::types::{LayoutConfig, OcrRunOptions};

use nanoserde::{DeJson, SerJson};

#[derive(Debug, Clone, SerJson, DeJson)]
pub struct DocBlock {
    pub region: LayoutRegion,
    pub text: Option<String>,
    pub html: Option<String>,
}

#[derive(Debug, Clone, SerJson, DeJson)]
pub struct DocResult {
    pub blocks: Vec<DocBlock>,
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
                        let height = block.region.bbox.y_max - block.region.bbox.y_min;
                        let level = if height > 100 {
                            1
                        } else if height > 60 {
                            2
                        } else {
                            3
                        };
                        let prefix = "#".repeat(level);
                        md.push_str(&format!("{} {}\n\n", prefix, text.trim()));
                    }
                    LayoutType::Header => {
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Footer => {
                        md.push_str(&format!("---\n{}\n\n", text.trim()));
                    }
                    LayoutType::FigureCaption => {
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::TableCaption => {
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::Equation | LayoutType::Reference => {
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Text => {
                        if Self::is_list_line(text) {
                            md.push_str(&format!("{}\n", text.trim()));
                        } else {
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
                match block.region.layout_type {
                    LayoutType::Figure => {
                        let has_caption = idx + 1 < self.blocks.len()
                            && matches!(self.blocks[idx + 1].region.layout_type, LayoutType::FigureCaption);

                        if !has_caption && !last_was_caption {
                            md.push_str("![Figure]\n\n");
                        } else if !last_was_caption {
                            md.push_str("![Figure]\n");
                        }
                        last_was_caption = false;
                    }
                    LayoutType::Table => {
                        if let Some(html) = &block.html {
                            md.push_str(&format!("\n{}\n\n", html));
                        } else {
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

    pub fn is_list_line(text: &str) -> bool {
        let trimmed = text.trim();
        trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("• ")
            || (trimmed.len() > 2
                && trimmed.chars().next().unwrap().is_ascii_digit()
                && (trimmed.chars().nth(1) == Some('.') || trimmed.chars().nth(1) == Some(')')))
    }
}

pub struct DocPipelineConfig {
    pub layout: LayoutConfig,
    pub ocr: InitializeConfig,
    pub table_detector: Option<TableDetectorConfig>,
    pub table_recognizer: Option<TableStructureConfig>,
}

pub struct DocPipeline<
    SL = Box<dyn InferenceSession>,
    SOcr = OcrPipeline,
    STD = Box<dyn InferenceSession>,
    STR = Box<dyn InferenceSession>,
> {
    pub layout: LayoutDetector<SL>,
    pub ocr: SOcr,
    pub table_detector: Option<TableDetector<STD>>,
    pub table_recognizer: Option<TableStructureRecognizer<STR>>,
}

pub trait OcrRunner {
    fn run_on_mat(&mut self, img: &Mat) -> Result<crate::pipeline::PipelineOutput, CoreError>;
}

impl OcrRunner for OcrPipeline {
    fn run_on_mat(&mut self, img: &Mat) -> Result<crate::pipeline::PipelineOutput, CoreError> {
        self.run_with_options(img, &OcrRunOptions::default())
    }
}

impl<SL: InferenceSession, SOcr: OcrRunner, STD: InferenceSession, STR: InferenceSession>
    DocPipeline<SL, SOcr, STD, STR>
{
    pub fn new(
        layout: LayoutDetector<SL>,
        ocr: SOcr,
        table_detector: Option<TableDetector<STD>>,
        table_recognizer: Option<TableStructureRecognizer<STR>>,
    ) -> Self {
        Self {
            layout,
            ocr,
            table_detector,
            table_recognizer,
        }
    }

    pub fn run<P: AsRef<Path>>(&mut self, image_path: P) -> Result<DocResult, CoreError> {
        let img = imread(image_path)?;
        self.run_on_mat(&img)
    }

    pub fn run_on_mat(&mut self, img: &Mat) -> Result<DocResult, CoreError> {
        let layout_output = self.layout.detect(img)?;
        let mut regions = layout_output.regions;
        sort_layout_regions_top_down(&mut regions, 20);

        let mut blocks = Vec::with_capacity(regions.len());
        for region in regions {
            let block = match region.layout_type {
                LayoutType::Figure => DocBlock { region, text: None, html: None },
                LayoutType::Table if self.table_detector.is_some() && self.table_recognizer.is_some() => {
                    let html = self.process_table(img, &region).ok();
                    DocBlock { region, text: None, html }
                }
                LayoutType::Table => DocBlock { region, text: None, html: None },
                _ => {
                    let crop = crop_bbox(img, &region.bbox)
                        .map_err(|e| CoreError::Image(e.to_string()))?;
                    let ocr_res = self.ocr.run_on_mat(&crop)?;
                    DocBlock {
                        region,
                        text: Some(ocr_res.txts.join(" ")),
                        html: None,
                    }
                }
            };
            blocks.push(block);
        }

        Ok(DocResult { blocks })
    }

    fn process_table(&mut self, img: &Mat, region: &LayoutRegion) -> Result<String, CoreError> {
        let table_crop = crop_bbox(img, &region.bbox)
            .map_err(|e| CoreError::Image(e.to_string()))?;
        let recognizer = self.table_recognizer.as_mut()
            .ok_or_else(|| CoreError::Preprocess("Table recognizer missing".to_string()))?;
        let mut structure = recognizer.recognize(&table_crop)?;
        let (img_h, img_w) = (table_crop.rows(), table_crop.cols());

        for row in &mut structure.rows {
            for cell in &mut row.cells {
                let cell_bbox = cell.scaled_bbox(img_w, img_h);
                if let Ok(cell_crop) = crop_bbox(&table_crop, &cell_bbox) {
                    let ocr_res = self.ocr.run_on_mat(&cell_crop);
                    cell.text = ocr_res.map(|r| r.txts.join(" ")).unwrap_or_default();
                } else {
                    cell.text.clear();
                }
            }
        }

        Ok(structure.to_html())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{BBox, LayoutRegion, LayoutType};

    #[test]
    fn test_doc_result_to_markdown() {
        let doc = DocResult {
            blocks: vec![
                DocBlock {
                    region: LayoutRegion {
                        bbox: BBox::new(0, 0, 100, 120),
                        layout_type: LayoutType::Title,
                        confidence: 0.99,
                    },
                    text: Some("Document Title".to_string()),
                    html: None,
                },
                DocBlock {
                    region: LayoutRegion {
                        bbox: BBox::new(0, 130, 100, 160),
                        layout_type: LayoutType::Text,
                        confidence: 0.95,
                    },
                    text: Some("- First item in list".to_string()),
                    html: None,
                },
                DocBlock {
                    region: LayoutRegion {
                        bbox: BBox::new(0, 170, 100, 200),
                        layout_type: LayoutType::Table,
                        confidence: 0.9,
                    },
                    text: None,
                    html: Some("<table><tr><td>data</td></tr></table>".to_string()),
                },
            ],
        };

        let md = doc.to_markdown();
        assert!(md.contains("# Document Title"));
        assert!(md.contains("- First item in list"));
        assert!(md.contains("<table><tr><td>data</td></tr></table>"));
    }
}
