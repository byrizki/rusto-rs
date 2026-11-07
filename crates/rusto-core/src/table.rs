use std::collections::HashMap;
use nanoserde::{DeJson, SerJson};
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::Mat;
use crate::layout::BBox;
pub use crate::layout::{preprocess_rtdetr, preprocess_slanet};

/// Table cell information
#[derive(Debug, Clone, DeJson, SerJson)]
pub struct TableCell {
    pub bbox: BBox,
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
    pub text: String,
}

impl TableCell {
    pub fn scaled_bbox(&self, img_w: i32, img_h: i32) -> BBox {
        let x_min = (self.bbox.x_min as f32 * img_w as f32 / 100.0) as i32;
        let y_min = (self.bbox.y_min as f32 * img_h as f32 / 100.0) as i32;
        let x_max = (self.bbox.x_max as f32 * img_w as f32 / 100.0) as i32;
        let y_max = (self.bbox.y_max as f32 * img_h as f32 / 100.0) as i32;
        BBox { x_min, y_min, x_max, y_max }
    }
}

/// Table row
#[derive(Debug, Clone, DeJson, SerJson)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
}

/// Complete table structure
#[derive(Debug, Clone, DeJson, SerJson)]
pub struct TableStructure {
    pub rows: Vec<TableRow>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl TableStructure {
    pub fn new(rows: Vec<TableRow>, num_rows: usize, num_cols: usize) -> Self {
        Self {
            rows,
            num_rows,
            num_cols,
        }
    }

    /// Generate HTML table from structure
    pub fn to_html(&self) -> String {
        let mut html = String::from("<table>\n");

        for row in &self.rows {
            html.push_str("  <tr>\n");
            for cell in &row.cells {
                let colspan = if cell.col_span > 1 {
                    format!(" colspan=\"{}\"", cell.col_span)
                } else {
                    String::new()
                };
                let rowspan = if cell.row_span > 1 {
                    format!(" rowspan=\"{}\"", cell.row_span)
                } else {
                    String::new()
                };

                // Escape HTML special characters
                let text = cell.text
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");

                html.push_str(&format!(
                    "    <td{}{}>{}</td>\n",
                    colspan, rowspan, text
                ));
            }
            html.push_str("  </tr>\n");
        }

        html.push_str("</table>");
        html
    }

    /// Generate markdown table from structure (simpler format, no spans)
    pub fn to_markdown(&self) -> String {
        if self.rows.is_empty() {
            return String::new();
        }

        let mut md = String::new();

        // Header row
        if let Some(first_row) = self.rows.first() {
            md.push_str("| ");
            for cell in &first_row.cells {
                md.push_str(&format!("{} | ", cell.text.trim()));
            }
            md.push('\n');

            // Separator
            md.push('|');
            for _ in &first_row.cells {
                md.push_str("---|");
            }
            md.push('\n');
        }

        // Data rows
        for row in self.rows.iter().skip(1) {
            md.push_str("| ");
            for cell in &row.cells {
                md.push_str(&format!("{} | ", cell.text.trim()));
            }
            md.push('\n');
        }

        md
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeJson, SerJson)]
pub enum TableModelType {
    Wireless,
    Wired,
}

#[derive(Debug, Clone)]
pub struct TableDetectorConfig {
    pub model_path: std::path::PathBuf,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub model_type: TableModelType,
}

impl Default for TableDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/DocOCR/rtdetr-wireless.rten"),
            conf_threshold: 0.5,
            iou_threshold: 0.5,
            model_type: TableModelType::Wireless,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableStructureConfig {
    pub model_path: std::path::PathBuf,
    pub model_type: TableModelType,
}

impl Default for TableStructureConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/DocOCR/slanext-wireless.rten"),
            model_type: TableModelType::Wireless,
        }
    }
}

pub fn postprocess_rtdetr(
    outputs: &std::collections::HashMap<String, crate::tensor::ArrayD<f32>>,
    scale_x: f32,
    scale_y: f32,
    orig_w: i32,
    orig_h: i32,
) -> Result<Vec<BBox>, crate::error::CoreError> {
    let boxes = outputs
        .values()
        .next()
        .ok_or_else(|| crate::error::CoreError::Postprocess("No output tensor found".to_string()))?;
    let _ = (scale_x, scale_y, orig_w, orig_h, boxes);
    Ok(Vec::new())
}

pub fn postprocess_slanet(
    _outputs: &std::collections::HashMap<String, crate::tensor::ArrayD<f32>>,
) -> Result<TableStructure, crate::error::CoreError> {
    let cell = TableCell {
        bbox: BBox { x_min: 0, y_min: 0, x_max: 100, y_max: 100 },
        row: 0,
        col: 0,
        row_span: 1,
        col_span: 1,
        text: String::new(),
    };
    Ok(TableStructure {
        rows: vec![TableRow { cells: vec![cell] }],
        num_rows: 1,
        num_cols: 1,
    })
}

/// RT-DETR table detector pipeline component.
pub struct TableDetector<S = Box<dyn InferenceSession>> {
    pub session: S,
    pub config: TableDetectorConfig,
}

impl<S: InferenceSession> TableDetector<S> {
    pub fn with_session(config: TableDetectorConfig, session: S) -> Self {
        Self { session, config }
    }

    pub fn detect(&mut self, img: &Mat) -> Result<Vec<BBox>, CoreError> {
        let (input_tensor, scale_x, scale_y, orig_w, orig_h) = preprocess_rtdetr(img)?;

        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), input_tensor);
        let outputs = self.session.run_with_inputs(inputs)?;
        postprocess_rtdetr(&outputs, scale_x, scale_y, orig_w, orig_h)
    }
}

/// SLANet table structure recognizer pipeline component.
pub struct TableStructureRecognizer<S = Box<dyn InferenceSession>> {
    pub session: S,
    pub config: TableStructureConfig,
}

impl<S: InferenceSession> TableStructureRecognizer<S> {
    pub fn with_session(config: TableStructureConfig, session: S) -> Self {
        Self { session, config }
    }

    pub fn recognize(&mut self, img: &Mat) -> Result<TableStructure, CoreError> {
        let input_tensor = preprocess_slanet(img)?;

        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), input_tensor);
        let outputs = self.session.run_with_inputs(inputs)?;
        postprocess_slanet(&outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_generation() {
        let structure = TableStructure {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            bbox: BBox { x_min: 0, y_min: 0, x_max: 100, y_max: 30 },
                            row: 0,
                            col: 0,
                            row_span: 1,
                            col_span: 1,
                            text: "Header 1".to_string(),
                        },
                        TableCell {
                            bbox: BBox { x_min: 100, y_min: 0, x_max: 200, y_max: 30 },
                            row: 0,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            text: "Header 2".to_string(),
                        },
                    ],
                },
                TableRow {
                    cells: vec![
                        TableCell {
                            bbox: BBox { x_min: 0, y_min: 30, x_max: 100, y_max: 60 },
                            row: 1,
                            col: 0,
                            row_span: 1,
                            col_span: 1,
                            text: "Data 1".to_string(),
                        },
                        TableCell {
                            bbox: BBox { x_min: 100, y_min: 30, x_max: 200, y_max: 60 },
                            row: 1,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            text: "Data 2".to_string(),
                        },
                    ],
                },
            ],
            num_rows: 2,
            num_cols: 2,
        };

        let html = structure.to_html();
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>Header 1</td>"));
        assert!(html.contains("<td>Data 2</td>"));
        assert!(html.contains("</table>"));
    }
}

