use std::collections::HashMap;
use std::time::Instant;
use crate::tensor::{Array2, Array3, Array4, ArrayD, Axis};
use nanoserde::{DeJson, SerJson};
use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::{self, Mat, Size, INTER_LINEAR};
use crate::types::LayoutConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeJson, SerJson)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

impl BBox {
    pub fn new(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    pub fn width(&self) -> i32 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> i32 {
        self.y_max - self.y_min
    }

    pub fn area(&self) -> i32 {
        self.width().max(0) * self.height().max(0)
    }

    pub fn intersection(&self, other: &BBox) -> Option<BBox> {
        let x_min = self.x_min.max(other.x_min);
        let y_min = self.y_min.max(other.y_min);
        let x_max = self.x_max.min(other.x_max);
        let y_max = self.y_max.min(other.y_max);

        if x_min < x_max && y_min < y_max {
            Some(BBox {
                x_min,
                y_min,
                x_max,
                y_max,
            })
        } else {
            None
        }
    }
}

/// Layout element types
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeJson, SerJson)]
pub enum LayoutType {
    Text = 0,
    Title = 1,
    Figure = 2,
    FigureCaption = 3,
    Table = 4,
    TableCaption = 5,
    Header = 6,
    Footer = 7,
    Reference = 8,
    Equation = 9,
}

impl LayoutType {
    pub fn from_class(class_id: usize) -> Option<Self> {
        match class_id {
            0 => Some(LayoutType::Text),
            1 => Some(LayoutType::Title),
            2 => Some(LayoutType::Figure),
            3 => Some(LayoutType::FigureCaption),
            4 => Some(LayoutType::Table),
            5 => Some(LayoutType::TableCaption),
            6 => Some(LayoutType::Header),
            7 => Some(LayoutType::Footer),
            8 => Some(LayoutType::Reference),
            9 => Some(LayoutType::Equation),
            _ => None,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "text" | "paragraph" | "ocrtext" | "list" => Some(LayoutType::Text),
            "title" | "paragraph_title" => Some(LayoutType::Title),
            "figure" | "image" => Some(LayoutType::Figure),
            "figure_caption" => Some(LayoutType::FigureCaption),
            "table" => Some(LayoutType::Table),
            "table_caption" => Some(LayoutType::TableCaption),
            "header" => Some(LayoutType::Header),
            "footer" => Some(LayoutType::Footer),
            "reference" | "tablefootnote" => Some(LayoutType::Reference),
            "equation" | "formula" | "isolateformula" | "inlineformula" | "isolatedformula" => {
                Some(LayoutType::Equation)
            }
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LayoutType::Text => "text",
            LayoutType::Title => "title",
            LayoutType::Figure => "figure",
            LayoutType::FigureCaption => "figure_caption",
            LayoutType::Table => "table",
            LayoutType::TableCaption => "table_caption",
            LayoutType::Header => "header",
            LayoutType::Footer => "footer",
            LayoutType::Reference => "reference",
            LayoutType::Equation => "equation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, DeJson, SerJson)]
pub struct LayoutRegion {
    pub bbox: BBox,
    pub layout_type: LayoutType,
    pub confidence: f32,
}

pub struct LayoutOutput {
    pub regions: Vec<LayoutRegion>,
    pub elapse: f64,
}

pub fn box_iou(a: &BBox, b: &BBox) -> f32 {
    let inter_w = (a.x_max.min(b.x_max) - a.x_min.max(b.x_min)).max(0);
    let inter_h = (a.y_max.min(b.y_max) - a.y_min.max(b.y_min)).max(0);
    let inter_area = (inter_w * inter_h) as f32;

    let a_area = ((a.x_max - a.x_min).max(0) * (a.y_max - a.y_min).max(0)) as f32;
    let b_area = ((b.x_max - b.x_min).max(0) * (b.y_max - b.y_min).max(0)) as f32;
    let union_area = a_area + b_area - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

pub fn nms_layout_regions(regions: &[LayoutRegion], iou_thresh: f32) -> Vec<LayoutRegion> {
    let mut sorted = regions.to_vec();
    sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(sorted[i].clone());

        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            if box_iou(&sorted[i].bbox, &sorted[j].bbox) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Preprocesses an image for layout detection models by resizing to target dimensions and normalizing.
pub fn preprocess_layout_image(
    image: &Mat,
    new_w: i32,
    new_h: i32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(Array4<f32>, f32, f32), CoreError> {
    let size = image.size()?;
    let (h, w) = (size.height, size.width);

    let scale_w = new_w as f32 / w as f32;
    let scale_h = new_h as f32 / h as f32;

    let mut resized = Mat::default();
    image::resize(
        image,
        &mut resized,
        Size::new(new_w, new_h),
        INTER_LINEAR,
    )?;

    let hw = new_h as usize * new_w as usize;
    let mut data = vec![0.0f32; 3 * hw];

    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let idx = y * (new_w as usize) + x;
            data[idx] = (b - mean[0]) / std[0];
            data[hw + idx] = (g - mean[1]) / std[1];
            data[2 * hw + idx] = (r - mean[2]) / std[2];
        }
    }

    let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;

    Ok((arr.insert_axis(Axis(0)), scale_w, scale_h))
}

pub fn build_layout_model_inputs(
    input_tensor: Array4<f32>,
    model_path: &std::path::Path,
    new_h: i32,
    new_w: i32,
    scale_h: f32,
    scale_w: f32,
) -> Result<std::collections::HashMap<String, ArrayD<f32>>, CoreError> {
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("image".to_string(), input_tensor.into_dyn());

    let p_str = model_path.to_string_lossy();
    if p_str.contains("doclayout") {
        let im_shape = Array2::from_shape_vec((1, 2), vec![new_h as f32, new_w as f32])
            .map_err(|e| CoreError::InvalidShape(e.to_string()))?;
        inputs.insert("im_shape".to_string(), im_shape.into_dyn());

        let scale_factor = Array2::from_shape_vec((1, 2), vec![scale_h, scale_w])
            .map_err(|e| CoreError::InvalidShape(e.to_string()))?;
        inputs.insert("scale_factor".to_string(), scale_factor.into_dyn());
    }

    Ok(inputs)
}

#[derive(Debug, Clone)]
pub struct IndexedRegion {
    pub original_idx: usize,
    pub region: LayoutRegion,
}

/// Sorts a slice of text lines into natural reading order (top-to-bottom, left-to-right, respecting multi-column layouts).
pub fn sort_lines_reading_order(lines: &mut [crate::types::TextResult]) {
    if lines.len() <= 1 {
        return;
    }

    let min_x = lines.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
    let max_x = lines
        .iter()
        .map(|l| l.frame.left + l.frame.width)
        .fold(f32::NEG_INFINITY, f32::max);
    let doc_w = (max_x - min_x).max(1.0);

    let mut heights: Vec<f32> = lines.iter().map(|l| l.frame.height).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_h = heights[heights.len() / 2].max(1.0);
    let y_tolerance = median_h * 0.5;

    // 1. Sort lines primarily by top (y), then left (x)
    let mut sorted = lines.to_vec();
    sorted.sort_by(|a, b| {
        a.frame
            .top
            .partial_cmp(&b.frame.top)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.frame
                    .left
                    .partial_cmp(&b.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // 2. Partition into vertical sections divided by spanning lines (width > 0.65 * doc_w)
    let mut sections: Vec<Vec<crate::types::TextResult>> = Vec::new();
    let mut current_section: Vec<crate::types::TextResult> = Vec::new();
    let mut section_max_y = f32::NEG_INFINITY;

    for item in sorted {
        let is_spanning = item.frame.width > 0.65 * doc_w;
        let item_bottom = item.frame.top + item.frame.height;

        if current_section.is_empty() {
            section_max_y = item_bottom;
            current_section.push(item);
        } else {
            let prev_was_spanning = current_section.last().unwrap().frame.width > 0.65 * doc_w;
            if is_spanning || prev_was_spanning {
                sections.push(current_section);
                section_max_y = item_bottom;
                current_section = vec![item];
            } else {
                section_max_y = section_max_y.max(item_bottom);
                current_section.push(item);
            }
        }
    }
    if !current_section.is_empty() {
        sections.push(current_section);
    }

    // 3. For each section, partition lines into columns based on horizontal overlap
    let mut ordered_results: Vec<crate::types::TextResult> = Vec::with_capacity(lines.len());

    for section in sections {
        if section.len() <= 1 {
            ordered_results.extend(section);
            continue;
        }

        let mut columns: Vec<Vec<crate::types::TextResult>> = Vec::new();

        for item in section {
            let item_x1 = item.frame.left;
            let item_x2 = item.frame.left + item.frame.width;
            let item_w = item.frame.width.max(1.0);

            let mut best_col: Option<usize> = None;
            let mut best_overlap = 0.0f32;

            for (col_idx, col) in columns.iter().enumerate() {
                let col_x1 = col.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
                let col_x2 = col
                    .iter()
                    .map(|l| l.frame.left + l.frame.width)
                    .fold(f32::NEG_INFINITY, f32::max);
                let col_w = (col_x2 - col_x1).max(1.0);

                let overlap_x1 = item_x1.max(col_x1);
                let overlap_x2 = item_x2.min(col_x2);
                let overlap = (overlap_x2 - overlap_x1).max(0.0);

                let overlap_ratio = overlap / item_w.min(col_w);
                if overlap_ratio > 0.35 && overlap_ratio > best_overlap {
                    best_overlap = overlap_ratio;
                    best_col = Some(col_idx);
                }
            }

            if let Some(col_idx) = best_col {
                columns[col_idx].push(item);
            } else {
                columns.push(vec![item]);
            }
        }

        // Merge columns that have significant horizontal overlap
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..columns.len() {
                for j in (i + 1)..columns.len() {
                    let a_x1 = columns[i].iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
                    let a_x2 = columns[i]
                        .iter()
                        .map(|l| l.frame.left + l.frame.width)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let b_x1 = columns[j].iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
                    let b_x2 = columns[j]
                        .iter()
                        .map(|l| l.frame.left + l.frame.width)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let a_w = (a_x2 - a_x1).max(1.0);
                    let b_w = (b_x2 - b_x1).max(1.0);

                    let ov_x1 = a_x1.max(b_x1);
                    let ov_x2 = a_x2.min(b_x2);
                    let ov = (ov_x2 - ov_x1).max(0.0);
                    if ov / a_w.min(b_w) > 0.35 {
                        let to_merge = columns.remove(j);
                        columns[i].extend(to_merge);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }

        // Sort columns left-to-right by their min X
        columns.sort_by(|col_a, col_b| {
            let a_x1 = col_a.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
            let b_x1 = col_b.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
            a_x1.partial_cmp(&b_x1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Within each column, sort lines top-to-bottom, left-to-right using y_tolerance
        for mut col in columns {
            col.sort_by(|a, b| {
                a.frame
                    .top
                    .partial_cmp(&b.frame.top)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.frame
                            .left
                            .partial_cmp(&b.frame.left)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });

            let mut col_rows: Vec<Vec<crate::types::TextResult>> = Vec::new();
            let mut current_row: Vec<crate::types::TextResult> = Vec::new();
            let mut current_row_y_center = 0.0f32;

            for item in col {
                let item_y_center = item.frame.top + item.frame.height / 2.0;
                if current_row.is_empty() {
                    current_row.push(item);
                    current_row_y_center = item_y_center;
                } else {
                    let avg_y = current_row_y_center / current_row.len() as f32;
                    if (item_y_center - avg_y).abs() <= y_tolerance {
                        current_row_y_center += item_y_center;
                        current_row.push(item);
                    } else {
                        col_rows.push(current_row);
                        current_row = vec![item];
                        current_row_y_center = item_y_center;
                    }
                }
            }
            if !current_row.is_empty() {
                col_rows.push(current_row);
            }

            for row in col_rows.iter_mut() {
                row.sort_by(|a, b| {
                    a.frame
                        .left
                        .partial_cmp(&b.frame.left)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            ordered_results.extend(col_rows.into_iter().flatten());
        }
    }

    lines.clone_from_slice(&ordered_results);
}

/// Cluster and sort text results in natural reading order across multi-column documents.
/// Groups words belonging to the same line within each column while strictly preserving column
/// boundaries to avoid cross-column merging.
pub fn cluster_and_sort_reading_order(
    lines: Vec<crate::types::TextResult>,
    line_y_threshold: Option<f32>,
    word_x_threshold: Option<f32>,
) -> Vec<crate::types::TextResult> {
    if lines.len() <= 1 {
        return lines;
    }

    let doc_w = lines
        .iter()
        .map(|l| l.frame.left + l.frame.width)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let mut heights: Vec<f32> = lines.iter().map(|l| l.frame.height).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_h = heights[heights.len() / 2].max(1.0);
    let y_tolerance = median_h * line_y_threshold.unwrap_or(0.5);
    let word_x = word_x_threshold.unwrap_or(0.4);

    // 1. Sort lines primarily by top (y), then left (x)
    let mut sorted = lines;
    sorted.sort_by(|a, b| {
        a.frame
            .top
            .partial_cmp(&b.frame.top)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.frame
                    .left
                    .partial_cmp(&b.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // 2. Partition into vertical sections divided by spanning lines (width > 0.65 * doc_w)
    let mut sections: Vec<Vec<crate::types::TextResult>> = Vec::new();
    let mut current_section: Vec<crate::types::TextResult> = Vec::new();
    let mut section_max_y = f32::NEG_INFINITY;

    for item in sorted {
        let is_spanning = item.frame.width > 0.65 * doc_w;
        let item_bottom = item.frame.top + item.frame.height;

        if current_section.is_empty() {
            section_max_y = item_bottom;
            current_section.push(item);
        } else {
            let prev_was_spanning = current_section.last().unwrap().frame.width > 0.65 * doc_w;
            if is_spanning || prev_was_spanning {
                sections.push(current_section);
                section_max_y = item_bottom;
                current_section = vec![item];
            } else {
                section_max_y = section_max_y.max(item_bottom);
                current_section.push(item);
            }
        }
    }
    if !current_section.is_empty() {
        sections.push(current_section);
    }

    // 3. For each section, partition lines into columns based on horizontal overlap
    let mut ordered_results: Vec<crate::types::TextResult> = Vec::with_capacity(section_max_y as usize);

    for section in sections {
        if section.len() <= 1 {
            ordered_results.extend(section);
            continue;
        }

        let mut columns: Vec<Vec<crate::types::TextResult>> = Vec::new();

        for item in section {
            let item_x1 = item.frame.left;
            let item_x2 = item.frame.left + item.frame.width;
            let item_w = item.frame.width.max(1.0);

            let mut best_col: Option<usize> = None;
            let mut best_overlap = 0.0f32;

            for (col_idx, col) in columns.iter().enumerate() {
                let col_x1 = col.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
                let col_x2 = col
                    .iter()
                    .map(|l| l.frame.left + l.frame.width)
                    .fold(f32::NEG_INFINITY, f32::max);
                let col_w = (col_x2 - col_x1).max(1.0);

                let overlap_x1 = item_x1.max(col_x1);
                let overlap_x2 = item_x2.min(col_x2);
                let overlap = (overlap_x2 - overlap_x1).max(0.0);

                let overlap_ratio = overlap / item_w.min(col_w);
                if overlap_ratio > 0.35 && overlap_ratio > best_overlap {
                    best_overlap = overlap_ratio;
                    best_col = Some(col_idx);
                }
            }

            if let Some(col_idx) = best_col {
                columns[col_idx].push(item);
            } else {
                columns.push(vec![item]);
            }
        }

        // Merge columns that have significant horizontal overlap
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..columns.len() {
                for j in (i + 1)..columns.len() {
                    let a_x1 = columns[i].iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
                    let a_x2 = columns[i]
                        .iter()
                        .map(|l| l.frame.left + l.frame.width)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let b_x1 = columns[j].iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
                    let b_x2 = columns[j]
                        .iter()
                        .map(|l| l.frame.left + l.frame.width)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let a_w = (a_x2 - a_x1).max(1.0);
                    let b_w = (b_x2 - b_x1).max(1.0);

                    let ov_x1 = a_x1.max(b_x1);
                    let ov_x2 = a_x2.min(b_x2);
                    let ov = (ov_x2 - ov_x1).max(0.0);
                    if ov / a_w.min(b_w) > 0.35 {
                        let to_merge = columns.remove(j);
                        columns[i].extend(to_merge);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }

        // Sort columns left-to-right by their min X
        columns.sort_by(|col_a, col_b| {
            let a_x1 = col_a.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
            let b_x1 = col_b.iter().map(|l| l.frame.left).fold(f32::INFINITY, f32::min);
            a_x1.partial_cmp(&b_x1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Within each column, sort lines top-to-bottom, left-to-right using y_tolerance
        for mut col in columns {
            col.sort_by(|a, b| {
                a.frame
                    .top
                    .partial_cmp(&b.frame.top)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.frame
                            .left
                            .partial_cmp(&b.frame.left)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });

            let mut col_rows: Vec<Vec<crate::types::TextResult>> = Vec::new();
            let mut current_row: Vec<crate::types::TextResult> = Vec::new();
            let mut current_row_y_center = 0.0f32;

            for item in col {
                let item_y_center = item.frame.top + item.frame.height / 2.0;
                if current_row.is_empty() {
                    current_row.push(item);
                    current_row_y_center = item_y_center;
                } else {
                    let avg_y = current_row_y_center / current_row.len() as f32;
                    if (item_y_center - avg_y).abs() <= y_tolerance {
                        current_row_y_center += item_y_center;
                        current_row.push(item);
                    } else {
                        col_rows.push(current_row);
                        current_row = vec![item];
                        current_row_y_center = item_y_center;
                    }
                }
            }
            if !current_row.is_empty() {
                col_rows.push(current_row);
            }

            for mut row in col_rows {
                row.sort_by(|a, b| {
                    a.frame
                        .left
                        .partial_cmp(&b.frame.left)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut merged_row: Vec<crate::types::TextResult> = Vec::new();
                for item in row {
                    if let Some(prev) = merged_row.last_mut() {
                        let prev_right = prev.frame.left + prev.frame.width;
                        let gap = item.frame.left - prev_right;
                        let avg_height = (prev.frame.height + item.frame.height) / 2.0;
                        let max_gap = (avg_height * word_x).max(1.0);
                        if gap >= 0.0 && gap <= max_gap {
                            prev.text.push(' ');
                            prev.text.push_str(&item.text);
                            prev.score = prev.score.min(item.score);
                            let left = prev.frame.left.min(item.frame.left);
                            let top = prev.frame.top.min(item.frame.top);
                            let right = prev_right.max(item.frame.left + item.frame.width);
                            let bottom = (prev.frame.top + prev.frame.height)
                                .max(item.frame.top + item.frame.height);
                            let box_points = [(left, top), (right, top), (right, bottom), (left, bottom)];
                            prev.box_points = box_points;
                            prev.frame = crate::types::Frame::from_points(&box_points);
                            continue;
                        }
                    }
                    merged_row.push(item);
                }

                ordered_results.extend(merged_row);
            }
        }
    }

    ordered_results
}

/// Orders layout regions into natural reading order across arbitrary column layouts (1 to N columns).
pub fn order_layout_regions(regions: &[LayoutRegion], img_w: f32) -> Vec<IndexedRegion> {
    if regions.len() <= 1 {
        return regions
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, r)| IndexedRegion {
                original_idx: i,
                region: r,
            })
            .collect();
    }

    let mut indexed: Vec<IndexedRegion> = regions
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, r)| IndexedRegion {
            original_idx: i,
            region: r,
        })
        .collect();

    indexed.sort_by(|a, b| {
        a.region
            .bbox
            .y_min
            .cmp(&b.region.bbox.y_min)
            .then_with(|| a.region.bbox.x_min.cmp(&b.region.bbox.x_min))
    });

    let mut sections: Vec<Vec<IndexedRegion>> = Vec::new();
    let mut current_section: Vec<IndexedRegion> = Vec::new();
    let mut section_max_y = i32::MIN;

    for item in indexed {
        let w = (item.region.bbox.x_max - item.region.bbox.x_min) as f32;
        let is_spanning = w > 0.65 * img_w
            || item.region.layout_type == LayoutType::Header
            || item.region.layout_type == LayoutType::Footer;

        if current_section.is_empty() {
            section_max_y = item.region.bbox.y_max;
            current_section.push(item);
        } else {
            let prev_was_spanning = {
                let prev = current_section.last().unwrap();
                let pw = (prev.region.bbox.x_max - prev.region.bbox.x_min) as f32;
                pw > 0.65 * img_w
                    || prev.region.layout_type == LayoutType::Header
                    || prev.region.layout_type == LayoutType::Footer
            };

            if is_spanning || prev_was_spanning || item.region.bbox.y_min >= section_max_y {
                sections.push(current_section);
                section_max_y = item.region.bbox.y_max;
                current_section = vec![item];
            } else {
                section_max_y = section_max_y.max(item.region.bbox.y_max);
                current_section.push(item);
            }
        }
    }
    if !current_section.is_empty() {
        sections.push(current_section);
    }

    let mut ordered_regions = Vec::with_capacity(regions.len());

    for section in sections {
        if section.len() <= 1 {
            ordered_regions.extend(section);
            continue;
        }

        let mut columns: Vec<Vec<IndexedRegion>> = Vec::new();

        for item in section {
            let item_x1 = item.region.bbox.x_min;
            let item_x2 = item.region.bbox.x_max;
            let item_w = (item_x2 - item_x1).max(1);

            let mut best_col: Option<usize> = None;
            let mut best_overlap = 0.0f32;

            for (col_idx, col) in columns.iter().enumerate() {
                let col_x1 = col.iter().map(|r| r.region.bbox.x_min).min().unwrap();
                let col_x2 = col.iter().map(|r| r.region.bbox.x_max).max().unwrap();
                let col_w = (col_x2 - col_x1).max(1);

                let overlap_x1 = item_x1.max(col_x1);
                let overlap_x2 = item_x2.min(col_x2);
                let overlap = (overlap_x2 - overlap_x1).max(0) as f32;

                let overlap_ratio = overlap / (item_w.min(col_w) as f32);
                if overlap_ratio > 0.35 && overlap_ratio > best_overlap {
                    best_overlap = overlap_ratio;
                    best_col = Some(col_idx);
                }
            }

            if let Some(col_idx) = best_col {
                columns[col_idx].push(item);
            } else {
                columns.push(vec![item]);
            }
        }

        columns.sort_by_key(|col| col.iter().map(|r| r.region.bbox.x_min).min().unwrap_or(0));

        for mut col in columns {
            col.sort_by_key(|r| r.region.bbox.y_min);
            ordered_regions.extend(col);
        }
    }

    ordered_regions
}

/// Orders text results based on document layout regions across arbitrary column configurations.
pub fn order_text_results_by_layout(
    results: &[crate::types::TextResult],
    regions: &[LayoutRegion],
    img_w: f32,
) -> Vec<crate::types::TextResult> {
    if regions.is_empty() || results.is_empty() {
        let mut sorted = results.to_vec();
        sort_lines_reading_order(&mut sorted);
        return sorted;
    }

    let ordered_regions = order_layout_regions(regions, img_w);
    let mut region_buckets: Vec<Vec<crate::types::TextResult>> = vec![Vec::new(); ordered_regions.len()];
    let mut unassigned: Vec<crate::types::TextResult> = Vec::new();

    for res in results {
        let cx = res.frame.left + res.frame.width / 2.0;
        let cy = res.frame.top + res.frame.height / 2.0;

        let mut best_match: Option<usize> = None;
        let mut best_overlap_area = 0.0f32;

        for (i, item) in ordered_regions.iter().enumerate() {
            let reg = &item.region;
            let rx1 = reg.bbox.x_min as f32 - 4.0;
            let ry1 = reg.bbox.y_min as f32 - 4.0;
            let rx2 = reg.bbox.x_max as f32 + 4.0;
            let ry2 = reg.bbox.y_max as f32 + 4.0;

            if cx >= rx1 && cx <= rx2 && cy >= ry1 && cy <= ry2 {
                let ix1 = res.frame.left.max(reg.bbox.x_min as f32);
                let iy1 = res.frame.top.max(reg.bbox.y_min as f32);
                let ix2 = (res.frame.left + res.frame.width).min(reg.bbox.x_max as f32);
                let iy2 = (res.frame.top + res.frame.height).min(reg.bbox.y_max as f32);
                let inter_w = (ix2 - ix1).max(0.0);
                let inter_h = (iy2 - iy1).max(0.0);
                let area = inter_w * inter_h;

                if area >= best_overlap_area {
                    best_overlap_area = area;
                    best_match = Some(i);
                }
            }
        }

        if let Some(i) = best_match {
            let mut line = res.clone();
            line.region_id = Some(ordered_regions[i].original_idx as i32);
            line.layout_type = Some(ordered_regions[i].region.layout_type);
            region_buckets[i].push(line);
        } else {
            let mut line = res.clone();
            line.region_id = None;
            line.layout_type = None;
            unassigned.push(line);
        }
    }

    let mut final_ordered = Vec::with_capacity(results.len());

    for mut bucket in region_buckets {
        sort_lines_reading_order(&mut bucket);
        final_ordered.extend(bucket);
    }

    if !unassigned.is_empty() {
        sort_lines_reading_order(&mut unassigned);
        final_ordered.extend(unassigned);
    }

    final_ordered
}

/// Resize and normalize an image for RT-DETR (640×640, ImageNet normalization, pure-Rust).
/// Returns `(tensor [1,3,640,640], scale, scale, orig_w, orig_h)`.
pub fn preprocess_rtdetr(
    image: &Mat,
) -> Result<(ArrayD<f32>, f32, f32, i32, i32), CoreError> {
    let size = image.size()?;
    let (orig_h, orig_w) = (size.height, size.width);
    let target = 640usize;
    let scale = target as f32 / orig_h.max(orig_w) as f32;
    let new_h = (orig_h as f32 * scale) as i32;
    let new_w = (orig_w as f32 * scale) as i32;

    let mut resized = Mat::default();
    image::resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut data = vec![0.0f32; 3 * target * target];

    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                data[c * target * target + y * target + x] = (val - mean[c]) / std[c];
            }
        }
    }

    let arr = Array3::from_shape_vec((3, target, target), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok((arr.insert_axis(Axis(0)).into_dyn(), scale, scale, orig_w, orig_h))
}

/// Resize and normalize an image for SLANet (488×488, [0,1] range, pure-Rust).
pub fn preprocess_slanet(image: &Mat) -> Result<ArrayD<f32>, CoreError> {
    let (target_h, target_w) = (488usize, 488usize);
    let mut resized = Mat::default();
    image::resize(image, &mut resized, Size::new(target_w as i32, target_h as i32), INTER_LINEAR)?;

    let hw = target_h * target_w;
    let mut data = vec![0.0f32; 3 * hw];

    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for (c, &pv) in pixel.iter().enumerate() {
                data[c * hw + y * target_w + x] = pv as f32 / 255.0;
            }
        }
    }

    let arr = Array3::from_shape_vec((3, target_h, target_w), data)
        .map_err(|e| CoreError::Preprocess(e.to_string()))?;
    Ok(arr.insert_axis(Axis(0)).into_dyn())
}

/// IoU between two axis-aligned boxes `[x1,y1,x2,y2]`.
#[inline]
pub fn calculate_box_iou(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let x1 = b1[0].max(b2[0]);
    let y1 = b1[1].max(b2[1]);
    let x2 = b1[2].min(b2[2]);
    let y2 = b1[3].min(b2[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let a1 = (b1[2] - b1[0]).max(0.0) * (b1[3] - b1[1]).max(0.0);
    let a2 = (b2[2] - b2[0]).max(0.0) * (b2[3] - b2[1]).max(0.0);
    let union = a1 + a2 - inter;
    if union > 0.0 { inter / (union + 1e-6) } else { 0.0 }
}

pub struct PicoDetInput<'a> {
    pub outputs: &'a HashMap<String, ArrayD<f32>>,
    pub orig_w: i32,
    pub orig_h: i32,
    pub new_w: i32,
    pub new_h: i32,
    pub scale_w: f32,
    pub scale_h: f32,
    pub conf_thresh: f32,
    pub iou_thresh: f32,
    pub labels: Option<&'a [String]>,
}

/// Decode PicoDet multi-scale outputs into `LayoutRegion` list.
#[allow(clippy::too_many_arguments)]
pub fn postprocess_picodet(input: PicoDetInput<'_>) -> Result<Vec<LayoutRegion>, CoreError> {
    let PicoDetInput {
        outputs,
        orig_w,
        orig_h,
        new_w,
        new_h,
        scale_w,
        scale_h,
        conf_thresh,
        iou_thresh,
        labels,
    } = input;
    let strides = [8usize, 16, 32, 64];

    struct StrideData<'a> {
        stride: usize,
        exp_points: usize,
        exp_w: usize,
        scores: std::borrow::Cow<'a, [f32]>,
        num_classes: usize,
        boxes: std::borrow::Cow<'a, [f32]>,
    }

    let mut stride_data_list: Vec<StrideData<'_>> = Vec::new();

    for &s in &strides {
        let exp_h = ((new_h as f32) / (s as f32)).ceil() as usize;
        let exp_w = ((new_w as f32) / (s as f32)).ceil() as usize;
        let exp_points = exp_h * exp_w;

        let mut matched_scores: Option<(std::borrow::Cow<'_, [f32]>, usize)> = None;
        let mut matched_boxes: Option<std::borrow::Cow<'_, [f32]>> = None;

        for tensor in outputs.values() {
            let shape = tensor.shape();
            let (points, last_dim) = match shape.len() {
                2 => (shape[0], shape[1]),
                3 if shape[0] == 1 => (shape[1], shape[2]),
                _ => continue,
            };
            if points != exp_points { continue; }
            let cow = std::borrow::Cow::Borrowed(tensor.as_slice());
            if last_dim == 32 {
                matched_boxes = Some(cow);
            } else {
                matched_scores = Some((cow, last_dim));
            }
        }

        if let (Some((scores, num_classes)), Some(boxes)) = (matched_scores, matched_boxes) {
            stride_data_list.push(StrideData { stride: s, exp_points, exp_w, scores, num_classes, boxes });
        }
    }

    if stride_data_list.is_empty() {
        let shapes: Vec<String> = outputs.iter()
            .map(|(k, v)| format!("{}: {:?}", k, v.shape()))
            .collect();
        return Err(CoreError::Preprocess(format!(
            "No PicoDet tensors found. Outputs: {}", shapes.join(", ")
        )));
    }

    let num_classes = stride_data_list[0].num_classes;

    struct Candidate {
        box_coords: [f32; 4],
        scores: Vec<f32>,
        max_score: f32,
    }

    let mut all_candidates: Vec<Candidate> = Vec::new();

    for data in &stride_data_list {
        let s = data.stride as f32;
        let mut stride_candidates: Vec<Candidate> = Vec::with_capacity(data.exp_points);

        for p in 0..data.exp_points {
            let r = (p / data.exp_w) as f32;
            let c = (p % data.exp_w) as f32;
            let cx = (c + 0.5) * s;
            let cy = (r + 0.5) * s;

            let mut d = [0.0f32; 4];
            for (side, d_side) in d.iter_mut().enumerate() {
                let base = p * 32 + side * 8;
                let max_val = (0..8).map(|k| data.boxes[base + k]).fold(f32::NEG_INFINITY, f32::max);
                let mut exp_vals = [0.0f32; 8];
                let mut sum_exp = 0.0f32;
                for (k, ev) in exp_vals.iter_mut().enumerate() {
                    *ev = (data.boxes[base + k] - max_val).exp();
                    sum_exp += *ev;
                }
                let inv_sum = 1.0 / (sum_exp + 1e-9);
                *d_side = exp_vals.iter().enumerate().map(|(k, &ev)| k as f32 * ev * inv_sum).sum::<f32>() * s;
            }

            let box_coords = [cx - d[0], cy - d[1], cx + d[2], cy + d[3]];
            let s_base = p * data.num_classes;
            let p_scores: Vec<f32> = (0..data.num_classes).map(|ci| data.scores[s_base + ci]).collect();
            let max_score = p_scores.iter().cloned().fold(0.0f32, f32::max);
            stride_candidates.push(Candidate { box_coords, scores: p_scores, max_score });
        }

        stride_candidates.sort_by(|a, b| b.max_score.partial_cmp(&a.max_score).unwrap_or(std::cmp::Ordering::Equal));
        if stride_candidates.len() > 1000 { stride_candidates.truncate(1000); }
        all_candidates.extend(stride_candidates);
    }

    let mut regions = Vec::new();

    for class_idx in 0..num_classes {
        let mut candidates_for_class: Vec<usize> = (0..all_candidates.len())
            .filter(|&i| all_candidates[i].scores[class_idx] > conf_thresh)
            .collect();
        if candidates_for_class.is_empty() { continue; }

        candidates_for_class.sort_by(|&a, &b| {
            all_candidates[b].scores[class_idx].partial_cmp(&all_candidates[a].scores[class_idx])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept: Vec<usize> = Vec::new();
        for &idx in &candidates_for_class {
            let b1 = &all_candidates[idx].box_coords;
            if !kept.iter().any(|&k| calculate_box_iou(b1, &all_candidates[k].box_coords) > iou_thresh) {
                kept.push(idx);
                if kept.len() >= 100 { break; }
            }
        }

        for idx in kept {
            let cand = &all_candidates[idx];
            let b = cand.box_coords;
            let cx1 = b[0].max(0.0).min(new_w as f32);
            let cy1 = b[1].max(0.0).min(new_h as f32);
            let cx2 = b[2].max(0.0).min(new_w as f32);
            let cy2 = b[3].max(0.0).min(new_h as f32);

            let x_min = ((cx1 / scale_w).round() as i32).max(0).min(orig_w);
            let y_min = ((cy1 / scale_h).round() as i32).max(0).min(orig_h);
            let x_max = ((cx2 / scale_w).round() as i32).max(0).min(orig_w);
            let y_max = ((cy2 / scale_h).round() as i32).max(0).min(orig_h);

            if x_max <= x_min || y_max <= y_min { continue; }

            let layout_type = if let Some(chars) = &labels {
                if class_idx < chars.len() {
                    LayoutType::from_name(&chars[class_idx]).unwrap_or(LayoutType::Text)
                } else {
                    LayoutType::from_class(class_idx).unwrap_or(LayoutType::Text)
                }
            } else if num_classes == 10 {
                match class_idx {
                    0 => LayoutType::Text, 1 => LayoutType::Title, 2 => LayoutType::Figure,
                    3 => LayoutType::FigureCaption, 4 => LayoutType::Table,
                    5 => LayoutType::TableCaption, 6 => LayoutType::Header,
                    7 => LayoutType::Footer, 8 => LayoutType::Reference,
                    9 => LayoutType::Equation, _ => LayoutType::Text,
                }
            } else if num_classes == 5 {
                match class_idx {
                    0 => LayoutType::Text, 1 => LayoutType::Title,
                    2 => LayoutType::Text, 3 => LayoutType::Table,
                    4 => LayoutType::Figure, _ => LayoutType::Text,
                }
            } else {
                LayoutType::from_class(class_idx).unwrap_or(LayoutType::Text)
            };

            regions.push(LayoutRegion {
                bbox: BBox { x_min, y_min, x_max, y_max },
                layout_type,
                confidence: cand.scores[class_idx],
            });
        }
    }

    Ok(regions)
}

pub fn resolve_layout_target_dimensions(
    model_path: &std::path::Path,
    target_height: Option<i32>,
    target_width: Option<i32>,
    target_size: i32,
) -> (i32, i32) {
    match (target_height, target_width) {
        (Some(h), Some(w)) => (h, w),
        (Some(h), None) => (h, (h * 608) / 800),
        (None, Some(w)) => ((w * 800) / 608, w),
        (None, None) => {
            let p_str = model_path.to_string_lossy();
            if p_str.contains("doclayout_s") {
                (480, 480)
            } else if p_str.contains("doclayout_l") || p_str.contains("doclayout_plus_l") {
                (800, 800)
            } else if p_str.contains("yolo") {
                (640, 640)
            } else if target_size > 0 && target_size != 800 {
                (target_size, target_size)
            } else {
                (800, 608)
            }
        }
    }
}

pub fn postprocess_direct_layout_boxes(
    outputs: &HashMap<String, ArrayD<f32>>,
    orig_w: i32,
    orig_h: i32,
    scale_w: f32,
    scale_h: f32,
    conf_thresh: f32,
    iou_thresh: f32,
) -> Option<Vec<LayoutRegion>> {
    let mut raw_boxes = Vec::new();
    let mut found_direct = false;
    for tensor in outputs.values() {
        let shape = tensor.shape();
        if shape.len() == 2 && shape[1] == 6 {
            for row in tensor.outer_iter() {
                raw_boxes.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
            }
            found_direct = true;
            break;
        } else if shape.len() == 3 && shape[2] == 6 {
            let n_boxes = shape[1];
            let slice = tensor.as_slice();
            for i in 0..n_boxes {
                let row = &slice[i * 6..(i + 1) * 6];
                raw_boxes.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
            }
            found_direct = true;
            break;
        }
    }

    if !found_direct {
        return None;
    }

    let indices = crate::geometry::nms(&raw_boxes, iou_thresh, conf_thresh);

    let mut regions = Vec::new();
    for idx in indices {
        let row = &raw_boxes[idx];
        let class_id = row[0] as usize;
        let score = row[1];
        let x1 = row[2];
        let y1 = row[3];
        let x2 = row[4];
        let y2 = row[5];

        let layout_type = match class_id {
            0 => LayoutType::Title,
            1 | 15 => LayoutType::Text,
            2 => continue,
            3 => LayoutType::Figure,
            4 => LayoutType::FigureCaption,
            5 => LayoutType::Table,
            6 => LayoutType::TableCaption,
            7 => LayoutType::Reference,
            8 | 13 | 14 => LayoutType::Equation,
            9 => LayoutType::Text,
            _ => LayoutType::Text,
        };

        let x_min = ((x1 / scale_w) as i32).max(0).min(orig_w);
        let y_min = ((y1 / scale_h) as i32).max(0).min(orig_h);
        let x_max = ((x2 / scale_w) as i32).max(0).min(orig_w);
        let y_max = ((y2 / scale_h) as i32).max(0).min(orig_h);

        if x_max <= x_min || y_max <= y_min {
            continue;
        }

        regions.push(LayoutRegion {
            bbox: BBox {
                x_min,
                y_min,
                x_max,
                y_max,
            },
            layout_type,
            confidence: score,
        });
    }

    Some(regions)
}

pub fn sort_layout_regions_top_down(regions: &mut [LayoutRegion], line_tolerance: i32) {
    regions.sort_by(|a, b| {
        let y_diff = (a.bbox.y_min - b.bbox.y_min).abs();
        if y_diff < line_tolerance {
            a.bbox.x_min.cmp(&b.bbox.x_min)
        } else {
            a.bbox.y_min.cmp(&b.bbox.y_min)
        }
    });
}

/// Layout detector pipeline component executing PicoDet or direct layout detection.
pub struct LayoutDetector<S = Box<dyn InferenceSession>> {
    pub session: S,
    pub config: LayoutConfig,
}

impl<S: InferenceSession> LayoutDetector<S> {
    pub fn with_session(config: LayoutConfig, session: S) -> Self {
        Self { session, config }
    }

    pub fn detect(&mut self, image: &Mat) -> Result<LayoutOutput, CoreError> {
        let start = Instant::now();
        let (orig_h, orig_w) = (image.rows(), image.cols());

        let (new_h, new_w) = self.target_dimensions();
        let (input_tensor, scale_w, scale_h) = self.preprocess(image)?;
        let inputs = build_layout_model_inputs(
            input_tensor,
            &self.config.model_path,
            new_h,
            new_w,
            scale_h,
            scale_w,
        )?;

        let outputs = self.session.run_with_inputs(inputs)?;
        let regions = self.postprocess(outputs, orig_w, orig_h, new_w, new_h, scale_w, scale_h)?;
        let elapse = start.elapsed().as_secs_f64();

        Ok(LayoutOutput { regions, elapse })
    }

    pub fn target_dimensions(&self) -> (i32, i32) {
        resolve_layout_target_dimensions(
            &self.config.model_path,
            self.config.target_height,
            self.config.target_width,
            self.config.target_size,
        )
    }

    pub fn preprocess(&self, image: &Mat) -> Result<(Array4<f32>, f32, f32), CoreError> {
        let (new_h, new_w) = self.target_dimensions();
        preprocess_layout_image(image, new_w, new_h, self.config.mean, self.config.std)
    }

    pub fn postprocess(
        &self,
        outputs: HashMap<String, ArrayD<f32>>,
        orig_w: i32,
        orig_h: i32,
        new_w: i32,
        new_h: i32,
        scale_w: f32,
        scale_h: f32,
    ) -> Result<Vec<LayoutRegion>, CoreError> {
        if let Some(regions) = postprocess_direct_layout_boxes(
            &outputs,
            orig_w,
            orig_h,
            scale_w,
            scale_h,
            self.config.conf_thresh,
            self.config.iou_thresh,
        ) {
            return Ok(regions);
        }

        let chars_opt = self.session.get_character_list("character");
        postprocess_picodet(PicoDetInput {
            outputs: &outputs,
            orig_w,
            orig_h,
            new_w,
            new_h,
            scale_w,
            scale_h,
            conf_thresh: self.config.conf_thresh,
            iou_thresh: self.config.iou_thresh,
            labels: chars_opt.as_deref(),
        })
    }
}



