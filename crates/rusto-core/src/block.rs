use nanoserde::{DeJson, SerJson};

use crate::layout::LayoutType;
use crate::types::{Frame, TextResult};

/// An aggregated document block representing a layout region and its member text lines.
#[derive(Debug, Clone, PartialEq, DeJson, SerJson)]
pub struct BlockResult {
    /// Identifier of the layout region.
    pub region_id: i32,
    /// Semantic layout category (e.g. Text, Title, Header, Footer, Table, Figure).
    pub layout_type: LayoutType,
    /// Axis-aligned bounding box frame enclosing all member lines in the block.
    pub frame: Frame,
    /// 4 polygon vertices enclosing the block [top-left, top-right, bottom-right, bottom-left].
    pub box_points: [(f32, f32); 4],
    /// Mean recognition confidence score across all member lines.
    pub score: f32,
    /// Member text lines in reading order.
    pub lines: Vec<TextResult>,
}

impl BlockResult {
    /// Returns the combined text of all lines in this block, separated by newlines.
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Clusters a flat list of `TextResult` lines into hierarchical `BlockResult` blocks.
///
/// Lines sharing a `region_id` are grouped into the same block, preserving reading order.
/// If lines do not have an assigned `region_id`, each line forms an individual block.
pub fn cluster_text_results_into_blocks(results: &[TextResult]) -> Vec<BlockResult> {
    if results.is_empty() {
        return Vec::new();
    }

    let mut blocks: Vec<BlockResult> = Vec::new();
    let mut region_map: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut next_unassigned_id: i32 = -1;

    for item in results {
        if let Some(rid) = item.region_id {
            if let Some(&block_idx) = region_map.get(&rid) {
                // Append line to existing block and update union bounding box
                let block = &mut blocks[block_idx];
                let line_left = item.frame.left;
                let line_top = item.frame.top;
                let line_right = line_left + item.frame.width;
                let line_bottom = line_top + item.frame.height;

                let cur_left = block.frame.left.min(line_left);
                let cur_top = block.frame.top.min(line_top);
                let cur_right = (block.frame.left + block.frame.width).max(line_right);
                let cur_bottom = (block.frame.top + block.frame.height).max(line_bottom);

                block.frame.left = cur_left;
                block.frame.top = cur_top;
                block.frame.width = cur_right - cur_left;
                block.frame.height = cur_bottom - cur_top;

                block.box_points = [
                    (cur_left, cur_top),
                    (cur_right, cur_top),
                    (cur_right, cur_bottom),
                    (cur_left, cur_bottom),
                ];

                block.lines.push(item.clone());
            } else {
                let block_idx = blocks.len();
                region_map.insert(rid, block_idx);
                let l_type = item.layout_type.unwrap_or(LayoutType::Text);
                blocks.push(BlockResult {
                    region_id: rid,
                    layout_type: l_type,
                    frame: item.frame,
                    box_points: item.box_points,
                    score: item.score,
                    lines: vec![item.clone()],
                });
            }
        } else {
            // Unassigned line forms its own block
            let rid = next_unassigned_id;
            next_unassigned_id -= 1;
            let l_type = item.layout_type.unwrap_or(LayoutType::Text);
            blocks.push(BlockResult {
                region_id: rid,
                layout_type: l_type,
                frame: item.frame,
                box_points: item.box_points,
                score: item.score,
                lines: vec![item.clone()],
            });
        }
    }

    // Recompute average scores for blocks with multiple lines
    for block in &mut blocks {
        if !block.lines.is_empty() {
            let total_score: f32 = block.lines.iter().map(|l| l.score).sum();
            block.score = total_score / block.lines.len() as f32;
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_text_results_into_blocks_with_regions() {
        let line1 = TextResult {
            text: "Hello".to_string(),
            score: 0.90,
            box_points: [(10.0, 10.0), (50.0, 10.0), (50.0, 25.0), (10.0, 25.0)],
            frame: Frame { left: 10.0, top: 10.0, width: 40.0, height: 15.0 },
            region_id: Some(1),
            layout_type: Some(LayoutType::Title),
        };
        let line2 = TextResult {
            text: "World".to_string(),
            score: 0.80,
            box_points: [(10.0, 30.0), (60.0, 30.0), (60.0, 45.0), (10.0, 45.0)],
            frame: Frame { left: 10.0, top: 30.0, width: 50.0, height: 15.0 },
            region_id: Some(1),
            layout_type: Some(LayoutType::Title),
        };
        let line3 = TextResult {
            text: "Paragraph content".to_string(),
            score: 0.95,
            box_points: [(10.0, 60.0), (120.0, 60.0), (120.0, 80.0), (10.0, 80.0)],
            frame: Frame { left: 10.0, top: 60.0, width: 110.0, height: 20.0 },
            region_id: Some(2),
            layout_type: Some(LayoutType::Text),
        };

        let blocks = cluster_text_results_into_blocks(&[line1, line2, line3]);
        assert_eq!(blocks.len(), 2);

        // Block 1 check
        assert_eq!(blocks[0].region_id, 1);
        assert_eq!(blocks[0].layout_type, LayoutType::Title);
        assert_eq!(blocks[0].lines.len(), 2);
        assert_eq!(blocks[0].text(), "Hello\nWorld");
        assert_eq!(blocks[0].frame.left, 10.0);
        assert_eq!(blocks[0].frame.top, 10.0);
        assert_eq!(blocks[0].frame.width, 50.0);
        assert_eq!(blocks[0].frame.height, 35.0);
        assert!((blocks[0].score - 0.85).abs() < 1e-4);

        // Block 2 check
        assert_eq!(blocks[1].region_id, 2);
        assert_eq!(blocks[1].layout_type, LayoutType::Text);
        assert_eq!(blocks[1].lines.len(), 1);
        assert_eq!(blocks[1].text(), "Paragraph content");
    }

    #[test]
    fn test_cluster_text_results_into_blocks_unassigned_lines() {
        let line = TextResult {
            text: "Solo line".to_string(),
            score: 0.99,
            box_points: [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            frame: Frame { left: 0.0, top: 0.0, width: 10.0, height: 10.0 },
            region_id: None,
            layout_type: None,
        };

        let blocks = cluster_text_results_into_blocks(&[line]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].lines.len(), 1);
        assert_eq!(blocks[0].layout_type, LayoutType::Text);
        assert_eq!(blocks[0].text(), "Solo line");
    }
}

