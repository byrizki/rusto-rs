use crate::image::Point2f;
use crate::types::{Frame, TextResult};

#[derive(Debug, Clone)]
struct SpatialToken {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    text: String,
}

/// Convert detected bounding boxes and recognized texts into spatial formatted string.
/// Reconstructs the visual 2D page layout using whitespace indentation and blank lines.
pub fn to_spatial_text(
    boxes: &[[Point2f; 4]],
    texts: &[String],
    y_threshold_multiplier: Option<f32>,
    x_threshold_multiplier: Option<f32>,
) -> String {
    if boxes.is_empty() || texts.is_empty() {
        return String::new();
    }

    let y_mult = y_threshold_multiplier.unwrap_or(0.5);
    let x_mult = x_threshold_multiplier.unwrap_or(0.4);

    let mut tokens: Vec<SpatialToken> = boxes
        .iter()
        .zip(texts.iter())
        .map(|(bbox, text)| {
            let x = bbox[0].x;
            let y = bbox[0].y;
            let width = (bbox[1].x - bbox[0].x)
                .abs()
                .max((bbox[2].x - bbox[3].x).abs());
            let height = (bbox[3].y - bbox[0].y)
                .abs()
                .max((bbox[2].y - bbox[1].y).abs());
            SpatialToken {
                x,
                y,
                width,
                height,
                text: text.clone(),
            }
        })
        .collect();

    let median_height = if !tokens.is_empty() {
        let mut heights: Vec<f32> = tokens.iter().map(|t| t.height).collect();
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        heights[heights.len() / 2]
    } else {
        10.0
    };

    let avg_char_width = if !tokens.is_empty() {
        let total_chars: usize = tokens.iter().map(|t| t.text.len()).sum();
        let total_width: f32 = tokens.iter().map(|t| t.width).sum();
        if total_chars > 0 {
            total_width / total_chars as f32
        } else {
            median_height * 0.5
        }
    } else {
        10.0
    };

    let x_gap_threshold = avg_char_width * x_mult;
    let y_tolerance = median_height * y_mult;

    tokens.sort_by(|a, b| {
        a.y.partial_cmp(&b.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut lines: Vec<Vec<SpatialToken>> = Vec::new();
    let mut current_line: Vec<SpatialToken> = Vec::new();
    let mut current_line_y_sum: f32 = 0.0;

    for token in tokens {
        if !current_line.is_empty() {
            let current_line_avg_y = current_line_y_sum / current_line.len() as f32;

            if (token.y - current_line_avg_y).abs() <= y_tolerance {
                current_line.push(token.clone());
                current_line_y_sum += token.y;
            } else {
                lines.push(current_line);
                current_line = vec![token.clone()];
                current_line_y_sum = token.y;
            }
        } else {
            current_line = vec![token.clone()];
            current_line_y_sum = token.y;
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    let mut result = String::new();
    let mut prev_line_avg_y: Option<f32> = None;

    let min_x = if !lines.is_empty() {
        lines
            .iter()
            .flatten()
            .map(|t| t.x)
            .fold(f32::INFINITY, f32::min)
    } else {
        0.0
    };

    for line in lines {
        if line.is_empty() {
            continue;
        }

        let current_line_avg_y = line.iter().map(|t| t.y).sum::<f32>() / line.len() as f32;
        if let Some(prev_y) = prev_line_avg_y {
            let vertical_gap = current_line_avg_y - prev_y;
            if vertical_gap > median_height * 1.5 {
                let num_blank_lines =
                    ((vertical_gap - median_height) / median_height).round() as usize;
                for _ in 0..num_blank_lines.max(1) - 1 {
                    result.push('\n');
                }
            }
        }
        prev_line_avg_y = Some(current_line_avg_y);

        let mut line_sorted = line.clone();
        line_sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let mut current_char_pos: f32 = 0.0;
        let mut prev_token_end_x: f32 = min_x;

        for (i, token) in line_sorted.iter().enumerate() {
            let target_char_pos = (token.x - min_x) / avg_char_width;

            let spaces_needed = if i == 0 {
                target_char_pos.max(0.0)
            } else {
                let physical_gap = token.x - prev_token_end_x;
                if physical_gap < x_gap_threshold {
                    1.0
                } else {
                    target_char_pos - current_char_pos
                }
            };

            let spaces_to_insert = if i > 0 {
                spaces_needed.round().max(1.0) as usize
            } else {
                spaces_needed.round() as usize
            };

            for _ in 0..spaces_to_insert {
                result.push(' ');
            }

            result.push_str(&token.text);
            current_char_pos += spaces_to_insert as f32 + token.text.len() as f32;
            prev_token_end_x = token.x + token.width;
        }
        result.push('\n');
    }

    result
}

/// Groups candidates into rows based on vertical center tolerance.
pub fn group_candidates_into_lines(
    items: &[TextResult],
    line_y_threshold: f32,
) -> Vec<Vec<TextResult>> {
    let mut sorted = items.to_vec();
    sorted.sort_by(|a, b| {
        a.frame
            .top
            .partial_cmp(&b.frame.top)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.frame.left.partial_cmp(&b.frame.left).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut lines: Vec<Vec<TextResult>> = Vec::new();
    let mut current_line: Vec<TextResult> = Vec::new();
    let mut current_avg_y = 0.0f32;

    for item in sorted {
        let item_center_y = item.frame.top + item.frame.height / 2.0;
        if current_line.is_empty() {
            current_line = vec![item];
            current_avg_y = item_center_y;
        } else {
            let avg_y = current_avg_y / current_line.len() as f32;
            let avg_h = current_line.iter().map(|c| c.frame.height).sum::<f32>() / current_line.len() as f32;
            let tolerance = (avg_h * line_y_threshold).max(1.0);

            if (item_center_y - avg_y).abs() <= tolerance {
                current_line.push(item);
                current_avg_y += item_center_y;
            } else {
                lines.push(current_line);
                current_line = vec![item];
                current_avg_y = item_center_y;
            }
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}

/// Clusters line segments by merging adjacent word bounding boxes if horizontal gap is small.
pub fn cluster_line_segments(
    line: &[TextResult],
    word_x_threshold: f32,
) -> Vec<TextResult> {
    if line.is_empty() {
        return Vec::new();
    }

    let mut sorted = line.to_vec();
    sorted.sort_by(|a, b| a.frame.left.partial_cmp(&b.frame.left).unwrap_or(std::cmp::Ordering::Equal));

    let mut clusters: Vec<TextResult> = Vec::new();
    let mut current = sorted[0].clone();

    for next in sorted.into_iter().skip(1) {
        let avg_h = (current.frame.height + next.frame.height) / 2.0;
        let gap = next.frame.left - (current.frame.left + current.frame.width);
        let max_gap = avg_h * word_x_threshold;

        if gap >= 0.0 && gap <= max_gap {
            // Merge into current
            current.text.push(' ');
            current.text.push_str(&next.text);
            current.score = (current.score + next.score) / 2.0;

            let min_x = current.frame.left.min(next.frame.left);
            let min_y = current.frame.top.min(next.frame.top);
            let max_x = (current.frame.left + current.frame.width).max(next.frame.left + next.frame.width);
            let max_y = (current.frame.top + current.frame.height).max(next.frame.top + next.frame.height);

            current.frame = Frame {
                left: min_x,
                top: min_y,
                width: max_x - min_x,
                height: max_y - min_y,
            };
            current.box_points = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ];
        } else {
            clusters.push(current);
            current = next;
        }
    }
    clusters.push(current);

    clusters
}

/// Export OCR results as ASCII format.
/// Format: `[x,y] confidence% text`
pub fn to_raw(boxes: &[[Point2f; 4]], texts: &[String], scores: &[f32]) -> String {
    if boxes.is_empty() {
        return String::new();
    }

    let mut entries: Vec<(f32, f32, String, f32)> = boxes
        .iter()
        .zip(texts.iter())
        .zip(scores.iter())
        .map(|((bbox, text), &score)| {
            let x = bbox[0].x;
            let y = bbox[0].y;
            (y, x, text.clone(), score)
        })
        .collect();

    entries.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut result = String::new();
    for (y, x, text, score) in entries {
        result.push_str(&format!(
            "[{:.0},{:.0}] {:.2}% {}\n",
            x,
            y,
            score * 100.0,
            text
        ));
    }
    result
}

/// Export OCR results as TSV format (tab-separated values).
/// Format: `line_id\tcolumn_id\ttext`
pub fn to_tsv(boxes: &[[Point2f; 4]], texts: &[String]) -> String {
    if boxes.is_empty() {
        return "line_id\tcolumn_id\ttext\n".to_string();
    }

    #[derive(Debug, Clone)]
    struct Token {
        x: f32,
        y: f32,
        text: String,
    }

    let mut tokens: Vec<Token> = boxes
        .iter()
        .zip(texts.iter())
        .map(|(bbox, text)| Token {
            x: bbox[0].x,
            y: bbox[0].y,
            text: text.clone(),
        })
        .collect();

    tokens.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));

    let y_diffs: Vec<f32> = tokens
        .windows(2)
        .map(|w| w[1].y - w[0].y)
        .filter(|&d| d > 0.0)
        .collect();
    let typical_y_diff = if !y_diffs.is_empty() {
        let mut sorted = y_diffs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    } else {
        10.0
    };
    let y_tolerance = typical_y_diff * 0.6;

    let mut lines: Vec<Vec<Token>> = Vec::new();
    let mut current_line: Vec<Token> = Vec::new();
    let mut current_y: Option<f32> = None;

    for token in tokens {
        if let Some(cy) = current_y {
            if (token.y - cy).abs() <= y_tolerance {
                current_line.push(token.clone());
                current_y = Some(
                    (cy * (current_line.len() - 1) as f32 + token.y)
                        / current_line.len() as f32,
                );
            } else {
                lines.push(current_line.clone());
                current_line = vec![token.clone()];
                current_y = Some(token.y);
            }
        } else {
            current_line = vec![token.clone()];
            current_y = Some(token.y);
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    let mut tsv_rows = Vec::new();
    for (line_id, mut line) in lines.into_iter().enumerate() {
        line.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        if line.len() == 1 {
            tsv_rows.push((line_id, 0, line[0].text.clone()));
            continue;
        }

        let x_gaps: Vec<f32> = line.windows(2).map(|w| w[1].x - w[0].x).collect();
        let median_gap = if !x_gaps.is_empty() {
            let mut sorted = x_gaps.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2].max(1.0)
        } else {
            1.0
        };
        let gap_threshold = median_gap * 1.3;

        let mut columns: Vec<Vec<Token>> = Vec::new();
        let mut current_col = vec![line[0].clone()];
        for i in 1..line.len() {
            let gap = line[i].x - line[i - 1].x;
            if gap > gap_threshold {
                columns.push(current_col.clone());
                current_col = vec![line[i].clone()];
            } else {
                current_col.push(line[i].clone());
            }
        }
        columns.push(current_col);

        for (col_id, col) in columns.into_iter().enumerate() {
            let text = col
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            tsv_rows.push((line_id, col_id, text));
        }
    }

    let mut result = String::from("line_id\tcolumn_id\ttext\n");
    for (line_id, col_id, text) in tsv_rows {
        let safe_text = text.replace('\t', " ").replace('\n', " ");
        result.push_str(&format!("{}\t{}\t{}\n", line_id, col_id, safe_text));
    }
    result
}

/// Export results as plain text with position info.
pub fn to_text_with_position(boxes: &[[Point2f; 4]], texts: &[String], scores: &[f32]) -> String {
    let mut indexed: Vec<(&[Point2f; 4], &String, f32)> = boxes
        .iter()
        .zip(texts.iter())
        .zip(scores.iter())
        .map(|((b, t), &s)| (b, t, s))
        .collect();

    indexed.sort_by(|a, b| {
        let ay = (a.0[0].y + a.0[2].y) / 2.0;
        let by = (b.0[0].y + b.0[2].y) / 2.0;
        let ax = (a.0[0].x + a.0[2].x) / 2.0;
        let bx = (b.0[0].x + b.0[2].x) / 2.0;

        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut result = String::new();
    for (bbox, text, score) in &indexed {
        let x = (bbox[0].x + bbox[2].x) / 2.0;
        let y = (bbox[0].y + bbox[2].y) / 2.0;
        result.push_str(&format!(
            "[{:.0},{:.0}] {:.2}% {}\n",
            x,
            y,
            score * 100.0,
            text
        ));
    }

    result
}

pub struct AlignedWord {
    pub result: TextResult,
    /// Recognizer emitted whitespace before this entry. Spatial merging must
    /// never cross this boundary.
    pub whitespace_before: bool,
}

pub fn group_words(
    word_lines: Vec<Vec<TextResult>>,
    line_y_threshold: f32,
    word_x_threshold: f32,
) -> Vec<TextResult> {
    let mut words: Vec<AlignedWord> = word_lines
        .into_iter()
        .flat_map(|line| {
            line.into_iter()
                .enumerate()
                .map(|(index, result)| AlignedWord {
                    result,
                    whitespace_before: index > 0,
                })
        })
        .collect();
    if words.is_empty() {
        return Vec::new();
    }
    let mut heights: Vec<f32> = words.iter().map(|word| word.result.frame.height).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let line_tolerance = heights[heights.len() / 2] * line_y_threshold;
    words.sort_by(|a, b| {
        a.result
            .frame
            .top
            .partial_cmp(&b.result.frame.top)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.result
                    .frame
                    .left
                    .partial_cmp(&b.result.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let mut lines: Vec<Vec<AlignedWord>> = Vec::new();
    for word in words {
        if let Some(line) = lines.last_mut() {
            let center = line
                .iter()
                .map(|item| item.result.frame.top + item.result.frame.height / 2.0)
                .sum::<f32>()
                / line.len() as f32;
            if ((word.result.frame.top + word.result.frame.height / 2.0) - center).abs()
                <= line_tolerance
            {
                line.push(word);
                continue;
            }
        }
        lines.push(vec![word]);
    }
    lines
        .into_iter()
        .flat_map(|mut line| {
            line.sort_by(|a, b| {
                a.result
                    .frame
                    .left
                    .partial_cmp(&b.result.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut grouped: Vec<Vec<TextResult>> = Vec::new();
            for word in line {
                let char_width = (word.result.frame.width
                    / word.result.text.chars().count().max(1) as f32)
                    .max(1e-6);
                if !word.whitespace_before {
                    if let Some(current) = grouped.last_mut() {
                        let previous = current.last().expect("non-empty word group");
                        let gap =
                            word.result.frame.left - (previous.frame.left + previous.frame.width);
                        if gap <= char_width * word_x_threshold {
                            current.push(word.result);
                            continue;
                        }
                    }
                }
                grouped.push(vec![word.result]);
            }
            grouped
                .into_iter()
                .map(|group| merge_text_results(group, ""))
        })
        .collect()
}

pub fn merge_text_results(mut entries: Vec<TextResult>, separator: &str) -> TextResult {
    if entries.len() == 1 {
        return entries.remove(0);
    }
    entries.sort_by(|a, b| {
        a.frame
            .left
            .partial_cmp(&b.frame.left)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let text = entries
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>()
        .join(separator);
    let score = entries.iter().map(|item| item.score).fold(1.0, f32::min);
    let left = entries
        .iter()
        .map(|item| item.frame.left)
        .fold(f32::INFINITY, f32::min);
    let top = entries
        .iter()
        .map(|item| item.frame.top)
        .fold(f32::INFINITY, f32::min);
    let right = entries
        .iter()
        .map(|item| item.frame.left + item.frame.width)
        .fold(f32::NEG_INFINITY, f32::max);
    let bottom = entries
        .iter()
        .map(|item| item.frame.top + item.frame.height)
        .fold(f32::NEG_INFINITY, f32::max);
    let box_points = [(left, top), (right, top), (right, bottom), (left, bottom)];
    let region_id = entries.first().and_then(|e| e.region_id);
    let layout_type = entries.first().and_then(|e| e.layout_type);
    TextResult {
        text,
        score,
        box_points,
        frame: Frame::from_points(&box_points),
        region_id,
        layout_type,
    }
}

pub fn cluster_text_results_into_lines(
    mut entries: Vec<TextResult>,
    line_y_threshold: f32,
    word_x_threshold: f32,
) -> Vec<TextResult> {
    if entries.is_empty() {
        return entries;
    }
    let mut heights: Vec<f32> = entries.iter().map(|entry| entry.frame.height).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tolerance = heights[heights.len() / 2] * line_y_threshold;
    entries.sort_by(|a, b| {
        a.frame
            .top
            .partial_cmp(&b.frame.top)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.frame
                    .left
                    .partial_cmp(&b.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let mut lines: Vec<Vec<TextResult>> = Vec::new();
    for entry in entries {
        if let Some(line) = lines.last_mut() {
            let y = line
                .iter()
                .map(|item| item.frame.top + item.frame.height / 2.0)
                .sum::<f32>()
                / line.len() as f32;
            if ((entry.frame.top + entry.frame.height / 2.0) - y).abs() <= tolerance {
                line.push(entry);
                continue;
            }
        }
        lines.push(vec![entry]);
    }
    lines
        .into_iter()
        .flat_map(|mut line| {
            if line.len() == 1 {
                return line;
            }
            line.sort_by(|a, b| {
                a.frame
                    .left
                    .partial_cmp(&b.frame.left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut segments: Vec<TextResult> = Vec::new();
            for entry in line {
                if let Some(prev) = segments.last_mut() {
                    if prev.region_id.is_some()
                        && entry.region_id.is_some()
                        && prev.region_id != entry.region_id
                    {
                        segments.push(entry);
                        continue;
                    }
                    let prev_right = prev.frame.left + prev.frame.width;
                    let gap = entry.frame.left - prev_right;
                    let avg_height = (prev.frame.height + entry.frame.height) / 2.0;
                    let max_gap = (avg_height * word_x_threshold).max(1.0);
                    if gap >= 0.0 && gap <= max_gap {
                        prev.text.push(' ');
                        prev.text.push_str(&entry.text);
                        prev.score = prev.score.min(entry.score);
                        let left = prev.frame.left.min(entry.frame.left);
                        let top = prev.frame.top.min(entry.frame.top);
                        let right = prev_right.max(entry.frame.left + entry.frame.width);
                        let bottom = (prev.frame.top + prev.frame.height)
                            .max(entry.frame.top + entry.frame.height);
                        let box_points = [(left, top), (right, top), (right, bottom), (left, bottom)];
                        prev.box_points = box_points;
                        prev.frame = Frame::from_points(&box_points);
                        continue;
                    }
                }
                segments.push(entry);
            }
            segments
        })
        .collect()
}

pub fn build_text_results(
    boxes: &[[Point2f; 4]],
    texts: &[String],
    scores: &[f32],
) -> Vec<TextResult> {
    let count = boxes.len().min(texts.len()).min(scores.len());
    let mut results = Vec::with_capacity(count);
    for i in 0..count {
        let b = &boxes[i];
        let box_points = [(b[0].x, b[0].y), (b[1].x, b[1].y), (b[2].x, b[2].y), (b[3].x, b[3].y)];
        let frame = Frame::from_points(&box_points);
        results.push(TextResult {
            text: texts[i].clone(),
            score: scores[i],
            box_points,
            frame,
            region_id: None,
            layout_type: None,
        });
    }
    results
}

pub fn build_word_results(
    word_results: &[Vec<(String, f32, [Point2f; 4])>],
) -> Vec<Vec<TextResult>> {
    word_results
        .iter()
        .map(|words| {
            words
                .iter()
                .map(|(text, score, quad)| {
                    let box_points = [
                        (quad[0].x, quad[0].y),
                        (quad[1].x, quad[1].y),
                        (quad[2].x, quad[2].y),
                        (quad[3].x, quad[3].y),
                    ];
                    TextResult {
                        text: text.clone(),
                        score: *score,
                        box_points,
                        frame: Frame::from_points(&box_points),
                        region_id: None,
                        layout_type: None,
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_text_results_prevents_negative_gap_merge() {
        let word1 = TextResult::from_coords(10.0, 10.0, 50.0, 30.0, "Hello".to_string(), 0.95);
        let word2 = TextResult::from_coords(55.0, 10.0, 100.0, 30.0, "World".to_string(), 0.95);
        let overlapping = TextResult::from_coords(20.0, 12.0, 80.0, 32.0, "Overlap".to_string(), 0.90);

        let entries = vec![word1, overlapping, word2];
        let clustered = cluster_text_results_into_lines(entries, 0.5, 0.4);

        assert!(clustered.iter().any(|r| r.text == "Overlap"));
    }

    #[test]
    fn test_cluster_line_segments_prevents_negative_gap_merge() {
        let seg1 = TextResult::from_coords(10.0, 10.0, 50.0, 30.0, "Part1".to_string(), 0.95);
        let seg_overlap = TextResult::from_coords(20.0, 10.0, 60.0, 30.0, "Part2".to_string(), 0.90);

        let line = vec![seg1, seg_overlap];
        let clustered = cluster_line_segments(&line, 0.4);
        assert_eq!(clustered.len(), 2);
        assert_eq!(clustered[0].text, "Part1");
        assert_eq!(clustered[1].text, "Part2");
    }
}



