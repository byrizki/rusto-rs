//! Line-level post-processing for LineNet recognition: column-gutter cuts,
//! wide-gap splitting and leader / list-marker tidying.

use crate::engine::InferenceSession;
use crate::geometry::linenet_crop_x_span;
use crate::image::Point2f;
use crate::order::predict_reading_order;

const GUTTER_GAP: f32 = 1.6;
const GUTTER_EDGES: usize = 3;
const MARKER_GAP: f32 = 2.5;
const LINE_SPLIT_GAP: f32 = 3.0;
const GUTTER_EDGE_INSET: f32 = 0.35;

const LEADER_CHARS: &str = ".·…‥_-–—=";
const DASHES: &str = "-–—";
const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";

/// A recognized text line: axis-aligned box, text and the page x of every character.
#[derive(Clone, Debug)]
pub struct OcrLine {
    pub rect: [f32; 4],
    pub text: String,
    pub score: f32,
    pub xs: Vec<f32>,
}

impl OcrLine {
    fn height(&self) -> f32 {
        (self.rect[3] - self.rect[1]).max(1.0)
    }

    fn chars_with_x(&self) -> Vec<(char, f32)> {
        self.text.chars().zip(self.xs.iter().copied()).collect()
    }
}

fn is_space(c: char) -> bool {
    c == ' '
}

/// Per line, the x positions of gaps that sit on a column gutter: a gap wider than
/// `GUTTER_GAP` line heights where at least `GUTTER_EDGES` other lines end at its
/// left side or start at its right side.
pub fn gutter_cuts(lines: &[OcrLine]) -> Vec<Vec<f32>> {
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let h = line.height();
            let xs: Vec<f32> = line
                .chars_with_x()
                .into_iter()
                .filter(|&(c, _)| !is_space(c))
                .map(|(_, x)| x)
                .collect();
            let mut cuts = Vec::new();
            for pair in xs.windows(2) {
                let (a, c) = (pair[0], pair[1]);
                if c - a <= GUTTER_GAP * h {
                    continue;
                }
                let gl = a + GUTTER_EDGE_INSET * h;
                let gr = c - GUTTER_EDGE_INSET * h;
                let near = lines
                    .iter()
                    .enumerate()
                    .filter(|&(j, other)| {
                        j != i
                            && ((other.rect[2] - gl).abs() < 0.5 * h
                                || (other.rect[0] - gr).abs() < 0.5 * h)
                    })
                    .count();
                if near >= GUTTER_EDGES {
                    cuts.push((a + c) / 2.0);
                }
            }
            cuts
        })
        .collect()
}

fn segment_text(segment: &[(char, f32)]) -> String {
    segment.iter().map(|&(c, _)| c).collect::<String>().trim().to_string()
}

/// Cuts a line where two neighbouring characters are more than `gap` line heights
/// apart or straddle one of the `at` positions. Returns the line itself when nothing is cut.
pub fn split_line(line: &OcrLine, gap: f32, at: &[f32]) -> Vec<OcrLine> {
    let h = line.height();
    let mut segments: Vec<Vec<(char, f32)>> = vec![Vec::new()];
    let mut prev: Option<f32> = None;
    for (c, x) in line.chars_with_x() {
        if !is_space(c) {
            let cut = prev.is_some_and(|p| x - p > gap * h || at.iter().any(|&a| p < a && a < x));
            if cut {
                segments.push(Vec::new());
            }
            prev = Some(x);
        }
        segments.last_mut().expect("segments is never empty").push((c, x));
    }
    if segments.len() == 1 {
        return vec![line.clone()];
    }

    segments
        .into_iter()
        .filter_map(|segment| {
            let text = segment_text(&segment);
            let xs: Vec<f32> = segment.iter().filter(|&&(c, _)| !is_space(c)).map(|&(_, x)| x).collect();
            if text.is_empty() || xs.is_empty() {
                return None;
            }
            let min_x = xs.iter().copied().fold(f32::INFINITY, f32::min);
            let max_x = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let trimmed_start = segment.iter().take_while(|&&(c, _)| is_space(c)).count();
            let trimmed_len = text.chars().count();
            let positions = segment
                .iter()
                .skip(trimmed_start)
                .take(trimmed_len)
                .map(|&(_, x)| x)
                .collect();
            Some(OcrLine {
                rect: [
                    line.rect[0].max(min_x - GUTTER_EDGE_INSET * h),
                    line.rect[1],
                    line.rect[2].min(max_x + GUTTER_EDGE_INSET * h),
                    line.rect[3],
                ],
                text,
                score: line.score,
                xs: positions,
            })
        })
        .collect()
}

fn is_enumerator(t: &str) -> bool {
    let chars: Vec<char> = t.chars().collect();
    let n = chars.len();
    if n < 2 {
        return false;
    }
    let is_label = |s: &[char]| {
        (s.len() == 1 && s[0].is_ascii_alphabetic())
            || (!s.is_empty() && s.len() <= 3 && s.iter().all(|c| c.is_ascii_digit()))
    };
    if matches!(chars[n - 1], '.' | ')') && is_label(&chars[..n - 1]) {
        return true;
    }
    if chars[0] == '(' && chars[n - 1] == ')' {
        let inner = &chars[1..n - 1];
        let roman = !inner.is_empty() && inner.len() <= 5 && inner.iter().all(|c| "ivxlcIVXLC".contains(*c));
        return is_label(inner) || roman;
    }
    false
}

fn is_leader_only(line: &OcrLine) -> bool {
    let core: Vec<char> = line.text.chars().filter(|&c| c != ' ').collect();
    if core.is_empty() || !core.iter().all(|c| LEADER_CHARS.contains(*c)) {
        return false;
    }
    let wide = (line.rect[2] - line.rect[0]) >= 3.0 * (line.rect[3] - line.rect[1]);
    core.len() >= 3 || wide || !core.iter().any(|c| DASHES.contains(*c))
}

/// Drops leader-only lines (dot leaders, rules, specks) and joins a lone list marker
/// to the line that starts within `MARKER_GAP` line heights after it on the same row.
pub fn tidy_lines(mut lines: Vec<OcrLine>) -> Vec<OcrLine> {
    let n = lines.len();
    let mut keep: Vec<bool> = lines.iter().map(|l| !is_leader_only(l)).collect();

    for i in 0..n {
        let t = lines[i].text.trim().to_string();
        let is_marker = (t.chars().count() == 1 && BULLETS.contains(t.as_str())) || is_enumerator(&t);
        if !keep[i] || !is_marker {
            continue;
        }
        let ri = lines[i].rect;
        let hi = ri[3] - ri[1];
        let mut best: Option<(usize, f32)> = None;
        for j in 0..n {
            if j == i || !keep[j] || lines[j].text.trim().is_empty() {
                continue;
            }
            let rj = lines[j].rect;
            let hj = rj[3] - rj[1];
            let overlap = ri[3].min(rj[3]) - ri[1].max(rj[1]);
            if overlap < 0.5 * hi.min(hj) {
                continue;
            }
            let gap = rj[0] - ri[2];
            if gap >= -0.3 * hi
                && gap <= MARKER_GAP * hi.max(hj)
                && best.is_none_or(|(_, best_gap)| gap < best_gap)
            {
                best = Some((j, gap));
            }
        }
        let Some((j, _)) = best else { continue };

        let marker = lines[i].clone();
        let target = &mut lines[j];
        let marker_xs = marker.xs.clone();
        let split_x = (marker.rect[2] + target.rect[0]) / 2.0;
        let mut xs = marker_xs;
        if !xs.is_empty() && !target.xs.is_empty() {
            xs.push(split_x);
            xs.extend(target.xs.iter().copied());
            target.xs = xs;
        }
        target.text = format!("{} {}", t, target.text.trim());
        target.rect = [
            marker.rect[0].min(target.rect[0]),
            marker.rect[1].min(target.rect[1]),
            marker.rect[2].max(target.rect[2]),
            marker.rect[3].max(target.rect[3]),
        ];
        keep[i] = false;
    }

    lines
        .into_iter()
        .zip(keep)
        .filter_map(|(line, keep)| keep.then_some(line))
        .collect()
}

/// Axis-aligned `[x0, y0, x1, y1]` bounds of a detection quad.
pub fn quad_to_rect(b: &[Point2f; 4]) -> [f32; 4] {
    [
        b[0].x.min(b[3].x),
        b[0].y.min(b[1].y),
        b[1].x.max(b[2].x),
        b[2].y.max(b[3].y),
    ]
}

/// Quad `[top-left, top-right, bottom-right, bottom-left]` of an axis-aligned rectangle.
pub fn rect_to_quad(r: [f32; 4]) -> [Point2f; 4] {
    [
        Point2f::new(r[0], r[1]),
        Point2f::new(r[2], r[1]),
        Point2f::new(r[2], r[3]),
        Point2f::new(r[0], r[3]),
    ]
}

/// Builds page-space lines from recognition results. A line under `min_score` keeps its
/// box (layout and reading order) but loses its text.
pub fn build_lines(
    boxes: &[[Point2f; 4]],
    texts: &[(String, f32)],
    fractional_positions: &[Vec<f32>],
    pad_x_factor: f32,
    min_score: f32,
) -> Vec<OcrLine> {
    boxes
        .iter()
        .zip(texts)
        .zip(fractional_positions)
        .map(|((quad, (text, score)), fractions)| {
            let (ix0, cw) = linenet_crop_x_span(quad, pad_x_factor);
            let readable = *score >= min_score && !text.is_empty();
            OcrLine {
                rect: quad_to_rect(quad),
                text: if readable { text.clone() } else { String::new() },
                score: *score,
                xs: if readable {
                    fractions.iter().map(|f| ix0 + f * cw).collect()
                } else {
                    Vec::new()
                },
            }
        })
        .collect()
}

/// Reorders lines with the neural reading order model; returns `None` when it fails.
pub fn order_lines(
    session: &mut dyn InferenceSession,
    lines: Vec<OcrLine>,
    page_w: f32,
    page_h: f32,
) -> (Vec<OcrLine>, bool) {
    if lines.is_empty() {
        return (lines, false);
    }
    let rects: Vec<[f32; 4]> = lines.iter().map(|l| l.rect).collect();
    let texts: Vec<String> = lines.iter().map(|l| l.text.clone()).collect();
    match predict_reading_order(session, &rects, page_w, page_h, &texts) {
        Ok(order) if order.len() == lines.len() => {
            let ordered = order.into_iter().map(|i| lines[i].clone()).collect();
            (ordered, true)
        }
        _ => (lines, false),
    }
}

/// Applies gutter cuts, wide-gap splitting and tidying to recognized lines.
pub fn refine_lines(lines: Vec<OcrLine>) -> Vec<OcrLine> {
    let cuts = gutter_cuts(&lines);
    let split: Vec<OcrLine> = lines
        .iter()
        .zip(cuts.iter())
        .flat_map(|(line, at)| split_line(line, LINE_SPLIT_GAP, at))
        .collect();
    tidy_lines(split)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(rect: [f32; 4], text: &str, x_start: f32, step: f32) -> OcrLine {
        let xs = (0..text.chars().count()).map(|i| x_start + i as f32 * step).collect();
        OcrLine { rect, text: text.to_string(), score: 0.9, xs }
    }

    #[test]
    fn test_gutter_cut_requires_aligned_neighbours() {
        let mut spanning = line([50.0, 100.0, 750.0, 120.0], "abc de", 0.0, 0.0);
        spanning.xs = vec![50.0, 60.0, 70.0, 75.0, 700.0, 710.0];
        let mut lines = vec![spanning];

        assert!(gutter_cuts(&lines)[0].is_empty());

        for k in 0..3 {
            let y = 140.0 + k as f32 * 30.0;
            lines.push(line([50.0, y, 80.0, y + 20.0], "xx", 50.0, 10.0));
        }
        let cuts = gutter_cuts(&lines);
        assert_eq!(cuts[0].len(), 1);
        assert!((cuts[0][0] - 385.0).abs() < 1.0);
        assert!(cuts[1].is_empty());
    }

    #[test]
    fn test_split_line_at_wide_gap() {
        let mut l = line([0.0, 0.0, 400.0, 10.0], "aa bb", 0.0, 10.0);
        l.xs = vec![0.0, 10.0, 20.0, 200.0, 210.0];
        let parts = split_line(&l, 3.0, &[]);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].text, "aa");
        assert_eq!(parts[1].text, "bb");
        assert_eq!(parts[1].xs, vec![200.0, 210.0]);
        assert!(parts[0].rect[2] <= parts[1].rect[0]);
    }

    #[test]
    fn test_split_line_keeps_close_text() {
        let l = line([0.0, 0.0, 100.0, 10.0], "ab cd", 0.0, 10.0);
        let parts = split_line(&l, 3.0, &[]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].text, "ab cd");
    }

    #[test]
    fn test_tidy_drops_leaders_and_merges_markers() {
        let lines = vec![
            line([0.0, 0.0, 200.0, 10.0], ".........", 0.0, 10.0),
            line([0.0, 20.0, 8.0, 30.0], "1.", 0.0, 4.0),
            line([14.0, 20.0, 100.0, 30.0], "Item text", 14.0, 8.0),
            line([0.0, 40.0, 6.0, 50.0], "-", 0.0, 4.0),
        ];
        let tidy = tidy_lines(lines);
        assert_eq!(tidy.len(), 2);
        assert_eq!(tidy[0].text, "1. Item text");
        assert_eq!(tidy[0].rect[0], 0.0);
        assert_eq!(tidy[1].text, "-");
    }

    #[test]
    fn test_enumerator_patterns() {
        for t in ["1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)"] {
            assert!(is_enumerator(t), "{t}");
        }
        for t in ["1", "abc.", "1234.", "()"] {
            assert!(!is_enumerator(t), "{t}");
        }
    }
}
