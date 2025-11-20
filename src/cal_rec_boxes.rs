#[cfg(feature = "use-opencv")]
use opencv::core::Point2f;
#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Point2f;

use crate::rec::{TextRecOutput, WordInfo, WordType};

pub struct CalRecBoxes;

impl CalRecBoxes {
    pub fn new() -> Self {
        Self
    }

    pub fn calc_word_boxes(
        &self,
        dt_boxes: &[[Point2f; 4]],
        rec_res: &TextRecOutput,
        return_single_char_box: bool,
    ) -> Vec<Vec<(String, f32, [Point2f; 4])>> {
        let mut all_word_results: Vec<Vec<(String, f32, [Point2f; 4])>> = Vec::new();

        let num = dt_boxes.len()
            .min(rec_res.txts.len())
            .min(rec_res.word_infos.len());

        for idx in 0..num {
            let txt = &rec_res.txts[idx];
            let word_info_opt = &rec_res.word_infos[idx];

            if txt.is_empty() {
                all_word_results.push(Vec::new());
                continue;
            }

            let Some(word_info) = word_info_opt else {
                all_word_results.push(Vec::new());
                continue;
            };

            if word_info.line_txt_len <= 0.0 {
                all_word_results.push(Vec::new());
                continue;
            }

            let box_pts = &dt_boxes[idx];
            let bbox_rect = quads_to_rect_bbox(box_pts);

            let (word_contents, rects, confs) = self.cal_ocr_word_box(
                txt,
                bbox_rect,
                word_info,
                return_single_char_box,
            );

            let (bx0, by0, bx1, by1) = bbox_rect;
            let mut line_results: Vec<(String, f32, [Point2f; 4])> = Vec::new();

            let n = word_contents
                .len()
                .min(rects.len())
                .min(confs.len());

            for i in 0..n {
                let (x0, y0, x1, y1) = rects[i];
                let quad = rect_to_quad_in_box(box_pts, (bx0, by0, bx1, by1), (x0, y0, x1, y1));
                line_results.push((word_contents[i].clone(), confs[i], quad));
            }

            all_word_results.push(line_results);
        }

        all_word_results
    }

    fn cal_ocr_word_box(
        &self,
        rec_txt: &str,
        bbox_points: (f32, f32, f32, f32),
        word_info: &WordInfo,
        return_single_char_box: bool,
    ) -> (Vec<String>, Vec<(f32, f32, f32, f32)>, Vec<f32>) {
        let (x0, y0, x1, y1) = bbox_points;

        if rec_txt.is_empty() || word_info.line_txt_len == 0.0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let avg_col_width = (x1 - x0) / word_info.line_txt_len.max(1e-6);

        let is_all_en_num = word_info
            .word_types
            .iter()
            .all(|t| *t == WordType::EnNum);

        let mut line_cols: Vec<Vec<usize>> = Vec::new();
        let mut flat_cols: Vec<usize> = Vec::new();
        let mut char_widths: Vec<f32> = Vec::new();
        let mut word_contents: Vec<String> = Vec::new();

        for (word, word_col) in word_info.words.iter().zip(word_info.word_cols.iter()) {
            if is_all_en_num && !return_single_char_box {
                line_cols.push(word_col.clone());
                let s: String = word.iter().cloned().collect();
                word_contents.push(s);
            } else {
                flat_cols.extend(word_col.iter().copied());
                for ch in word.iter() {
                    word_contents.push(ch.clone());
                }
            }

            if word_col.len() == 1 {
                continue;
            }

            let avg_width = calc_avg_char_width(word_col, avg_col_width);
            char_widths.push(avg_width);
        }

        let txt_len = rec_txt.chars().count();
        let avg_char_width = calc_all_char_avg_width(&char_widths, x0, x1, txt_len);

        let rects: Vec<(f32, f32, f32, f32)> = if is_all_en_num && !return_single_char_box {
            calc_en_num_box(&line_cols, avg_char_width, avg_col_width, (x0, y0, x1, y1))
        } else {
            calc_box(&flat_cols, avg_char_width, avg_col_width, (x0, y0, x1, y1))
        };

        let confs = word_info.confs.clone();
        (word_contents, rects, confs)
    }
}

fn quads_to_rect_bbox(quad: &[Point2f; 4]) -> (f32, f32, f32, f32) {
    let mut xs = [quad[0].x, quad[1].x, quad[2].x, quad[3].x];
    let mut ys = [quad[0].y, quad[1].y, quad[2].y, quad[3].y];
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let x_min = xs[0];
    let x_max = xs[3];
    let y_min = ys[0];
    let y_max = ys[3];
    (x_min, y_min, x_max, y_max)
}

fn rect_to_quad_in_box(
    box_pts: &[Point2f; 4],
    bbox_rect: (f32, f32, f32, f32),
    rect: (f32, f32, f32, f32),
) -> [Point2f; 4] {
    let (bx0, by0, bx1, by1) = bbox_rect;
    let (x0, y0, x1, y1) = rect;

    let width = (bx1 - bx0).abs().max(1e-6);
    let height = (by1 - by0).abs().max(1e-6);

    let u0 = (x0 - bx0) / width;
    let u1 = (x1 - bx0) / width;
    let v0 = (y0 - by0) / height;
    let v1 = (y1 - by0) / height;

    let p0 = box_pts[0];
    let p1 = box_pts[1];
    let p2 = box_pts[2];
    let p3 = box_pts[3];

    fn lerp(a: Point2f, b: Point2f, t: f32) -> Point2f {
        Point2f::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
    }

    fn bilinear(
        p0: Point2f,
        p1: Point2f,
        p2: Point2f,
        p3: Point2f,
        u: f32,
        v: f32,
    ) -> Point2f {
        let top = lerp(p0, p1, u);
        let bottom = lerp(p3, p2, u);
        lerp(top, bottom, v)
    }

    let q0 = bilinear(p0, p1, p2, p3, u0, v0);
    let q1 = bilinear(p0, p1, p2, p3, u1, v0);
    let q2 = bilinear(p0, p1, p2, p3, u1, v1);
    let q3 = bilinear(p0, p1, p2, p3, u0, v1);

    [q0, q1, q2, q3]
}

fn calc_box(
    line_cols: &[usize],
    avg_char_width: f32,
    avg_col_width: f32,
    bbox_points: (f32, f32, f32, f32),
) -> Vec<(f32, f32, f32, f32)> {
    let (x0, y0, x1, y1) = bbox_points;
    let mut results = Vec::new();

    for &col_idx in line_cols {
        let center_x = (col_idx as f32 + 0.5) * avg_col_width;
        let mut char_x0 = center_x - avg_char_width / 2.0;
        let mut char_x1 = center_x + avg_char_width / 2.0;

        if char_x0 < 0.0 {
            char_x0 = 0.0;
        }
        if char_x1 > (x1 - x0) {
            char_x1 = x1 - x0;
        }

        let rx0 = char_x0 + x0;
        let rx1 = char_x1 + x0;

        results.push((rx0, y0, rx1, y1));
    }

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn calc_en_num_box(
    line_cols: &[Vec<usize>],
    avg_char_width: f32,
    avg_col_width: f32,
    bbox_points: (f32, f32, f32, f32),
) -> Vec<(f32, f32, f32, f32)> {
    let mut results = Vec::new();

    for one_col in line_cols {
        let cells = calc_box(one_col, avg_char_width, avg_col_width, bbox_points);
        if cells.is_empty() {
            continue;
        }

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (x0, y0, x1, y1) in &cells {
            xs.push(*x0);
            xs.push(*x1);
            ys.push(*y0);
            ys.push(*y1);
        }

        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let rx0 = xs.first().copied().unwrap_or(0.0);
        let rx1 = xs.last().copied().unwrap_or(rx0);
        let ry0 = ys.first().copied().unwrap_or(0.0);
        let ry1 = ys.last().copied().unwrap_or(ry0);

        results.push((rx0, ry0, rx1, ry1));
    }

    results
}

fn calc_avg_char_width(word_col: &[usize], each_col_width: f32) -> f32 {
    if word_col.len() <= 1 {
        return each_col_width;
    }

    let char_total_length = (word_col[word_col.len() - 1] - word_col[0]) as f32 * each_col_width;
    char_total_length / (word_col.len() as f32 - 1.0)
}

fn calc_all_char_avg_width(
    width_list: &[f32],
    bbox_x0: f32,
    bbox_x1: f32,
    txt_len: usize,
) -> f32 {
    if txt_len == 0 {
        return 0.0;
    }

    if !width_list.is_empty() {
        let sum: f32 = width_list.iter().copied().sum();
        return sum / (width_list.len() as f32);
    }

    (bbox_x1 - bbox_x0) / (txt_len as f32)
}
