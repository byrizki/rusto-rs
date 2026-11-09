use crate::tensor::Array4;
use crate::error::CoreError;
use crate::image::{self, Mat, Point2f};
use crate::contours;

pub struct TextDetOutput {
    pub img: Option<Mat>,
    pub boxes: Option<Vec<[Point2f; 4]>>,
    pub scores: Option<Vec<f32>>,
    pub elapse: f64,
}

impl TextDetOutput {
    pub fn empty() -> Self {
        Self {
            img: None,
            boxes: None,
            scores: None,
            elapse: 0.0,
        }
    }

    pub fn new(img: Mat, boxes: Vec<[Point2f; 4]>, scores: Vec<f32>, elapse: f64) -> Self {
        Self {
            img: Some(img),
            boxes: Some(boxes),
            scores: Some(scores),
            elapse,
        }
    }

    pub fn len(&self) -> usize {
        self.boxes.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct DBPostProcess {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f64,
    pub min_size: f32,
    pub use_dilation: bool,
    pub score_mode: String,
}

impl DBPostProcess {
    pub fn new(
        thresh: f32,
        box_thresh: f32,
        max_candidates: i32,
        unclip_ratio: f32,
        use_dilation: bool,
        score_mode: impl Into<String>,
    ) -> Self {
        Self {
            thresh,
            box_thresh,
            max_candidates: max_candidates as usize,
            unclip_ratio: unclip_ratio as f64,
            min_size: 3.0,
            use_dilation,
            score_mode: score_mode.into(),
        }
    }

    pub fn process(
        &self,
        pred: &Array4<f32>,
        ori_h: i32,
        ori_w: i32,
    ) -> Result<(Vec<[Point2f; 4]>, Vec<f32>), CoreError> {
        let (_, _, h, w) = pred.dim();
        if h == 0 || w == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut mask_img = ::image::GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let v = pred[[0, 0, y, x]];
                let val: u8 = if v > self.thresh { 255 } else { 0 };
                mask_img.put_pixel(x as u32, y as u32, ::image::Luma([val]));
            }
        }

        let mask_for_contours = if self.use_dilation {
            dilate_2x2(&mask_img)
        } else {
            mask_img
        };

        let mut contours = if self.score_mode == "slow" {
            contours::find_contours_with_pred(&mask_for_contours, Some(pred))
        } else {
            contours::find_contours(&mask_for_contours)
        };

        contours.sort_by(|a, b| b.area.cmp(&a.area));

        let num_contours = contours.len().min(self.max_candidates);
        let mut boxes = Vec::with_capacity(num_contours.min(100));
        let mut scores = Vec::with_capacity(num_contours.min(100));

        for contour in contours.iter().take(num_contours) {
            if contour.len() < 4 {
                continue;
            }

            let mut points = Vec::with_capacity(contour.points.len());
            for &(x, y) in &contour.points {
                points.push(Point2f::new(x as f32, y as f32));
            }

            let (center, size, angle) = match image::min_area_rect(&points) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let rect_points = image::box_points(center, size, angle);
            let side_len = size.width.min(size.height) as f32;

            if side_len < 3.0 {
                continue;
            }

            let score = if self.score_mode == "slow" {
                contour.score
            } else {
                self.box_score_fast(pred, &rect_points, w, h)?
            };
            if score < self.box_thresh {
                continue;
            }

            let unclipped = self.unclip_pure(&rect_points)?;
            if unclipped.len() < 4 {
                continue;
            }

            let (center2, size2, angle2) = match image::min_area_rect(&unclipped) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let box_pts_raw = image::box_points(center2, size2, angle2);

            let mut sorted_pts2 = box_pts_raw.to_vec();
            sorted_pts2.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

            let (mut tl2, mut bl2) = (sorted_pts2[0], sorted_pts2[1]);
            if bl2.y < tl2.y {
                std::mem::swap(&mut tl2, &mut bl2);
            }

            let (mut tr2, mut br2) = (sorted_pts2[2], sorted_pts2[3]);
            if br2.y < tr2.y {
                std::mem::swap(&mut tr2, &mut br2);
            }

            let box_pts = [tl2, tr2, br2, bl2];
            let sside = size2.width.min(size2.height).abs() as f32;
            if sside < 3.0 {
                continue;
            }

            let mut final_box = [Point2f::default(); 4];
            for (i, pt) in box_pts.iter().enumerate() {
                final_box[i] = Point2f::new(
                    pt.x * (ori_w as f32 / w as f32),
                    pt.y * (ori_h as f32 / h as f32),
                );
            }

            final_box = self.order_points_clockwise_pure(final_box);
            final_box = self.clip_det_res_pure(final_box, ori_h, ori_w);

            let rect_width = ((final_box[0].x - final_box[1].x).powi(2)
                + (final_box[0].y - final_box[1].y).powi(2))
            .sqrt() as i32;
            let rect_height = ((final_box[0].x - final_box[3].x).powi(2)
                + (final_box[0].y - final_box[3].y).powi(2))
            .sqrt() as i32;

            if rect_width <= 3 || rect_height <= 3 {
                continue;
            }

            boxes.push(final_box);
            scores.push(score);
        }

        Ok((boxes, scores))
    }

    pub fn process_raw_slice(
        &self,
        pred_slice: &[f32],
        h: usize,
        w: usize,
        ori_h: i32,
        ori_w: i32,
    ) -> Result<(Vec<[Point2f; 4]>, Vec<f32>), CoreError> {
        if h == 0 || w == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let pred_arr = crate::tensor::Tensor::from_slice([1, 1, h, w], pred_slice);

        self.process(&pred_arr, ori_h, ori_w)
    }

    fn box_score_fast(
        &self,
        pred: &Array4<f32>,
        box_pts: &[Point2f; 4],
        w: usize,
        h: usize,
    ) -> Result<f32, CoreError> {
        let xmin = box_pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min).floor() as i32;
        let xmax = box_pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max).ceil() as i32;
        let ymin = box_pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min).floor() as i32;
        let ymax = box_pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max).ceil() as i32;

        let xmin = xmin.max(0).min(w as i32 - 1);
        let xmax = xmax.max(0).min(w as i32 - 1);
        let ymin = ymin.max(0).min(h as i32 - 1);
        let ymax = ymax.max(0).min(h as i32 - 1);

        if xmin >= xmax || ymin >= ymax {
            return Ok(0.0);
        }

        let mut sum = 0.0f32;
        let mut count = 0;

        for y in ymin..=ymax {
            for x in xmin..=xmax {
                if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, box_pts) {
                    sum += pred[[0, 0, y as usize, x as usize]];
                    count += 1;
                }
            }
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(sum / count as f32)
        }
    }

    fn unclip_pure(&self, box_pts: &[Point2f; 4]) -> Result<Vec<Point2f>, CoreError> {
        Ok(unclip_polygon_analytic(box_pts, self.unclip_ratio))
    }

    fn order_points_clockwise_pure(&self, pts: [Point2f; 4]) -> [Point2f; 4] {
        let mut pts_vec: Vec<Point2f> = pts.to_vec();
        pts_vec.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let (mut tl, mut bl) = (pts_vec[0], pts_vec[1]);
        if tl.y > bl.y {
            std::mem::swap(&mut tl, &mut bl);
        }

        let (mut tr, mut br) = (pts_vec[2], pts_vec[3]);
        if tr.y > br.y {
            std::mem::swap(&mut tr, &mut br);
        }

        [tl, tr, br, bl]
    }

    fn clip_det_res_pure(
        &self,
        mut points: [Point2f; 4],
        img_height: i32,
        img_width: i32,
    ) -> [Point2f; 4] {
        for p in &mut points {
            p.x = p.x.max(0.0).min((img_width - 1) as f32);
            p.y = p.y.max(0.0).min((img_height - 1) as f32);
        }
        points
    }
}

pub fn decode_linenet_det_boxes(
    preds: &Array4<f32>,
    orig_w: f32,
    orig_h: f32,
    scale_w: f32,
    scale_h: f32,
    det_thresh: f32,
    box_thresh: f32,
) -> (Vec<[Point2f; 4]>, Vec<f32>) {
    let (_, _, out_h, out_w) = preds.dim();
    let plane = out_h * out_w;

    let mut prob = vec![0.0f32; plane];
    for y in 0..out_h {
        for x in 0..out_w {
            let logit = preds[[0, 0, y, x]];
            prob[y * out_w + x] = 1.0 / (1.0 + (-logit).exp());
        }
    }

    let mut visited = vec![false; plane];
    let mut boxes = Vec::new();
    let mut scores = Vec::new();

    for y in 0..out_h {
        for x in 0..out_w {
            let idx = y * out_w + x;
            if visited[idx] || prob[idx] < det_thresh {
                continue;
            }

            let mut queue = std::collections::VecDeque::new();
            queue.push_back((x, y));
            visited[idx] = true;

            let ys0 = (y as f32 + 0.5) * 2.0;
            let mut sum_top = ys0 - preds[[0, 1, y, x]];
            let mut sum_bot = ys0 + preds[[0, 2, y, x]];
            let mut sum_prob = prob[idx];
            let mut count = 1usize;
            let mut min_x = x;
            let mut max_x = x;

            while let Some((cx, cy)) = queue.pop_front() {
                if cx < min_x { min_x = cx; }
                if cx > max_x { max_x = cx; }

                for (nx, ny) in [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ] {
                    if nx < out_w && ny < out_h {
                        let nidx = ny * out_w + nx;
                        if !visited[nidx] && prob[nidx] > det_thresh {
                            let nys = (ny as f32 + 0.5) * 2.0;
                            let n_y0 = nys - preds[[0, 1, ny, nx]];
                            let n_y1 = nys + preds[[0, 2, ny, nx]];

                            visited[nidx] = true;
                            sum_top += n_y0;
                            sum_bot += n_y1;
                            sum_prob += prob[nidx];
                            count += 1;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            if count < 2 { continue; }
            let avg_prob = sum_prob / count as f32;
            if avg_prob < box_thresh { continue; }

            let y0_in = sum_top / count as f32;
            let y1_in = sum_bot / count as f32;
            let lh = (y1_in - y0_in).max(1.0);
            let ex = (0.15 * lh).min(0.5 * lh);
            let x0_in = (min_x as f32 * 2.0) - ex;
            let x1_in = ((max_x + 1) as f32 * 2.0) + ex;

            let x0 = (x0_in / scale_w).clamp(0.0, orig_w);
            let y0 = (y0_in / scale_h).clamp(0.0, orig_h);
            let x1 = (x1_in / scale_w).clamp(0.0, orig_w);
            let y1 = (y1_in / scale_h).clamp(0.0, orig_h);

            if x1 > x0 && y1 > y0 {
                let box_quad = [
                    Point2f::new(x0, y0),
                    Point2f::new(x1, y0),
                    Point2f::new(x1, y1),
                    Point2f::new(x0, y1),
                ];
                boxes.push(box_quad);
                scores.push(avg_prob);
            }
        }
    }

    (boxes, scores)
}

fn dilate_2x2(img: &::image::GrayImage) -> ::image::GrayImage {
    let (width, height) = img.dimensions();
    let mut result = ::image::GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;
            for dy in 0..=1 {
                for dx in 0..=1 {
                    let nx = ((x as i32 + dx).min(width as i32 - 1)) as u32;
                    let ny = ((y as i32 + dy).min(height as i32 - 1)) as u32;
                    max_val = max_val.max(img.get_pixel(nx, ny)[0]);
                }
            }
            result.put_pixel(x, y, ::image::Luma([max_val]));
        }
    }

    result
}

fn point_in_polygon(x: f32, y: f32, polygon: &[Point2f]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;

    for i in 0..polygon.len() {
        let xi = polygon[i].x;
        let yi = polygon[i].y;
        let xj = polygon[j].x;
        let yj = polygon[j].y;

        let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }

    inside
}

pub fn sort_det_boxes(dt_boxes: &mut [[Point2f; 4]]) {
    dt_boxes.sort_by(|a, b| {
        let ay = a[0].y as i32;
        let by = b[0].y as i32;
        if ay != by {
            ay.cmp(&by)
        } else {
            let ax = a[0].x as i32;
            let bx = b[0].x as i32;
            ax.cmp(&bx)
        }
    });
}

/// Fast BFS connected component box extraction for DBNet heatmap
pub fn decode_dbnet_boxes_fast(
    heat_map: &[f32],
    target_w: usize,
    target_h: usize,
    orig_w: usize,
    orig_h: usize,
    scale_w: f32,
    scale_h: f32,
    det_thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
) -> Vec<([f32; 4], f32)> {
    let mut raw_boxes = Vec::new();
    let mut visited = vec![false; target_w * target_h];
    let mut queue = std::collections::VecDeque::with_capacity(256);

    for y in 0..target_h {
        for x in 0..target_w {
            let idx = y * target_w + x;
            if visited[idx] || heat_map[idx] < det_thresh {
                continue;
            }

            queue.clear();
            queue.push_back((x, y));
            visited[idx] = true;

            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;
            let mut sum_prob = 0.0f32;
            let mut count = 0usize;

            while let Some((cx, cy)) = queue.pop_front() {
                let cidx = cy * target_w + cx;
                sum_prob += heat_map[cidx];
                count += 1;

                if cx < min_x { min_x = cx; }
                if cx > max_x { max_x = cx; }
                if cy < min_y { min_y = cy; }
                if cy > max_y { max_y = cy; }

                for (nx, ny) in [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ] {
                    if nx < target_w && ny < target_h {
                        let nidx = ny * target_w + nx;
                        if !visited[nidx] && heat_map[nidx] >= det_thresh {
                            visited[nidx] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            if count < 16 || (max_x - min_x) < 4 || (max_y - min_y) < 3 {
                continue;
            }

            let avg_score = sum_prob / (count as f32);
            if avg_score < box_thresh {
                continue;
            }

            let bw = (max_x - min_x + 1) as f32;
            let bh = (max_y - min_y + 1) as f32;
            let peri = 2.0 * (bw + bh);
            let area = bw * bh;
            let dist = (area * unclip_ratio) / peri.max(1.0);

            let x0 = (min_x as f32 - dist).max(0.0) / scale_w;
            let y0 = (min_y as f32 - dist).max(0.0) / scale_h;
            let x1 = ((max_x as f32 + dist).min(target_w as f32)) / scale_w;
            let y1 = ((max_y as f32 + dist).min(target_h as f32)) / scale_h;

            let clamped_x0 = x0.clamp(0.0, orig_w as f32);
            let clamped_y0 = y0.clamp(0.0, orig_h as f32);
            let clamped_x1 = x1.clamp(0.0, orig_w as f32);
            let clamped_y1 = y1.clamp(0.0, orig_h as f32);

            if (clamped_x1 - clamped_x0) > 4.0 && (clamped_y1 - clamped_y0) > 4.0 {
                raw_boxes.push(([clamped_x0, clamped_y0, clamped_x1, clamped_y1], avg_score));
            }
        }
    }

    raw_boxes.sort_by(|a, b| {
        a.0[1].total_cmp(&b.0[1]).then_with(|| a.0[0].total_cmp(&b.0[0]))
    });

    raw_boxes
}

/// Decode LineNet detection boxes from raw 3-channel flat float buffer (e.g. from Session output)
pub fn decode_linenet_det_boxes_slice(
    det_data: &[f32],
    out_h: usize,
    out_w: usize,
    orig_w: f32,
    orig_h: f32,
    scale_w: f32,
    scale_h: f32,
    det_thresh: f32,
    box_thresh: f32,
) -> Vec<([f32; 4], f32)> {
    let plane = out_h * out_w;
    if det_data.len() < 3 * plane {
        return Vec::new();
    }

    let logit_guard = if det_thresh > 0.0 && det_thresh < 1.0 {
        -((1.0 - det_thresh) / det_thresh).ln() - 1e-4
    } else {
        f32::NEG_INFINITY
    };

    let prob_at = |idx: usize| -> f32 {
        let logit = det_data[idx];
        if logit < logit_guard {
            0.0
        } else {
            1.0 / (1.0 + (-logit).exp())
        }
    };

    let mut visited = vec![false; plane];
    let mut raw_boxes = Vec::new();
    let mut queue = std::collections::VecDeque::with_capacity(256);

    for y in 0..out_h {
        for x in 0..out_w {
            let idx = y * out_w + x;
            if visited[idx] || prob_at(idx) < det_thresh {
                continue;
            }
            queue.clear();
            queue.push_back((x, y));
            visited[idx] = true;

            let ys0 = (y as f32 + 0.5) * 2.0;
            let mut sum_top = ys0 - det_data[plane + idx];
            let mut sum_bot = ys0 + det_data[2 * plane + idx];
            let mut sum_prob = prob_at(idx);
            let mut count = 1usize;
            let mut min_x = x;
            let mut max_x = x;

            while let Some((cx, cy)) = queue.pop_front() {
                if cx < min_x { min_x = cx; }
                if cx > max_x { max_x = cx; }

                for (nx, ny) in [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ] {
                    if nx < out_w && ny < out_h {
                        let nidx = ny * out_w + nx;
                        if !visited[nidx] && prob_at(nidx) > det_thresh {
                            let nys = (ny as f32 + 0.5) * 2.0;
                            let n_y0 = nys - det_data[plane + nidx];
                            let n_y1 = nys + det_data[2 * plane + nidx];

                            visited[nidx] = true;
                            sum_top += n_y0;
                            sum_bot += n_y1;
                            sum_prob += prob_at(nidx);
                            count += 1;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            if count < 2 { continue; }
            let avg_prob = sum_prob / count as f32;
            if avg_prob < box_thresh { continue; }

            let y0_in = sum_top / count as f32;
            let y1_in = sum_bot / count as f32;
            let lh = (y1_in - y0_in).max(1.0);
            let ex = (0.15 * lh).min(0.5 * lh);
            let x0_in = (min_x as f32 * 2.0) - ex;
            let x1_in = ((max_x + 1) as f32 * 2.0) + ex;

            let x0 = (x0_in / scale_w).clamp(0.0, orig_w);
            let y0 = (y0_in / scale_h).clamp(0.0, orig_h);
            let x1 = (x1_in / scale_w).clamp(0.0, orig_w);
            let y1 = (y1_in / scale_h).clamp(0.0, orig_h);

            if x1 > x0 && y1 > y0 {
                raw_boxes.push(([x0, y0, x1, y1], avg_prob));
            }
        }
    }

    raw_boxes.sort_by(|a, b| {
        a.0[1].total_cmp(&b.0[1]).then_with(|| a.0[0].total_cmp(&b.0[0]))
    });

    raw_boxes
}

pub fn decode_det_predictions(
    preds: &Array4<f32>,
    is_linenet: bool,
    ori_w: i32,
    ori_h: i32,
    scale_w: f32,
    scale_h: f32,
    params: &crate::preprocess::ResolvedDetParams,
) -> Result<(Vec<[Point2f; 4]>, Vec<f32>), CoreError> {
    if is_linenet {
        Ok(decode_linenet_det_boxes(
            preds,
            ori_w as f32,
            ori_h as f32,
            scale_w,
            scale_h,
            params.thresh,
            params.box_thresh,
        ))
    } else {
        let db = DBPostProcess::new(
            params.thresh,
            params.box_thresh,
            params.max_candidates as i32,
            params.unclip_ratio,
            params.use_dilation,
            &params.score_mode,
        );
        db.process(preds, ori_h, ori_w)
    }
}

pub fn unclip_polygon_analytic(box_pts: &[Point2f; 4], unclip_ratio: f64) -> Vec<Point2f> {
    let mut signed_area = 0.0f64;
    let mut length = 0.0f64;
    for i in 0..4 {
        let j = (i + 1) % 4;
        let dx = (box_pts[j].x - box_pts[i].x) as f64;
        let dy = (box_pts[j].y - box_pts[i].y) as f64;
        signed_area += (box_pts[i].x as f64) * (box_pts[j].y as f64) - (box_pts[j].x as f64) * (box_pts[i].y as f64);
        length += (dx * dx + dy * dy).sqrt();
    }
    let area = signed_area.abs() / 2.0;
    let distance = area * unclip_ratio / length.max(1e-6);

    let sign = if signed_area >= 0.0 { 1.0 } else { -1.0 };

    let mut normals = [(0.0f64, 0.0f64); 4];
    for i in 0..4 {
        let j = (i + 1) % 4;
        let dx = (box_pts[j].x - box_pts[i].x) as f64;
        let dy = (box_pts[j].y - box_pts[i].y) as f64;
        let edge_len = (dx * dx + dy * dy).sqrt().max(1e-6);
        normals[i] = (sign * dy / edge_len, -sign * dx / edge_len);
    }

    let mut result = Vec::with_capacity(4);
    for i in 0..4 {
        let prev = (i + 3) % 4;
        let n_prev = normals[prev];
        let n_curr = normals[i];
        let dot = n_prev.0 * n_curr.0 + n_prev.1 * n_curr.1;
        let denom = (1.0 + dot).max(1e-4);
        let mx = (n_prev.0 + n_curr.0) / denom;
        let my = (n_prev.1 + n_curr.1) / denom;

        result.push(Point2f::new(
            (box_pts[i].x as f64 + distance * mx) as f32,
            (box_pts[i].y as f64 + distance * my) as f32,
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{box_points, min_area_rect, Size};

    #[test]
    fn test_unclip_analytic_geometry() {
        for angle in [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0, 270.0] {
            for (w, h) in [(50, 20), (120, 35), (200, 10), (80, 80)] {
                let center = Point2f::new(200.0, 200.0);
                let size = Size::new(w, h);
                let pts = box_points(center, size, angle);

                let unclipped = unclip_polygon_analytic(&pts, 1.5);
                assert_eq!(unclipped.len(), 4, "Analytic unclip should produce 4 vertices");

                let (c, s, _a) = min_area_rect(&unclipped).expect("rect");
                assert!((c.x - center.x).abs() < 0.5, "Center X mismatch at angle {}: {} vs {}", angle, c.x, center.x);
                assert!((c.y - center.y).abs() < 0.5, "Center Y mismatch at angle {}: {} vs {}", angle, c.y, center.y);
                let max_orig = w.max(h);
                let min_orig = w.min(h);
                let max_exp = s.width.max(s.height);
                let min_exp = s.width.min(s.height);
                assert!(max_exp >= max_orig, "Expanded max side should be >= original: {} vs {}", max_exp, max_orig);
                assert!(min_exp >= min_orig, "Expanded min side should be >= original: {} vs {}", min_exp, min_orig);
            }
        }
    }
}
