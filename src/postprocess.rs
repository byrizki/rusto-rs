#[cfg(feature = "use-opencv")]
use opencv::{core, imgproc, prelude::*};

#[cfg(feature = "use-opencv")]
use ndarray::Array4;

#[cfg(feature = "use-opencv")]
use opencv::core::Mat;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

#[cfg(feature = "use-opencv")]
use geo_clipper::{Clipper, EndType, JoinType};
#[cfg(feature = "use-opencv")]
use geo_types::{Coord, LineString, Polygon};

use crate::engine::EngineError;

#[cfg(feature = "use-opencv")]
pub struct TextDetOutput {
    pub img: Option<Mat>,
    pub boxes: Option<Vec<[core::Point2f; 4]>>,
    pub scores: Option<Vec<f32>>,
    pub elapse: f64,
}

#[cfg(not(feature = "use-opencv"))]
pub struct TextDetOutput {
    pub img: Option<Mat>,
    pub boxes: Option<Vec<[crate::image_impl::Point2f; 4]>>,
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

    #[cfg(feature = "use-opencv")]
    pub fn new(img: Mat, boxes: Vec<[core::Point2f; 4]>, scores: Vec<f32>, elapse: f64) -> Self {
        Self {
            img: Some(img),
            boxes: Some(boxes),
            scores: Some(scores),
            elapse,
        }
    }

    #[cfg(not(feature = "use-opencv"))]
    pub fn new(img: Mat, boxes: Vec<[crate::image_impl::Point2f; 4]>, scores: Vec<f32>, elapse: f64) -> Self {
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
}

pub struct DBPostProcess {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f64,
    pub min_size: f32,
    pub use_dilation: bool,
}

#[cfg(feature = "use-opencv")]
impl DBPostProcess {
    pub fn new(
        thresh: f32,
        box_thresh: f32,
        max_candidates: i32,
        unclip_ratio: f32,
        use_dilation: bool,
    ) -> Self {
        Self {
            thresh,
            box_thresh,
            max_candidates: max_candidates as usize,
            unclip_ratio: unclip_ratio as f64,
            min_size: 3.0,
            use_dilation,
        }
    }

    pub fn process(
        &self,
        pred: &Array4<f32>,
        ori_h: i32,
        ori_w: i32,
    ) -> Result<(Vec<[core::Point2f; 4]>, Vec<f32>), EngineError> {
        let (_, _, h, w) = pred.dim();
        if h == 0 || w == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Build mask from prediction
        // pred is NCHW: (1, 1, height, width) where height=736, width=1184
        // OpenCV Mat is (rows, cols) where rows=height, cols=width
        let mut mask_mat = Mat::new_rows_cols_with_default(
            h as i32,  // rows = height
            w as i32,  // cols = width
            core::CV_8UC1,
            core::Scalar::all(0.0),
        )?;
        
        // Fill mask row by row
        for y in 0..h {
            for x in 0..w {
                let v = pred[[0, 0, y, x]];
                let val: u8 = if v > self.thresh { 255 } else { 0 };
                *mask_mat.at_2d_mut::<u8>(y as i32, x as i32)? = val;
            }
        }
        
        // Optional dilation, like Python's use_dilation with a 2x2 kernel of ones
        let mut dilated = Mat::default();
        let mut mask_for_contours: &Mat = &mask_mat;
        if self.use_dilation {
            let kernel = Mat::from_slice_2d(&[[1u8, 1u8], [1u8, 1u8]])?;
            imgproc::dilate(
                &mask_mat,
                &mut dilated,
                &kernel,
                core::Point::new(-1, -1),
                1,
                core::BORDER_CONSTANT,
                core::Scalar::all(0.0),
            )?;
            mask_for_contours = &dilated;
        }

        let mut contours = core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(
            mask_for_contours,
            &mut contours,
            imgproc::RETR_LIST,
            imgproc::CHAIN_APPROX_SIMPLE,
            core::Point::new(0, 0),
        )?;
        
        let (boxes, scores) = self.boxes_from_bitmap(pred, &contours, w, h, ori_w, ori_h)?;
        let (boxes, scores) = self.filter_det_res(boxes, scores, ori_h, ori_w);

        Ok((boxes, scores))
    }

    fn boxes_from_bitmap(
        &self,
        pred: &Array4<f32>,
        contours: &core::Vector<core::Vector<core::Point>>,
        width: usize,
        height: usize,
        dest_width: i32,
        dest_height: i32,
    ) -> Result<(Vec<[core::Point2f; 4]>, Vec<f32>), EngineError> {
        let num_contours = contours.len().min(self.max_candidates);
        let mut boxes = Vec::new();
        let mut scores = Vec::new();

        for i in 0..num_contours {
            let contour = contours.get(i)?;
            if contour.len() < 3 {
                continue;
            }

            let (box_pts, sside) = self.get_mini_box(&contour)?;
            if sside < self.min_size {
                continue;
            }

            let score = self.box_score_fast(pred, &box_pts, height, width)?;
            if score < self.box_thresh {
                continue;
            }

            let unclipped = self.unclip(&box_pts)?;
            if unclipped.is_empty() {
                continue;
            }
            let (box_pts2, sside2) = self.get_mini_box_points(&unclipped)?;
            if sside2 < self.min_size + 2.0 {
                continue;
            }

            let src_h = dest_height as f32;
            let src_w = dest_width as f32;
            let mut scaled = box_pts2;
            for p in &mut scaled {
                p.x = (p.x / width as f32 * src_w).round().clamp(0.0, src_w);
                p.y = (p.y / height as f32 * src_h).round().clamp(0.0, src_h);
            }

            boxes.push(scaled);
            scores.push(score);
        }

        Ok((boxes, scores))
    }

    fn get_mini_box(
        &self,
        contour: &core::Vector<core::Point>,
    ) -> opencv::Result<([core::Point2f; 4], f32)> {
        let rect = imgproc::min_area_rect(contour)?;

        // Use RotatedRect::points to get the 4 vertices directly into a fixed array,
        // avoiding ToOutputArray-based imgproc::box_points, which was triggering
        // an internal matrix_wrap assertion.
        let mut pts_arr = [
            core::Point2f::new(0.0, 0.0),
            core::Point2f::new(0.0, 0.0),
            core::Point2f::new(0.0, 0.0),
            core::Point2f::new(0.0, 0.0),
        ];
        rect.points(&mut pts_arr)?;

        let mut pts: Vec<core::Point2f> = pts_arr.to_vec();
        pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let (mut tl, mut bl) = (pts[0], pts[1]);
        if bl.y < tl.y {
            std::mem::swap(&mut tl, &mut bl);
        }

        let (mut tr, mut br) = (pts[2], pts[3]);
        if br.y < tr.y {
            std::mem::swap(&mut tr, &mut br);
        }

        let box_pts = [tl, tr, br, bl];
        let size = rect.size;
        let sside = size.width.min(size.height).abs();
        Ok((box_pts, sside))
    }

    fn get_mini_box_points(
        &self,
        pts: &[core::Point2f],
    ) -> opencv::Result<([core::Point2f; 4], f32)> {
        // Convert Point2f to Point for minAreaRect
        let contour: core::Vector<core::Point> = pts
            .iter()
            .map(|p| core::Point::new(p.x as i32, p.y as i32))
            .collect();
        
        // Use minAreaRect to get the proper bounding box for any number of points
        let rect = imgproc::min_area_rect(&contour)?;
        
        // Get the 4 corner points
        let mut pts_arr = [core::Point2f::default(); 4];
        rect.points(&mut pts_arr)?;

        // Sort and order them consistently
        let mut pts: Vec<core::Point2f> = pts_arr.to_vec();
        pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let (mut tl, mut bl) = (pts[0], pts[1]);
        if bl.y < tl.y {
            std::mem::swap(&mut tl, &mut bl);
        }

        let (mut tr, mut br) = (pts[2], pts[3]);
        if br.y < tr.y {
            std::mem::swap(&mut tr, &mut br);
        }

        let box_pts = [tl, tr, br, bl];
        let size = rect.size;
        let sside = size.width.min(size.height).abs();
        Ok((box_pts, sside))
    }

    fn box_score_fast(
        &self,
        pred: &Array4<f32>,
        box_pts: &[core::Point2f; 4],
        h: usize,
        w: usize,
    ) -> Result<f32, EngineError> {
        // Use polygon-based scoring like Python's box_score_fast with fillPoly
        let mut xs: Vec<f32> = box_pts.iter().map(|p| p.x).collect();
        let mut ys: Vec<f32> = box_pts.iter().map(|p| p.y).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let xmin = xs.first().copied().unwrap_or(0.0).floor().max(0.0).min((w - 1) as f32) as i32;
        let xmax = xs.last().copied().unwrap_or(0.0).ceil().max(0.0).min(w as f32) as i32;
        let ymin = ys.first().copied().unwrap_or(0.0).floor().max(0.0).min((h - 1) as f32) as i32;
        let ymax = ys.last().copied().unwrap_or(0.0).ceil().max(0.0).min(h as f32) as i32;

        if xmax <= xmin || ymax <= ymin {
            return Ok(0.0);
        }

        let mask_h = (ymax - ymin + 1) as i32;
        let mask_w = (xmax - xmin + 1) as i32;

        // Create mask for the polygon
        let mut mask = Mat::zeros(mask_h, mask_w, core::CV_8UC1)?.to_mat()?;

        // Adjust box points relative to the mask
        let box_adjusted: Vec<core::Point> = box_pts
            .iter()
            .map(|p| core::Point::new((p.x - xmin as f32) as i32, (p.y - ymin as f32) as i32))
            .collect();
        let pts = core::Vector::<core::Point>::from(box_adjusted);
        let pts_vec = core::Vector::<core::Vector<core::Point>>::from(vec![pts]);

        // Fill polygon
        imgproc::fill_poly(
            &mut mask,
            &pts_vec,
            core::Scalar::all(1.0),
            imgproc::LINE_8,
            0,
            core::Point::new(0, 0),
        )?;

        // Extract region from pred and compute mean with mask
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for yy in ymin..ymax + 1 {
            for xx in xmin..xmax + 1 {
                let mask_val = mask.at_2d::<u8>((yy - ymin) as i32, (xx - xmin) as i32)?;
                if *mask_val > 0 {
                    sum += pred[[0, 0, yy as usize, xx as usize]];
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

    fn unclip(&self, box_pts: &[core::Point2f; 4]) -> Result<Vec<core::Point2f>, EngineError> {
        // Compute polygon area and perimeter (shoelace formula + edge lengths), as in Python.
        let mut area = 0.0f64;
        let mut length = 0.0f64;
        for i in 0..4 {
            let p1 = box_pts[i];
            let p2 = box_pts[(i + 1) % 4];
            area += (p1.x as f64) * (p2.y as f64) - (p2.x as f64) * (p1.y as f64);
            let dx = p1.x as f64 - p2.x as f64;
            let dy = p1.y as f64 - p2.y as f64;
            length += (dx * dx + dy * dy).sqrt();
        }
        area = (area * 0.5).abs();
        if area <= 0.0 || length <= 0.0 {
            return Ok(box_pts.to_vec());
        }

        let distance = area * self.unclip_ratio / length;

        // Build a geo-types polygon from the 4-point box.
        let coords: Vec<Coord<f64>> = box_pts
            .iter()
            .map(|p| Coord { x: p.x as f64, y: p.y as f64 })
            .collect();

        // Ensure closed ring for polygon (first point == last point).
        let mut ring = coords.clone();
        if let Some(first) = coords.first() {
            if coords.last().map(|c| c.x != first.x || c.y != first.y).unwrap_or(false) {
                ring.push(*first);
            }
        }

        let poly = Polygon::new(LineString::from(ring), vec![]);

        // Use geo-clipper offset with round joins and closed polygon, like pyclipper JT_ROUND/ET_CLOSEDPOLYGON.
        let mpoly = poly.offset(distance, JoinType::Round(1.0), EndType::ClosedPolygon, 1.0f64);
        let first_poly = match mpoly.0.first() {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };

        let mut result = Vec::new();
        // Return all points from the expanded polygon, not just 4
        for coord in first_poly.exterior().0.iter() {
            result.push(core::Point2f::new(coord.x as f32, coord.y as f32));
        }

        // Remove last point if it's a duplicate of first (closed ring)
        if result.len() > 1 {
            let first = result[0];
            let last = result[result.len() - 1];
            if (first.x - last.x).abs() < 0.01 && (first.y - last.y).abs() < 0.01 {
                result.pop();
            }
        }

        Ok(result)
    }

    fn filter_det_res(
        &self,
        dt_boxes: Vec<[core::Point2f; 4]>,
        scores: Vec<f32>,
        img_height: i32,
        img_width: i32,
    ) -> (Vec<[core::Point2f; 4]>, Vec<f32>) {
        let mut dt_boxes_new = Vec::new();
        let mut new_scores = Vec::new();

        for (mut box_pts, score) in dt_boxes.into_iter().zip(scores.into_iter()) {
            // Order points clockwise
            box_pts = self.order_points_clockwise(box_pts);
            // Clip to image bounds
            box_pts = self.clip_det_res(box_pts, img_height, img_width);

            // Check rectangle width and height
            let rect_width = ((box_pts[0].x - box_pts[1].x).powi(2)
                + (box_pts[0].y - box_pts[1].y).powi(2))
            .sqrt() as i32;
            let rect_height = ((box_pts[0].x - box_pts[3].x).powi(2)
                + (box_pts[0].y - box_pts[3].y).powi(2))
            .sqrt() as i32;

            if rect_width <= 3 || rect_height <= 3 {
                continue;
            }

            dt_boxes_new.push(box_pts);
            new_scores.push(score);
        }

        (dt_boxes_new, new_scores)
    }

    fn order_points_clockwise(&self, pts: [core::Point2f; 4]) -> [core::Point2f; 4] {
        // Sort by x coordinate
        let mut pts_vec: Vec<core::Point2f> = pts.to_vec();
        pts_vec.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        // Left-most and right-most points
        let mut left_most = [pts_vec[0], pts_vec[1]];
        let mut right_most = [pts_vec[2], pts_vec[3]];

        // Sort left-most by y to get top-left and bottom-left
        left_most.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
        let tl = left_most[0];
        let bl = left_most[1];

        // Sort right-most by y to get top-right and bottom-right
        right_most.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
        let tr = right_most[0];
        let br = right_most[1];

        [tl, tr, br, bl]
    }

    fn clip_det_res(
        &self,
        mut points: [core::Point2f; 4],
        img_height: i32,
        img_width: i32,
    ) -> [core::Point2f; 4] {
        for p in &mut points {
            p.x = p.x.max(0.0).min((img_width - 1) as f32);
            p.y = p.y.max(0.0).min((img_height - 1) as f32);
        }
        points
    }
}

// Pure Rust implementation of DBPostProcess
#[cfg(not(feature = "use-opencv"))]
impl DBPostProcess {
    pub fn new(
        thresh: f32,
        box_thresh: f32,
        max_candidates: i32,
        unclip_ratio: f32,
        use_dilation: bool,
    ) -> Self {
        Self {
            thresh,
            box_thresh,
            max_candidates: max_candidates as usize,
            unclip_ratio: unclip_ratio as f64,
            min_size: 3.0,
            use_dilation,
        }
    }

    pub fn process(
        &self,
        pred: &ndarray::Array4<f32>,
        ori_h: i32,
        ori_w: i32,
    ) -> Result<(Vec<[crate::image_impl::Point2f; 4]>, Vec<f32>), EngineError> {
        use crate::contours::find_contours;
        use crate::image_impl::{Point2f, min_area_rect, box_points};
        use image::{GrayImage, Luma};
        
        let (_, _, h, w) = pred.dim();
        if h == 0 || w == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Create binary mask from prediction
        let mut binary_img = GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let val = pred[[0, 0, y, x]];
                if val > self.thresh {
                    binary_img.put_pixel(x as u32, y as u32, Luma([255]));
                } else {
                    binary_img.put_pixel(x as u32, y as u32, Luma([0]));
                }
            }
        }

        // Optionally dilate to connect nearby regions (2x2 kernel like OpenCV)
        let img_for_contours = if self.use_dilation {
            dilate_2x2(&binary_img)
        } else {
            binary_img.clone()
        };

        // Find contours
        let mut contours = find_contours(&img_for_contours);
        
        // Sort contours by area (descending) - largest first
        contours.sort_by(|a, b| {
            let area_a = calculate_contour_area(a);
            let area_b = calculate_contour_area(b);
            area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let num_contours = contours.len().min(self.max_candidates);
        let mut boxes = Vec::new();
        let mut scores = Vec::new();

        for contour in contours.iter().take(num_contours) {
            if contour.len() < 4 {
                continue;
            }

            // Convert contour points to f32
            let points: Vec<Point2f> = contour
                .points
                .iter()
                .map(|&(x, y)| Point2f::new(x as f32, y as f32))
                .collect();

            // Get minimum area rectangle
            let (center, size, angle) = min_area_rect(&points)?;
            let rect_points = box_points(center, size, angle);
            let side_len = size.width.min(size.height) as f32;
            
            if side_len < 3.0 {
                continue;
            }

            // Calculate score for this box
            let score = self.box_score_fast_pure(pred, &rect_points, h, w)?;
            if score < self.box_thresh {
                continue;
            }

            // Unclip the box
            let unclipped = self.unclip_pure(&rect_points)?;
            if unclipped.len() < 4 {
                continue;
            }

            // Get minimum area rectangle of unclipped points - matching get_mini_box_points
            let (center2, size2, angle2) = min_area_rect(&unclipped).map_err(|e| {
                EngineError::ImageError(e.to_string())
            })?;
            let box_pts_raw = box_points(center2, size2, angle2);
            
            // Sort and order to match OpenCV
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
            
            // Scale to original image size
            let mut final_box = [Point2f::default(); 4];
            for (i, pt) in box_pts.iter().enumerate() {
                final_box[i] = Point2f::new(
                    pt.x * (ori_w as f32 / w as f32),
                    pt.y * (ori_h as f32 / h as f32),
                );
            }

            final_box = self.order_points_clockwise_pure(final_box);
            final_box = self.clip_det_res_pure(final_box, ori_h, ori_w);

            // Check rectangle width and height (same as OpenCV filter_det_res)
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

    fn box_score_fast_pure(
        &self,
        pred: &ndarray::Array4<f32>,
        box_pts: &[crate::image_impl::Point2f; 4],
        h: usize,
        w: usize,
    ) -> Result<f32, EngineError> {
        // Get bounding box
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

        // Compute mean score inside polygon using point-in-polygon test
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

    fn unclip_pure(&self, box_pts: &[crate::image_impl::Point2f; 4]) -> Result<Vec<crate::image_impl::Point2f>, EngineError> {
        use geo_clipper::Clipper;
        use geo_types::{Coord, LineString, Polygon};

        // Compute area and perimeter
        let mut area = 0.0f64;
        let mut length = 0.0f64;
        for i in 0..4 {
            let j = (i + 1) % 4;
            let dx = (box_pts[j].x - box_pts[i].x) as f64;
            let dy = (box_pts[j].y - box_pts[i].y) as f64;
            area += box_pts[i].x as f64 * box_pts[j].y as f64 - box_pts[j].x as f64 * box_pts[i].y as f64;
            length += (dx * dx + dy * dy).sqrt();
        }
        area = area.abs() / 2.0;

        let distance = area * self.unclip_ratio / length;

        let coords: Vec<Coord<f64>> = box_pts
            .iter()
            .map(|p| Coord { x: p.x as f64, y: p.y as f64 })
            .collect();

        let mut ring = coords.clone();
        ring.push(coords[0]);
        let line_string = LineString(ring);
        let poly = Polygon::new(line_string, vec![]);

        let expanded = poly.offset(distance, geo_clipper::JoinType::Miter(2.0), geo_clipper::EndType::ClosedPolygon, 2.0);

        let mut result = Vec::new();
        if !expanded.0.is_empty() {
            let first_poly = &expanded.0[0];
            for coord in first_poly.exterior().0.iter() {
                result.push(crate::image_impl::Point2f::new(coord.x as f32, coord.y as f32));
            }
            if let Some(last) = result.last() {
                if let Some(first) = result.first() {
                    if (last.x - first.x).abs() < 0.1 && (last.y - first.y).abs() < 0.1 {
                        result.pop();
                    }
                }
            }
        }

        Ok(result)
    }

    fn order_points_clockwise_pure(&self, pts: [crate::image_impl::Point2f; 4]) -> [crate::image_impl::Point2f; 4] {
        let mut pts_vec: Vec<crate::image_impl::Point2f> = pts.to_vec();
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
        mut points: [crate::image_impl::Point2f; 4],
        img_height: i32,
        img_width: i32,
    ) -> [crate::image_impl::Point2f; 4] {
        for p in &mut points {
            p.x = p.x.max(0.0).min((img_width - 1) as f32);
            p.y = p.y.max(0.0).min((img_height - 1) as f32);
        }
        points
    }
}

// Helper functions for pure Rust implementation
#[cfg(not(feature = "use-opencv"))]
fn dilate_2x2(img: &image::GrayImage) -> image::GrayImage {
    use image::{Luma, GrayImage};
    let (width, height) = img.dimensions();
    let mut result = GrayImage::new(width, height);

    // 2x2 kernel dilation matching OpenCV's [[1,1],[1,1]] kernel
    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;
            // Check 2x2 neighborhood
            for dy in 0..=1 {
                for dx in 0..=1 {
                    let nx = ((x as i32 + dx).min(width as i32 - 1)) as u32;
                    let ny = ((y as i32 + dy).min(height as i32 - 1)) as u32;
                    max_val = max_val.max(img.get_pixel(nx, ny)[0]);
                }
            }
            result.put_pixel(x, y, Luma([max_val]));
        }
    }

    result
}

#[cfg(not(feature = "use-opencv"))]
fn point_in_polygon(x: f32, y: f32, polygon: &[crate::image_impl::Point2f; 4]) -> bool {
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

#[cfg(not(feature = "use-opencv"))]
fn calculate_contour_area(contour: &crate::contours::Contour) -> f32 {
    if contour.points.len() < 3 {
        return 0.0;
    }
    
    // Use shoelace formula to calculate polygon area
    let mut area = 0.0f32;
    let n = contour.points.len();
    
    for i in 0..n {
        let j = (i + 1) % n;
        let (x1, y1) = contour.points[i];
        let (x2, y2) = contour.points[j];
        area += (x1 as f32 * y2 as f32) - (x2 as f32 * y1 as f32);
    }
    
    (area * 0.5).abs()
}

