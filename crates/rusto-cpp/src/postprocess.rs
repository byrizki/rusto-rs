#[cfg(feature = "opencv")]
pub use opencv_postprocess::*;

#[cfg(feature = "opencv")]
mod opencv_postprocess {
    use opencv::{core, imgproc, prelude::*};
    use ndarray::Array4;
    use opencv::core::Mat;
    use clipper2_rust::{inflate_paths_d, EndType, JoinType, PathD, PointD};
    use rusto_core::CoreError;

    pub struct TextDetOutput {
        pub img: Option<Mat>,
        pub boxes: Option<Vec<[core::Point2f; 4]>>,
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

        pub fn new(img: Mat, boxes: Vec<[core::Point2f; 4]>, scores: Vec<f32>, elapse: f64) -> Self {
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
        ) -> Result<(Vec<[core::Point2f; 4]>, Vec<f32>), CoreError> {
            let (_, _, h, w) = pred.dim();
            if h == 0 || w == 0 {
                return Ok((Vec::new(), Vec::new()));
            }

            let mut mask_mat = Mat::new_rows_cols_with_default(
                h as i32,
                w as i32,
                core::CV_8UC1,
                core::Scalar::all(0.0),
            ).map_err(|e| CoreError::Postprocess(e.to_string()))?;

            for y in 0..h {
                for x in 0..w {
                    let v = pred[[0, 0, y, x]];
                    let val: u8 = if v > self.thresh { 255 } else { 0 };
                    *mask_mat.at_2d_mut::<u8>(y as i32, x as i32).map_err(|e| CoreError::Postprocess(e.to_string()))? = val;
                }
            }

            let mut dilated = Mat::default();
            let mut mask_for_contours: &Mat = &mask_mat;
            if self.use_dilation {
                let kernel = Mat::from_slice_2d(&[[1u8, 1u8], [1u8, 1u8]]).map_err(|e| CoreError::Postprocess(e.to_string()))?;
                imgproc::dilate(
                    &mask_mat,
                    &mut dilated,
                    &kernel,
                    core::Point::new(-1, -1),
                    1,
                    core::BORDER_CONSTANT,
                    core::Scalar::all(0.0),
                ).map_err(|e| CoreError::Postprocess(e.to_string()))?;
                mask_for_contours = &dilated;
            }

            let mut contours = core::Vector::<core::Vector<core::Point>>::new();
            imgproc::find_contours(
                mask_for_contours,
                &mut contours,
                imgproc::RETR_LIST,
                imgproc::CHAIN_APPROX_SIMPLE,
                core::Point::new(0, 0),
            ).map_err(|e| CoreError::Postprocess(e.to_string()))?;

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
        ) -> Result<(Vec<[core::Point2f; 4]>, Vec<f32>), CoreError> {
            let num_contours = contours.len().min(self.max_candidates);
            let mut boxes = Vec::new();
            let mut scores = Vec::new();

            for i in 0..num_contours {
                let contour = contours.get(i).map_err(|e| CoreError::Postprocess(e.to_string()))?;
                if contour.len() < 3 {
                    continue;
                }

                let (box_pts, sside) = self.get_mini_box(&contour).map_err(|e| CoreError::Postprocess(e.to_string()))?;
                if sside < self.min_size {
                    continue;
                }

                let score = if self.score_mode == "slow" {
                    self.box_score_slow(pred, &contour, height, width)?
                } else {
                    self.box_score_fast(pred, &box_pts, height, width)?
                };
                if score < self.box_thresh {
                    continue;
                }

                let unclipped = self.unclip(&box_pts)?;
                if unclipped.is_empty() {
                    continue;
                }
                let (box_pts2, sside2) = self.get_mini_box_points(&unclipped).map_err(|e| CoreError::Postprocess(e.to_string()))?;
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
            let contour: core::Vector<core::Point> = pts
                .iter()
                .map(|p| core::Point::new(p.x as i32, p.y as i32))
                .collect();

            let rect = imgproc::min_area_rect(&contour)?;
            let mut pts_arr = [core::Point2f::default(); 4];
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

        fn box_score_fast(
            &self,
            pred: &Array4<f32>,
            box_pts: &[core::Point2f; 4],
            h: usize,
            w: usize,
        ) -> Result<f32, CoreError> {
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

            let mut mask = Mat::zeros(mask_h, mask_w, core::CV_8UC1)
                .map_err(|e| CoreError::Postprocess(e.to_string()))?
                .to_mat()
                .map_err(|e| CoreError::Postprocess(e.to_string()))?;

            let box_adjusted: Vec<core::Point> = box_pts
                .iter()
                .map(|p| core::Point::new((p.x - xmin as f32) as i32, (p.y - ymin as f32) as i32))
                .collect();
            let pts = core::Vector::<core::Point>::from(box_adjusted);
            let pts_vec = core::Vector::<core::Vector<core::Point>>::from(vec![pts]);

            imgproc::fill_poly(
                &mut mask,
                &pts_vec,
                core::Scalar::all(1.0),
                imgproc::LINE_8,
                0,
                core::Point::new(0, 0),
            ).map_err(|e| CoreError::Postprocess(e.to_string()))?;

            let mut sum = 0.0f32;
            let mut count = 0usize;
            for yy in ymin..ymax + 1 {
                for xx in xmin..xmax + 1 {
                    let mask_val = mask.at_2d::<u8>((yy - ymin) as i32, (xx - xmin) as i32)
                        .map_err(|e| CoreError::Postprocess(e.to_string()))?;
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

        fn box_score_slow(
            &self,
            pred: &Array4<f32>,
            contour: &core::Vector<core::Point>,
            h: usize,
            w: usize,
        ) -> Result<f32, CoreError> {
            if contour.is_empty() {
                return Ok(0.0);
            }
            let mut min_x = i32::MAX;
            let mut max_x = i32::MIN;
            let mut min_y = i32::MAX;
            let mut max_y = i32::MIN;

            for i in 0..contour.len() {
                let pt = contour.get(i).map_err(|e| CoreError::Postprocess(e.to_string()))?;
                min_x = min_x.min(pt.x);
                max_x = max_x.max(pt.x);
                min_y = min_y.min(pt.y);
                max_y = max_y.max(pt.y);
            }

            let xmin = min_x.clamp(0, (w - 1) as i32);
            let xmax = max_x.clamp(0, (w - 1) as i32);
            let ymin = min_y.clamp(0, (h - 1) as i32);
            let ymax = max_y.clamp(0, (h - 1) as i32);

            if xmax <= xmin || ymax <= ymin {
                return Ok(0.0);
            }

            let mask_h = ymax - ymin + 1;
            let mask_w = xmax - xmin + 1;

            let mut mask = Mat::zeros(mask_h, mask_w, core::CV_8UC1)
                .map_err(|e| CoreError::Postprocess(e.to_string()))?
                .to_mat()
                .map_err(|e| CoreError::Postprocess(e.to_string()))?;

            let mut contour_adjusted = Vec::with_capacity(contour.len());
            for i in 0..contour.len() {
                let pt = contour.get(i).map_err(|e| CoreError::Postprocess(e.to_string()))?;
                contour_adjusted.push(core::Point::new(pt.x - xmin, pt.y - ymin));
            }
            let pts = core::Vector::<core::Point>::from(contour_adjusted);
            let pts_vec = core::Vector::<core::Vector<core::Point>>::from(vec![pts]);

            imgproc::fill_poly(
                &mut mask,
                &pts_vec,
                core::Scalar::all(1.0),
                imgproc::LINE_8,
                0,
                core::Point::new(0, 0),
            ).map_err(|e| CoreError::Postprocess(e.to_string()))?;

            let mut sum = 0.0f32;
            let mut count = 0usize;
            for yy in ymin..=ymax {
                for xx in xmin..=xmax {
                    let mask_val = mask.at_2d::<u8>(yy - ymin, xx - xmin)
                        .map_err(|e| CoreError::Postprocess(e.to_string()))?;
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

        fn unclip(&self, box_pts: &[core::Point2f; 4]) -> Result<Vec<core::Point2f>, CoreError> {
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

            let poly: PathD = box_pts
                .iter()
                .map(|p| PointD::new(p.x as f64, p.y as f64))
                .collect();

            let paths = vec![poly];
            let inflated = inflate_paths_d(&paths, distance, JoinType::Round, EndType::Polygon, 2.0, 2, 0.25);
            let first_path = match inflated.first() {
                Some(p) => p,
                None => return Ok(Vec::new()),
            };

            let mut result = Vec::with_capacity(first_path.len());
            for pt in first_path {
                result.push(core::Point2f::new(pt.x as f32, pt.y as f32));
            }

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
                box_pts = self.order_points_clockwise(box_pts);
                box_pts = self.clip_det_res(box_pts, img_height, img_width);

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
            let mut pts_vec: Vec<core::Point2f> = pts.to_vec();
            pts_vec.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

            let mut left_most = [pts_vec[0], pts_vec[1]];
            let mut right_most = [pts_vec[2], pts_vec[3]];

            left_most.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
            let tl = left_most[0];
            let bl = left_most[1];

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

    pub fn sort_det_boxes(dt_boxes: &mut [[core::Point2f; 4]]) {
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

    pub fn decode_det_predictions(
        preds: &Array4<f32>,
        _is_linenet: bool,
        ori_w: i32,
        ori_h: i32,
        _scale_w: f32,
        _scale_h: f32,
        params: &rusto_core::preprocess::ResolvedDetParams,
    ) -> Result<(Vec<[core::Point2f; 4]>, Vec<f32>), CoreError> {
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

