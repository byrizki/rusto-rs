use std::time::Instant;

use ndarray::{Array4, ArrayD};

use crate::engine::{EngineError, EngineSession, InferenceSession};
use crate::postprocess::{DBPostProcess, TextDetOutput};
use crate::preprocess::DetPreProcess;
use crate::rusto_ocr::{DetectionRunOptions, PostprocessRunOptions};
use crate::types::DetConfig;

#[cfg(feature = "use-opencv")]
use opencv::{core::Mat, core::Point2f, prelude::MatTraitConst};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::{Mat, Point2f};

pub struct TextDetector {
    pub cfg: DetConfig,
    pub session: EngineSession,
}

impl TextDetector {
    pub fn new(cfg: DetConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_det_config(&cfg)?;
        Ok(Self { cfg, session })
    }

    pub fn is_linenet(&self) -> bool {
        self.session.input_channels() == 1
    }

    /// Construct detector with a custom inference session implementation
    pub fn with_custom_session(cfg: DetConfig, session: impl InferenceSession + 'static) -> Self {
        Self {
            cfg,
            session: EngineSession::from_custom(session),
        }
    }

    pub fn run(&mut self, img: &Mat) -> Result<TextDetOutput, EngineError> {
        self.run_with_options(img, None, None)
    }

    pub fn run_with_options(
        &mut self,
        img: &Mat,
        options: Option<&DetectionRunOptions>,
        postprocess: Option<&PostprocessRunOptions>,
    ) -> Result<TextDetOutput, EngineError> {
        let start = Instant::now();
        // Per-call fields override only supplied values; omitted fields retain
        // initialized detector defaults for this request.
        let options = options.cloned().unwrap_or_default();
        let limit_type = options.limit_type.unwrap_or_else(|| self.cfg.limit_type.clone());
        let configured_limit_side_len = options.limit_side_len.unwrap_or(self.cfg.limit_side_len);
        let mean = options.mean.unwrap_or(self.cfg.mean);
        let std = options.std.unwrap_or(self.cfg.std);
        // Merge postprocessing options: top-level postprocess overrides nested detection.postprocess,
        // which overrides detector defaults. Never mutate `self.cfg`.
        let postprocess_options = match (postprocess, options.postprocess.as_ref()) {
            (Some(root), Some(nested)) => PostprocessRunOptions {
                threshold: root.threshold.or(nested.threshold),
                box_threshold: root.box_threshold.or(nested.box_threshold),
                max_candidates: root.max_candidates.or(nested.max_candidates),
                unclip_ratio: root.unclip_ratio.or(nested.unclip_ratio),
                use_dilation: root.use_dilation.or(nested.use_dilation),
                score_mode: root.score_mode.clone().or_else(|| nested.score_mode.clone()),
            },
            (Some(root), None) => root.clone(),
            (None, Some(nested)) => nested.clone(),
            (None, None) => PostprocessRunOptions::default(),
        };

        let ori_h = img.rows();
        let ori_w = img.cols();
        let max_wh = ori_h.max(ori_w);

        let limit_side_len = if let Some(explicit_limit) = options.limit_side_len {
            explicit_limit
        } else if limit_type == "min" {
            configured_limit_side_len
        } else if self.cfg.limit_side_len > 0 {
            self.cfg.limit_side_len
        } else if max_wh < 960 {
            960
        } else if max_wh < 1500 {
            1500
        } else {
            2000
        };

        if self.session.input_channels() == 1 {
            let (target_w, target_h, scale_w, scale_h) =
                calculate_linenet_dims(ori_w, ori_h, limit_side_len, &limit_type);

            let input = prepare_linenet_det_input(img, target_w, target_h)?;
            let preds_dyn = self.session.run(input.into_dyn())?;
            let preds: Array4<f32> = preds_dyn
                .into_dimensionality()
                .map_err(|_| EngineError::InvalidInputShape)?;

            let (mut boxes, scores) = decode_linenet_det_boxes(
                &preds,
                ori_w as f32,
                ori_h as f32,
                scale_w,
                scale_h,
                postprocess_options.threshold.unwrap_or(self.cfg.thresh),
                postprocess_options.box_threshold.unwrap_or(self.cfg.box_thresh),
            );

            if boxes.is_empty() {
                return Ok(TextDetOutput::empty());
            }

            self.sorted_boxes(&mut boxes);
            let elapse = start.elapsed().as_secs_f64();

            return Ok(TextDetOutput {
                img: None,
                boxes: Some(boxes),
                scores: Some(scores),
                elapse,
            });
        }

        let pre = DetPreProcess::new(limit_side_len, limit_type, mean, std);
        let input = pre.run(img)?;
        let input_dyn: ArrayD<f32> = input.into_dyn();
        let preds_dyn = self.session.run(input_dyn)?;
        let preds: Array4<f32> = preds_dyn
            .into_dimensionality()
            .map_err(|_| EngineError::InvalidInputShape)?;
        let postprocess = DBPostProcess::new(
            postprocess_options.threshold.unwrap_or(self.cfg.thresh),
            postprocess_options.box_threshold.unwrap_or(self.cfg.box_thresh),
            postprocess_options.max_candidates.unwrap_or(self.cfg.max_candidates),
            postprocess_options.unclip_ratio.unwrap_or(self.cfg.unclip_ratio),
            postprocess_options.use_dilation.unwrap_or(self.cfg.use_dilation),
            postprocess_options.score_mode.as_deref().unwrap_or(&self.cfg.score_mode),
        );
        let (mut boxes, scores) = postprocess.process(&preds, ori_h, ori_w)?;
        if boxes.is_empty() {
            return Ok(TextDetOutput::empty());
        }

        self.sorted_boxes(&mut boxes);
        let elapse = start.elapsed().as_secs_f64();

        Ok(TextDetOutput {
            img: None,
            boxes: Some(boxes),
            scores: Some(scores),
            elapse,
        })
    }

    #[cfg(feature = "use-opencv")]
    pub fn sorted_boxes(&self, dt_boxes: &mut [[opencv::core::Point2f; 4]]) {
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

    #[cfg(not(feature = "use-opencv"))]
    pub fn sorted_boxes(&self, dt_boxes: &mut [[crate::image_impl::Point2f; 4]]) {
        sort_det_boxes(dt_boxes);
    }
}

#[cfg(not(feature = "use-opencv"))]
pub fn sort_det_boxes(dt_boxes: &mut [[crate::image_impl::Point2f; 4]]) {
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

fn calculate_linenet_dims(
    ori_w: i32,
    ori_h: i32,
    limit_side_len: i32,
    limit_type: &str,
) -> (usize, usize, f32, f32) {
    let ratio = if limit_type == "max" {
        let max_side = ori_h.max(ori_w) as f32;
        if max_side > limit_side_len as f32 {
            limit_side_len as f32 / max_side
        } else {
            1.0
        }
    } else {
        let min_side = ori_h.min(ori_w) as f32;
        if min_side < limit_side_len as f32 {
            limit_side_len as f32 / min_side
        } else {
            1.0
        }
    };

    let mut resize_h = ((ori_h as f32 * ratio / 32.0).round() * 32.0) as usize;
    let mut resize_w = ((ori_w as f32 * ratio / 32.0).round() * 32.0) as usize;
    if resize_h == 0 {
        resize_h = 32;
    }
    if resize_w == 0 {
        resize_w = 32;
    }

    let scale_w = resize_w as f32 / ori_w as f32;
    let scale_h = resize_h as f32 / ori_h as f32;

    (resize_w, resize_h, scale_w, scale_h)
}

fn prepare_linenet_det_input(
    img: &Mat,
    target_w: usize,
    target_h: usize,
) -> Result<Array4<f32>, EngineError> {
    let orig_w = img.cols() as f32;
    let orig_h = img.rows() as f32;
    let scale_w = target_w as f32 / orig_w;
    let scale_h = target_h as f32 / orig_h;

    let mut input = Array4::<f32>::zeros((1, 1, target_h, target_w));

    #[cfg(feature = "use-opencv")]
    {
        for ty in 0..target_h {
            let sy = ((ty as f32 + 0.5) / scale_h - 0.5).clamp(0.0, orig_h - 1.0);
            let y0 = sy.floor() as i32;
            let y1 = (y0 + 1).min(img.rows() - 1);
            let fy = sy - y0 as f32;

            for tx in 0..target_w {
                let sx = ((tx as f32 + 0.5) / scale_w - 0.5).clamp(0.0, orig_w - 1.0);
                let x0 = sx.floor() as i32;
                let x1 = (x0 + 1).min(img.cols() - 1);
                let fx = sx - x0 as f32;

                let p00 = img.at_2d::<opencv::core::Vec3b>(y0, x0)?;
                let p10 = img.at_2d::<opencv::core::Vec3b>(y0, x1)?;
                let p01 = img.at_2d::<opencv::core::Vec3b>(y1, x0)?;
                let p11 = img.at_2d::<opencv::core::Vec3b>(y1, x1)?;

                let b = (p00[0] as f32 * (1.0 - fx) + p10[0] as f32 * fx) * (1.0 - fy)
                    + (p01[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
                let g = (p00[1] as f32 * (1.0 - fx) + p10[1] as f32 * fx) * (1.0 - fy)
                    + (p01[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
                let r = (p00[2] as f32 * (1.0 - fx) + p10[2] as f32 * fx) * (1.0 - fy)
                    + (p01[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                input[[0, 0, ty, tx]] = ink;
            }
        }
    }

    #[cfg(not(feature = "use-opencv"))]
    {
        for ty in 0..target_h {
            let sy = ((ty as f32 + 0.5) / scale_h - 0.5).clamp(0.0, orig_h - 1.0);
            let y0 = sy.floor() as u32;
            let y1 = (y0 + 1).min(img.rows() as u32 - 1);
            let fy = sy - y0 as f32;

            for tx in 0..target_w {
                let sx = ((tx as f32 + 0.5) / scale_w - 0.5).clamp(0.0, orig_w - 1.0);
                let x0 = sx.floor() as u32;
                let x1 = (x0 + 1).min(img.cols() as u32 - 1);
                let fx = sx - x0 as f32;

                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x1, y0);
                let p01 = img.get_pixel(x0, y1);
                let p11 = img.get_pixel(x1, y1);

                let r = (p00[0] as f32 * (1.0 - fx) + p10[0] as f32 * fx) * (1.0 - fy)
                    + (p01[0] as f32 * (1.0 - fx) + p11[0] as f32 * fx) * fy;
                let g = (p00[1] as f32 * (1.0 - fx) + p10[1] as f32 * fx) * (1.0 - fy)
                    + (p01[1] as f32 * (1.0 - fx) + p11[1] as f32 * fx) * fy;
                let b = (p00[2] as f32 * (1.0 - fx) + p10[2] as f32 * fx) * (1.0 - fy)
                    + (p01[2] as f32 * (1.0 - fx) + p11[2] as f32 * fx) * fy;

                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let ink = (1.0 - gray / 255.0).clamp(0.0, 1.0);
                input[[0, 0, ty, tx]] = ink;
            }
        }
    }

    Ok(input)
}

fn decode_linenet_det_boxes(
    preds: &Array4<f32>,
    orig_w: f32,
    orig_h: f32,
    scale_w: f32,
    scale_h: f32,
    det_thresh: f32,
    box_thresh: f32,
) -> (Vec<[Point2f; 4]>, Vec<f32>) {
    let (_, channels, out_h, out_w) = preds.dim();
    if channels < 3 {
        return (Vec::new(), Vec::new());
    }

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

            let mut min_x = x;
            let mut max_x = x;
            let mut sum_top = 0.0f32;
            let mut sum_bot = 0.0f32;
            let mut sum_prob = 0.0f32;
            let mut count = 0usize;

            while let Some((cx, cy)) = queue.pop_front() {
                let ys = (cy as f32 + 0.5) * 2.0;
                sum_top += ys - preds[[0, 1, cy, cx]];
                sum_bot += ys + preds[[0, 2, cy, cx]];
                sum_prob += prob[cy * out_w + cx];
                count += 1;

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
                        if !visited[nidx] && prob[nidx] >= det_thresh {
                            visited[nidx] = true;
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

            if (x1 - x0) > 4.0 && (y1 - y0) > 4.0 {
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
