use std::time::Instant;

use ndarray::{Array4, ArrayD};

use crate::engine::{EngineError, MnnSession};
use crate::postprocess::{DBPostProcess, TextDetOutput};
use crate::preprocess::DetPreProcess;
use crate::rusto_ocr::{DetectionRunOptions, PostprocessRunOptions};
use crate::types::DetConfig;

#[cfg(feature = "use-opencv")]
use opencv::{core::Mat, prelude::MatTraitConst};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

pub struct TextDetector {
    pub cfg: DetConfig,
    pub session: MnnSession,
}

impl TextDetector {
    pub fn new(cfg: DetConfig) -> Result<Self, EngineError> {
        let session = MnnSession::from_det_config(&cfg)?;
        Ok(Self { cfg, session })
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
        // Same merge rule for postprocessing. Never mutate `self.cfg`.
        let postprocess_options = postprocess.cloned().unwrap_or_default();

        let ori_h = img.rows();
        let ori_w = img.cols();
        let max_wh = ori_h.max(ori_w);

        let limit_side_len = if limit_type == "min" {
            configured_limit_side_len
        } else if max_wh < 960 {
            960
        } else if max_wh < 1500 {
            1500
        } else if options.limit_side_len.is_some() {
            configured_limit_side_len
        } else {
            2000
        };

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
    fn sorted_boxes(&self, dt_boxes: &mut Vec<[opencv::core::Point2f; 4]>) {
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
    fn sorted_boxes(&self, dt_boxes: &mut Vec<[crate::image_impl::Point2f; 4]>) {
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
}
