use std::time::Instant;

use ndarray::{Array4, ArrayD};

use crate::engine::{EngineError, OrtSession};
use crate::postprocess::{DBPostProcess, TextDetOutput};
use crate::preprocess::DetPreProcess;
use crate::types::DetConfig;

#[cfg(feature = "use-opencv")]
use opencv::{core::Mat, prelude::MatTraitConst};

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

pub struct TextDetector {
    pub cfg: DetConfig,
    pub session: OrtSession,
    pub postprocess: DBPostProcess,
}

impl TextDetector {
    pub fn new(cfg: DetConfig) -> Result<Self, EngineError> {
        let session = OrtSession::from_det_config(&cfg)?;
        let postprocess = DBPostProcess::new(
            cfg.thresh,
            cfg.box_thresh,
            cfg.max_candidates,
            cfg.unclip_ratio,
            cfg.use_dilation,
        );
        Ok(Self {
            cfg,
            session,
            postprocess,
        })
    }

    pub fn run(&mut self, img: &Mat) -> Result<TextDetOutput, EngineError> {
        let start = Instant::now();

        let ori_h = img.rows();
        let ori_w = img.cols();
        let max_wh = ori_h.max(ori_w);

        let limit_side_len = if self.cfg.limit_type == "min" {
            self.cfg.limit_side_len
        } else if max_wh < 960 {
            960
        } else if max_wh < 1500 {
            1500
        } else {
            2000
        };

        let pre = DetPreProcess::new(
            limit_side_len,
            self.cfg.limit_type.clone(),
            self.cfg.mean,
            self.cfg.std,
        );
        let input = pre.run(img)?;
        let input_dyn: ArrayD<f32> = input.into_dyn();
        let preds_dyn = self.session.run(input_dyn)?;
        let preds: Array4<f32> = preds_dyn
            .into_dimensionality()
            .map_err(|_| EngineError::InvalidInputShape)?;
        let (mut boxes, scores) = self.postprocess.process(&preds, ori_h, ori_w)?;
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
