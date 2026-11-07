use std::time::Instant;

use crate::tensor::Array4;

use crate::engine::InferenceSession;
use crate::error::CoreError;
use crate::image::{Mat, Point2f};
pub use crate::postprocess::{decode_det_predictions, decode_linenet_det_boxes, sort_det_boxes, TextDetOutput};
pub use crate::preprocess::{calculate_linenet_dims, prepare_det_input, prepare_linenet_det_input};
use crate::preprocess::resolve_det_params;
use crate::types::{DetConfig, DetectionRunOptions, PostprocessRunOptions};

/// Text detection pipeline component executing DBNet or LineNet detection.
pub struct TextDetector<S = Box<dyn InferenceSession>> {
    pub cfg: DetConfig,
    pub session: S,
}

impl<S: InferenceSession> TextDetector<S> {
    pub fn with_session(cfg: DetConfig, session: S) -> Self {
        Self { cfg, session }
    }

    pub fn is_linenet(&self) -> bool {
        self.session.input_channels() == 1
    }

    pub fn run(&mut self, img: &Mat) -> Result<TextDetOutput, CoreError> {
        self.run_with_options(img, None, None)
    }

    pub fn run_with_options(
        &mut self,
        img: &Mat,
        options: Option<&DetectionRunOptions>,
        postprocess: Option<&PostprocessRunOptions>,
    ) -> Result<TextDetOutput, CoreError> {
        let start = Instant::now();
        let (ori_h, ori_w) = (img.rows(), img.cols());
        let params = resolve_det_params(&self.cfg, options, postprocess, ori_w, ori_h);

        let (input, scale_w, scale_h) =
            prepare_det_input(img, self.is_linenet(), &params, ori_w, ori_h)?;
        let preds: Array4<f32> = self
            .session
            .run(input.into_dyn())?
            .into_dimensionality()
            .map_err(|_| CoreError::InvalidShape("detection output must be 4D".into()))?;

        let (mut boxes, scores) = decode_det_predictions(
            &preds,
            self.is_linenet(),
            ori_w,
            ori_h,
            scale_w,
            scale_h,
            &params,
        )?;

        if boxes.is_empty() {
            return Ok(TextDetOutput::empty());
        }

        sort_det_boxes(&mut boxes);

        Ok(TextDetOutput {
            img: None,
            boxes: Some(boxes),
            scores: Some(scores),
            elapse: start.elapsed().as_secs_f64(),
        })
    }

    pub fn sorted_boxes(&self, dt_boxes: &mut [[Point2f; 4]]) {
        sort_det_boxes(dt_boxes);
    }
}
