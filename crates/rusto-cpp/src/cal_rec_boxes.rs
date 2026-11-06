//! Word bounding box calculation with OpenCV types.

#[cfg(feature = "opencv")]
use opencv::core::Point2f;
#[cfg(feature = "opencv")]
use rusto_core::ctc::WordInfo;

#[cfg(feature = "opencv")]
pub struct CalRecBoxes;

#[cfg(feature = "opencv")]
impl Default for CalRecBoxes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "opencv")]
impl CalRecBoxes {
    pub fn new() -> Self {
        Self
    }

    pub fn calc_word_boxes(
        &self,
        dt_boxes: &[[Point2f; 4]],
        txts: &[String],
        word_infos: &[Option<WordInfo>],
        return_single_char_box: bool,
    ) -> Vec<Vec<(String, f32, [Point2f; 4])>> {
        let converted_boxes: Vec<[rusto_core::image::Point2f; 4]> = dt_boxes
            .iter()
            .map(|b| [
                rusto_core::image::Point2f::new(b[0].x, b[0].y),
                rusto_core::image::Point2f::new(b[1].x, b[1].y),
                rusto_core::image::Point2f::new(b[2].x, b[2].y),
                rusto_core::image::Point2f::new(b[3].x, b[3].y),
            ])
            .collect();
        let res = rusto_core::cal_rec_boxes::CalRecBoxes::new().calc_word_boxes(
            &converted_boxes,
            txts,
            word_infos,
            return_single_char_box,
        );
        res.into_iter()
            .map(|line| {
                line.into_iter()
                    .map(|(text, score, pts)| {
                        let cv_pts = [
                            Point2f::new(pts[0].x, pts[0].y),
                            Point2f::new(pts[1].x, pts[1].y),
                            Point2f::new(pts[2].x, pts[2].y),
                            Point2f::new(pts[3].x, pts[3].y),
                        ];
                        (text, score, cv_pts)
                    })
                    .collect()
            })
            .collect()
    }
}

