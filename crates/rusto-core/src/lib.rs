pub mod error;
pub mod image;
pub mod types;
pub mod config;
pub mod calibration;
pub mod contours;
pub mod geometry;
pub mod preprocess;
pub mod postprocess;
pub mod ctc;
pub mod cal_rec_boxes;
pub mod layout;
pub mod orient;
pub mod table;
#[doc(hidden)]
pub mod spatial;
pub mod unwarp;
pub mod models;
pub mod det;
pub mod rec;
pub mod doc;
pub mod tensor;
pub mod engine;
pub mod pipeline;
pub mod order;
pub mod order_blocks;
pub mod lines;
pub mod resample;
pub mod block;
pub use pipeline as pipelines;

pub use error::CoreError;
pub use image::{Mat, Point2f, Rect, Size, sample_bilinear};
pub use types::*;
pub use config::*;
pub use calibration::*;
pub use contours::*;
pub use geometry::*;
pub use preprocess::*;
pub use postprocess::*;
pub use ctc::*;
pub use cal_rec_boxes::*;
pub use det::TextDetOutput;
pub use rec::TextRecOutput;
pub use layout::{BBox, LayoutOutput, LayoutRegion, LayoutType};
pub use block::{cluster_text_results_into_blocks, BlockResult};
pub use doc::{DocBlock, DocPipelineConfig, DocResult};
pub use orient::{OrientOutput, Orientation};
pub use table::{
    TableCell, TableDetectorConfig, TableModelType, TableRow, TableStructure,
    TableStructureConfig,
};
pub use unwarp::UnwarpOutput;
pub use tensor::{Array2, Array3, Array4, ArrayD, Axis, Tensor};
pub use engine::*;
pub use order::*;

/// Returns the engine library version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

