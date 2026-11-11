use smallvec::SmallVec;

pub use crate::kernels::activation::FusedActivation;
pub use crate::kernels::binary::BinaryOpType;
pub use crate::kernels::interp::ResizeType;
pub use crate::kernels::pool::PoolType;
pub use crate::kernels::reduction::ReductionOp;
pub use crate::kernels::unary::UnaryOpType;
pub use crate::schema::PadMode;
use crate::tensor::{DataType, Tensor};

pub use super::prepack::{PrepackedConv, PrepackedDeconv};

#[derive(Clone, Debug)]
pub struct PoolParams {
    pub pool_type: PoolType,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_mode: PadMode,
    pub pad_x: usize,
    pub pad_y: usize,
    pub pads: Option<Vec<i32>>,
    pub is_global: bool,
}

#[derive(Clone, Debug)]
pub struct BinaryParams {
    pub op_type: BinaryOpType,
    pub activation: FusedActivation,
}

#[derive(Clone, Debug)]
pub struct UnaryParams {
    pub op_type: UnaryOpType,
    pub activation: FusedActivation,
    pub table: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct InterpParams {
    pub resize_type: ResizeType,
    pub width_scale: f32,
    pub height_scale: f32,
    pub output_w: i32,
    pub output_h: i32,
    pub align_corners: bool,
    pub half_pixel_centers: bool,
}

#[derive(Clone, Debug)]
pub struct MatMulParams {
    pub trans_a: bool,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Clone, Debug)]
pub struct GridSampleParams {
    pub mode: i32,
    pub padding_mode: i32,
    pub align_corners: bool,
}

#[derive(Clone, Debug)]
pub struct ConvIntegerParams {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub pad_mode: PadMode,
    pub pad_x: usize,
    pub pad_y: usize,
    pub pads: Option<Vec<i32>>,
    pub group: usize,
    pub has_x_zp: bool,
    pub has_w_zp: bool,
}

#[derive(Clone, Debug)]
pub struct MatMulIntegerParams {
    pub has_a_zp: bool,
    pub has_b_zp: bool,
}

#[derive(Clone, Debug)]
pub struct ConvInt8Params {
    pub weight: Vec<i8>,
    pub bias: Vec<i32>,
    pub scale: Vec<f32>,
    pub zero_point: i32,
    pub output_zero_point: i32,
    pub clamp_min: i8,
    pub clamp_max: i8,
    pub output_float: bool,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilate_h: usize,
    pub dilate_w: usize,
    pub pad_mode: PadMode,
    pub pad_x: usize,
    pub pad_y: usize,
    pub pads: Option<Vec<i32>>,
    pub group: usize,
    pub out_channels: usize,
    pub is_depthwise: bool,
}

#[derive(Clone, Debug)]
pub struct Int8ScaleParams {
    pub scale: Vec<f32>,
    pub zero_points: Vec<f32>,
    pub clamp_min: i8,
    pub clamp_max: i8,
}

#[derive(Clone, Debug)]
pub struct LayerNormParams {
    pub axis: i32,
    pub epsilon: f32,
    pub use_rms_norm: bool,
    pub gamma: Option<Vec<f32>>,
    pub beta: Option<Vec<f32>>,
}

#[derive(Clone)]
pub enum OpKind {
    Input,
    Const(Tensor),
    ConvertTensor,
    Conv(PrepackedConv),
    Deconv(PrepackedDeconv),
    Pool(PoolParams),
    Binary(BinaryParams),
    Unary(UnaryParams),
    Relu { slope: f32 },
    Relu6 { min_val: f32, max_val: f32 },
    Clip {
        min_val: f32,
        max_val: f32,
        has_min_input: bool,
        has_max_input: bool,
    },
    Reduction { op: ReductionOp, axes: Vec<i32>, keep_dims: bool },
    Interp(InterpParams),
    Reshape { dims: Option<Vec<i32>> },
    Permute { dims: Vec<usize> },
    Concat { axis: i32 },
    Slice { axis: i32, slice_points: Vec<i32> },
    SliceTf,
    StridedSlice {
        begin_mask: i32,
        end_mask: i32,
        ellipsis_mask: i32,
        new_axis_mask: i32,
        shrink_axis_mask: i32,
        starts: Option<Vec<i32>>,
        ends: Option<Vec<i32>>,
        axes: Option<Vec<i32>>,
        strides: Option<Vec<i32>>,
    },
    Squeeze { axes: Vec<i32> },
    Unsqueeze { axes: Vec<i32> },
    Scale { scale: Vec<f32>, bias: Option<Vec<f32>> },
    DepthToSpace { block_size: usize, mode: i32 },
    Flatten { axis: i32, end_axis: i32 },
    Softmax { axis: i32 },
    LogSoftmax { axis: i32 },
    ArgMax { axis: i32, out_max_val: bool },
    TopKV2 { k: usize, axis: i32, largest: bool, sorted: bool },
    Pad { pad_value: f32, is_interleaved: bool },
    Shape,
    Rank,
    Range,
    Select,
    Gather { axis: i32 },
    Cast { dst_type: DataType },
    GridSample(GridSampleParams),
    MatMul(MatMulParams),
    PRelu { slopes: Option<Vec<f32>> },
    ConstantOfShape { value: f32, int_value: Option<i32> },
    LayerNorm(LayerNormParams),
    Expand,
    ScatterND,
    DynamicQuant { signed: bool },
    ConvInteger(ConvIntegerParams),
    MatMulInteger(MatMulIntegerParams),
    ConvInt8(ConvInt8Params),
    Int8ToFloat(Int8ScaleParams),
    FloatToInt8(Int8ScaleParams),
    NoOp,
}

impl OpKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Const(_) => "Const",
            Self::ConvertTensor => "ConvertTensor",
            Self::Conv(c) => if c.is_depthwise { "ConvolutionDepthwise" } else { "Convolution" },
            Self::Deconv(_) => "Deconvolution",
            Self::Pool(_) => "Pooling",
            Self::Binary(_) => "BinaryOp",
            Self::Unary(_) => "UnaryOp",
            Self::Relu { .. } => "ReLU",
            Self::Relu6 { .. } => "ReLU6",
            Self::Clip { .. } => "Clip",
            Self::Reduction { .. } => "Reduction",
            Self::Interp(_) => "Interp",
            Self::Reshape { .. } => "Reshape",
            Self::Permute { .. } => "Permute",
            Self::Concat { .. } => "Concat",
            Self::Slice { .. } => "Slice",
            Self::SliceTf => "SliceTf",
            Self::StridedSlice { .. } => "StridedSlice",
            Self::Squeeze { .. } => "Squeeze",
            Self::Unsqueeze { .. } => "Unsqueeze",
            Self::Scale { .. } => "Scale",
            Self::DepthToSpace { .. } => "DepthToSpace",
            Self::Flatten { .. } => "Flatten",
            Self::Softmax { .. } => "Softmax",
            Self::LogSoftmax { .. } => "LogSoftmax",
            Self::ArgMax { .. } => "ArgMax",
            Self::TopKV2 { .. } => "TopKV2",
            Self::Pad { .. } => "Padding",
            Self::Shape => "Shape",
            Self::Rank => "Rank",
            Self::Range => "Range",
            Self::Select => "Select",
            Self::Gather { .. } => "Gather",
            Self::Cast { .. } => "Cast",
            Self::GridSample(_) => "GridSample",
            Self::MatMul(_) => "MatMul",
            Self::PRelu { .. } => "PRelu",
            Self::ConstantOfShape { .. } => "ConstantOfShape",
            Self::LayerNorm(_) => "LayerNorm",
            Self::Expand => "Expand",
            Self::ScatterND => "ScatterND",
            Self::DynamicQuant { .. } => "DynamicQuant",
            Self::ConvInteger(_) => "ConvInteger",
            Self::MatMulInteger(_) => "MatMulInteger",
            Self::ConvInt8(c) => if c.is_depthwise { "DepthwiseConvInt8" } else { "ConvInt8" },
            Self::Int8ToFloat(_) => "Int8ToFloat",
            Self::FloatToInt8(_) => "FloatToInt8",
            Self::NoOp => "NoOp",
        }
    }
}

#[derive(Clone)]
pub struct Node {
    pub name: String,
    pub in_indices: SmallVec<[usize; 4]>,
    pub out_indices: SmallVec<[usize; 2]>,
    pub op: OpKind,
}

#[derive(Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensor_names: Vec<String>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub input_channels: usize,
    pub last_use: Vec<usize>,
}

impl Graph {
    pub fn new(
        nodes: Vec<Node>,
        tensor_names: Vec<String>,
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_channels: usize,
    ) -> Self {
        let mut graph = Self {
            nodes,
            tensor_names,
            input_names,
            output_names,
            input_channels,
            last_use: Vec::new(),
        };
        super::optimize::optimize_graph(&mut graph);
        graph
    }
}
