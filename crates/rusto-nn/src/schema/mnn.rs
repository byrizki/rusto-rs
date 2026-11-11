#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms)]

use super::reader::{FromTable, Scalar, ScalarVec, StringVec, Table, TableVec};

/// Implements `from_*` via bounded discriminant transmute and `Scalar` bit-coding
/// for contiguous C-like MNN schema enums (`0..=$max`).
macro_rules! impl_contiguous_enum_scalar {
    ($ty:ident, i8, $max:expr, $default:expr, $from_fn:ident) => {
        impl $ty {
            #[inline]
            pub fn $from_fn(v: i8) -> Self {
                if (v as u8) <= ($max as u8) {
                    // Safety: `$ty` is `#[repr(i8)]` with contiguous discriminants `0..=$max`.
                    unsafe { std::mem::transmute(v) }
                } else {
                    $default
                }
            }
        }
        impl Scalar for $ty {
            #[inline]
            fn to_bits(self) -> u64 {
                self as i8 as u8 as u64
            }
            #[inline]
            fn from_bits(bits: u64) -> Self {
                Self::$from_fn(bits as u8 as i8)
            }
        }
    };
    ($ty:ident, u8, $max:expr, $default:expr, $from_fn:ident) => {
        impl $ty {
            #[inline]
            pub fn $from_fn(v: u8) -> Self {
                if v <= $max {
                    // Safety: `$ty` is `#[repr(u8)]` with contiguous discriminants `0..=$max`.
                    unsafe { std::mem::transmute(v) }
                } else {
                    $default
                }
            }
        }
        impl Scalar for $ty {
            #[inline]
            fn to_bits(self) -> u64 {
                self as u8 as u64
            }
            #[inline]
            fn from_bits(bits: u64) -> Self {
                Self::$from_fn(bits as u8)
            }
        }
    };
    ($ty:ident, i32, $max:expr, $default:expr, $from_fn:ident) => {
        impl $ty {
            #[inline]
            pub fn $from_fn(v: i32) -> Self {
                if (v as u32) <= ($max as u32) {
                    // Safety: `$ty` is `#[repr(i32)]` with contiguous discriminants `0..=$max`.
                    unsafe { std::mem::transmute(v) }
                } else {
                    $default
                }
            }
        }
        impl Scalar for $ty {
            #[inline]
            fn to_bits(self) -> u64 {
                self as i32 as u32 as u64
            }
            #[inline]
            fn from_bits(bits: u64) -> Self {
                Self::$from_fn(bits as u32 as i32)
            }
        }
    };
}

macro_rules! define_op_types {
    ($($name:ident = $val:expr),* $(,)?) => {
        impl OpType {
            $(
                #[allow(non_upper_case_globals)]
                pub const $name: Self = Self($val);
            )*

            fn variant_name(v: i32) -> Option<&'static str> {
                match v {
                    $($val => Some(stringify!($name)),)*
                    _ => None,
                }
            }
        }
    };
}

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum NetSource {
    CAFFE = 0,
    TENSORFLOW = 1,
    TFLITE = 2,
    ONNX = 3,
    TORCH = 4,
}

impl_contiguous_enum_scalar!(NetSource, i8, 4, Self::CAFFE, from_i8);

#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DataType {
    DT_INVALID = 0,
    DT_FLOAT = 1,
    DT_DOUBLE = 2,
    DT_INT32 = 3,
    DT_UINT8 = 4,
    DT_INT16 = 5,
    DT_INT8 = 6,
    DT_STRING = 7,
    DT_COMPLEX64 = 8,
    DT_INT64 = 9,
    DT_BOOL = 10,
    DT_QINT8 = 11,
    DT_QUINT8 = 12,
    DT_QINT32 = 13,
    DT_BFLOAT16 = 14,
    DT_QINT16 = 15,
    DT_QUINT16 = 16,
    DT_UINT16 = 17,
    DT_COMPLEX128 = 18,
    DT_HALF = 19,
    DT_RESOURCE = 20,
    DT_VARIANT = 21,
}

impl_contiguous_enum_scalar!(DataType, i32, 21, Self::DT_INVALID, from_i32);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MNN_DATA_FORMAT {
    NCHW = 0,
    NHWC = 1,
    NC4HW4 = 2,
    NHWC4 = 3,
    UNKNOWN = 4,
}

impl_contiguous_enum_scalar!(MNN_DATA_FORMAT, i8, 4, Self::UNKNOWN, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PadMode {
    CAFFE = 0,
    VALID = 1,
    SAME = 2,
}

impl_contiguous_enum_scalar!(PadMode, i8, 2, Self::CAFFE, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PoolType {
    MAXPOOL = 0,
    AVEPOOL = 1,
}

impl_contiguous_enum_scalar!(PoolType, i8, 1, Self::MAXPOOL, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PoolPadType {
    CAFFE = 0,
    VALID = 1,
    SAME = 2,
}

impl_contiguous_enum_scalar!(PoolPadType, i8, 2, Self::CAFFE, from_i8);

#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BinaryOpOperation {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    MAX_TEMP = 4,
    MIN_TEMP = 5,
    POW = 6,
    REALDIV = 7,
    MINIMUM = 8,
    MAXIMUM = 9,
    GREATER = 10,
    GREATER_EQUAL = 11,
    LESS = 12,
    FLOORDIV = 13,
    SquaredDifference = 14,
    EQUAL = 15,
    LESS_EQUAL = 16,
    FLOORMOD = 17,
    MOD = 19,
    ATAN2 = 20,
    LOGICALOR = 21,
    NOTEQUAL = 22,
    BITWISE_AND = 23,
    BITWISE_OR = 24,
    BITWISE_XOR = 25,
    LOGICALXOR = 26,
    LEFTSHIFT = 27,
    RIGHTSHIFT = 28,
    MUL_SILU = 29,
}

impl BinaryOpOperation {
    #[inline]
    pub fn from_i32(v: i32) -> Self {
        if (v as u32) <= 29 && v != 18 {
            // Safety: `BinaryOpOperation` is `#[repr(i32)]` covering `0..=29` excluding `18`.
            unsafe { std::mem::transmute(v) }
        } else {
            Self::ADD
        }
    }
}

impl Scalar for BinaryOpOperation {
    #[inline]
    fn to_bits(self) -> u64 {
        self as i32 as u32 as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        Self::from_i32(bits as u32 as i32)
    }
}

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ReductionType {
    SUM = 0,
    ASUM = 1,
    SUMSQ = 2,
    MEAN = 3,
    MAXIMUM = 4,
    MINIMUM = 5,
    PROD = 6,
    ANY = 7,
    ALL = 8,
}

impl_contiguous_enum_scalar!(ReductionType, i8, 8, Self::SUM, from_i8);

#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum UnaryOpOperation {
    ABS = 0,
    NEG = 1,
    FLOOR = 2,
    CEIL = 3,
    SQUARE = 4,
    SQRT = 5,
    RSQRT = 6,
    EXP = 7,
    LOG = 8,
    SIN = 9,
    COS = 10,
    TAN = 11,
    ASIN = 12,
    ACOS = 13,
    ATAN = 14,
    RECIPROCAL = 15,
    LOG1P = 16,
    BNLL = 17,
    ACOSH = 18,
    SINH = 19,
    ASINH = 20,
    ATANH = 21,
    SIGN = 22,
    ROUND = 23,
    COSH = 24,
    ERF = 25,
    ERFC = 26,
    ERFINV = 27,
    EXPM1 = 28,
    SIGMOID = 29,
    TANH = 30,
    HARDSWISH = 31,
    GELU = 32,
    GELU_STANDARD = 33,
    SILU = 34,
}

impl_contiguous_enum_scalar!(UnaryOpOperation, i32, 34, Self::ABS, from_i32);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DepthToSpaceMode {
    DCR = 0,
    CRD = 1,
}

impl_contiguous_enum_scalar!(DepthToSpaceMode, i8, 1, Self::DCR, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SampleMode {
    BILINEAR = 0,
    NEAREST = 1,
}

impl_contiguous_enum_scalar!(SampleMode, i8, 1, Self::BILINEAR, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BorderMode {
    ZEROS = 0,
    CLAMP = 1,
    REFLECTION = 2,
    CUBE = 3,
}

impl_contiguous_enum_scalar!(BorderMode, i8, 3, Self::ZEROS, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ForwardType {
    CPU = 0,
    METAL = 1,
    CUDA = 2,
    OPENCL = 3,
    AUTO = 4,
    NNAPI = 5,
    OPENGLES = 6,
    VULKAN = 7,
}

impl_contiguous_enum_scalar!(ForwardType, i8, 7, Self::CPU, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PadValueMode {
    CONSTANT = 0,
    REFLECT = 1,
    SYMMETRIC = 2,
    EDGE = 3,
}

impl_contiguous_enum_scalar!(PadValueMode, i8, 3, Self::CONSTANT, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum QuantizeAlgo {
    DEFAULT = 0,
    OVERFLOW_AWARE = 1,
    WINOGRAD_AWARE = 2,
}

impl_contiguous_enum_scalar!(QuantizeAlgo, i8, 2, Self::DEFAULT, from_i8);

#[allow(non_camel_case_types)]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ScaleStorageType {
    FP32 = 0,
    FP16 = 1,
}

impl_contiguous_enum_scalar!(ScaleStorageType, i8, 1, Self::FP32, from_i8);

/// MNN operator opcode stored as an `i32` FlatBuffers wire scalar.
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpType(pub i32);

impl OpType {
    #[inline]
    pub const fn from_i32(v: i32) -> Self {
        Self(v)
    }
}

impl Scalar for OpType {
    #[inline]
    fn to_bits(self) -> u64 {
        self.0 as u32 as u64
    }
    #[inline]
    fn from_bits(bits: u64) -> Self {
        Self(bits as u32 as i32)
    }
}

impl std::fmt::Debug for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = Self::variant_name(self.0) {
            f.write_str(name)
        } else {
            write!(f, "_Unknown({})", self.0)
        }
    }
}

define_op_types!(
    AbsVal = 0,
    QuantizedAdd = 1,
    ArgMax = 2,
    AsString = 3,
    InstanceNorm = 4,
    BatchToSpaceND = 5,
    Copy = 6,
    BinaryOp = 7,
    Bnll = 8,
    Cast = 9,
    Concat = 10,
    Const = 11,
    Convolution = 12,
    ConvolutionDepthwise = 13,
    Crop = 14,
    CropAndResize = 15,
    ImageProcess = 16,
    Deconvolution = 17,
    DeconvolutionDepthwise = 18,
    Dequantize = 19,
    DetectionOutput = 20,
    Dropout = 21,
    Eltwise = 22,
    ELU = 23,
    Unique = 24,
    Exp = 25,
    ExpandDims = 26,
    Fill = 27,
    Flatten = 28,
    Im2Col = 29,
    Gather = 30,
    GatherV2 = 31,
    Im2Seq = 32,
    InnerProduct = 33,
    Input = 34,
    Interp = 35,
    Log = 36,
    LRN = 37,
    LSTM = 38,
    MatMul = 39,
    MoE = 40,
    NonMaxSuppression = 41,
    NonMaxSuppressionV2 = 42,
    Normalize = 43,
    Pack = 44,
    Padding = 45,
    Permute = 46,
    Pooling = 47,
    Power = 48,
    PReLU = 49,
    PriorBox = 50,
    Proposal = 51,
    QuantizedAvgPool = 52,
    QuantizedBiasAdd = 53,
    QuantizedConcat = 54,
    QuantizedDepthwiseConv2D = 55,
    QuantizedLogistic = 56,
    RasterAndInterpolate = 57,
    QuantizedMaxPool = 58,
    Texture = 59,
    RasterDiff = 60,
    QuantizedReshape = 61,
    QuantizedSoftmax = 62,
    QuantizeMaxMin = 63,
    QuantizeV2 = 64,
    Range = 65,
    Rank = 66,
    ReduceJoin = 67,
    Reduction = 68,
    ReLU = 69,
    ReLU6 = 70,
    RequantizationRange = 71,
    Requantize = 72,
    Reshape = 73,
    Resize = 74,
    RNN = 75,
    ROIPooling = 76,
    Scale = 77,
    Selu = 78,
    Seq2Out = 79,
    Shape = 80,
    Sigmoid = 81,
    Size = 82,
    Slice = 83,
    SliceTf = 84,
    Softmax = 85,
    SpaceToBatchND = 86,
    SpatialProduct = 87,
    Col2Im = 88,
    Segment = 89,
    Squeeze = 90,
    StridedSlice = 91,
    CastLike = 92,
    StringSplit = 93,
    StringToNumber = 94,
    TanH = 95,
    TfQuantizedConv2D = 96,
    Threshold = 97,
    Tile = 98,
    TopKV2 = 99,
    Transpose = 100,
    UnaryOp = 101,
    Unpack = 102,
    Where = 103,
    Moments = 104,
    RNNSequenceGRU = 105,
    BatchMatMul = 106,
    Unsqueeze = 107,
    CosineSimilarity = 108,
    DepthToSpace = 109,
    SpaceToDepth = 110,
    ReverseSequence = 111,
    Pooling3D = 112,
    Convolution3D = 113,
    MatrixBandPart = 114,
    GatherND = 115,
    DetectionPostProcess = 116,
    UnravelIndex = 117,
    ScatterNd = 118,
    OneHot = 119,
    BroadcastTo = 120,
    Dilation2D = 121,
    Interp3D = 122,
    Raster = 128,
    ConvertTensor = 129,
    ArgMin = 130,
    LinSpace = 131,
    RandomUniform = 132,
    TensorArray = 133,
    TensorArraySize = 134,
    TensorArrayRead = 135,
    TensorArrayWrite = 136,
    TensorArrayGather = 137,
    TensorArrayScatter = 138,
    TensorArraySplit = 139,
    TensorArrayConcat = 140,
    LSTMBlockCell = 141,
    Reverse = 142,
    ROIAlign = 143,
    RandomNormal = 144,
    TensorArrayInsert = 145,
    TensorArrayErase = 146,
    EyeLike = 147,
    CumSum = 148,
    Det = 149,
    CumProd = 150,
    ScatterElements = 151,
    GatherElements = 152,
    Svd = 153,
    Histogram = 154,
    DynamicQuant = 155,
    Stft = 156,
    Plugin = 256,
    Select = 257,
    ZerosLike = 258,
    Broastcast = 259,
    SetDiff1D = 260,
    ReluGrad = 261,
    Identity = 262,
    PoolGrad = 263,
    SoftmaxGrad = 264,
    Conv2DBackPropFilter = 265,
    TrainableParam = 266,
    BatchNorm = 267,
    ConvTranspose3D = 268,
    ZeroGrad = 269,
    Attention = 299,
    FmhaV2 = 300,
    Fmhca = 301,
    SeqLen2Spatial = 302,
    SplitGeLU = 303,
    GroupNorm = 304,
    LinearAttention = 305,
    RoPE = 306,
    FusedLinear = 307,
    GatedRMSNorm = 308,
    IndexedLinear = 309,
    Extra = 512,
    ConvInt8 = 513,
    Int8ToFloat = 514,
    DepthwiseConvInt8 = 515,
    FloatToInt8 = 517,
    EltwiseInt8 = 518,
    While = 600,
    If = 601,
    LayerNorm = 603,
    GridSample = 604,
);

#[allow(non_camel_case_types)]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum OpParameter {
    NONE = 0,
    QuantizedAdd = 1,
    ArgMax = 2,
    AsString = 3,
    Axis = 4,
    BatchNorm = 5,
    BinaryOp = 6,
    Blob = 7,
    CastParam = 8,
    Convolution2D = 9,
    Crop = 10,
    CropAndResize = 11,
    Dequantize = 12,
    DetectionOutput = 13,
    Eltwise = 14,
    ExpandDims = 15,
    Fill = 16,
    Flatten = 17,
    Gather = 18,
    GatherV2 = 19,
    InnerProduct = 20,
    Input = 21,
    Interp = 22,
    LRN = 23,
    LSTM = 24,
    MatMul = 25,
    NonMaxSuppressionV2 = 26,
    Normalize = 27,
    PackParam = 28,
    Permute = 29,
    Plugin = 30,
    Pool = 31,
    PRelu = 32,
    PriorBox = 33,
    Proposal = 34,
    QuantizedAvgPool = 35,
    QuantizedBiasAdd = 36,
    QuantizedConcat = 37,
    QuantizedLogistic = 38,
    QuantizedMatMul = 39,
    QuantizedMaxPool = 40,
    QuantizedRelu = 41,
    QuantizedRelu6 = 42,
    QuantizedReshape = 43,
    QuantizedSoftmax = 44,
    QuantizeMaxMin = 45,
    QuantizeV2 = 46,
    Range = 47,
    Rank = 48,
    ReduceJoin = 49,
    ReductionParam = 50,
    Relu = 51,
    Relu6 = 52,
    RequantizationRange = 53,
    Requantize = 54,
    Reshape = 55,
    Resize = 56,
    RoiParameters = 57,
    Scale = 58,
    Selu = 59,
    Size = 60,
    Slice = 61,
    SliceTf = 62,
    SpaceBatch = 63,
    SqueezeParam = 64,
    StridedSliceParam = 65,
    TensorConvertInfo = 66,
    TfQuantizedConv2D = 67,
    TopKV2 = 68,
    Transpose = 69,
    UnaryOp = 70,
    MomentsParam = 71,
    RNNParam = 72,
    BatchMatMulParam = 73,
    QuantizedFloatParam = 74,
    DepthSpaceParam = 75,
    EltwiseInt8 = 76,
    ReverseSequenceParam = 77,
    Extra = 78,
    Pool3D = 79,
    Convolution3D = 80,
    ELU = 81,
    DetectionPostProcessParam = 82,
    OneHotParam = 83,
    PadParam = 84,
    WhileParam = 85,
    IfParam = 86,
    RandomUniform = 87,
    LayerNorm = 88,
    TensorArray = 89,
    LSTMBlockCell = 90,
    GridSample = 91,
    LoopParam = 92,
    ImageProcessParam = 93,
    CumSum = 94,
    GroupNorm = 95,
    FmhaV2Param = 96,
    FmhcaParam = 97,
    AttentionParam = 98,
    StftParam = 99,
    LinearAttentionParam = 100,
    ShapeParam = 101,
    RoPEParam = 102,
    FusedLinearParam = 103,
    IndexedLinearParam = 104,
}

impl_contiguous_enum_scalar!(OpParameter, u8, 104, Self::NONE, from_u8);

// ---------------------------------------------------------------------------
// Typed Table Wrappers
// ---------------------------------------------------------------------------

macro_rules! define_table {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(transparent)]
        pub struct $name<'a> {
            pub _tab: Table<'a>,
        }

        impl<'a> FromTable<'a> for $name<'a> {
            #[inline(always)]
            fn from_table(table: Table<'a>) -> Self {
                Self { _tab: table }
            }
        }

        impl<'a> $name<'a> {
            #[inline(always)]
            pub fn init_from_table(table: Table<'a>) -> Self {
                Self { _tab: table }
            }
        }
    };
}

define_table!(Net);
define_table!(Op);
define_table!(Input);
define_table!(Blob);
define_table!(Convolution2D);
define_table!(Convolution2DCommon);
define_table!(Pool);
define_table!(Pool3D);
define_table!(BinaryOp);
define_table!(UnaryOp);
define_table!(Relu);
define_table!(PRelu);
define_table!(Relu6);
define_table!(ReductionParam);
define_table!(Interp);
define_table!(Reshape);
define_table!(Permute);
define_table!(Axis);
define_table!(Slice);
define_table!(StridedSliceParam);
define_table!(Scale);
define_table!(DepthSpaceParam);
define_table!(Flatten);
define_table!(ArgMax);
define_table!(TopKV2);
define_table!(GridSample);
define_table!(MatMul);
define_table!(LayerNorm);
define_table!(SqueezeParam);
define_table!(CastParam);
define_table!(IDSTQuan);
define_table!(QuantizedFloatParam);

// --- Net Methods ---
impl<'a> Net<'a> {
    pub const VT_BIZCODE: usize = 4;
    pub const VT_OPLISTS: usize = 10;
    pub const VT_OUTPUTNAME: usize = 12;
    pub const VT_PREFERFORWARDTYPE: usize = 14;
    pub const VT_SOURCETYPE: usize = 16;
    pub const VT_TENSORNAME: usize = 18;
    pub const VT_TENSORNUMBER: usize = 20;

    pub fn bizCode(&self) -> Option<&'a str> {
        self._tab.get_string(Self::VT_BIZCODE)
    }

    pub fn oplists(&self) -> Option<TableVec<'a, Op<'a>>> {
        self._tab.get_table_vec(Self::VT_OPLISTS)
    }

    pub fn outputName(&self) -> Option<StringVec<'a>> {
        self._tab.get_string_vec(Self::VT_OUTPUTNAME)
    }

    pub fn preferForwardType(&self) -> ForwardType {
        self._tab.get_scalar(Self::VT_PREFERFORWARDTYPE, ForwardType::CPU)
    }

    pub fn sourceType(&self) -> NetSource {
        self._tab.get_scalar(Self::VT_SOURCETYPE, NetSource::CAFFE)
    }

    pub fn tensorName(&self) -> Option<StringVec<'a>> {
        self._tab.get_string_vec(Self::VT_TENSORNAME)
    }

    pub fn tensorNumber(&self) -> i32 {
        self._tab.get_scalar(Self::VT_TENSORNUMBER, 0)
    }
}

// --- Op Methods ---
impl<'a> Op<'a> {
    pub const VT_INPUTINDEXES: usize = 4;
    pub const VT_MAIN_TYPE: usize = 6;
    pub const VT_MAIN: usize = 8;
    pub const VT_NAME: usize = 10;
    pub const VT_OUTPUTINDEXES: usize = 12;
    pub const VT_TYPE_: usize = 14;
    pub const VT_DEFAULTDIMENTIONFORMAT: usize = 16;
    pub const VT_EXTERNALPATH: usize = 18;

    pub fn inputIndexes(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_INPUTINDEXES)
    }

    pub fn main_type(&self) -> OpParameter {
        self._tab.get_scalar(Self::VT_MAIN_TYPE, OpParameter::NONE)
    }

    pub fn main(&self) -> Option<Table<'a>> {
        self._tab.get_table(Self::VT_MAIN)
    }

    pub fn name(&self) -> Option<&'a str> {
        self._tab.get_string(Self::VT_NAME)
    }

    pub fn outputIndexes(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_OUTPUTINDEXES)
    }

    pub fn type_(&self) -> OpType {
        self._tab.get_scalar(Self::VT_TYPE_, OpType::AbsVal)
    }

    pub fn defaultDimentionFormat(&self) -> MNN_DATA_FORMAT {
        self._tab.get_scalar(Self::VT_DEFAULTDIMENTIONFORMAT, MNN_DATA_FORMAT::NHWC)
    }

    pub fn externalPath(&self) -> Option<&'a str> {
        self._tab.get_string(Self::VT_EXTERNALPATH)
    }

    #[inline(never)]
    fn main_as_table(&self, expected: OpParameter) -> Option<Table<'a>> {
        if self.main_type() == expected {
            self.main()
        } else {
            None
        }
    }
}

macro_rules! define_op_main_as {
    ($($method:ident => $variant:ident : $ty:ident),* $(,)?) => {
        impl<'a> Op<'a> {
            $(
                pub fn $method(&self) -> Option<$ty<'a>> {
                    self.main_as_table(OpParameter::$variant)
                        .map($ty::init_from_table)
                }
            )*
        }
    };
}

define_op_main_as!(
    main_as_input => Input: Input,
    main_as_blob => Blob: Blob,
    main_as_convolution_2d => Convolution2D: Convolution2D,
    main_as_pool => Pool: Pool,
    main_as_pool_3d => Pool3D: Pool3D,
    main_as_binary_op => BinaryOp: BinaryOp,
    main_as_unary_op => UnaryOp: UnaryOp,
    main_as_relu => Relu: Relu,
    main_as_prelu => PRelu: PRelu,
    main_as_relu_6 => Relu6: Relu6,
    main_as_reduction_param => ReductionParam: ReductionParam,
    main_as_interp => Interp: Interp,
    main_as_reshape => Reshape: Reshape,
    main_as_permute => Permute: Permute,
    main_as_axis => Axis: Axis,
    main_as_slice => Slice: Slice,
    main_as_strided_slice_param => StridedSliceParam: StridedSliceParam,
    main_as_squeeze_param => SqueezeParam: SqueezeParam,
    main_as_scale => Scale: Scale,
    main_as_depth_space_param => DepthSpaceParam: DepthSpaceParam,
    main_as_flatten => Flatten: Flatten,
    main_as_arg_max => ArgMax: ArgMax,
    main_as_top_kv2 => TopKV2: TopKV2,
    main_as_grid_sample => GridSample: GridSample,
    main_as_mat_mul => MatMul: MatMul,
    main_as_layer_norm => LayerNorm: LayerNorm,
    main_as_cast_param => CastParam: CastParam,
    main_as_quantized_float_param => QuantizedFloatParam: QuantizedFloatParam,
);

// --- Input Methods ---
impl<'a> Input<'a> {
    pub const VT_DIMS: usize = 4;
    pub const VT_DTYPE: usize = 6;
    pub const VT_DFORMAT: usize = 8;

    pub fn dims(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_DIMS)
    }

    pub fn dtype(&self) -> DataType {
        self._tab.get_scalar(Self::VT_DTYPE, DataType::DT_FLOAT)
    }

    pub fn dformat(&self) -> MNN_DATA_FORMAT {
        self._tab.get_scalar(Self::VT_DFORMAT, MNN_DATA_FORMAT::NC4HW4)
    }
}

// --- Blob Methods ---
impl<'a> Blob<'a> {
    pub const VT_DIMS: usize = 4;
    pub const VT_DATAFORMAT: usize = 6;
    pub const VT_DATATYPE: usize = 8;
    pub const VT_UINT8S: usize = 10;
    pub const VT_INT8S: usize = 12;
    pub const VT_INT32S: usize = 14;
    pub const VT_INT64S: usize = 16;
    pub const VT_FLOAT32S: usize = 18;
    pub const VT_STRINGS: usize = 20;

    pub fn dims(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_DIMS)
    }

    pub fn dataFormat(&self) -> MNN_DATA_FORMAT {
        self._tab.get_scalar(Self::VT_DATAFORMAT, MNN_DATA_FORMAT::NCHW)
    }

    pub fn dataType(&self) -> DataType {
        self._tab.get_scalar(Self::VT_DATATYPE, DataType::DT_FLOAT)
    }

    pub fn uint8s(&self) -> Option<&'a [u8]> {
        self._tab.get_byte_slice(Self::VT_UINT8S)
    }

    pub fn int8s(&self) -> Option<&'a [i8]> {
        self._tab.get_i8_slice(Self::VT_INT8S)
    }

    pub fn int32s(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_INT32S)
    }

    pub fn int64s(&self) -> Option<ScalarVec<'a, i64>> {
        self._tab.get_scalar_vec(Self::VT_INT64S)
    }

    pub fn float32s(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_FLOAT32S)
    }

    pub fn strings(&self) -> Option<StringVec<'a>> {
        self._tab.get_string_vec(Self::VT_STRINGS)
    }
}

// --- Convolution2D Methods ---
impl<'a> Convolution2D<'a> {
    pub const VT_COMMON: usize = 4;
    pub const VT_WEIGHT: usize = 6;
    pub const VT_BIAS: usize = 8;
    pub const VT_QUANPARAMETER: usize = 10;
    pub const VT_SYMMETRICQUAN: usize = 12;

    pub fn common(&self) -> Option<Convolution2DCommon<'a>> {
        self._tab.get_table(Self::VT_COMMON)
    }

    pub fn weight(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_WEIGHT)
    }

    pub fn bias(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_BIAS)
    }

    pub fn quanParameter(&self) -> Option<IDSTQuan<'a>> {
        self._tab.get_table(Self::VT_QUANPARAMETER)
    }

    pub fn symmetricQuan(&self) -> Option<QuantizedFloatParam<'a>> {
        self._tab.get_table(Self::VT_SYMMETRICQUAN)
    }
}

// --- Convolution2DCommon Methods ---
impl<'a> Convolution2DCommon<'a> {
    pub const VT_PADX: usize = 4;
    pub const VT_PADY: usize = 6;
    pub const VT_KERNELX: usize = 8;
    pub const VT_KERNELY: usize = 10;
    pub const VT_STRIDEX: usize = 12;
    pub const VT_STRIDEY: usize = 14;
    pub const VT_DILATEX: usize = 16;
    pub const VT_DILATEY: usize = 18;
    pub const VT_PADMODE: usize = 20;
    pub const VT_GROUP: usize = 22;
    pub const VT_OUTPUTCOUNT: usize = 24;
    pub const VT_INPUTCOUNT: usize = 26;
    pub const VT_RELU: usize = 28;
    pub const VT_RELU6: usize = 30;
    pub const VT_PADS: usize = 32;
    pub const VT_OUTPADS: usize = 34;

    pub fn padX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_PADX, 0)
    }

    pub fn padY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_PADY, 0)
    }

    pub fn kernelX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_KERNELX, 1)
    }

    pub fn kernelY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_KERNELY, 1)
    }

    pub fn strideX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_STRIDEX, 1)
    }

    pub fn strideY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_STRIDEY, 1)
    }

    pub fn dilateX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_DILATEX, 1)
    }

    pub fn dilateY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_DILATEY, 1)
    }

    pub fn padMode(&self) -> PadMode {
        self._tab.get_scalar(Self::VT_PADMODE, PadMode::CAFFE)
    }

    pub fn group(&self) -> i32 {
        self._tab.get_scalar(Self::VT_GROUP, 1)
    }

    pub fn outputCount(&self) -> i32 {
        self._tab.get_scalar(Self::VT_OUTPUTCOUNT, 0)
    }

    pub fn inputCount(&self) -> i32 {
        self._tab.get_scalar(Self::VT_INPUTCOUNT, 0)
    }

    pub fn relu(&self) -> bool {
        self._tab.get_scalar(Self::VT_RELU, false)
    }

    pub fn relu6(&self) -> bool {
        self._tab.get_scalar(Self::VT_RELU6, false)
    }

    pub fn pads(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_PADS)
    }

    pub fn outPads(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_OUTPADS)
    }
}

// --- Pool Methods ---
impl<'a> Pool<'a> {
    pub const VT_PADX: usize = 4;
    pub const VT_PADY: usize = 6;
    pub const VT_ISGLOBAL: usize = 8;
    pub const VT_KERNELX: usize = 10;
    pub const VT_KERNELY: usize = 12;
    pub const VT_STRIDEX: usize = 14;
    pub const VT_STRIDEY: usize = 16;
    pub const VT_TYPE_: usize = 18;
    pub const VT_PADTYPE: usize = 20;
    pub const VT_PADS: usize = 26;

    pub fn padX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_PADX, 0)
    }

    pub fn padY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_PADY, 0)
    }

    pub fn isGlobal(&self) -> bool {
        self._tab.get_scalar(Self::VT_ISGLOBAL, false)
    }

    pub fn kernelX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_KERNELX, 0)
    }

    pub fn kernelY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_KERNELY, 0)
    }

    pub fn strideX(&self) -> i32 {
        self._tab.get_scalar(Self::VT_STRIDEX, 0)
    }

    pub fn strideY(&self) -> i32 {
        self._tab.get_scalar(Self::VT_STRIDEY, 0)
    }

    pub fn type_(&self) -> PoolType {
        self._tab.get_scalar(Self::VT_TYPE_, PoolType::MAXPOOL)
    }

    pub fn padType(&self) -> PoolPadType {
        self._tab.get_scalar(Self::VT_PADTYPE, PoolPadType::CAFFE)
    }

    pub fn pads(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_PADS)
    }
}

// --- Pool3D Methods ---
impl<'a> Pool3D<'a> {
    pub const VT_STRIDES: usize = 4;
    pub const VT_KERNELS: usize = 6;
    pub const VT_PADS: usize = 8;
    pub const VT_TYPE_: usize = 10;
    pub const VT_PADTYPE: usize = 12;
    pub const VT_ISGLOBAL: usize = 14;

    pub fn strides(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_STRIDES)
    }

    pub fn kernels(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_KERNELS)
    }

    pub fn pads(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_PADS)
    }

    pub fn type_(&self) -> PoolType {
        self._tab.get_scalar(Self::VT_TYPE_, PoolType::MAXPOOL)
    }

    pub fn padType(&self) -> PoolPadType {
        self._tab.get_scalar(Self::VT_PADTYPE, PoolPadType::CAFFE)
    }

    pub fn isGlobal(&self) -> bool {
        self._tab.get_scalar(Self::VT_ISGLOBAL, false)
    }
}

// --- BinaryOp Methods ---
impl<'a> BinaryOp<'a> {
    pub const VT_OPTYPE: usize = 4;
    pub const VT_T: usize = 6;
    pub const VT_ACTIVATIONTYPE: usize = 8;

    pub fn opType(&self) -> BinaryOpOperation {
        self._tab.get_scalar(Self::VT_OPTYPE, BinaryOpOperation::ADD)
    }

    pub fn T(&self) -> DataType {
        self._tab.get_scalar(Self::VT_T, DataType::DT_FLOAT)
    }

    pub fn activationType(&self) -> i32 {
        self._tab.get_scalar(Self::VT_ACTIVATIONTYPE, 0)
    }
}

// --- UnaryOp Methods ---
impl<'a> UnaryOp<'a> {
    pub const VT_OPTYPE: usize = 4;
    pub const VT_T: usize = 6;
    pub const VT_TABLEINT8: usize = 8;

    pub fn opType(&self) -> UnaryOpOperation {
        self._tab.get_scalar(Self::VT_OPTYPE, UnaryOpOperation::ABS)
    }

    pub fn T(&self) -> DataType {
        self._tab.get_scalar(Self::VT_T, DataType::DT_INVALID)
    }

    pub fn tableInt8(&self) -> Option<&'a [i8]> {
        self._tab.get_i8_slice(Self::VT_TABLEINT8)
    }
}

// --- Relu Methods ---
impl<'a> Relu<'a> {
    pub const VT_SLOPE: usize = 4;

    pub fn slope(&self) -> f32 {
        self._tab.get_scalar(Self::VT_SLOPE, 0.0)
    }
}

// --- PRelu Methods ---
impl<'a> PRelu<'a> {
    pub const VT_SLOPECOUNT: usize = 4;
    pub const VT_SLOPE: usize = 6;

    pub fn slopeCount(&self) -> i32 {
        self._tab.get_scalar(Self::VT_SLOPECOUNT, 0)
    }

    pub fn slope(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_SLOPE)
    }
}

// --- Relu6 Methods ---
impl<'a> Relu6<'a> {
    pub const VT_MINVALUE: usize = 4;
    pub const VT_MAXVALUE: usize = 6;

    pub fn minValue(&self) -> f32 {
        self._tab.get_scalar(Self::VT_MINVALUE, 0.0)
    }

    pub fn maxValue(&self) -> f32 {
        self._tab.get_scalar(Self::VT_MAXVALUE, 6.0)
    }
}

// --- ReductionParam Methods ---
impl<'a> ReductionParam<'a> {
    pub const VT_OPERATION: usize = 4;
    pub const VT_DIM: usize = 6;
    pub const VT_COEFF: usize = 8;
    pub const VT_KEEPDIMS: usize = 10;
    pub const VT_DTYPE: usize = 12;

    pub fn operation(&self) -> ReductionType {
        self._tab.get_scalar(Self::VT_OPERATION, ReductionType::SUM)
    }

    pub fn dim(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_DIM)
    }

    pub fn coeff(&self) -> f32 {
        self._tab.get_scalar(Self::VT_COEFF, 0.0)
    }

    pub fn keepDims(&self) -> bool {
        self._tab.get_scalar(Self::VT_KEEPDIMS, false)
    }

    pub fn dType(&self) -> DataType {
        self._tab.get_scalar(Self::VT_DTYPE, DataType::DT_FLOAT)
    }
}

// --- Interp Methods ---
impl<'a> Interp<'a> {
    pub const VT_WIDTHSCALE: usize = 4;
    pub const VT_HEIGHTSCALE: usize = 6;
    pub const VT_OUTPUTWIDTH: usize = 8;
    pub const VT_OUTPUTHEIGHT: usize = 10;
    pub const VT_RESIZETYPE: usize = 12;
    pub const VT_ALIGNCORNERS: usize = 14;
    pub const VT_HALFPIXELCENTERS: usize = 16;

    pub fn widthScale(&self) -> f32 {
        self._tab.get_scalar(Self::VT_WIDTHSCALE, 0.0)
    }

    pub fn heightScale(&self) -> f32 {
        self._tab.get_scalar(Self::VT_HEIGHTSCALE, 0.0)
    }

    pub fn outputWidth(&self) -> i32 {
        self._tab.get_scalar(Self::VT_OUTPUTWIDTH, 0)
    }

    pub fn outputHeight(&self) -> i32 {
        self._tab.get_scalar(Self::VT_OUTPUTHEIGHT, 0)
    }

    pub fn resizeType(&self) -> i32 {
        self._tab.get_scalar(Self::VT_RESIZETYPE, 0)
    }

    pub fn alignCorners(&self) -> bool {
        self._tab.get_scalar(Self::VT_ALIGNCORNERS, false)
    }

    pub fn halfPixelCenters(&self) -> bool {
        self._tab.get_scalar(Self::VT_HALFPIXELCENTERS, false)
    }
}

// --- Reshape Methods ---
impl<'a> Reshape<'a> {
    pub const VT_DIMS: usize = 4;
    pub const VT_DIMTYPE: usize = 6;

    pub fn dims(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_DIMS)
    }

    pub fn dimType(&self) -> MNN_DATA_FORMAT {
        self._tab.get_scalar(Self::VT_DIMTYPE, MNN_DATA_FORMAT::NCHW)
    }
}

// --- Permute Methods ---
impl<'a> Permute<'a> {
    pub const VT_DIMS: usize = 4;

    pub fn dims(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_DIMS)
    }
}

// --- Axis Methods ---
impl<'a> Axis<'a> {
    pub const VT_AXIS: usize = 4;

    pub fn axis(&self) -> i32 {
        self._tab.get_scalar(Self::VT_AXIS, 0)
    }
}

// --- Slice Methods ---
impl<'a> Slice<'a> {
    pub const VT_AXIS: usize = 4;
    pub const VT_SLICEPOINTS: usize = 6;
    pub const VT_SOURCETYPE: usize = 8;

    pub fn axis(&self) -> i32 {
        self._tab.get_scalar(Self::VT_AXIS, 0)
    }

    pub fn slicePoints(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_SLICEPOINTS)
    }

    pub fn sourceType(&self) -> NetSource {
        self._tab.get_scalar(Self::VT_SOURCETYPE, NetSource::CAFFE)
    }
}

// --- StridedSliceParam Methods ---
impl<'a> StridedSliceParam<'a> {
    pub const VT_INDEX: usize = 4;
    pub const VT_T: usize = 6;
    pub const VT_BEGINMASK: usize = 8;
    pub const VT_ENDMASK: usize = 10;
    pub const VT_ELLIPSISMASK: usize = 12;
    pub const VT_NEWAXISMASK: usize = 14;
    pub const VT_SHRINKAXISMASK: usize = 16;

    pub fn Index(&self) -> DataType {
        self._tab.get_scalar(Self::VT_INDEX, DataType::DT_INVALID)
    }

    pub fn T(&self) -> DataType {
        self._tab.get_scalar(Self::VT_T, DataType::DT_INVALID)
    }

    pub fn beginMask(&self) -> i32 {
        self._tab.get_scalar(Self::VT_BEGINMASK, 0)
    }

    pub fn endMask(&self) -> i32 {
        self._tab.get_scalar(Self::VT_ENDMASK, 0)
    }

    pub fn ellipsisMask(&self) -> i32 {
        self._tab.get_scalar(Self::VT_ELLIPSISMASK, 0)
    }

    pub fn newAxisMask(&self) -> i32 {
        self._tab.get_scalar(Self::VT_NEWAXISMASK, 0)
    }

    pub fn shrinkAxisMask(&self) -> i32 {
        self._tab.get_scalar(Self::VT_SHRINKAXISMASK, 0)
    }
}

// --- Scale Methods ---
impl<'a> Scale<'a> {
    pub const VT_CHANNELS: usize = 4;
    pub const VT_SCALEDATA: usize = 6;
    pub const VT_BIASDATA: usize = 8;

    pub fn channels(&self) -> i32 {
        self._tab.get_scalar(Self::VT_CHANNELS, 0)
    }

    pub fn scaleData(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_SCALEDATA)
    }

    pub fn biasData(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_BIASDATA)
    }
}

// --- DepthSpaceParam Methods ---
impl<'a> DepthSpaceParam<'a> {
    pub const VT_BLOCKSIZE: usize = 4;
    pub const VT_MODE: usize = 6;

    pub fn blockSize(&self) -> i32 {
        self._tab.get_scalar(Self::VT_BLOCKSIZE, 0)
    }

    pub fn mode(&self) -> DepthToSpaceMode {
        self._tab.get_scalar(Self::VT_MODE, DepthToSpaceMode::DCR)
    }
}

// --- Flatten Methods ---
impl<'a> Flatten<'a> {
    pub const VT_AXIS: usize = 4;
    pub const VT_ENDAXIS: usize = 6;

    pub fn axis(&self) -> i32 {
        self._tab.get_scalar(Self::VT_AXIS, 0)
    }

    pub fn endAxis(&self) -> i32 {
        self._tab.get_scalar(Self::VT_ENDAXIS, 0)
    }
}

// --- ArgMax Methods ---
impl<'a> ArgMax<'a> {
    pub const VT_OUTMAXVAL: usize = 4;
    pub const VT_TOPK: usize = 6;
    pub const VT_AXIS: usize = 8;
    pub const VT_SOFTMAXTHRESHOLD: usize = 10;

    pub fn outMaxVal(&self) -> i32 {
        self._tab.get_scalar(Self::VT_OUTMAXVAL, 0)
    }

    pub fn topK(&self) -> i32 {
        self._tab.get_scalar(Self::VT_TOPK, 0)
    }

    pub fn axis(&self) -> i32 {
        self._tab.get_scalar(Self::VT_AXIS, 0)
    }

    pub fn softmaxThreshold(&self) -> i32 {
        self._tab.get_scalar(Self::VT_SOFTMAXTHRESHOLD, 0)
    }
}

// --- TopKV2 Methods ---
impl<'a> TopKV2<'a> {
    pub const VT_T: usize = 4;
    pub const VT_SORTED: usize = 6;
    pub const VT_LARGEST: usize = 8;

    pub fn T(&self) -> DataType {
        self._tab.get_scalar(Self::VT_T, DataType::DT_FLOAT)
    }

    pub fn sorted(&self) -> bool {
        self._tab.get_scalar(Self::VT_SORTED, false)
    }

    pub fn largest(&self) -> bool {
        self._tab.get_scalar(Self::VT_LARGEST, true)
    }
}

// --- GridSample Methods ---
impl<'a> GridSample<'a> {
    pub const VT_MODE: usize = 4;
    pub const VT_PADDINGMODE: usize = 6;
    pub const VT_ALIGNCORNERS: usize = 8;
    pub const VT_BACKWARD: usize = 10;

    pub fn mode(&self) -> SampleMode {
        self._tab.get_scalar(Self::VT_MODE, SampleMode::BILINEAR)
    }

    pub fn paddingMode(&self) -> BorderMode {
        self._tab.get_scalar(Self::VT_PADDINGMODE, BorderMode::ZEROS)
    }

    pub fn alignCorners(&self) -> bool {
        self._tab.get_scalar(Self::VT_ALIGNCORNERS, false)
    }

    pub fn backward(&self) -> bool {
        self._tab.get_scalar(Self::VT_BACKWARD, false)
    }
}

// --- MatMul Methods ---
impl<'a> MatMul<'a> {
    pub const VT_T: usize = 4;
    pub const VT_TRANSPOSEA: usize = 6;
    pub const VT_TRANSPOSEB: usize = 8;
    pub const VT_WEIGHT: usize = 10;
    pub const VT_BIAS: usize = 12;

    pub fn T(&self) -> DataType {
        self._tab.get_scalar(Self::VT_T, DataType::DT_INVALID)
    }

    pub fn transposeA(&self) -> bool {
        self._tab.get_scalar(Self::VT_TRANSPOSEA, false)
    }

    pub fn transposeB(&self) -> bool {
        self._tab.get_scalar(Self::VT_TRANSPOSEB, false)
    }

    pub fn weight(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_WEIGHT)
    }

    pub fn bias(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_BIAS)
    }
}

// --- LayerNorm Methods ---
impl<'a> LayerNorm<'a> {
    pub const VT_AXIS: usize = 4;
    pub const VT_EPSILON: usize = 6;
    pub const VT_GAMMA: usize = 8;
    pub const VT_BETA: usize = 10;
    pub const VT_GROUP: usize = 12;
    pub const VT_USERMSNORM: usize = 16;

    pub fn axis(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_AXIS)
    }

    pub fn epsilon(&self) -> f32 {
        self._tab.get_scalar(Self::VT_EPSILON, 0.0)
    }

    pub fn gamma(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_GAMMA)
    }

    pub fn beta(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_BETA)
    }

    pub fn group(&self) -> i32 {
        self._tab.get_scalar(Self::VT_GROUP, 1)
    }

    pub fn useRMSNorm(&self) -> bool {
        self._tab.get_scalar(Self::VT_USERMSNORM, false)
    }
}

// --- SqueezeParam Methods ---
impl<'a> SqueezeParam<'a> {
    pub const VT_SQUEEZEDIMS: usize = 4;

    pub fn squeezeDims(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_SQUEEZEDIMS)
    }
}

// --- CastParam Methods ---
impl<'a> CastParam<'a> {
    pub const VT_SRCT: usize = 4;
    pub const VT_DSTT: usize = 6;

    pub fn srcT(&self) -> DataType {
        self._tab.get_scalar(Self::VT_SRCT, DataType::DT_INVALID)
    }

    pub fn dstT(&self) -> DataType {
        self._tab.get_scalar(Self::VT_DSTT, DataType::DT_INVALID)
    }
}

// --- IDSTQuan Methods ---
impl<'a> IDSTQuan<'a> {
    pub const VT_BUFFER: usize = 4;
    pub const VT_ALPHA: usize = 6;
    pub const VT_TYPE_: usize = 8;
    pub const VT_USEINT32: usize = 10;
    pub const VT_QUANTSCALE: usize = 12;
    pub const VT_SCALEIN: usize = 14;
    pub const VT_SCALEOUT: usize = 16;
    pub const VT_AMAXORBITS: usize = 18;
    pub const VT_AMIN: usize = 20;
    pub const VT_READTYPE: usize = 22;
    pub const VT_HAS_SCALEINT: usize = 24;
    pub const VT_SHAPEINT32: usize = 26;
    pub const VT_WEIGHTSIZE: usize = 28;
    pub const VT_INDEX: usize = 30;
    pub const VT_ALPHAFP16: usize = 32;
    pub const VT_SCALESTORAGE: usize = 34;

    pub fn buffer(&self) -> Option<&'a [i8]> {
        self._tab.get_i8_slice(Self::VT_BUFFER)
    }

    pub fn alpha(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_ALPHA)
    }

    pub fn type_(&self) -> i32 {
        self._tab.get_scalar(Self::VT_TYPE_, 0)
    }

    pub fn useInt32(&self) -> bool {
        self._tab.get_scalar(Self::VT_USEINT32, false)
    }

    pub fn quantScale(&self) -> f32 {
        self._tab.get_scalar(Self::VT_QUANTSCALE, 0.0)
    }

    pub fn scaleIn(&self) -> f32 {
        self._tab.get_scalar(Self::VT_SCALEIN, 0.0)
    }

    pub fn scaleOut(&self) -> f32 {
        self._tab.get_scalar(Self::VT_SCALEOUT, 0.0)
    }

    pub fn aMaxOrBits(&self) -> i32 {
        self._tab.get_scalar(Self::VT_AMAXORBITS, 0)
    }

    pub fn aMin(&self) -> i32 {
        self._tab.get_scalar(Self::VT_AMIN, 0)
    }

    pub fn readType(&self) -> i32 {
        self._tab.get_scalar(Self::VT_READTYPE, 0)
    }

    pub fn has_scaleInt(&self) -> bool {
        self._tab.get_scalar(Self::VT_HAS_SCALEINT, false)
    }

    pub fn shapeInt32(&self) -> bool {
        self._tab.get_scalar(Self::VT_SHAPEINT32, false)
    }

    pub fn weightSize(&self) -> u32 {
        self._tab.get_scalar(Self::VT_WEIGHTSIZE, 0)
    }

    pub fn index(&self) -> Option<ScalarVec<'a, u32>> {
        self._tab.get_scalar_vec(Self::VT_INDEX)
    }

    pub fn alphaFp16(&self) -> Option<ScalarVec<'a, u16>> {
        self._tab.get_scalar_vec(Self::VT_ALPHAFP16)
    }

    pub fn scaleStorage(&self) -> ScaleStorageType {
        self._tab.get_scalar(Self::VT_SCALESTORAGE, ScaleStorageType::FP32)
    }
}

// --- QuantizedFloatParam Methods ---
impl<'a> QuantizedFloatParam<'a> {
    pub const VT_WEIGHT: usize = 4;
    pub const VT_BIAS: usize = 6;
    pub const VT_SCALE: usize = 8;
    pub const VT_TENSORSCALE: usize = 10;
    pub const VT_METHOD: usize = 12;
    pub const VT_NBITS: usize = 14;
    pub const VT_ZEROPOINT: usize = 16;
    pub const VT_OUTPUTZEROPOINT: usize = 18;
    pub const VT_CLAMPMIN: usize = 20;
    pub const VT_CLAMPMAX: usize = 22;
    pub const VT_WINOGRADATTR: usize = 24;
    pub const VT_OUTPUTDATATYPE: usize = 26;
    pub const VT_FLOATZEROS: usize = 28;

    pub fn weight(&self) -> Option<&'a [i8]> {
        self._tab.get_i8_slice(Self::VT_WEIGHT)
    }

    pub fn bias(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_BIAS)
    }

    pub fn scale(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_SCALE)
    }

    pub fn tensorScale(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_TENSORSCALE)
    }

    pub fn method(&self) -> QuantizeAlgo {
        self._tab.get_scalar(Self::VT_METHOD, QuantizeAlgo::DEFAULT)
    }

    pub fn nbits(&self) -> i32 {
        self._tab.get_scalar(Self::VT_NBITS, 8)
    }

    pub fn zeroPoint(&self) -> i8 {
        self._tab.get_scalar(Self::VT_ZEROPOINT, 0)
    }

    pub fn outputZeroPoint(&self) -> i8 {
        self._tab.get_scalar(Self::VT_OUTPUTZEROPOINT, 0)
    }

    pub fn clampMin(&self) -> i8 {
        self._tab.get_scalar(Self::VT_CLAMPMIN, -128)
    }

    pub fn clampMax(&self) -> i8 {
        self._tab.get_scalar(Self::VT_CLAMPMAX, 127)
    }

    pub fn winogradAttr(&self) -> Option<ScalarVec<'a, i32>> {
        self._tab.get_scalar_vec(Self::VT_WINOGRADATTR)
    }

    pub fn outputDataType(&self) -> DataType {
        self._tab.get_scalar(Self::VT_OUTPUTDATATYPE, DataType::DT_INT8)
    }

    pub fn floatzeros(&self) -> Option<ScalarVec<'a, f32>> {
        self._tab.get_scalar_vec(Self::VT_FLOATZEROS)
    }
}

/// Parse root Net from byte buffer.
pub fn parse_net<'a>(buf: &'a [u8]) -> Net<'a> {
    if buf.len() < 4 {
        return Net {
            _tab: Table::new(buf, 0),
        };
    }
    let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    Net {
        _tab: Table::new(buf, root_offset),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_layout_and_debug() {
        assert_eq!(std::mem::size_of::<OpType>(), 4);
        assert_eq!(OpType::WIRE_SIZE, 4);
        assert_eq!(OpType::from_bits(OpType::Convolution.to_bits()), OpType::Convolution);
        assert_eq!(OpType::from_i32(12), OpType::Convolution);
        assert_eq!(OpType::from_i32(-5).0, -5);
        assert_eq!(format!("{:?}", OpType::Input), "Input");
        assert_eq!(format!("{:?}", OpType::from_i32(9999)), "_Unknown(9999)");
    }

    #[test]
    fn test_contiguous_and_sparse_enum_scalar_roundtrips() {
        for v in 0..=4i8 {
            assert_eq!(NetSource::from_i8(v) as i8, v);
            assert_eq!(MNN_DATA_FORMAT::from_i8(v) as i8, v);
        }
        assert_eq!(NetSource::from_i8(-1), NetSource::CAFFE);
        assert_eq!(NetSource::from_i8(5), NetSource::CAFFE);
        assert_eq!(MNN_DATA_FORMAT::from_i8(-1), MNN_DATA_FORMAT::UNKNOWN);
        assert_eq!(MNN_DATA_FORMAT::from_i8(5), MNN_DATA_FORMAT::UNKNOWN);
        assert_eq!(
            MNN_DATA_FORMAT::from_bits(MNN_DATA_FORMAT::NHWC.to_bits()),
            MNN_DATA_FORMAT::NHWC
        );

        for v in 0..=21i32 {
            let dt = DataType::from_i32(v);
            assert_eq!(dt as i32, v);
            assert_eq!(DataType::from_bits(dt.to_bits()), dt);
        }
        assert_eq!(DataType::from_i32(-1), DataType::DT_INVALID);
        assert_eq!(DataType::from_i32(22), DataType::DT_INVALID);

        for v in 0..=34i32 {
            let u = UnaryOpOperation::from_i32(v);
            assert_eq!(u as i32, v);
            assert_eq!(UnaryOpOperation::from_bits(u.to_bits()), u);
        }
        assert_eq!(UnaryOpOperation::from_i32(35), UnaryOpOperation::ABS);

        for v in 0..=29i32 {
            let b = BinaryOpOperation::from_i32(v);
            if v == 18 {
                assert_eq!(b, BinaryOpOperation::ADD);
            } else {
                assert_eq!(b as i32, v);
                assert_eq!(BinaryOpOperation::from_bits(b.to_bits()), b);
            }
        }
        assert_eq!(BinaryOpOperation::from_i32(30), BinaryOpOperation::ADD);

        for v in 0..=1i8 {
            let s = ScaleStorageType::from_i8(v);
            assert_eq!(s as i8, v);
            assert_eq!(ScaleStorageType::from_bits(s.to_bits()), s);
        }
        assert_eq!(ScaleStorageType::from_i8(2), ScaleStorageType::FP32);

        for v in 0..=104u8 {
            let p = OpParameter::from_u8(v);
            assert_eq!(p as u8, v);
            assert_eq!(OpParameter::from_bits(p.to_bits()), p);
        }
        assert_eq!(OpParameter::from_u8(105), OpParameter::NONE);
        assert_eq!(OpParameter::from_u8(255), OpParameter::NONE);
    }
}
