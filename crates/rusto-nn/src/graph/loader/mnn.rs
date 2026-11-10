use smallvec::SmallVec;

use crate::error::MnnError;
use crate::graph::ir::{
    BinaryParams, Graph, GridSampleParams, InterpParams, LayerNormParams, MatMulParams, Node, OpKind, PoolParams,
    UnaryParams,
};
use crate::graph::prepack::prepack_op;
use crate::kernels::activation::FusedActivation;
use crate::kernels::binary::BinaryOpType;
use crate::kernels::interp::ResizeType;
use crate::kernels::pool::PoolType as CorePoolType;
use crate::kernels::reduction::ReductionOp;
use crate::kernels::unary::UnaryOpType;
use crate::schema::{
    self, BinaryOpOperation, OpType, PoolType as MnnPoolType, ReductionType, UnaryOpOperation,
};
use crate::tensor::{DataType, Tensor, TensorLayout};

pub fn load_mnn(slice: &[u8]) -> Result<Graph, MnnError> {
    if slice.len() < 4 {
        return Err(MnnError::InvalidModel("Buffer too small for MNN header".into()));
    }

    let net = schema::parse_net(slice);

    let mut tensor_names = Vec::new();
    if let Some(names) = net.tensorName() {
        tensor_names.reserve(names.len());
        for i in 0..names.len() {
            tensor_names.push(names.get(i).to_string());
        }
    }

    let mut input_names = Vec::new();
    let mut output_names = Vec::new();
    let mut input_channels = 3usize;

    if let Some(oplists) = net.oplists() {
        for i in 0..oplists.len() {
            let op = oplists.get(i);
            if op.type_() == OpType::Input {
                if let Some(name) = op.name() {
                    input_names.push(name.to_string());
                }
                if let Some(inp) = op.main_as_input() {
                    if let Some(dims) = inp.dims() {
                        if dims.len() >= 2 {
                            let c = dims.get(1);
                            if c > 0 {
                                input_channels = c as usize;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(outs) = net.outputName() {
        output_names.reserve(outs.len());
        for i in 0..outs.len() {
            output_names.push(outs.get(i).to_string());
        }
    }

    let mut nodes = Vec::new();
    if let Some(oplists) = net.oplists() {
        nodes.reserve(oplists.len());
        for i in 0..oplists.len() {
            let op = oplists.get(i);
            let name = op.name().unwrap_or("").to_string();

            let in_indices: SmallVec<[usize; 4]> = op
                .inputIndexes()
                .map(|v| v.iter().map(|idx| idx as usize).collect())
                .unwrap_or_default();

            let out_indices: SmallVec<[usize; 2]> = op
                .outputIndexes()
                .map(|v| v.iter().map(|idx| idx as usize).collect())
                .unwrap_or_default();

            let op_kind = match op.type_() {
                OpType::Input => OpKind::Input,

                OpType::Const => {
                    let tensor = if let Some(blob) = op.main_as_blob() {
                        let dims: Vec<usize> = blob
                            .dims()
                            .map(|d| d.iter().map(|v| v.max(0) as usize).collect())
                            .unwrap_or_else(|| vec![1]);

                        if let Some(floats) = blob.float32s() {
                            let v: Vec<f32> = floats.iter().collect();
                            Tensor::from_floats(&dims, v, TensorLayout::NCHW)
                        } else if let Some(ints) = blob.int32s() {
                            let v: Vec<i32> = ints.iter().collect();
                            Tensor::from_ints(&dims, v)
                        } else if let Some(i64s) = blob.int64s() {
                            let v: Vec<i32> = i64s.iter().map(|x| x as i32).collect();
                            Tensor::from_ints(&dims, v)
                        } else if let Some(u8s) = blob.uint8s() {
                            let v: Vec<i32> = u8s.iter().map(|&x| x as i32).collect();
                            Tensor::from_ints(&dims, v)
                        } else {
                            Tensor::zeros(&dims, TensorLayout::NCHW)
                        }
                    } else {
                        Tensor::zeros(&[1], TensorLayout::NCHW)
                    };
                    OpKind::Const(tensor.into_shared())
                }

                OpType::ConvertTensor => OpKind::ConvertTensor,

                OpType::Convolution | OpType::ConvolutionDepthwise => {
                    if let Some(crate::graph::prepack::PrepackedOp::Conv(conv)) = prepack_op(&op) {
                        OpKind::Conv(conv)
                    } else {
                        OpKind::NoOp
                    }
                }

                OpType::Deconvolution => {
                    if let Some(crate::graph::prepack::PrepackedOp::Deconv(deconv)) = prepack_op(&op) {
                        OpKind::Deconv(deconv)
                    } else {
                        OpKind::NoOp
                    }
                }

                OpType::Pooling | OpType::Pooling3D => {
                    let pool = op.main_as_pool();
                    let (pool_type, kernel_h, kernel_w, stride_h, stride_w, pad_mode, pad_x, pad_y, pads, is_global) = if let Some(p) = pool {
                        let p_type = match p.type_() {
                            MnnPoolType::AVEPOOL => CorePoolType::AvgPool,
                            _ => CorePoolType::MaxPool,
                        };
                        let kh = p.kernelY().max(1) as usize;
                        let kw = p.kernelX().max(1) as usize;
                        let sh = p.strideY().max(1) as usize;
                        let sw = p.strideX().max(1) as usize;
                        let pad_x = p.padX().max(0) as usize;
                        let pad_y = p.padY().max(0) as usize;
                        let pads = p.pads().map(|v| v.iter().collect());
                        let is_global = p.isGlobal();
                        let pad_m = match p.padType() {
                            schema::PoolPadType::SAME => schema::PadMode::SAME,
                            schema::PoolPadType::VALID => schema::PadMode::VALID,
                            _ => schema::PadMode::CAFFE,
                        };
                        (p_type, kh, kw, sh, sw, pad_m, pad_x, pad_y, pads, is_global)
                    } else if let Some(p3d) = op.main_as_pool_3d() {
                        let p_type = match p3d.type_() {
                            MnnPoolType::AVEPOOL => CorePoolType::AvgPool,
                            _ => CorePoolType::MaxPool,
                        };
                        let (kh, kw) = if let Some(k) = p3d.kernels() {
                            if k.len() >= 3 {
                                (k.get(1).max(1) as usize, k.get(2).max(1) as usize)
                            } else if k.len() == 2 {
                                (k.get(0).max(1) as usize, k.get(1).max(1) as usize)
                            } else if k.len() == 1 {
                                (k.get(0).max(1) as usize, k.get(0).max(1) as usize)
                            } else {
                                (1, 1)
                            }
                        } else {
                            (1, 1)
                        };
                        let (sh, sw) = if let Some(s) = p3d.strides() {
                            if s.len() >= 3 {
                                (s.get(1).max(1) as usize, s.get(2).max(1) as usize)
                            } else if s.len() == 2 {
                                (s.get(0).max(1) as usize, s.get(1).max(1) as usize)
                            } else if s.len() == 1 {
                                (s.get(0).max(1) as usize, s.get(0).max(1) as usize)
                            } else {
                                (1, 1)
                            }
                        } else {
                            (1, 1)
                        };
                        let pads = p3d.pads().map(|v| v.iter().collect());
                        let is_global = p3d.isGlobal();
                        let pad_m = match p3d.padType() {
                            schema::PoolPadType::SAME => schema::PadMode::SAME,
                            schema::PoolPadType::VALID => schema::PadMode::VALID,
                            _ => schema::PadMode::CAFFE,
                        };
                        (p_type, kh, kw, sh, sw, pad_m, 0, 0, pads, is_global)
                    } else {
                        (CorePoolType::MaxPool, 1, 1, 1, 1, schema::PadMode::CAFFE, 0, 0, None, false)
                    };

                    OpKind::Pool(PoolParams {
                        pool_type,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_mode,
                        pad_x,
                        pad_y,
                        pads,
                        is_global,
                    })
                }

                OpType::BinaryOp => {
                    let (op_type, act) = if let Some(b) = op.main_as_binary_op() {
                        let t = match b.opType() {
                            BinaryOpOperation::ADD => BinaryOpType::Add,
                            BinaryOpOperation::SUB => BinaryOpType::Sub,
                            BinaryOpOperation::MUL => BinaryOpType::Mul,
                            BinaryOpOperation::DIV | BinaryOpOperation::REALDIV => BinaryOpType::Div,
                            BinaryOpOperation::POW => BinaryOpType::Pow,
                            BinaryOpOperation::MIN_TEMP | BinaryOpOperation::MINIMUM => BinaryOpType::Min,
                            BinaryOpOperation::MAX_TEMP | BinaryOpOperation::MAXIMUM => BinaryOpType::Max,
                            BinaryOpOperation::GREATER => BinaryOpType::Greater,
                            BinaryOpOperation::GREATER_EQUAL => BinaryOpType::GreaterEqual,
                            BinaryOpOperation::LESS => BinaryOpType::Less,
                            BinaryOpOperation::LESS_EQUAL => BinaryOpType::LessEqual,
                            BinaryOpOperation::EQUAL => BinaryOpType::Equal,
                            BinaryOpOperation::NOTEQUAL => BinaryOpType::NotEqual,
                            BinaryOpOperation::FLOORDIV => BinaryOpType::FloorDiv,
                            BinaryOpOperation::FLOORMOD | BinaryOpOperation::MOD => BinaryOpType::FloorMod,
                            BinaryOpOperation::SquaredDifference => BinaryOpType::SquaredDifference,
                            BinaryOpOperation::MUL_SILU => BinaryOpType::MulSilu,
                            _ => BinaryOpType::Add,
                        };
                        let act = if b.activationType() == 1 {
                            FusedActivation::Relu
                        } else if b.activationType() == 2 {
                            FusedActivation::Relu6
                        } else {
                            FusedActivation::None
                        };
                        (t, act)
                    } else {
                        (BinaryOpType::Add, FusedActivation::None)
                    };

                    OpKind::Binary(BinaryParams {
                        op_type,
                        activation: act,
                    })
                }

                OpType::UnaryOp => {
                    let (op_type, table) = if let Some(u) = op.main_as_unary_op() {
                        let t = match u.opType() {
                            UnaryOpOperation::ABS => UnaryOpType::Abs,
                            UnaryOpOperation::NEG => UnaryOpType::Neg,
                            UnaryOpOperation::FLOOR => UnaryOpType::Floor,
                            UnaryOpOperation::CEIL => UnaryOpType::Ceil,
                            UnaryOpOperation::SQUARE => UnaryOpType::Square,
                            UnaryOpOperation::SQRT => UnaryOpType::Sqrt,
                            UnaryOpOperation::RSQRT => UnaryOpType::Rsqrt,
                            UnaryOpOperation::EXP => UnaryOpType::Exp,
                            UnaryOpOperation::LOG => UnaryOpType::Log,
                            UnaryOpOperation::SIN => UnaryOpType::Sin,
                            UnaryOpOperation::COS => UnaryOpType::Cos,
                            UnaryOpOperation::TAN => UnaryOpType::Tan,
                            UnaryOpOperation::RECIPROCAL => UnaryOpType::Reciprocal,
                            UnaryOpOperation::SIGMOID => UnaryOpType::Sigmoid,
                            UnaryOpOperation::TANH => UnaryOpType::Tanh,
                            UnaryOpOperation::HARDSWISH => UnaryOpType::HardSwish,
                            UnaryOpOperation::GELU => UnaryOpType::Gelu,
                            UnaryOpOperation::GELU_STANDARD => UnaryOpType::GeluStandard,
                            UnaryOpOperation::SILU => UnaryOpType::Silu,
                            UnaryOpOperation::ROUND => UnaryOpType::Round,
                            UnaryOpOperation::SIGN => UnaryOpType::Sign,
                            UnaryOpOperation::ERF => UnaryOpType::Erf,
                            UnaryOpOperation::BNLL => UnaryOpType::Bnll,
                            _ => UnaryOpType::Abs,
                        };
                        let tbl: Option<Vec<f32>> = u.tableInt8().map(|t| t.iter().map(|x| *x as f32).collect());
                        (t, tbl)
                    } else {
                        (UnaryOpType::Abs, None)
                    };

                    OpKind::Unary(UnaryParams {
                        op_type,
                        activation: FusedActivation::None,
                        table,
                    })
                }

                OpType::ReLU => {
                    let slope = op.main_as_relu().map(|r| r.slope()).unwrap_or(0.0);
                    OpKind::Relu { slope }
                }

                OpType::PReLU => {
                    let slopes = op.main_as_prelu().and_then(|p| p.slope().map(|s| s.iter().collect()));
                    OpKind::PRelu { slopes }
                }

                OpType::ReLU6 => {
                    let (min_val, max_val) = if let Some(r) = op.main_as_relu_6() {
                        (r.minValue(), r.maxValue())
                    } else {
                        (0.0, 6.0)
                    };
                    OpKind::Relu6 { min_val, max_val }
                }

                OpType::Reduction => {
                    let (red_op, axes, keep_dims) = if let Some(r) = op.main_as_reduction_param() {
                        let rop = match r.operation() {
                            ReductionType::SUM | ReductionType::ASUM | ReductionType::SUMSQ => ReductionOp::Sum,
                            ReductionType::MEAN => ReductionOp::Mean,
                            ReductionType::MAXIMUM => ReductionOp::Max,
                            ReductionType::MINIMUM => ReductionOp::Min,
                            ReductionType::PROD => ReductionOp::Prod,
                            _ => ReductionOp::Sum,
                        };
                        let ax = r.dim().map(|d| d.iter().collect()).unwrap_or_default();
                        (rop, ax, r.keepDims())
                    } else {
                        (ReductionOp::Sum, Vec::new(), false)
                    };
                    OpKind::Reduction {
                        op: red_op,
                        axes,
                        keep_dims,
                    }
                }

                OpType::Interp => {
                    let (resize_type, width_scale, height_scale, output_w, output_h, align_corners, half_pixel_centers) =
                        if let Some(interp) = op.main_as_interp() {
                            let rtype = match interp.resizeType() {
                                1 => ResizeType::Nearest,
                                2 => ResizeType::Bilinear,
                                3 => ResizeType::Cubic,
                                _ => ResizeType::Bilinear,
                            };
                            (
                                rtype,
                                interp.widthScale(),
                                interp.heightScale(),
                                interp.outputWidth(),
                                interp.outputHeight(),
                                interp.alignCorners(),
                                interp.halfPixelCenters(),
                            )
                        } else {
                            (ResizeType::Bilinear, 0.0, 0.0, 0, 0, false, false)
                        };

                    OpKind::Interp(InterpParams {
                        resize_type,
                        width_scale,
                        height_scale,
                        output_w,
                        output_h,
                        align_corners,
                        half_pixel_centers,
                    })
                }

                OpType::Reshape => {
                    let dims = op.main_as_reshape().and_then(|r| r.dims()).map(|d| d.iter().collect());
                    OpKind::Reshape { dims }
                }

                OpType::Permute => {
                    let dims = op
                        .main_as_permute()
                        .and_then(|p| p.dims())
                        .map(|d| d.iter().map(|v| v.max(0) as usize).collect())
                        .unwrap_or_default();
                    OpKind::Permute { dims }
                }

                OpType::Concat => {
                    let axis = op.main_as_axis().map(|a| a.axis()).unwrap_or(1);
                    OpKind::Concat { axis }
                }

                OpType::Slice => {
                    let (axis, slice_points) = if let Some(s) = op.main_as_slice() {
                        let ax = s.axis();
                        let pts = s.slicePoints().map(|p| p.iter().collect()).unwrap_or_default();
                        (ax, pts)
                    } else {
                        (1, Vec::new())
                    };
                    OpKind::Slice {
                        axis,
                        slice_points,
                    }
                }

                OpType::SliceTf => OpKind::SliceTf,

                OpType::StridedSlice => {
                    let (begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask) =
                        if let Some(param) = op.main_as_strided_slice_param() {
                            (
                                param.beginMask(),
                                param.endMask(),
                                param.ellipsisMask(),
                                param.newAxisMask(),
                                param.shrinkAxisMask(),
                            )
                        } else {
                            (0, 0, 0, 0, 0)
                        };
                    OpKind::StridedSlice {
                        begin_mask,
                        end_mask,
                        ellipsis_mask,
                        new_axis_mask,
                        shrink_axis_mask,
                        starts: None,
                        ends: None,
                        axes: None,
                        strides: None,
                    }
                }

                OpType::Squeeze => {
                    let axes = op
                        .main_as_squeeze_param()
                        .and_then(|p| p.squeezeDims())
                        .map(|d| d.iter().collect())
                        .unwrap_or_default();
                    OpKind::Squeeze { axes }
                }

                OpType::Unsqueeze => {
                    let axes = op
                        .main_as_squeeze_param()
                        .and_then(|p| p.squeezeDims())
                        .map(|d| d.iter().collect())
                        .unwrap_or_default();
                    OpKind::Unsqueeze { axes }
                }

                OpType::Scale => {
                    let (scale, bias) = if let Some(s) = op.main_as_scale() {
                        let sc = s.scaleData().map(|d| d.iter().collect()).unwrap_or_default();
                        let bi = s.biasData().map(|d| d.iter().collect());
                        (sc, bi)
                    } else {
                        (Vec::new(), None)
                    };
                    OpKind::Scale { scale, bias }
                }

                OpType::DepthToSpace => {
                    let (block_size, mode) = if let Some(d) = op.main_as_depth_space_param() {
                        (d.blockSize().max(1) as usize, d.mode() as i32)
                    } else {
                        (1, 0)
                    };
                    OpKind::DepthToSpace { block_size, mode }
                }

                OpType::Flatten => {
                    let (axis, end_axis) = if let Some(fl) = op.main_as_flatten() {
                        (fl.axis(), fl.endAxis())
                    } else if let Some(a) = op.main_as_axis() {
                        (a.axis(), 0)
                    } else {
                        (1, 0)
                    };
                    OpKind::Flatten { axis, end_axis }
                }

                OpType::Softmax => {
                    let axis = op.main_as_axis().map(|a| a.axis()).unwrap_or(1);
                    OpKind::Softmax { axis }
                }

                OpType::ArgMax => {
                    let (axis, out_max_val) = if let Some(a) = op.main_as_arg_max() {
                        (a.axis(), a.outMaxVal() != 0)
                    } else {
                        (0, false)
                    };
                    OpKind::ArgMax { axis, out_max_val }
                }

                OpType::TopKV2 => {
                    let (largest, sorted) = if let Some(t) = op.main_as_top_kv2() {
                        (t.largest(), t.sorted())
                    } else {
                        (true, true)
                    };
                    OpKind::TopKV2 {
                        k: 1,
                        axis: -1,
                        largest,
                        sorted,
                    }
                }

                OpType::Padding => {
                    let pad_value = 0.0f32;
                    OpKind::Pad { pad_value, is_interleaved: true }
                }

                OpType::Shape => OpKind::Shape,
                OpType::Rank => OpKind::Rank,
                OpType::Range => OpKind::Range,
                OpType::Select | OpType::Where => OpKind::Select,
                OpType::Fill => OpKind::ConstantOfShape { value: 0.0 },
                OpType::BroadcastTo => OpKind::Expand,

                OpType::Gather | OpType::GatherV2 => {
                    let axis = op.main_as_axis().map(|a| a.axis()).unwrap_or(0);
                    OpKind::Gather { axis }
                }

                OpType::Cast | OpType::CastLike => OpKind::Cast {
                    dst_type: DataType::Float,
                },

                OpType::GridSample => {
                    let (mode, padding_mode, align_corners) = if let Some(param) = op.main_as_grid_sample() {
                        (param.mode() as i32, param.paddingMode() as i32, param.alignCorners())
                    } else {
                        (0, 0, false)
                    };
                    OpKind::GridSample(GridSampleParams {
                        mode,
                        padding_mode,
                        align_corners,
                    })
                }

                OpType::MatMul | OpType::BatchMatMul => {
                    let (trans_a, trans_b) = if let Some(m) = op.main_as_mat_mul() {
                        (m.transposeA(), m.transposeB())
                    } else {
                        (false, false)
                    };
                    OpKind::MatMul(MatMulParams {
                        trans_a,
                        trans_b,
                        alpha: 1.0,
                        beta: 0.0,
                    })
                }

                OpType::LayerNorm => {
                    if let Some(ln) = op.main_as_layer_norm() {
                        let axis = ln.axis().and_then(|a| a.iter().next()).unwrap_or(-1);
                        let epsilon = ln.epsilon();
                        let use_rms_norm = ln.useRMSNorm();
                        let gamma = ln.gamma().map(|g| g.iter().collect());
                        let beta = ln.beta().map(|b| b.iter().collect());
                        OpKind::LayerNorm(LayerNormParams {
                            axis,
                            epsilon,
                            use_rms_norm,
                            gamma,
                            beta,
                        })
                    } else {
                        OpKind::NoOp
                    }
                }

                _ => OpKind::NoOp,
            };

            nodes.push(Node {
                name,
                in_indices,
                out_indices,
                op: op_kind,
            });
        }
    }

    Ok(Graph::new(
        nodes,
        tensor_names,
        input_names,
        output_names,
        input_channels,
    ))
}
