use rusto_nn::kernels::activation::FusedActivation;
use rusto_nn::kernels::binary::{binary_op, BinaryOpType};
use rusto_nn::kernels::classification::{argmax, softmax, topk};
use rusto_nn::kernels::conv::{conv2d, Conv2dConfig};
use rusto_nn::kernels::deconv::{deconv2d, Deconv2dConfig};
use rusto_nn::kernels::depthwise::{depthwise_conv2d_nchw, DepthwiseConv2dConfig};
use rusto_nn::kernels::interp::{interp2d, InterpConfig, ResizeType};
use rusto_nn::kernels::pool::{pool2d, Pool2dConfig, PoolType};
use rusto_nn::kernels::reduction::{reduce, ReductionOp};
use rusto_nn::kernels::tensor_ops::{compute_reshape_dims, concat, permute, slice_tf};
use rusto_nn::kernels::unary::{unary_op, UnaryOpType};

#[test]
fn test_conv2d_1x1_and_3x3() {
    // 1x1 conv
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [1, 2, 2, 2]
    let weight = vec![1.0, 2.0]; // [1, 2, 1, 1]
    let mut output = vec![0.0; 4];
    let config = Conv2dConfig {
        in_channels: 2,
        out_channels: 1,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_top: 0,
        pad_left: 0,
        pad_bottom: 0,
        pad_right: 0,
        dilate_h: 1,
        dilate_w: 1,
        group: 1,
        activation: FusedActivation::None,
    };
    conv2d(&input, 1, 2, 2, &weight, None, &mut output, &config);
    // output[i] = input[0, i] * 1.0 + input[1, i] * 2.0
    // [1 + 10, 2 + 12, 3 + 14, 4 + 16] = [11, 14, 17, 20]
    assert_eq!(output, vec![11.0, 14.0, 17.0, 20.0]);

    // 3x3 conv with pad=1
    let in3 = vec![1.0; 9]; // [1, 1, 3, 3]
    let w3 = vec![1.0; 9]; // [1, 1, 3, 3]
    let mut out3 = vec![0.0; 9];
    let cfg3 = Conv2dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_top: 1,
        pad_left: 1,
        pad_bottom: 1,
        pad_right: 1,
        dilate_h: 1,
        dilate_w: 1,
        group: 1,
        activation: FusedActivation::None,
    };
    conv2d(&in3, 1, 3, 3, &w3, None, &mut out3, &cfg3);
    // Center pixel has all 9 neighbors = 9.0
    assert_eq!(out3[4], 9.0);
    // Corner pixel has 4 neighbors = 4.0
    assert_eq!(out3[0], 4.0);
}

#[test]
fn test_depthwise_conv2d() {
    let input = vec![1.0; 18]; // [1, 2, 3, 3]
    let weight = vec![1.0; 18]; // 2 channels * 9
    let mut output = vec![0.0; 18];
    let cfg = DepthwiseConv2dConfig {
        channels: 2,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_top: 1,
        pad_left: 1,
        pad_bottom: 1,
        pad_right: 1,
        dilate_h: 1,
        dilate_w: 1,
        activation: FusedActivation::None,
    };
    depthwise_conv2d_nchw(&input, 1, 3, 3, &weight, None, &mut output, &cfg);
    assert_eq!(output[4], 9.0);
    assert_eq!(output[9 + 4], 9.0);
}

#[test]
fn test_deconv2d() {
    // 1x1 input to 2x2 output via 2x2 kernel with stride 2
    let input = vec![2.0]; // [1, 1, 1, 1]
    let weight = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
    let mut output = vec![0.0; 4];
    let cfg = Deconv2dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 2,
        stride_w: 2,
        pad_top: 0,
        pad_left: 0,
        pad_bottom: 0,
        pad_right: 0,
        out_pad_h: 0,
        out_pad_w: 0,
        dilate_h: 1,
        dilate_w: 1,
        group: 1,
        activation: FusedActivation::None,
    };
    deconv2d(&input, 1, 1, 1, &weight, None, &mut output, &cfg);
    assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_pool2d_max_and_avg() {
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]; // [1, 1, 4, 4]

    let mut out_max = vec![0.0; 4];
    let max_cfg = Pool2dConfig {
        pool_type: PoolType::MaxPool,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 2,
        stride_w: 2,
        ..Default::default()
    };
    pool2d(&input, 1, 1, 4, 4, &mut out_max, &max_cfg);
    assert_eq!(out_max, vec![6.0, 8.0, 14.0, 16.0]);

    let mut out_avg = vec![0.0; 4];
    let avg_cfg = Pool2dConfig {
        pool_type: PoolType::AvgPool,
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 2,
        stride_w: 2,
        ..Default::default()
    };
    pool2d(&input, 1, 1, 4, 4, &mut out_avg, &avg_cfg);
    assert_eq!(out_avg, vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn test_interp2d_nearest_and_bilinear() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
    let mut output = vec![0.0; 16]; // [1, 1, 4, 4]
    let cfg = InterpConfig {
        resize_type: ResizeType::Nearest,
        align_corners: false,
        half_pixel_centers: true,
    };
    interp2d(&input, 1, 1, 2, 2, 4, 4, &mut output, &cfg);
    assert_eq!(output[0], 1.0);
    assert_eq!(output[15], 4.0);
}

#[test]
fn test_reduction_mean_and_max() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let mut out_shape = Vec::new();
    let mut output = Vec::new();

    reduce(&input, &[2, 3], &[1], false, ReductionOp::Mean, &mut out_shape, &mut output);
    assert_eq!(out_shape, vec![2]);
    assert_eq!(output, vec![2.0, 5.0]);

    reduce(&input, &[2, 3], &[0], true, ReductionOp::Max, &mut out_shape, &mut output);
    assert_eq!(out_shape, vec![1, 3]);
    assert_eq!(output, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_ops_permute_concat_slice() {
    // Permute [2, 3] -> [3, 2]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b = vec![0.0; 6];
    permute(&a, &[2, 3], &[1, 0], &mut b);
    assert_eq!(b, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // Concat
    let t1 = vec![1.0, 2.0];
    let t2 = vec![3.0, 4.0, 5.0, 6.0];
    let mut cat_out = vec![0.0; 6];
    concat(&[(&t1, &[1, 2]), (&t2, &[1, 4])], 1, &mut cat_out);
    assert_eq!(cat_out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // SliceTf
    let mut slice_out = vec![0.0; 2];
    slice_tf(&cat_out, &[1, 6], &[0, 2], &[1, 2], &mut slice_out);
    assert_eq!(slice_out, vec![3.0, 4.0]);

    // Compute reshape dims
    let target = compute_reshape_dims(&[6, 4], &[-1, 4]).unwrap();
    assert_eq!(target, vec![6, 4]);
}

#[test]
fn test_softmax_and_argmax() {
    let input = vec![1.0, 3.0, 2.0, 0.5];
    let mut prob = vec![0.0; 4];
    softmax(&input, &[1, 4], -1, &mut prob);
    let sum: f32 = prob.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(prob[1] > prob[2] && prob[2] > prob[0]);

    let mut ids = vec![0i32; 1];
    argmax(&input, &[1, 4], -1, &mut ids);
    assert_eq!(ids[0], 1);

    let mut top_v = vec![0.0; 2];
    let mut top_i = vec![0i32; 2];
    topk(&input, 1, 4, 2, &mut top_v, &mut top_i);
    assert_eq!(top_i, vec![1, 2]);
    assert_eq!(top_v, vec![3.0, 2.0]);
}

#[test]
fn test_unary_and_binary() {
    let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut y = vec![0.0; 5];
    unary_op(&x, &mut y, UnaryOpType::HardSwish);
    // HardSwish: x * clamp((x + 3) / 6, 0, 1)
    // For -2: -2 * (1/6) = -0.333333
    // For 0: 0
    // For 2: 2 * (5/6) = 1.666667
    assert!((y[0] - (-1.0 / 3.0)).abs() < 1e-4);
    assert_eq!(y[2], 0.0);
    assert!((y[4] - (5.0 / 3.0)).abs() < 1e-4);

    let a = vec![1.0, 2.0, 3.0];
    let b = vec![10.0];
    let mut c = vec![0.0; 3];
    binary_op(&[3], &a, &[1], &b, &[3], &mut c, BinaryOpType::Add, FusedActivation::None);
    assert_eq!(c, vec![11.0, 12.0, 13.0]);
}
