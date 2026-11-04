use crate::kernels::activation::{apply_activation_scalar, FusedActivation};

/// Batch Normalization 2D: y = (x - mean) / sqrt(var + eps) * gamma + beta
pub fn batch_norm_2d(
    x: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    epsilon: f32,
    y: &mut [f32],
    act: FusedActivation,
) {
    let hw = height * width;
    assert!(x.len() >= batch * channels * hw);
    assert!(y.len() >= batch * channels * hw);
    assert!(gamma.len() >= channels);
    assert!(beta.len() >= channels);
    assert!(mean.len() >= channels);
    assert!(var.len() >= channels);

    // Precompute scale and shift for each channel: y = x * scale + shift
    let mut scale = vec![0.0f32; channels];
    let mut shift = vec![0.0f32; channels];

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + epsilon).sqrt();
        scale[c] = gamma[c] * inv_std;
        shift[c] = beta[c] - mean[c] * scale[c];
    }

    for n in 0..batch {
        for c in 0..channels {
            let offset = (n * channels + c) * hw;
            let s = scale[c];
            let b = shift[c];

            for i in 0..hw {
                let val = x[offset + i] * s + b;
                y[offset + i] = apply_activation_scalar(val, act);
            }
        }
    }
}

