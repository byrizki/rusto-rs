#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

pub fn reduce(
    input: &[f32],
    in_shape: &[usize],
    axes: &[i32],
    keep_dims: bool,
    op: ReductionOp,
    out_shape: &mut Vec<usize>,
    output: &mut Vec<f32>,
) {
    let ndim = in_shape.len();
    let mut normalized_axes: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.iter()
            .map(|&ax| {
                if ax < 0 {
                    (ndim as i32 + ax) as usize
                } else {
                    ax as usize
                }
            })
            .collect()
    };
    normalized_axes.sort_unstable();
    normalized_axes.dedup();

    // Determine output shape
    out_shape.clear();
    for (d, &size) in in_shape.iter().enumerate() {
        if normalized_axes.contains(&d) {
            if keep_dims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(size);
        }
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_len = out_shape.iter().product::<usize>();
    output.clear();
    output.resize(
        out_len,
        match op {
            ReductionOp::Sum | ReductionOp::Mean => 0.0,
            ReductionOp::Max => f32::NEG_INFINITY,
            ReductionOp::Min => f32::INFINITY,
            ReductionOp::Prod => 1.0,
        },
    );

    // Fast path 1: 4D Spatial reduction across dims [2, 3] (e.g. Global Average Pooling in MobileNet / SE-block)
    if ndim == 4 && normalized_axes == [2, 3] {
        let total_channels = in_shape[0] * in_shape[1];
        let hw = in_shape[2] * in_shape[3];

        let compute_ch = |c: usize| -> f32 {
            let slice = &input[c * hw..(c + 1) * hw];
            match op {
                ReductionOp::Mean => slice.iter().sum::<f32>() / (hw as f32),
                ReductionOp::Sum => slice.iter().sum::<f32>(),
                ReductionOp::Max => slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                ReductionOp::Min => slice.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                ReductionOp::Prod => slice.iter().fold(1.0f32, |a, &b| a * b),
            }
        };

        #[cfg(feature = "rayon")]
        {
            if total_channels >= 2 {
                output
                    .par_chunks_exact_mut(1)
                    .enumerate()
                    .for_each(|(c, out_elem)| {
                        out_elem[0] = compute_ch(c);
                    });
                return;
            }
        }

        for c in 0..total_channels {
            output[c] = compute_ch(c);
        }
        return;
    }

    // Fast path 2: Reduction across the final axis (contiguous trailing reduction)
    if normalized_axes.len() == 1 && normalized_axes[0] == ndim - 1 {
        let k = in_shape[ndim - 1];
        let outer_count = out_len;

        let compute_row = |r: usize| -> f32 {
            let slice = &input[r * k..(r + 1) * k];
            match op {
                ReductionOp::Mean => slice.iter().sum::<f32>() / (k as f32),
                ReductionOp::Sum => slice.iter().sum::<f32>(),
                ReductionOp::Max => slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                ReductionOp::Min => slice.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                ReductionOp::Prod => slice.iter().fold(1.0f32, |a, &b| a * b),
            }
        };

        #[cfg(feature = "rayon")]
        {
            if outer_count >= 4 {
                output
                    .par_chunks_exact_mut(1)
                    .enumerate()
                    .for_each(|(r, out_elem)| {
                        out_elem[0] = compute_row(r);
                    });
                return;
            }
        }

        for r in 0..outer_count {
            output[r] = compute_row(r);
        }
        return;
    }

    // Compute strides
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    let total_in_elems = in_shape.iter().product::<usize>();
    let reduce_count: usize = normalized_axes.iter().map(|&a| in_shape[a]).product();

    // General fallback
    let mut coord = vec![0usize; ndim];
    let mut counts = vec![0usize; out_len];

    for in_idx in 0..total_in_elems {
        let mut rem = in_idx;
        for d in 0..ndim {
            coord[d] = rem / in_strides[d];
            rem %= in_strides[d];
        }

        let mut out_idx = 0;
        let mut out_stride = 1;
        for d in (0..ndim).rev() {
            if !normalized_axes.contains(&d) {
                out_idx += coord[d] * out_stride;
                out_stride *= in_shape[d];
            }
        }

        let val = input[in_idx];
        match op {
            ReductionOp::Sum | ReductionOp::Mean => {
                output[out_idx] += val;
                counts[out_idx] += 1;
            }
            ReductionOp::Max => {
                if val > output[out_idx] {
                    output[out_idx] = val;
                }
            }
            ReductionOp::Min => {
                if val < output[out_idx] {
                    output[out_idx] = val;
                }
            }
            ReductionOp::Prod => {
                output[out_idx] *= val;
            }
        }
    }

    if op == ReductionOp::Mean && reduce_count > 0 {
        for (i, v) in output.iter_mut().enumerate() {
            let c = counts[i].max(1) as f32;
            *v /= c;
        }
    }
}
