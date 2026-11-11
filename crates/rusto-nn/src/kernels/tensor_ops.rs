#[cfg(feature = "parallel")]
use crate::par::prelude::*;

/// Permute (transpose) multidimensional tensor
pub fn permute(
    input: &[f32],
    in_shape: &[usize],
    perm: &[usize],
    output: &mut [f32],
) {
    let ndim = in_shape.len();
    assert_eq!(ndim, perm.len());
    let total_elems = in_shape.iter().product::<usize>();
    assert!(output.len() >= total_elems);

    // Fast path 1: NCHW -> NHWC [0, 2, 3, 1]
    if ndim == 4 && perm == [0, 2, 3, 1] {
        let n = in_shape[0];
        let c = in_shape[1];
        let h = in_shape[2];
        let w = in_shape[3];
        let hw = h * w;
        let hw_c = hw * c;

        let process_batch = |b: usize, out_b: &mut [f32]| {
            let in_b = &input[b * hw_c..(b + 1) * hw_c];
            for ch in 0..c {
                let in_ch = &in_b[ch * hw..(ch + 1) * hw];
                for i in 0..hw {
                    out_b[i * c + ch] = in_ch[i];
                }
            }
        };

        #[cfg(feature = "parallel")]
        {
            if crate::par::current_thread_index().is_none() && n >= 2 {
                output.par_chunks_exact_mut(hw_c).enumerate().for_each(|(b, out_b)| {
                    process_batch(b, out_b);
                });
                return;
            }
        }

        for b in 0..n {
            let out_b = &mut output[b * hw_c..(b + 1) * hw_c];
            process_batch(b, out_b);
        }
        return;
    }

    // Fast path 2: 2D/3D matrix transpose [0, 2, 1] or [1, 0]
    if (ndim == 3 && perm == [0, 2, 1]) || (ndim == 2 && perm == [1, 0]) {
        let (batch, rows, cols) = if ndim == 3 {
            (in_shape[0], in_shape[1], in_shape[2])
        } else {
            (1, in_shape[0], in_shape[1])
        };
        let rc = rows * cols;

        let process_matrix = |b: usize, out_b: &mut [f32]| {
            let in_b = &input[b * rc..(b + 1) * rc];
            for r in 0..rows {
                let in_row = &in_b[r * cols..(r + 1) * cols];
                for c in 0..cols {
                    out_b[c * rows + r] = in_row[c];
                }
            }
        };

        #[cfg(feature = "parallel")]
        {
            if crate::par::current_thread_index().is_none() && batch >= 2 {
                output.par_chunks_exact_mut(rc).enumerate().for_each(|(b, out_b)| {
                    process_matrix(b, out_b);
                });
                return;
            }
        }

        for b in 0..batch {
            let out_b = &mut output[b * rc..(b + 1) * rc];
            process_matrix(b, out_b);
        }
        return;
    }

    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Precompute perm_inv: which output dimension corresponds to input dimension d
    let mut perm_inv = vec![0usize; ndim];
    for (out_d, &in_d) in perm.iter().enumerate() {
        perm_inv[in_d] = out_d;
    }

    let mut coord = vec![0usize; ndim];
    let mut out_idx = 0;

    for &val in input.iter() {
        output[out_idx] = val;

        // Advance coordinates and update out_idx incrementally
        for d in (0..ndim).rev() {
            coord[d] += 1;
            let out_dim_stride = out_strides[perm_inv[d]];
            out_idx += out_dim_stride;

            if coord[d] < in_shape[d] {
                break;
            }
            coord[d] = 0;
            out_idx -= in_shape[d] * out_dim_stride;
        }
    }
}

/// Concatenate multiple tensors along specified axis
pub fn concat(
    inputs: &[(&[f32], &[usize])],
    axis: usize,
    output: &mut [f32],
) {
    if inputs.is_empty() {
        return;
    }

    let ndim = inputs[0].1.len();
    assert!(axis < ndim);

    // Compute total outer dimensions (before axis) and inner dimensions (after axis)
    let outer_size: usize = inputs[0].1[..axis].iter().product();
    let inner_size: usize = inputs[0].1[axis + 1..].iter().product();

    let mut out_axis_size = 0;
    for (_, shape) in inputs {
        assert_eq!(shape.len(), ndim);
        out_axis_size += shape[axis];
    }

    let out_stride_outer = out_axis_size * inner_size;
    if outer_size == 1 {
        #[cfg(feature = "parallel")]
        if crate::par::current_thread_index().is_none()
            && out_stride_outer >= 32768
            && inputs.len() >= 2
        {
            let mut rem = &mut output[..out_stride_outer];
            let mut slices: Vec<(&[f32], &mut [f32])> = Vec::with_capacity(inputs.len());
            for (data, shape) in inputs {
                let len = shape[axis] * inner_size;
                let (head, tail) = rem.split_at_mut(len);
                slices.push((&data[..len], head));
                rem = tail;
            }
            slices.into_par_iter().for_each(|(src, dst)| {
                dst.copy_from_slice(src);
            });
            return;
        }
    }

    let mut out_offset = 0;

    for (data, shape) in inputs {
        let axis_size = shape[axis];
        let in_stride_outer = axis_size * inner_size;

        for out_idx in 0..outer_size {
            let in_start = out_idx * in_stride_outer;
            let out_start = out_idx * out_stride_outer + out_offset;

            output[out_start..out_start + in_stride_outer]
                .copy_from_slice(&data[in_start..in_start + in_stride_outer]);
        }

        out_offset += axis_size * inner_size;
    }
}

fn copy_strided_nd(
    input: &[f32],
    in_shape: &[usize],
    actual_begins: &[usize],
    actual_steps: &[usize],
    out_shape: &[usize],
    out_len: usize,
    output: &mut [f32],
) {
    let ndim = in_shape.len();
    if out_len == 0 || input.is_empty() {
        return;
    }
    if ndim == 0 {
        output[0] = input[0];
        return;
    }

    let mut in_strides = [1usize; 8];
    let mut in_strides_vec;
    let strides_slice: &mut [usize] = if ndim <= 8 {
        &mut in_strides[..ndim]
    } else {
        in_strides_vec = vec![1usize; ndim];
        &mut in_strides_vec
    };
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides_slice[i] = strides_slice[i + 1] * in_shape[i + 1];
    }

    // Copy largest contiguous trailing block when trailing axes have unit stride
    if actual_steps[ndim - 1] == 1 {
        let mut contig_axis = ndim - 1;
        while contig_axis > 0
            && actual_begins[contig_axis] == 0
            && out_shape[contig_axis] == in_shape[contig_axis]
            && actual_steps[contig_axis - 1] == 1
        {
            contig_axis -= 1;
        }

        let chunk_len = out_shape[contig_axis] * strides_slice[contig_axis];
        let base_inner_offset = actual_begins[contig_axis] * strides_slice[contig_axis];

        if contig_axis == 0 {
            output[..chunk_len]
                .copy_from_slice(&input[base_inner_offset..base_inner_offset + chunk_len]);
            return;
        }

        let num_chunks = out_len / chunk_len;
        let mut coord_arr = [0usize; 8];
        let mut coord_vec;
        let coord: &mut [usize] = if contig_axis <= 8 {
            &mut coord_arr[..contig_axis]
        } else {
            coord_vec = vec![0usize; contig_axis];
            &mut coord_vec
        };

        for chunk_idx in 0..num_chunks {
            let mut in_idx = base_inner_offset;
            for d in 0..contig_axis {
                in_idx += (actual_begins[d] + coord[d] * actual_steps[d]) * strides_slice[d];
            }
            let out_start = chunk_idx * chunk_len;
            output[out_start..out_start + chunk_len]
                .copy_from_slice(&input[in_idx..in_idx + chunk_len]);

            for d in (0..contig_axis).rev() {
                coord[d] += 1;
                if coord[d] < out_shape[d] {
                    break;
                }
                coord[d] = 0;
            }
        }
        return;
    }

    let mut coord = vec![0usize; ndim];
    for out_idx in 0..out_len {
        let mut in_idx = 0;
        for d in 0..ndim {
            in_idx += (actual_begins[d] + coord[d] * actual_steps[d]) * strides_slice[d];
        }
        output[out_idx] = input[in_idx];

        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < out_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
}

/// TensorFlow-style Slice: begins and sizes
pub fn slice_tf(
    input: &[f32],
    in_shape: &[usize],
    begins: &[usize],
    sizes: &[usize],
    output: &mut [f32],
) {
    let ndim = in_shape.len();
    assert_eq!(ndim, begins.len());
    assert_eq!(ndim, sizes.len());

    let out_len: usize = sizes.iter().product();
    assert!(output.len() >= out_len);

    let unit_steps = [1usize; 8];
    let unit_vec;
    let steps: &[usize] = if ndim <= 8 {
        &unit_steps[..ndim]
    } else {
        unit_vec = vec![1usize; ndim];
        &unit_vec
    };

    copy_strided_nd(input, in_shape, begins, steps, sizes, out_len, output);
}

/// ScatterND updates into data tensor
pub fn scatter_nd(
    data: &[f32],
    data_shape: &[usize],
    indices: &[i32],
    indices_shape: &[usize],
    updates: &[f32],
    output: &mut [f32],
) {
    if output.len() >= data.len() {
        output[..data.len()].copy_from_slice(data);
    }
    if indices_shape.is_empty() || indices.is_empty() {
        return;
    }
    let k = indices_shape.last().copied().unwrap_or(1);
    let num_indices: usize = indices_shape[..indices_shape.len() - 1].iter().product::<usize>().max(1);

    let mut data_strides = vec![1usize; data_shape.len()];
    for d in (0..data_shape.len().saturating_sub(1)).rev() {
        data_strides[d] = data_strides[d + 1] * data_shape[d + 1];
    }

    let slice_size: usize = data_shape[k..].iter().product::<usize>().max(1);

    for i in 0..num_indices {
        let mut target_offset = 0usize;
        for j in 0..k {
            if j < data_shape.len() {
                let idx = indices[i * k + j];
                let actual_idx = if idx < 0 {
                    (data_shape[j] as i32 + idx).max(0) as usize
                } else {
                    (idx as usize).min(data_shape[j].saturating_sub(1))
                };
                target_offset += actual_idx * data_strides[j];
            }
        }

        let update_offset = i * slice_size;
        for s in 0..slice_size {
            if target_offset + s < output.len() && update_offset + s < updates.len() {
                output[target_offset + s] = updates[update_offset + s];
            }
        }
    }
}

/// Broadcast / expand a tensor to a target shape
pub fn expand(
    input: &[f32],
    in_shape: &[usize],
    target_shape: &[usize],
    output: &mut [f32],
) -> Vec<usize> {
    let out_ndim = target_shape.len();
    let in_ndim = in_shape.len();
    let max_ndim = out_ndim.max(in_ndim);

    let mut padded_in = vec![1usize; max_ndim];
    let in_offset = max_ndim - in_ndim;
    for (i, &dim) in in_shape.iter().enumerate() {
        padded_in[in_offset + i] = dim;
    }

    let mut resolved_out = vec![1usize; max_ndim];
    let out_offset = max_ndim - out_ndim;
    for (i, &dim) in target_shape.iter().enumerate() {
        resolved_out[out_offset + i] = dim;
    }
    for d in 0..max_ndim {
        resolved_out[d] = resolved_out[d].max(padded_in[d]);
    }

    let total_elems: usize = resolved_out.iter().product();
    if total_elems == 0 || input.is_empty() {
        return resolved_out;
    }

    let mut in_strides = vec![0usize; max_ndim];
    let mut stride = 1usize;
    for d in (0..max_ndim).rev() {
        if padded_in[d] > 1 {
            in_strides[d] = stride;
            stride *= padded_in[d];
        } else {
            in_strides[d] = 0;
        }
    }

    let mut coords = vec![0usize; max_ndim];
    for idx in 0..total_elems {
        let mut in_idx = 0usize;
        for d in 0..max_ndim {
            in_idx += coords[d] * in_strides[d];
        }
        output[idx] = input.get(in_idx).copied().unwrap_or(input[0]);

        for d in (0..max_ndim).rev() {
            coords[d] += 1;
            if coords[d] < resolved_out[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    resolved_out
}

/// Strided slice along arbitrary dimensions
pub fn strided_slice(
    input: &[f32],
    in_shape: &[usize],
    begins: &[i32],
    ends: &[i32],
    strides: &[i32],
    out_shape: &mut Vec<usize>,
    output: &mut Vec<f32>,
) {
    let ndim = in_shape.len();
    out_shape.clear();

    let mut actual_begins = vec![0usize; ndim];
    let mut actual_steps = vec![1usize; ndim];

    for d in 0..ndim {
        let dim_size = in_shape[d] as i32;
        let stride = if d < strides.len() { strides[d] } else { 1 }.max(1);
        let begin = if d < begins.len() {
            let b = begins[d];
            if b < 0 {
                (dim_size + b).max(0) as usize
            } else {
                (b as usize).min(in_shape[d])
            }
        } else {
            0
        };
        let end = if d < ends.len() {
            let e = ends[d];
            if e < 0 {
                (dim_size + e).max(0) as usize
            } else {
                (e as usize).min(in_shape[d])
            }
        } else {
            in_shape[d]
        };

        let count = if end > begin {
            (end - begin).div_ceil(stride as usize)
        } else {
            0
        };

        out_shape.push(count);
        actual_begins[d] = begin;
        actual_steps[d] = stride as usize;
    }

    let out_len = out_shape.iter().product::<usize>();
    output.clear();
    output.resize(out_len, 0.0);

    copy_strided_nd(
        input,
        in_shape,
        &actual_begins,
        &actual_steps,
        out_shape,
        out_len,
        output,
    );
}

/// Gather slices along specified axis
pub fn gather(
    input: &[f32],
    in_shape: &[usize],
    indices: &[i32],
    indices_shape: &[usize],
    axis: usize,
    out_shape: &mut Vec<usize>,
    output: &mut Vec<f32>,
) {
    let ndim = in_shape.len();
    assert!(axis < ndim);

    out_shape.clear();
    out_shape.extend_from_slice(&in_shape[..axis]);
    out_shape.extend_from_slice(indices_shape);
    out_shape.extend_from_slice(&in_shape[axis + 1..]);

    let outer_size: usize = in_shape[..axis].iter().product();
    let axis_size = in_shape[axis];
    let inner_size: usize = in_shape[axis + 1..].iter().product();

    let num_indices = indices.len();
    let out_len = outer_size * num_indices * inner_size;
    output.clear();
    output.resize(out_len, 0.0);

    for outer in 0..outer_size {
        for (i, &idx) in indices.iter().enumerate() {
            let k = if idx < 0 {
                (axis_size as i32 + idx).max(0) as usize
            } else {
                (idx as usize).min(axis_size - 1)
            };

            let in_start = (outer * axis_size + k) * inner_size;
            let out_start = (outer * num_indices + i) * inner_size;

            output[out_start..out_start + inner_size]
                .copy_from_slice(&input[in_start..in_start + inner_size]);
        }
    }
}

/// Compute Reshape target dimensions resolving 0 (copy input dim) and -1 (wildcard)
pub fn compute_reshape_dims(
    in_shape: &[usize],
    target_dims: &[i32],
) -> Result<Vec<usize>, String> {
    let in_len: usize = in_shape.iter().product();
    let mut resolved = Vec::with_capacity(target_dims.len());
    let mut wildcard_idx = None;
    let mut known_product = 1usize;

    for (i, &d) in target_dims.iter().enumerate() {
        if d == -1 {
            if wildcard_idx.is_some() {
                return Err("Multiple -1 wildcards in reshape".to_string());
            }
            wildcard_idx = Some(i);
            resolved.push(0);
        } else if d == 0 {
            let copy_dim = if i < in_shape.len() { in_shape[i] } else { 1 };
            known_product *= copy_dim;
            resolved.push(copy_dim);
        } else if d > 0 {
            known_product *= d as usize;
            resolved.push(d as usize);
        } else {
            return Err(format!("Invalid negative dimension in reshape: {}", d));
        }
    }

    if let Some(idx) = wildcard_idx {
        if known_product == 0 || !in_len.is_multiple_of(known_product) {
            return Err(format!(
                "Cannot infer -1 dimension for input size {} with known product {}",
                in_len, known_product
            ));
        }
        resolved[idx] = in_len / known_product;
    }

    Ok(resolved)
}

/// Pad an N-dimensional tensor with constant, reflect, symmetric, or edge mode.
/// `paddings` has length `ndim * 2`: [pad_before_0, pad_after_0, pad_before_1, pad_after_1, ...]
pub fn pad_tensor(
    input: &[f32],
    in_shape: &[usize],
    paddings: &[usize],
    mode: crate::schema::PadValueMode,
    fill_value: f32,
    output: &mut [f32],
) {
    let ndim = in_shape.len();
    assert_eq!(paddings.len(), ndim * 2);

    let mut out_shape = Vec::with_capacity(ndim);
    for d in 0..ndim {
        out_shape.push(in_shape[d] + paddings[2 * d] + paddings[2 * d + 1]);
    }
    let out_len: usize = out_shape.iter().product();
    assert!(output.len() >= out_len);

    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    let mut out_coord = vec![0usize; ndim];
    for out_idx in 0..out_len {
        let mut is_pad = false;
        let mut in_idx = 0;

        for d in 0..ndim {
            let oc = out_coord[d] as isize;
            let pb = paddings[2 * d] as isize;
            let idim = in_shape[d] as isize;
            let c = oc - pb;

            let ic = if c >= 0 && c < idim {
                c as usize
            } else {
                match mode {
                    crate::schema::PadValueMode::CONSTANT => {
                        is_pad = true;
                        0
                    }
                    crate::schema::PadValueMode::REFLECT => {
                        if c < 0 {
                            (-c).min(idim.saturating_sub(1)) as usize
                        } else {
                            (2 * idim - 2 - c).max(0) as usize
                        }
                    }
                    crate::schema::PadValueMode::SYMMETRIC => {
                        if c < 0 {
                            (-c - 1).min(idim.saturating_sub(1)) as usize
                        } else {
                            (2 * idim - 1 - c).max(0) as usize
                        }
                    }
                    crate::schema::PadValueMode::EDGE => {
                        c.clamp(0, idim.saturating_sub(1)) as usize
                    }
                }
            };

            if is_pad {
                break;
            }
            in_idx += ic * in_strides[d];
        }

        output[out_idx] = if is_pad { fill_value } else { input[in_idx] };

        // Advance out_coord
        for d in (0..ndim).rev() {
            out_coord[d] += 1;
            if out_coord[d] < out_shape[d] {
                break;
            }
            out_coord[d] = 0;
        }
    }
}

/// 2D Grid Sample (sampling input feature map / image based on flow field / mesh grid).
///
/// Input: [N, C, in_H, in_W]
/// Grid:  [N, out_H, out_W, 2]
/// Output: [N, C, out_H, out_W]
pub fn grid_sample_2d(
    input: &[f32],
    in_shape: &[usize],
    grid: &[f32],
    grid_shape: &[usize],
    mode: crate::schema::SampleMode,
    padding_mode: crate::schema::BorderMode,
    align_corners: bool,
    output: &mut [f32],
) {
    let n = in_shape[0];
    let c = in_shape[1];
    let in_h = in_shape[2];
    let in_w = in_shape[3];

    let out_h = grid_shape[1];
    let out_w = grid_shape[2];

    let in_hw = in_h * in_w;
    let out_hw = out_h * out_w;

    let a = if align_corners { 1.0f32 } else { 0.0f32 };
    let b = if align_corners { 0.0f32 } else { 1.0f32 };
    let kx = 0.5f32 * (in_w as f32 - a);
    let bx = 0.5f32 * (in_w as f32 - a - b);
    let ky = 0.5f32 * (in_h as f32 - a);
    let by = 0.5f32 * (in_h as f32 - a - b);

    let sample_pixel = |in_ch: &[f32], x: isize, y: isize| -> f32 {
        let (sx, sy) = match padding_mode {
            crate::schema::BorderMode::ZEROS => {
                if x < 0 || x >= in_w as isize || y < 0 || y >= in_h as isize {
                    return 0.0;
                }
                (x as usize, y as usize)
            }
            crate::schema::BorderMode::CLAMP => {
                (
                    x.clamp(0, in_w as isize - 1) as usize,
                    y.clamp(0, in_h as isize - 1) as usize,
                )
            }
            crate::schema::BorderMode::REFLECTION => {
                let rx = if in_w <= 1 {
                    0
                } else {
                    let mut cx = x;
                    if cx < 0 {
                        cx = -cx;
                    }
                    if cx >= in_w as isize {
                        cx = (2 * in_w as isize - 2 - cx).max(0);
                    }
                    cx.clamp(0, in_w as isize - 1) as usize
                };
                let ry = if in_h <= 1 {
                    0
                } else {
                    let mut cy = y;
                    if cy < 0 {
                        cy = -cy;
                    }
                    if cy >= in_h as isize {
                        cy = (2 * in_h as isize - 2 - cy).max(0);
                    }
                    cy.clamp(0, in_h as isize - 1) as usize
                };
                (rx, ry)
            }
            _ => {
                if x < 0 || x >= in_w as isize || y < 0 || y >= in_h as isize {
                    return 0.0;
                }
                (x as usize, y as usize)
            }
        };
        in_ch[sy * in_w + sx]
    };

    for b in 0..n {
        let in_batch = &input[b * c * in_hw..(b + 1) * c * in_hw];
        let grid_batch = &grid[b * out_hw * 2..(b + 1) * out_hw * 2];
        let out_batch = &mut output[b * c * out_hw..(b + 1) * c * out_hw];

        for oh in 0..out_h {
            for ow in 0..out_w {
                let grid_idx = (oh * out_w + ow) * 2;
                let gx = grid_batch[grid_idx];
                let gy = grid_batch[grid_idx + 1];

                let x = kx * gx + bx;
                let y = ky * gy + by;

                let out_pixel_idx = oh * out_w + ow;

                match mode {
                    crate::schema::SampleMode::NEAREST => {
                        let nx = x.round() as isize;
                        let ny = y.round() as isize;
                        for ch in 0..c {
                            let in_ch = &in_batch[ch * in_hw..(ch + 1) * in_hw];
                            out_batch[ch * out_hw + out_pixel_idx] = sample_pixel(in_ch, nx, ny);
                        }
                    }
                    _ => {
                        let x0 = x.floor() as isize;
                        let y0 = y.floor() as isize;
                        let x1 = x0 + 1;
                        let y1 = y0 + 1;

                        let wx = x - x0 as f32;
                        let wy = y - y0 as f32;

                        let w00 = (1.0 - wx) * (1.0 - wy);
                        let w01 = wx * (1.0 - wy);
                        let w10 = (1.0 - wx) * wy;
                        let w11 = wx * wy;

                        for ch in 0..c {
                            let in_ch = &in_batch[ch * in_hw..(ch + 1) * in_hw];
                            let v00 = sample_pixel(in_ch, x0, y0);
                            let v01 = sample_pixel(in_ch, x1, y0);
                            let v10 = sample_pixel(in_ch, x0, y1);
                            let v11 = sample_pixel(in_ch, x1, y1);

                            let interp = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                            out_batch[ch * out_hw + out_pixel_idx] = interp;
                        }
                    }
                }
            }
        }
    }
}
