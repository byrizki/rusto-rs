#[cfg(feature = "parallel")]
use crate::par::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_slice_avx2(slice: &[f32], op: ReductionOp) -> f32 {
    use core::arch::x86_64::*;

    let len = slice.len();
    let chunks = len / 8;
    let ptr = slice.as_ptr();

    match op {
        ReductionOp::Sum | ReductionOp::Mean => {
            let mut acc = _mm256_setzero_ps();
            for i in 0..chunks {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(ptr.add(i * 8)));
            }
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let s128 = _mm_add_ps(hi, lo);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 0x55));
            let mut sum = _mm_cvtss_f32(s32);
            for i in (chunks * 8)..len {
                sum += *ptr.add(i);
            }
            if op == ReductionOp::Mean {
                sum / (len as f32)
            } else {
                sum
            }
        }
        ReductionOp::Max => {
            let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                acc = _mm256_max_ps(acc, _mm256_loadu_ps(ptr.add(i * 8)));
            }
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let m128 = _mm_max_ps(hi, lo);
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 0x55));
            let mut res = _mm_cvtss_f32(m32);
            for i in (chunks * 8)..len {
                res = res.max(*ptr.add(i));
            }
            res
        }
        ReductionOp::Min => {
            let mut acc = _mm256_set1_ps(f32::INFINITY);
            for i in 0..chunks {
                acc = _mm256_min_ps(acc, _mm256_loadu_ps(ptr.add(i * 8)));
            }
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let m128 = _mm_min_ps(hi, lo);
            let m64 = _mm_min_ps(m128, _mm_movehl_ps(m128, m128));
            let m32 = _mm_min_ss(m64, _mm_shuffle_ps(m64, m64, 0x55));
            let mut res = _mm_cvtss_f32(m32);
            for i in (chunks * 8)..len {
                res = res.min(*ptr.add(i));
            }
            res
        }
        ReductionOp::Prod => {
            let mut acc = _mm256_set1_ps(1.0);
            for i in 0..chunks {
                acc = _mm256_mul_ps(acc, _mm256_loadu_ps(ptr.add(i * 8)));
            }
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let p128 = _mm_mul_ps(hi, lo);
            let p64 = _mm_mul_ps(p128, _mm_movehl_ps(p128, p128));
            let p32 = _mm_mul_ss(p64, _mm_shuffle_ps(p64, p64, 0x55));
            let mut res = _mm_cvtss_f32(p32);
            for i in (chunks * 8)..len {
                res *= *ptr.add(i);
            }
            res
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn reduce_slice_wasm(slice: &[f32], op: ReductionOp) -> f32 {
    use core::arch::wasm32::*;

    let len = slice.len();
    let chunks = len / 4;
    let ptr = slice.as_ptr();

    match op {
        ReductionOp::Sum | ReductionOp::Mean => {
            let mut acc = f32x4_splat(0.0);
            for i in 0..chunks {
                acc = f32x4_add(acc, v128_load(ptr.add(i * 4) as *const v128));
            }
            let mut sum = (f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc))
                + (f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc));
            for i in (chunks * 4)..len {
                sum += *ptr.add(i);
            }
            if op == ReductionOp::Mean {
                sum / (len as f32)
            } else {
                sum
            }
        }
        ReductionOp::Max => {
            let mut acc = f32x4_splat(f32::NEG_INFINITY);
            for i in 0..chunks {
                acc = f32x4_pmax(acc, v128_load(ptr.add(i * 4) as *const v128));
            }
            let mut res = f32x4_extract_lane::<0>(acc)
                .max(f32x4_extract_lane::<1>(acc))
                .max(f32x4_extract_lane::<2>(acc))
                .max(f32x4_extract_lane::<3>(acc));
            for i in (chunks * 4)..len {
                res = res.max(*ptr.add(i));
            }
            res
        }
        ReductionOp::Min => {
            let mut acc = f32x4_splat(f32::INFINITY);
            for i in 0..chunks {
                acc = f32x4_pmin(acc, v128_load(ptr.add(i * 4) as *const v128));
            }
            let mut res = f32x4_extract_lane::<0>(acc)
                .min(f32x4_extract_lane::<1>(acc))
                .min(f32x4_extract_lane::<2>(acc))
                .min(f32x4_extract_lane::<3>(acc));
            for i in (chunks * 4)..len {
                res = res.min(*ptr.add(i));
            }
            res
        }
        ReductionOp::Prod => {
            let mut acc = f32x4_splat(1.0);
            for i in 0..chunks {
                acc = f32x4_mul(acc, v128_load(ptr.add(i * 4) as *const v128));
            }
            let mut res = (f32x4_extract_lane::<0>(acc) * f32x4_extract_lane::<1>(acc))
                * (f32x4_extract_lane::<2>(acc) * f32x4_extract_lane::<3>(acc));
            for i in (chunks * 4)..len {
                res *= *ptr.add(i);
            }
            res
        }
    }
}

#[inline]
fn reduce_contiguous_slice(slice: &[f32], op: ReductionOp) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { reduce_slice_avx2(slice, op) };
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { reduce_slice_wasm(slice, op) };
    }
    #[allow(unreachable_code)]
    match op {
        ReductionOp::Mean => slice.iter().sum::<f32>() / (slice.len() as f32),
        ReductionOp::Sum => slice.iter().sum::<f32>(),
        ReductionOp::Max => slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        ReductionOp::Min => slice.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        ReductionOp::Prod => slice.iter().fold(1.0f32, |a, &b| a * b),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accum_row_avx2(acc: &mut [f32], row: &[f32], op: ReductionOp) {
    use core::arch::x86_64::*;

    let len = acc.len();
    let chunks = len / 8;
    let a_ptr = acc.as_mut_ptr();
    let r_ptr = row.as_ptr();

    match op {
        ReductionOp::Sum | ReductionOp::Mean => {
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a_ptr.add(i * 8));
                let vr = _mm256_loadu_ps(r_ptr.add(i * 8));
                _mm256_storeu_ps(a_ptr.add(i * 8), _mm256_add_ps(va, vr));
            }
            for i in (chunks * 8)..len {
                *a_ptr.add(i) += *r_ptr.add(i);
            }
        }
        ReductionOp::Max => {
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a_ptr.add(i * 8));
                let vr = _mm256_loadu_ps(r_ptr.add(i * 8));
                _mm256_storeu_ps(a_ptr.add(i * 8), _mm256_max_ps(va, vr));
            }
            for i in (chunks * 8)..len {
                *a_ptr.add(i) = (*a_ptr.add(i)).max(*r_ptr.add(i));
            }
        }
        ReductionOp::Min => {
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a_ptr.add(i * 8));
                let vr = _mm256_loadu_ps(r_ptr.add(i * 8));
                _mm256_storeu_ps(a_ptr.add(i * 8), _mm256_min_ps(va, vr));
            }
            for i in (chunks * 8)..len {
                *a_ptr.add(i) = (*a_ptr.add(i)).min(*r_ptr.add(i));
            }
        }
        ReductionOp::Prod => {
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a_ptr.add(i * 8));
                let vr = _mm256_loadu_ps(r_ptr.add(i * 8));
                _mm256_storeu_ps(a_ptr.add(i * 8), _mm256_mul_ps(va, vr));
            }
            for i in (chunks * 8)..len {
                *a_ptr.add(i) *= *r_ptr.add(i);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn accum_row_wasm(acc: &mut [f32], row: &[f32], op: ReductionOp) {
    use core::arch::wasm32::*;

    let len = acc.len();
    let chunks = len / 4;
    let a_ptr = acc.as_mut_ptr();
    let r_ptr = row.as_ptr();

    match op {
        ReductionOp::Sum | ReductionOp::Mean => {
            for i in 0..chunks {
                let va = v128_load(a_ptr.add(i * 4) as *const v128);
                let vr = v128_load(r_ptr.add(i * 4) as *const v128);
                v128_store(a_ptr.add(i * 4) as *mut v128, f32x4_add(va, vr));
            }
            for i in (chunks * 4)..len {
                *a_ptr.add(i) += *r_ptr.add(i);
            }
        }
        ReductionOp::Max => {
            for i in 0..chunks {
                let va = v128_load(a_ptr.add(i * 4) as *const v128);
                let vr = v128_load(r_ptr.add(i * 4) as *const v128);
                v128_store(a_ptr.add(i * 4) as *mut v128, f32x4_pmax(va, vr));
            }
            for i in (chunks * 4)..len {
                *a_ptr.add(i) = (*a_ptr.add(i)).max(*r_ptr.add(i));
            }
        }
        ReductionOp::Min => {
            for i in 0..chunks {
                let va = v128_load(a_ptr.add(i * 4) as *const v128);
                let vr = v128_load(r_ptr.add(i * 4) as *const v128);
                v128_store(a_ptr.add(i * 4) as *mut v128, f32x4_pmin(va, vr));
            }
            for i in (chunks * 4)..len {
                *a_ptr.add(i) = (*a_ptr.add(i)).min(*r_ptr.add(i));
            }
        }
        ReductionOp::Prod => {
            for i in 0..chunks {
                let va = v128_load(a_ptr.add(i * 4) as *const v128);
                let vr = v128_load(r_ptr.add(i * 4) as *const v128);
                v128_store(a_ptr.add(i * 4) as *mut v128, f32x4_mul(va, vr));
            }
            for i in (chunks * 4)..len {
                *a_ptr.add(i) *= *r_ptr.add(i);
            }
        }
    }
}

fn reduce_interior_block(
    in_block: &[f32],
    reduce_len: usize,
    inner: usize,
    op: ReductionOp,
    out_slice: &mut [f32],
) {
    out_slice.copy_from_slice(&in_block[..inner]);

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2");

    for r in 1..reduce_len {
        let row = &in_block[r * inner..(r + 1) * inner];
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            unsafe { accum_row_avx2(out_slice, row, op) };
            continue;
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe { accum_row_wasm(out_slice, row, op) };
            continue;
        }
        #[allow(unreachable_code)]
        match op {
            ReductionOp::Sum | ReductionOp::Mean => {
                for i in 0..inner {
                    out_slice[i] += row[i];
                }
            }
            ReductionOp::Max => {
                for i in 0..inner {
                    out_slice[i] = out_slice[i].max(row[i]);
                }
            }
            ReductionOp::Min => {
                for i in 0..inner {
                    out_slice[i] = out_slice[i].min(row[i]);
                }
            }
            ReductionOp::Prod => {
                for i in 0..inner {
                    out_slice[i] *= row[i];
                }
            }
        }
    }

    if op == ReductionOp::Mean {
        let inv = 1.0 / (reduce_len as f32);
        for v in out_slice.iter_mut() {
            *v *= inv;
        }
    }
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

    if ndim == 0 {
        if !input.is_empty() {
            output[0] = input[0];
        }
        return;
    }

    // Unified contiguous-axes fast path: handles any single axis (trailing, interior, or leading)
    // as well as contiguous multi-axis reductions such as 4D [2, 3] Global Pooling or full reduction.
    if !normalized_axes.is_empty() {
        let ax_start = normalized_axes[0];
        let ax_end = normalized_axes[normalized_axes.len() - 1];
        if ax_end - ax_start + 1 == normalized_axes.len() {
            let outer: usize = in_shape[..ax_start].iter().product();
            let reduce_len: usize = in_shape[ax_start..=ax_end].iter().product();
            let inner: usize = in_shape[ax_end + 1..].iter().product();

            if outer == 0 || inner == 0 || reduce_len == 0 {
                return;
            }

            if inner == 1 {
                #[cfg(feature = "parallel")]
                {
                    if outer >= 4 {
                        output
                            .par_chunks_exact_mut(1)
                            .enumerate()
                            .for_each(|(o, out_elem)| {
                                let slice = &input[o * reduce_len..(o + 1) * reduce_len];
                                out_elem[0] = reduce_contiguous_slice(slice, op);
                            });
                        return;
                    }
                }

                for o in 0..outer {
                    let slice = &input[o * reduce_len..(o + 1) * reduce_len];
                    output[o] = reduce_contiguous_slice(slice, op);
                }
                return;
            }

            let block_stride = reduce_len * inner;
            #[cfg(feature = "parallel")]
            {
                if outer >= 2 {
                    output
                        .par_chunks_exact_mut(inner)
                        .enumerate()
                        .for_each(|(o, out_slice)| {
                            let in_block = &input[o * block_stride..(o + 1) * block_stride];
                            reduce_interior_block(in_block, reduce_len, inner, op, out_slice);
                        });
                    return;
                }
            }

            for o in 0..outer {
                let in_block = &input[o * block_stride..(o + 1) * block_stride];
                let out_slice = &mut output[o * inner..(o + 1) * inner];
                reduce_interior_block(in_block, reduce_len, inner, op, out_slice);
            }
            return;
        }
    }

    // Disjoint multi-axis fallback (e.g. axes = [0, 2] on 4D)
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    let total_in_elems = in_shape.iter().product::<usize>();
    let reduce_count: usize = normalized_axes.iter().map(|&a| in_shape[a]).product();

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
