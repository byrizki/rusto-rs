//! Area-averaging image resampling for downscaled recognition crops.

fn area_weights(src_len: usize, dst_len: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = src_len as f32 / dst_len as f32;
    (0..dst_len)
        .map(|i| {
            let start = i as f32 * scale;
            let end = ((i + 1) as f32 * scale).min(src_len as f32);
            let first = start.floor() as usize;
            let last = (end.ceil() as usize).min(src_len);
            let mut taps = Vec::with_capacity(last.saturating_sub(first));
            let mut total = 0.0f32;
            for s in first..last {
                let overlap = (end.min(s as f32 + 1.0) - start.max(s as f32)).max(0.0);
                if overlap > 0.0 {
                    taps.push((s, overlap));
                    total += overlap;
                }
            }
            if total <= 0.0 {
                return vec![(first.min(src_len - 1), 1.0)];
            }
            for tap in &mut taps {
                tap.1 /= total;
            }
            taps
        })
        .collect()
}

/// Resizes a row-major single-channel buffer by averaging the source pixels covered by each output pixel.
pub fn area_resize_gray(src: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    let x_taps = area_weights(src_w, dst_w);
    let y_taps = area_weights(src_h, dst_h);

    let mut horizontal = vec![0.0f32; src_h * dst_w];
    for y in 0..src_h {
        let row = &src[y * src_w..(y + 1) * src_w];
        let out_row = &mut horizontal[y * dst_w..(y + 1) * dst_w];
        for (x, taps) in x_taps.iter().enumerate() {
            out_row[x] = taps.iter().map(|&(s, w)| row[s] * w).sum();
        }
    }

    let mut out = vec![0.0f32; dst_h * dst_w];
    for (y, taps) in y_taps.iter().enumerate() {
        let out_row = &mut out[y * dst_w..(y + 1) * dst_w];
        for &(s, w) in taps {
            let row = &horizontal[s * dst_w..(s + 1) * dst_w];
            for (o, &v) in out_row.iter_mut().zip(row) {
                *o += v * w;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area_resize_averages_blocks() {
        let src = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let out = area_resize_gray(&src, 4, 2, 2, 1);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_area_resize_fractional_ratio_preserves_constant() {
        let src = vec![0.25f32; 7 * 5];
        let out = area_resize_gray(&src, 7, 5, 3, 2);
        assert!(out.iter().all(|&v| (v - 0.25).abs() < 1e-5));
    }
}

