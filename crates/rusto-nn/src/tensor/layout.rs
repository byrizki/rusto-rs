/// Hardware tensor data layouts supported by MNN
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Planar batch-channel-height-width (standard ONNX / PyTorch)
    NCHW,
    /// Batch-height-width-channel (TensorFlow)
    NHWC,
    /// Channel-packed by 4 (optimized for ARM NEON and WASM SIMD128)
    NC4HW4,
    /// Channel-packed by 8 (optimized for x86_64 AVX2 / AVX-512)
    NC8HW8,
}

/// Transform NCHW buffer into NC8HW8 packed buffer
pub fn pack_nchw_to_nc8hw8(
    src: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    dst: &mut [f32],
) {
    let c_blocks = channels.div_ceil(8);
    let hw = height * width;
    let expected_dst_len = batch * c_blocks * hw * 8;
    assert!(dst.len() >= expected_dst_len);

    for n in 0..batch {
        let src_batch = &src[n * channels * hw..(n + 1) * channels * hw];
        let dst_batch = &mut dst[n * c_blocks * hw * 8..(n + 1) * c_blocks * hw * 8];

        for cb in 0..c_blocks {
            let c_start = cb * 8;
            let c_count = (channels - c_start).min(8);
            let dst_block = &mut dst_batch[cb * hw * 8..(cb + 1) * hw * 8];

            for p in 0..hw {
                let out_idx = p * 8;
                for i in 0..c_count {
                    dst_block[out_idx + i] = src_batch[(c_start + i) * hw + p];
                }
                for i in c_count..8 {
                    dst_block[out_idx + i] = 0.0;
                }
            }
        }
    }
}

/// Unpack NC8HW8 buffer into NCHW planar buffer
pub fn unpack_nc8hw8_to_nchw(
    src: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    dst: &mut [f32],
) {
    let c_blocks = channels.div_ceil(8);
    let hw = height * width;
    let expected_dst_len = batch * channels * hw;
    assert!(dst.len() >= expected_dst_len);

    for n in 0..batch {
        let src_batch = &src[n * c_blocks * hw * 8..(n + 1) * c_blocks * hw * 8];
        let dst_batch = &mut dst[n * channels * hw..(n + 1) * channels * hw];

        for cb in 0..c_blocks {
            let c_start = cb * 8;
            let c_count = (channels - c_start).min(8);
            let src_block = &src_batch[cb * hw * 8..(cb + 1) * hw * 8];

            for p in 0..hw {
                let in_idx = p * 8;
                for i in 0..c_count {
                    dst_batch[(c_start + i) * hw + p] = src_block[in_idx + i];
                }
            }
        }
    }
}

/// Transform NCHW buffer into NC4HW4 packed buffer (ARM NEON / WASM)
pub fn pack_nchw_to_nc4hw4(
    src: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    dst: &mut [f32],
) {
    let c_blocks = channels.div_ceil(4);
    let hw = height * width;
    let expected_dst_len = batch * c_blocks * hw * 4;
    assert!(dst.len() >= expected_dst_len);

    for n in 0..batch {
        let src_batch = &src[n * channels * hw..(n + 1) * channels * hw];
        let dst_batch = &mut dst[n * c_blocks * hw * 4..(n + 1) * c_blocks * hw * 4];

        for cb in 0..c_blocks {
            let c_start = cb * 4;
            let c_count = (channels - c_start).min(4);
            let dst_block = &mut dst_batch[cb * hw * 4..(cb + 1) * hw * 4];

            for p in 0..hw {
                let out_idx = p * 4;
                for i in 0..c_count {
                    dst_block[out_idx + i] = src_batch[(c_start + i) * hw + p];
                }
                for i in c_count..4 {
                    dst_block[out_idx + i] = 0.0;
                }
            }
        }
    }
}

/// Unpack NC4HW4 buffer into NCHW planar buffer
pub fn unpack_nc4hw4_to_nchw(
    src: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    dst: &mut [f32],
) {
    let c_blocks = channels.div_ceil(4);
    let hw = height * width;
    let expected_dst_len = batch * channels * hw;
    assert!(dst.len() >= expected_dst_len);

    for n in 0..batch {
        let src_batch = &src[n * c_blocks * hw * 4..(n + 1) * c_blocks * hw * 4];
        let dst_batch = &mut dst[n * channels * hw..(n + 1) * channels * hw];

        for cb in 0..c_blocks {
            let c_start = cb * 4;
            let c_count = (channels - c_start).min(4);
            let src_block = &src_batch[cb * hw * 4..(cb + 1) * hw * 4];

            for p in 0..hw {
                let in_idx = p * 4;
                for i in 0..c_count {
                    dst_batch[(c_start + i) * hw + p] = src_block[in_idx + i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nchw_nc8hw8_roundtrip() {
        let (batch, channels, height, width) = (2, 13, 3, 5);
        let len = batch * channels * height * width;
        let mut original = vec![0.0f32; len];
        for i in 0..len {
            original[i] = (i + 1) as f32 * 0.25;
        }

        let c_blocks = channels.div_ceil(8);
        let packed_len = batch * c_blocks * height * width * 8;
        let mut packed = vec![0.0f32; packed_len];

        pack_nchw_to_nc8hw8(&original, batch, channels, height, width, &mut packed);

        let mut roundtrip = vec![0.0f32; len];
        unpack_nc8hw8_to_nchw(&packed, batch, channels, height, width, &mut roundtrip);

        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_nchw_nc4hw4_roundtrip() {
        let (batch, channels, height, width) = (2, 7, 4, 4);
        let len = batch * channels * height * width;
        let mut original = vec![0.0f32; len];
        for i in 0..len {
            original[i] = (i + 1) as f32 * 0.5;
        }

        let c_blocks = channels.div_ceil(4);
        let packed_len = batch * c_blocks * height * width * 4;
        let mut packed = vec![0.0f32; packed_len];

        pack_nchw_to_nc4hw4(&original, batch, channels, height, width, &mut packed);

        let mut roundtrip = vec![0.0f32; len];
        unpack_nc4hw4_to_nchw(&packed, batch, channels, height, width, &mut roundtrip);

        assert_eq!(original, roundtrip);
    }
}

