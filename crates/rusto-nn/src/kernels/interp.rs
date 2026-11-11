#[cfg(feature = "parallel")]
use crate::par::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeType {
    Nearest = 1,
    Bilinear = 2,
    Cubic = 3,
}

#[derive(Debug, Clone)]
pub struct InterpConfig {
    pub resize_type: ResizeType,
    pub align_corners: bool,
    pub half_pixel_centers: bool,
}

impl Default for InterpConfig {
    fn default() -> Self {
        Self {
            resize_type: ResizeType::Bilinear,
            align_corners: false,
            half_pixel_centers: true,
        }
    }
}

pub fn interp2d(
    input: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    output: &mut [f32],
    config: &InterpConfig,
) {
    let hw_in = in_h * in_w;
    let hw_out = out_h * out_w;
    assert!(output.len() >= batch * channels * hw_out);

    let scale_y = in_h as f32 / out_h as f32;
    let scale_x = in_w as f32 / out_w as f32;

    // Precompute horizontal interpolation table
    struct XCoord {
        ix0: usize,
        ix1: usize,
        wx0: f32,
        wx1: f32,
        ix_nearest: usize,
    }

    let round_nearest = |coord: f32, in_len: usize| -> usize {
        let clamped = coord.clamp(0.0, (in_len - 1) as f32);
        let idx = if !config.half_pixel_centers {
            clamped.floor() as usize
        } else if (clamped.fract() - 0.5).abs() < 1e-5 {
            clamped.floor() as usize
        } else {
            clamped.round() as usize
        };
        idx.min(in_len - 1)
    };

    let x_coords: Vec<XCoord> = (0..out_w)
        .map(|ow| {
            let src_x = if config.half_pixel_centers {
                ((ow as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (in_w - 1) as f32)
            } else if config.align_corners {
                if out_w > 1 {
                    (ow as f32 * (in_w - 1) as f32 / (out_w - 1) as f32).clamp(0.0, (in_w - 1) as f32)
                } else {
                    0.0
                }
            } else {
                (ow as f32 * scale_x).clamp(0.0, (in_w - 1) as f32)
            };

            let ix_nearest = round_nearest(src_x, in_w);
            let x0 = (src_x.floor() as usize).min(in_w - 1);
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = (src_x - x0 as f32).clamp(0.0, 1.0);

            XCoord {
                ix0: x0,
                ix1: x1,
                wx0: 1.0 - wx,
                wx1: wx,
                ix_nearest,
            }
        })
        .collect();

    // Precompute vertical interpolation table
    struct YCoord {
        iy0: usize,
        iy1: usize,
        wy0: f32,
        wy1: f32,
        iy_nearest: usize,
    }

    let y_coords: Vec<YCoord> = (0..out_h)
        .map(|oh| {
            let src_y = if config.half_pixel_centers {
                ((oh as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (in_h - 1) as f32)
            } else if config.align_corners {
                if out_h > 1 {
                    (oh as f32 * (in_h - 1) as f32 / (out_h - 1) as f32).clamp(0.0, (in_h - 1) as f32)
                } else {
                    0.0
                }
            } else {
                (oh as f32 * scale_y).clamp(0.0, (in_h - 1) as f32)
            };

            let iy_nearest = round_nearest(src_y, in_h);
            let y0 = (src_y.floor() as usize).min(in_h - 1);
            let y1 = (y0 + 1).min(in_h - 1);
            let wy = (src_y - y0 as f32).clamp(0.0, 1.0);

            YCoord {
                iy0: y0,
                iy1: y1,
                wy0: 1.0 - wy,
                wy1: wy,
                iy_nearest,
            }
        })
        .collect();

    let total_channels = batch * channels;

    let process_channel = |in_ch: &[f32], out_ch: &mut [f32]| {
        for oh in 0..out_h {
            let y = &y_coords[oh];
            let out_row = &mut out_ch[oh * out_w..(oh + 1) * out_w];

            match config.resize_type {
                ResizeType::Nearest => {
                    let r0 = &in_ch[y.iy_nearest * in_w..(y.iy_nearest + 1) * in_w];
                    for ow in 0..out_w {
                        out_row[ow] = r0[x_coords[ow].ix_nearest];
                    }
                }
                ResizeType::Bilinear | ResizeType::Cubic => {
                    let r0 = &in_ch[y.iy0 * in_w..(y.iy0 + 1) * in_w];
                    let r1 = &in_ch[y.iy1 * in_w..(y.iy1 + 1) * in_w];
                    let wy0 = y.wy0;
                    let wy1 = y.wy1;

                    for ow in 0..out_w {
                        let x = &x_coords[ow];
                        let top = r0[x.ix0] * x.wx0 + r0[x.ix1] * x.wx1;
                        let bot = r1[x.ix0] * x.wx0 + r1[x.ix1] * x.wx1;
                        out_row[ow] = top * wy0 + bot * wy1;
                    }
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        if crate::par::current_thread_index().is_none() && total_channels >= 2 {
            output
                .par_chunks_exact_mut(hw_out)
                .enumerate()
                .for_each(|(c_idx, out_ch)| {
                    let in_ch = &input[c_idx * hw_in..(c_idx + 1) * hw_in];
                    process_channel(in_ch, out_ch);
                });
            return;
        }
    }

    for c_idx in 0..total_channels {
        let in_ch = &input[c_idx * hw_in..(c_idx + 1) * hw_in];
        let out_ch = &mut output[c_idx * hw_out..(c_idx + 1) * hw_out];
        process_channel(in_ch, out_ch);
    }
}
