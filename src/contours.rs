// Pure Rust contour detection implementation
// This module provides OpenCV-like contour detection for the pure Rust backend

use image::GrayImage;

#[derive(Clone, Copy)]
struct ImageDims {
    width: i32,
    height: i32,
    stride: usize,
}

#[derive(Debug, Clone)]
pub struct Contour {
    pub points: Vec<(i32, i32)>,
    pub area: usize,
    pub score: f32,
}

impl Contour {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            area: 0,
            score: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl Default for Contour {
    fn default() -> Self {
        Self::new()
    }
}

/// Find contours in a binary image
/// Extracts boundary pixels and component area in pure Rust
pub fn find_contours(binary_img: &GrayImage) -> Vec<Contour> {
    find_contours_with_pred(binary_img, None)
}

/// Find contours in a binary image, optionally calculating mean probability score from prediction map
pub fn find_contours_with_pred(
    binary_img: &GrayImage,
    pred: Option<&ndarray::Array4<f32>>,
) -> Vec<Contour> {
    let (width, height) = binary_img.dimensions();
    let w_usize = width as usize;
    let h_usize = height as usize;
    let mut label_map = vec![0u32; w_usize * h_usize];
    let mut component_areas = Vec::with_capacity(64);
    let mut label = 1u32;

    let w_i32 = width as i32;
    let h_i32 = height as i32;

    let dims = ImageDims {
        width: w_i32,
        height: h_i32,
        stride: w_usize,
    };

    // First pass: label connected components using flood fill and record component area
    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize) * w_usize + (x as usize);

            if binary_img.get_pixel(x, y)[0] > 127 && label_map[idx] == 0 {
                let area = flood_fill_label(
                    binary_img,
                    &mut label_map,
                    (x as i32, y as i32),
                    label,
                    dims,
                );
                component_areas.push(area);
                label += 1;
            }
        }
    }

    let num_labels = (label - 1) as usize;
    if num_labels == 0 {
        return Vec::new();
    }

    // Second pass: single O(W x H) pass collecting boundary pixels and score sums for each label
    let mut boundary_buckets = vec![Vec::with_capacity(64); num_labels];
    let mut score_sums = vec![0.0f32; num_labels];
    for y in 0..dims.height {
        for x in 0..dims.width {
            let idx = (y as usize) * dims.stride + (x as usize);
            let lbl = label_map[idx];
            if lbl > 0 {
                if let Some(p) = pred {
                    score_sums[(lbl - 1) as usize] += p[[0, 0, y as usize, x as usize]];
                }
                if is_boundary_pixel_label(&label_map, x, y, lbl, dims) {
                    boundary_buckets[(lbl - 1) as usize].push((x, y));
                }
            }
        }
    }

    let mut contours = Vec::with_capacity(num_labels);
    for (lbl_idx, boundary_pixels) in boundary_buckets.into_iter().enumerate() {
        if boundary_pixels.len() >= 3 {
            let area = component_areas[lbl_idx];
            let score = if area > 0 && pred.is_some() {
                score_sums[lbl_idx] / area as f32
            } else {
                0.0
            };
            contours.push(Contour {
                points: boundary_pixels,
                area,
                score,
            });
        }
    }

    contours
}

/// Flood fill to label a connected component, returning total pixel count
fn flood_fill_label(
    img: &GrayImage,
    labels: &mut [u32],
    start: (i32, i32),
    label: u32,
    dims: ImageDims,
) -> usize {
    let mut stack = Vec::with_capacity(256);
    stack.push(start);
    let mut count = 0usize;

    while let Some((x, y)) = stack.pop() {
        if x < 0 || x >= dims.width || y < 0 || y >= dims.height {
            continue;
        }

        let idx = (y as usize) * dims.stride + (x as usize);

        if labels[idx] != 0 {
            continue;
        }

        let pixel = img.get_pixel(x as u32, y as u32)[0];
        if pixel <= 127 {
            continue;
        }

        labels[idx] = label;
        count += 1;

        // 4-connected neighbors
        if x + 1 < dims.width {
            stack.push((x + 1, y));
        }
        if x > 0 {
            stack.push((x - 1, y));
        }
        if y + 1 < dims.height {
            stack.push((x, y + 1));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
    }

    count
}

/// Check if a pixel is on the boundary of its labeled region
fn is_boundary_pixel_label(
    label_map: &[u32],
    x: i32,
    y: i32,
    target_label: u32,
    dims: ImageDims,
) -> bool {
    // Check 4-connected neighbors
    for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
        let nx = x + dx;
        let ny = y + dy;

        if nx < 0 || nx >= dims.width || ny < 0 || ny >= dims.height {
            return true;
        }

        if label_map[(ny as usize) * dims.stride + (nx as usize)] != target_label {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_find_contours_simple() {
        let mut img = GrayImage::new(10, 10);

        // Draw a simple 6x6 rectangle (area = 36 pixels)
        for x in 2..8 {
            for y in 2..8 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        let contours = find_contours(&img);
        assert!(!contours.is_empty(), "Should find at least one contour");
        assert_eq!(contours[0].area, 36, "Area should match foreground pixel count");
    }
}
